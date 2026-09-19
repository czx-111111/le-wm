import os
from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from lightning.pytorch.callbacks import ModelCheckpoint

from einops import rearrange

from module import SIGReg
from utils import (
    build_lewm,
    get_column_normalizer,
    get_img_preprocessor,
    ModelObjectCallBack,
)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D) or (B, T, N, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    n_hi = output.get("n_hi")
    n_lo = output.get("n_lo")
    hi_ctx = n_hi[:, :ctx_len] if n_hi is not None else None
    lo_ctx = n_lo[:, :ctx_len] if n_lo is not None else None
    pred_emb = self.model.predict(ctx_emb, ctx_act, n_hi=hi_ctx, n_lo=lo_ctx)

    # LeWM loss (ordinary token MSE — dense over all spatial tokens when N>1)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    if emb.ndim == 4:
        # SIGReg expects (T, batch', D); flatten spatial into batch
        sigreg_in = rearrange(emb, "b t n d -> t (b n) d")
    else:
        sigreg_in = emb.transpose(0, 1)
    output["sigreg_loss"] = self.sigreg(sigreg_in)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    #########################
    ##       dataset       ##
    #########################

    dataset = instantiate(cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    plan_cfg = cfg.get("plan") or {}
    world_model = build_lewm(
        action_dim=effective_act_dim,
        history_size=cfg.wm.history_size,
        embed_dim=cfg.wm.get("embed_dim", 192),
        encoder_scale=cfg.encoder_scale,
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        predictor_cfg=dict(cfg.predictor),
        token_mode=cfg.wm.get("token_mode", "cls"),
        num_spatial_tokens=int(cfg.wm.get("num_spatial_tokens", 1)),
        spatial_grid=int(cfg.wm.get("spatial_grid", 4)),
        cost_mode=str(plan_cfg.get("cost_mode", "dense_terminal")),
        m_task_threshold=plan_cfg.get("m_task_threshold", 0.08),
        m_task_mode=str(plan_cfg.get("m_task_mode", "eps")),
        m_task_topk=int(plan_cfg.get("m_task_topk", 4)),
        m_task_assume_normalized=bool(plan_cfg.get("m_task_assume_normalized", True)),
        pool_mode=str(cfg.wm.get("pool_mode", "avg")),
        include_global_token=bool(cfg.wm.get("include_global_token", False)),
        pos_mode=str(cfg.wm.get("pos_mode", "learned")),
        inject_every_layer=bool(cfg.wm.get("inject_every_layer", False)),
        n_lo_mode=str(cfg.wm.get("n_lo_mode", "attn")),
        use_relative_bias=bool(cfg.wm.get("use_relative_bias", True)),
        abs_mode=str(cfg.wm.get("abs_mode", "shared")),
    )
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    ckpt_cfg = cfg.get("checkpoint") or {}
    monitor = str(ckpt_cfg.get("monitor", "validate/pred_loss"))
    mode = str(ckpt_cfg.get("mode", "min"))
    save_top_k = int(ckpt_cfg.get("save_top_k", 1))

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.get("save_object_every_n_epochs", 1)),
        monitor=monitor,
        mode=mode,
    )

    best_ckpt_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename=f"{cfg.output_model_name}_best",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, best_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    weights_ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        # Only resume when the weights file already exists; otherwise train
        # from scratch (Manager refuses a missing ckpt_path).
        ckpt_path=str(weights_ckpt.resolve()) if weights_ckpt.is_file() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
