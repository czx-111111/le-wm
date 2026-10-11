import math
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

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from einops import rearrange

from module import SIGReg
from utils import (
    build_lewm,
    get_column_normalizer,
    get_img_preprocessor,
    ModelObjectCallBack,
)


def _cfg_get(cfg, path, default=None):
    """OmegaConf-safe nested get with a default."""
    if OmegaConf.is_config(cfg):
        val = OmegaConf.select(cfg, path, default=None)
        return default if val is None else val
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return default if cur is None else cur


def _gather_sigreg_batch(sigreg_in: torch.Tensor) -> torch.Tensor:
    """DDP: gather batch dim so SIGReg sees global B."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            parts = torch.distributed.nn.functional.all_gather(sigreg_in.contiguous())
            return torch.cat(parts, dim=1)
    return sigreg_in


def _rearrange_sigreg(emb: torch.Tensor, per_token: bool) -> torch.Tensor:
    """emb (B,T,N,D) or (B,T,D) -> SIGReg layout (T' , B', D)."""
    if emb.ndim == 4:
        if per_token:
            return rearrange(emb, "b t n d -> (t n) b d")
        return rearrange(emb, "b t n d -> t (b n) d")
    return emb.transpose(0, 1)


def _token_balance_mse(pred, tgt, cfg, module, log_prefix=None):
    """MSE with optional CLS/slot group balance (stage-2 C2).

    Defaults match the old dense mean over all tokens.
    """
    mode = str(_cfg_get(cfg, "loss.token_balance.mode", "uniform"))
    alpha = float(_cfg_get(cfg, "loss.token_balance.alpha", 0.5))
    ema = float(_cfg_get(cfg, "loss.token_balance.ema", 0.99))

    # CLS / single-token: nothing to balance
    if pred.ndim != 4 or pred.size(2) < 2 or mode == "uniform":
        return (pred - tgt).pow(2).mean()

    err = (pred - tgt).pow(2)
    l_cls = err[:, :, 0].mean()
    l_slot = err[:, :, 1:].mean()

    if mode == "group_varnorm":
        with torch.no_grad():
            v_cls = tgt[:, :, 0].float().var(unbiased=False).clamp_min(1e-6)
            v_slot = tgt[:, :, 1:].float().var(unbiased=False).clamp_min(1e-6)
            if not hasattr(module, "_tb_ema_cls"):
                module._tb_ema_cls = v_cls.detach().clone()
                module._tb_ema_slot = v_slot.detach().clone()
            else:
                if module._tb_ema_cls.device != v_cls.device:
                    module._tb_ema_cls = module._tb_ema_cls.to(v_cls.device)
                    module._tb_ema_slot = module._tb_ema_slot.to(v_slot.device)
                module._tb_ema_cls.mul_(ema).add_(v_cls, alpha=1.0 - ema)
                module._tb_ema_slot.mul_(ema).add_(v_slot, alpha=1.0 - ema)
        l_cls = l_cls / module._tb_ema_cls.to(dtype=l_cls.dtype).clamp_min(1e-6)
        l_slot = l_slot / module._tb_ema_slot.to(dtype=l_slot.dtype).clamp_min(1e-6)
    elif mode != "group":
        raise ValueError(f"Unknown loss.token_balance.mode={mode!r}")

    loss = alpha * l_cls + (1.0 - alpha) * l_slot
    if log_prefix is not None:
        module.log(
            f"{log_prefix}/pred_loss_cls",
            l_cls.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        module.log(
            f"{log_prefix}/pred_loss_slot",
            l_slot.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
    return loss


def _teacher_force_pred_loss(
    self, emb, act_emb, n_hi, n_lo, ctx_len, n_preds, cfg, stage, mu=None, conf=None
):
    """Current LeWM teacher-forcing loss (rollout_k == 1)."""
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    hi_ctx = n_hi[:, :ctx_len] if n_hi is not None else None
    lo_ctx = n_lo[:, :ctx_len] if n_lo is not None else None
    mu_ctx = mu[:, :ctx_len] if mu is not None else None
    conf_ctx = conf[:, :ctx_len] if conf is not None else None
    pred_emb = self.model.predict(
        ctx_emb, ctx_act, n_hi=hi_ctx, n_lo=lo_ctx, mu=mu_ctx, conf=conf_ctx
    )
    return _token_balance_mse(pred_emb, tgt_emb, cfg, self, log_prefix=stage), pred_emb


def _rollout_pred_loss(
    self, emb, act_emb, n_hi, n_lo, ctx_len, rollout_k, cfg, stage, mu=None, conf=None
):
    """k-step autoregressive latent rollout loss (averaged over steps)."""
    assert emb.size(1) >= ctx_len + rollout_k, (
        f"need history+k={ctx_len + rollout_k} frames, got T={emb.size(1)}"
    )
    emb_ar = emb[:, :ctx_len]
    act_ar = act_emb[:, :ctx_len]
    hi_ar = n_hi[:, :ctx_len] if n_hi is not None else None
    lo_ar = n_lo[:, :ctx_len] if n_lo is not None else None
    mu_ar = mu[:, :ctx_len] if mu is not None else None
    conf_ar = conf[:, :ctx_len] if conf is not None else None

    step_losses = []
    for t in range(rollout_k):
        hi_trunc = hi_ar[:, -ctx_len:] if hi_ar is not None else None
        lo_trunc = lo_ar[:, -ctx_len:] if lo_ar is not None else None
        mu_trunc = mu_ar[:, -ctx_len:] if mu_ar is not None else None
        conf_trunc = conf_ar[:, -ctx_len:] if conf_ar is not None else None
        pred = self.model.predict(
            emb_ar[:, -ctx_len:],
            act_ar[:, -ctx_len:],
            n_hi=hi_trunc,
            n_lo=lo_trunc,
            mu=mu_trunc,
            conf=conf_trunc,
        )[:, -1:]
        tgt = emb[:, ctx_len + t : ctx_len + t + 1]
        step_losses.append(
            _token_balance_mse(pred, tgt, cfg, self, log_prefix=None)
        )
        emb_ar = torch.cat([emb_ar, pred], dim=1)
        # action at the predicted frame (same convention as JEPA.rollout)
        act_ar = torch.cat([act_ar, act_emb[:, ctx_len + t : ctx_len + t + 1]], dim=1)
        if hi_ar is not None:
            hi_ar = torch.cat([hi_ar, hi_ar[:, -1:]], dim=1)
            lo_ar = torch.cat([lo_ar, lo_ar[:, -1:]], dim=1)
        if mu_ar is not None:
            mu_ar = torch.cat([mu_ar, mu_ar[:, -1:]], dim=1)
        if conf_ar is not None:
            conf_ar = torch.cat([conf_ar, conf_ar[:, -1:]], dim=1)

    loss = torch.stack(step_losses).mean()
    self.log(
        f"{stage}/rollout_k",
        float(rollout_k),
        on_step=False,
        on_epoch=True,
        sync_dist=True,
    )
    return loss


def _sigreg_loss(self, emb, cfg):
    """SIGReg; optional CLS/slot grouping (stage-2 C2)."""
    per_token = bool(_cfg_get(cfg, "loss.sigreg.per_token", False))
    grouped = bool(_cfg_get(cfg, "loss.sigreg.grouped", False))
    fp32 = bool(_cfg_get(cfg, "loss.sigreg.fp32", False))
    lambd = float(_cfg_get(cfg, "loss.sigreg.weight", 0.09))

    def _run(sig_in):
        sig_in = _gather_sigreg_batch(sig_in)
        if fp32:
            with torch.autocast(device_type=emb.device.type, enabled=False):
                return self.sigreg(sig_in.float())
        return self.sigreg(sig_in)

    # Grouped only when we actually have a CLS + slots layout
    if grouped and emb.ndim == 4 and emb.size(2) >= 2:
        w_cls = float(_cfg_get(cfg, "loss.sigreg.cls_weight", lambd))
        w_slot = float(_cfg_get(cfg, "loss.sigreg.slot_weight", lambd))
        cls_loss = _run(_rearrange_sigreg(emb[:, :, :1], per_token))
        slot_loss = _run(_rearrange_sigreg(emb[:, :, 1:], per_token))
        # Log unweighted components; total mixes with separate weights.
        return (
            w_cls * cls_loss + w_slot * slot_loss,
            cls_loss,
            slot_loss,
            0.0,  # already weighted into returned loss; outer λ unused
        )

    sig = _run(_rearrange_sigreg(emb, per_token))
    return lambd * sig, sig, None, lambd


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    rollout_k = int(_cfg_get(cfg, "loss.rollout_k", 1))
    rollout_weight = float(_cfg_get(cfg, "loss.rollout_weight", 1.0))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D) or (B, T, N, D)
    act_emb = output["act_emb"]
    n_hi = output.get("n_hi")
    n_lo = output.get("n_lo")
    mu = output.get("mu")
    conf = output.get("conf")
    pred_emb = None

    if rollout_k <= 1:
        output["pred_loss"], pred_emb = _teacher_force_pred_loss(
            self, emb, act_emb, n_hi, n_lo, ctx_len, n_preds, cfg, stage, mu=mu, conf=conf
        )
    else:
        # k-step AR latent loss; data window must be history + k (wm.num_preds >= k)
        ar_loss = _rollout_pred_loss(
            self, emb, act_emb, n_hi, n_lo, ctx_len, rollout_k, cfg, stage, mu=mu, conf=conf
        )
        output["pred_loss"] = rollout_weight * ar_loss

    weighted_sig, sig_raw, sig_slot, _ = _sigreg_loss(self, emb, cfg)
    output["sigreg_loss"] = sig_raw
    if sig_slot is not None:
        output["sigreg_slot_loss"] = sig_slot
    output["loss"] = output["pred_loss"] + weighted_sig

    # Stage-3: position propagation aux loss (detach predicted emb → next-frame μ)
    pos_prop_on = bool(_cfg_get(cfg, "wm.pos_prop.enabled", False)) or bool(
        getattr(self.model, "pos_prop_enabled", False)
    )
    if (
        pos_prop_on
        and pred_emb is not None
        and mu is not None
        and getattr(self.model, "pos_prop_head", None) is not None
    ):
        tgt_mu = mu[:, n_preds:]
        pred_mu = self.model.predict_mu_from_emb(pred_emb.detach())
        # align lengths (teacher-forcing window)
        t = min(pred_mu.size(1), tgt_mu.size(1))
        w = float(_cfg_get(cfg, "wm.pos_prop.weight", getattr(self.model, "pos_prop_weight", 0.01)))
        output["pos_prop_loss"] = (pred_mu[:, :t] - tgt_mu[:, :t]).pow(2).mean()
        output["loss"] = output["loss"] + w * output["pos_prop_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)

    # spt Module sums micro-batch grads without rescaling; average them here.
    accum = int(cfg.trainer.get("accumulate_grad_batches", 1))
    if stage == "fit" and accum > 1:
        output["loss"] = output["loss"] / accum
    return output


class ResumeCheckpoint(Callback):
    """Full Lightning state (weights, optimizer, scheduler, loops) every epoch.

    spt's Manager redirects every ``ModelCheckpoint`` into its cache dir and
    deletes it after a successful run, so resume needs its own file.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        tmp = self.path.with_name(self.path.name + ".tmp")
        trainer.save_checkpoint(str(tmp))
        if trainer.is_global_zero:
            os.replace(tmp, self.path)


def optimizer_steps_total(cfg, train_loader) -> int:
    """Real optimizer steps for the whole run.

    spt's manual-optimization patch resets ``trainer.accumulate_grad_batches``
    to 1, so ``estimated_stepping_batches`` over-counts by the accumulation
    factor and the default cosine schedule never decays.
    """
    devices = cfg.trainer.get("devices", 1)
    if isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices):
        world = len(devices)
    else:
        try:
            world = int(devices)
        except (TypeError, ValueError):
            world = 1
    batches = math.ceil(len(train_loader) / max(world, 1))
    limit = cfg.trainer.get("limit_train_batches", None)
    if isinstance(limit, int) and not isinstance(limit, bool):
        batches = min(batches, limit)
    elif isinstance(limit, float) and limit < 1.0:
        batches = int(batches * limit)
    accum = int(cfg.trainer.get("accumulate_grad_batches", 1))
    return max(1, (batches // accum) * int(cfg.trainer.max_epochs))

@hydra.main(version_base=None, config_path="./config/train", config_name="r3/rope_cls")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    # Stage-2: k-step AR needs history + k frames in the HDF5 window.
    rollout_k = int(_cfg_get(cfg, "loss.rollout_k", 1))
    if rollout_k > int(cfg.wm.num_preds):
        with open_dict(cfg):
            cfg.wm.num_preds = rollout_k
        print(f"[loss] rollout_k={rollout_k} → wm.num_preds={cfg.wm.num_preds}")

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
    pos_prop_cfg = cfg.wm.get("pos_prop") or {}
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
        locality=str(cfg.wm.get("locality", "none")),
        locality_sigma_init=float(cfg.wm.get("locality_sigma_init", 2.0)),
        locality_sigma_min=float(cfg.wm.get("locality_sigma_min", 1.0)),
        halo=int(cfg.wm.get("halo", 1)),
        fine_mode=str(cfg.wm.get("fine_mode", "discrete")),
        softargmax_temp=float(cfg.wm.get("softargmax_temp", 1.0)),
        fine_fourier_freqs=int(cfg.wm.get("fine_fourier_freqs", 8)),
        fine_conf_gate=bool(cfg.wm.get("fine_conf_gate", True)),
        pos_prop_enabled=bool(pos_prop_cfg.get("enabled", False)),
        pos_prop_weight=float(pos_prop_cfg.get("weight", 0.01)),
    )
    if bool(cfg.wm.get("encoder_grad_ckpt", False)):
        world_model.encoder.gradient_checkpointing_enable()

    total_steps = optimizer_steps_total(cfg, train)
    warmup_frac = float(cfg.get("scheduler", {}).get("warmup_frac", 0.01))
    print(f"[sched] optimizer steps total={total_steps} warmup_frac={warmup_frac}")
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(warmup_frac * total_steps)),
                "max_steps": total_steps,
            },
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

    weights_ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    resume_ckpt = weights_ckpt.is_file()
    if resume_ckpt:
        print(f"[resume] full state from {weights_ckpt}")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, best_ckpt_callback, ResumeCheckpoint(weights_ckpt)],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        # Only resume when the weights file already exists; otherwise train
        # from scratch (Manager refuses a missing ckpt_path).
        ckpt_path=str(weights_ckpt.resolve()) if resume_ckpt else None,
        # Full Lightning state (optimizer / scheduler / epoch / RNG). Default
        # Manager weights_only=True would only reload model weights and change
        # the training trajectory after an interrupt.
        weights_only=False if resume_ckpt else True,
    )

    manager()
    return


if __name__ == "__main__":
    run()
