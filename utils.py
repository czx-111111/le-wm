import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def aggregate_dct(dcts):
    """Stack a list of dicts into a dict of stacked arrays/tensors."""
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct


class _CompatResize(dt.transforms.Resize):
    """SPT Resize calls ``self.transform``; torchvision>=0.20 renamed it to ``_transform``."""

    def __call__(self, x):
        img = self.nested_get(x, self.source)
        if hasattr(self, "transform"):
            out = self.transform(img, [])
        else:
            params = self.make_params([]) if hasattr(self, "make_params") else {}
            out = self._transform(img, params)
        self.nested_set(x, out, self.target)
        return x


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """ImageNet-normalize CHW tensors from HDF5 (already permuted, often float 0-255).

    Avoids SPT ``ToImage``+``Resize`` which break on torchvision>=0.20 and on
    pre-permuted CHW batches.
    """
    stats = dt.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)

    def _norm(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        # HDF5 path already returns (T, C, H, W); keep a HWC fallback just in case.
        if x.ndim >= 3 and x.shape[-1] in (1, 3) and x.shape[-3] not in (1, 3):
            x = x.movedim(-1, -3)
        if img_size is not None and tuple(x.shape[-2:]) != (img_size, img_size):
            batched = x.ndim == 4
            y = x if batched else x.unsqueeze(0)
            y = torch.nn.functional.interpolate(
                y, size=(img_size, img_size), mode="bilinear", align_corners=False
            )
            x = y if batched else y.squeeze(0)
        if x.max() > 1.5:
            x = x / 255.0
        m = mean.to(device=x.device, dtype=x.dtype)
        s = std.to(device=x.device, dtype=x.dtype)
        if x.ndim == 3:
            m, s = m[0], s[0]
        return (x - m) / s

    return dt.transforms.WrapTorchTransform(_norm, source=source, target=target)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Periodically pickle the model; also keep the best-by-val object ckpt."""

    def __init__(
        self,
        dirpath,
        filename="model_object",
        epoch_interval: int = 1,
        monitor: str = "validate/pred_loss",
        mode: str = "min",
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = max(1, int(epoch_interval))
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self._better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def _metric_candidates(self):
        return (
            self.monitor,
            f"{self.monitor}_epoch",
            "validate/pred_loss",
            "validate/pred_loss_epoch",
        )

    def _read_score(self, trainer):
        metrics = trainer.callback_metrics
        for key in self._metric_candidates():
            if key not in metrics:
                continue
            value = metrics[key]
            return float(value.item() if hasattr(value, "item") else value)
        return None

    def _seed_best_score_from_trainer(self, trainer):
        """After resume, restore best_score so a worse sanity-val cannot clobber best_object."""
        if self.best_score is not None:
            return
        score_path = self.dirpath / f"{self.filename}_best_score.txt"
        if score_path.is_file():
            try:
                self.best_score = float(score_path.read_text().strip())
                print(
                    f"[ModelObjectCallBack] seeded best_score={self.best_score:.6f} "
                    f"from {score_path.name}"
                )
                return
            except Exception:
                pass
        for cb in trainer.callbacks:
            best = getattr(cb, "best_model_score", None)
            if best is None:
                continue
            try:
                self.best_score = float(best.item() if hasattr(best, "item") else best)
                print(
                    f"[ModelObjectCallBack] seeded best_score={self.best_score:.6f} "
                    f"from {type(cb).__name__}"
                )
                return
            except Exception:
                continue

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._seed_best_score_from_trainer(trainer)
        score = self._read_score(trainer)
        if score is None:
            return
        if self.best_score is not None and not self._better(score, self.best_score):
            return
        self.best_score = score
        best_path = self.dirpath / f"{self.filename}_best_object.ckpt"
        self._dump_model(pl_module.model, best_path)
        score_path = self.dirpath / f"{self.filename}_best_score.txt"
        try:
            score_path.write_text(f"{score:.8f}\n")
        except Exception:
            pass
        print(
            f"[ModelObjectCallBack] saved best object "
            f"(epoch={trainer.current_epoch + 1}, {self.monitor}={score:.6f}) -> {best_path}"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch_1based = trainer.current_epoch + 1
        output_path = self.dirpath / f"{self.filename}_epoch_{epoch_1based}_object.ckpt"

        if not trainer.is_global_zero:
            return
        if epoch_1based % self.epoch_interval == 0 or epoch_1based == trainer.max_epochs:
            self._dump_model(pl_module.model, output_path)
            # 训练结束再存一份固定名，正式评估用最后一轮（best 仍单独保留）
            if epoch_1based == trainer.max_epochs:
                last_path = self.dirpath / f"{self.filename}_last_object.ckpt"
                self._dump_model(pl_module.model, last_path)
                print(
                    f"[ModelObjectCallBack] saved last object "
                    f"(epoch={epoch_1based}) -> {last_path}"
                )

    def _dump_model(self, model, path):
        try:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


def load_lewm_weights_into_model(model, ckpt_path: str):
    """Load Lightning weights / state_dict / pickled JEPA into ``model``.

    Returns the model to use (may be the pickled module itself).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ModelObjectCallBack saves the raw JEPA module
    if isinstance(checkpoint, torch.nn.Module):
        print(f"[INFO] loaded pickled model object from {ckpt_path}")
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)} ({ckpt_path})")

    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k[6:]] = v
        elif k.startswith("module.model."):
            model_state_dict[k[13:]] = v
        else:
            model_state_dict[k] = v

    result = model.load_state_dict(model_state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARN] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:8]}...")
    if result.unexpected_keys:
        print(
            f"[WARN] Unexpected keys ({len(result.unexpected_keys)}): "
            f"{result.unexpected_keys[:8]}..."
        )
    return model


def build_lewm(
    *,
    action_dim: int,
    history_size: int = 3,
    embed_dim: int = 192,
    encoder_scale: str = "tiny",
    patch_size: int = 14,
    img_size: int = 224,
    predictor_cfg=None,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
    predictor_emb_dropout: float = 0.0,
    smoothed_dim: int = 10,
    token_mode: str = "cls",
    num_spatial_tokens: int = 1,
    spatial_grid: int = 4,
    cost_mode: str = "dense_terminal",
    m_task_threshold=0.08,
    m_task_mode: str = "eps",
    m_task_topk: int = 4,
    m_task_assume_normalized: bool = True,
    pool_mode: str = "avg",
    include_global_token: bool = False,
    pos_mode: str = "learned",
    inject_every_layer: bool = False,
    n_lo_mode: str = "attn",
    use_relative_bias: bool = True,
    abs_mode: str = "shared",
    locality: str = "none",
    locality_sigma_init: float = 2.0,
    locality_sigma_min: float = 1.0,
    halo: int = 1,
    fine_mode: str = "discrete",
    softargmax_temp: float = 1.0,
    fine_fourier_freqs: int = 8,
    fine_conf_gate: bool = True,
    pos_prop_enabled: bool = False,
    pos_prop_weight: float = 0.01,
):
    """Construct a JEPA world model (CLS / C1 / S1 query / S2 NibblePE)."""
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size

    pred_kwargs = {
        "depth": predictor_depth,
        "heads": predictor_heads,
        "mlp_dim": predictor_mlp_dim,
        "dim_head": predictor_dim_head,
        "dropout": predictor_dropout,
        "emb_dropout": predictor_emb_dropout,
    }
    if predictor_cfg:
        pred_kwargs.update(predictor_cfg)

    n_spatial = int(num_spatial_tokens)
    if token_mode in ("spatial16", "spatial"):
        n_spatial = max(n_spatial, spatial_grid * spatial_grid)
    include_global = bool(include_global_token) and token_mode in ("spatial16", "spatial")
    n_seq = n_spatial + int(include_global) if token_mode in ("spatial16", "spatial") else 1

    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_spatial_tokens=n_seq,
        pos_mode=pos_mode,
        include_global_token=include_global,
        inject_every_layer=bool(inject_every_layer),
        n_lo_mode=str(n_lo_mode),
        use_relative_bias=bool(use_relative_bias),
        abs_mode=str(abs_mode),
        spatial_grid=spatial_grid,
        fine_mode=str(fine_mode),
        fine_fourier_freqs=int(fine_fourier_freqs),
        fine_conf_gate=bool(fine_conf_gate),
        **pred_kwargs,
    )

    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=smoothed_dim,
        emb_dim=embed_dim,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        token_mode=token_mode,
        num_spatial_tokens=n_spatial if token_mode in ("spatial16", "spatial") else 1,
        spatial_grid=spatial_grid,
        cost_mode=cost_mode,
        m_task_threshold=m_task_threshold,
        m_task_mode=m_task_mode,
        m_task_topk=m_task_topk,
        m_task_assume_normalized=m_task_assume_normalized,
        pool_mode=pool_mode,
        include_global_token=include_global,
        pos_mode=pos_mode,
        n_lo_mode=str(n_lo_mode),
        encoder_hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        locality=str(locality),
        locality_sigma_init=float(locality_sigma_init),
        locality_sigma_min=float(locality_sigma_min),
        halo=int(halo),
        fine_mode=str(fine_mode),
        softargmax_temp=float(softargmax_temp),
        fine_fourier_freqs=int(fine_fourier_freqs),
        fine_conf_gate=bool(fine_conf_gate),
        pos_prop_enabled=bool(pos_prop_enabled),
        pos_prop_weight=float(pos_prop_weight),
    )