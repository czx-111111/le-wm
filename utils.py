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


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


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

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        score = self._read_score(trainer)
        if score is None:
            return
        if self.best_score is not None and not self._better(score, self.best_score):
            return
        self.best_score = score
        best_path = self.dirpath / f"{self.filename}_best_object.ckpt"
        self._dump_model(pl_module.model, best_path)
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
):
    """Construct a JEPA world model (CLS or C1 spatial-16)."""
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

    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_spatial_tokens=n_spatial if token_mode in ("spatial16", "spatial") else 1,
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
    )