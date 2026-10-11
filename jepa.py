"""JEPA Implementation"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def denormalize_imagenet(pixels: torch.Tensor) -> torch.Tensor:
    """Map ImageNet-normalized CHW tensors back to approximately [0, 1] RGB."""
    while pixels.ndim > 4:
        pixels = pixels[:, 0]
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (pixels.float() * std + mean).clamp(0.0, 1.0)


def sincos_2d_pe(grid_h: int, grid_w: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """(grid_h * grid_w, dim) 2D sine-cosine positional encoding."""
    if dim % 4 != 0:
        raise ValueError(f"sincos PE dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    out_y = torch.einsum("h,d->hd", y.reshape(-1), omega)
    out_x = torch.einsum("w,d->wd", x.reshape(-1), omega)
    pe = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    if dtype is not None:
        pe = pe.to(dtype=dtype)
    return pe


def nibble_codes_from_attn(
    attn: torch.Tensor,
    spatial_grid: int = 4,
    fine_grid: int = 16,
    softargmax_temp: float = 1.0,
):
    """Quantize per-query attention into radix-16 nibble codes.

    attn: (BT, N, P) over a square patch grid.
    Returns n_hi, n_lo, n_hi_content, mu (BT,N,2) in [0,cell), conf (BT,N).
    """
    bt, n, p = attn.shape
    side = int(math.sqrt(p))
    if side * side != p:
        raise ValueError(f"patch count {p} is not a square grid")
    attn_map = attn.reshape(bt, n, side, side)
    attn_fine = F.interpolate(
        attn_map, size=(fine_grid, fine_grid), mode="bilinear", align_corners=False
    )
    g = int(spatial_grid)
    cell = fine_grid // g
    device = attn.device
    n_hi = torch.arange(n, device=device).view(1, n).expand(bt, -1)

    windows = []
    for i in range(n):
        row, col = divmod(i, g)
        windows.append(
            attn_fine[:, i, row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
        )
    window = torch.stack(windows, dim=1)  # (BT, N, cell, cell)
    flat = window.flatten(2)
    mass = flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    temp = max(float(softargmax_temp), 1e-4)
    # temperature on window mass (soft-argmax); temp=1 recovers mass-normalized centroid
    prob = (flat / mass).clamp_min(1e-8)
    prob = (prob.log() / temp).softmax(dim=-1).view(bt, n, cell, cell)
    ys = torch.arange(cell, device=device, dtype=prob.dtype).view(1, 1, cell, 1)
    xs = torch.arange(cell, device=device, dtype=prob.dtype).view(1, 1, 1, cell)
    cy = (prob * ys).sum(dim=(2, 3))
    cx = (prob * xs).sum(dim=(2, 3))
    mu = torch.stack([cy, cx], dim=-1)
    n_lo = (
        cy.round().long().clamp(0, cell - 1) * cell
        + cx.round().long().clamp(0, cell - 1)
    )

    fine_mass = attn_fine.flatten(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # confidence = share of attention mass inside the slot window
    conf = (mass.squeeze(-1) / fine_mass.squeeze(-1)).clamp(0.0, 1.0)
    fine_prob = (attn_fine.flatten(2) / fine_mass).view(bt, n, fine_grid, fine_grid)
    fy = torch.arange(fine_grid, device=device, dtype=fine_prob.dtype).view(
        1, 1, fine_grid, 1
    )
    fx = torch.arange(fine_grid, device=device, dtype=fine_prob.dtype).view(
        1, 1, 1, fine_grid
    )
    gy = (fine_prob * fy).sum(dim=(2, 3))
    gx = (fine_prob * fx).sum(dim=(2, 3))
    hi_r = (gy / cell).floor().long().clamp(0, g - 1)
    hi_c = (gx / cell).floor().long().clamp(0, g - 1)
    n_hi_content = hi_r * g + hi_c
    return n_hi, n_lo, n_hi_content, mu, conf


class SpatialQueryPool(nn.Module):
    """Single-head cross-attention from a 4x4 query grid onto ViT patches."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 16,
        spatial_grid: int = 4,
        locality: str = "none",
        locality_sigma_init: float = 2.0,
        locality_sigma_min: float = 1.0,
        halo: int = 1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.spatial_grid = int(spatial_grid)
        self.locality = str(locality)
        self.locality_sigma_min = float(locality_sigma_min)
        self.halo = int(halo)
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, self.hidden_dim))
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale = self.hidden_dim**-0.5
        # learnable log-σ (patch units); unused when locality == none
        init = max(float(locality_sigma_init), float(locality_sigma_min), 1e-3)
        self.log_sigma = nn.Parameter(
            torch.full((self.num_queries,), math.log(init), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self):
        pe = sincos_2d_pe(self.spatial_grid, self.spatial_grid, self.hidden_dim)
        with torch.no_grad():
            self.query.copy_(pe.unsqueeze(0))
            self.query.add_(0.02 * torch.randn_like(self.query))

    def _patch_query_geometry(self, num_patches: int, device, dtype):
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(f"patch count {num_patches} is not a square grid")
        g = self.spatial_grid
        cell = side // g
        yy, xx = torch.meshgrid(
            torch.arange(side, device=device, dtype=dtype),
            torch.arange(side, device=device, dtype=dtype),
            indexing="ij",
        )
        patch_xy = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (P, 2)
        centers = []
        for i in range(self.num_queries):
            r, c = divmod(i, g)
            centers.append([r * cell + (cell - 1) * 0.5, c * cell + (cell - 1) * 0.5])
        centers = torch.tensor(centers, device=device, dtype=dtype)  # (N, 2)
        d2 = ((patch_xy[None, :, :] - centers[:, None, :]) ** 2).sum(dim=-1)  # (N, P)
        return d2, side, cell

    def forward(self, patches: torch.Tensor):
        """patches: (BT, P, H) -> slots (BT, N, H), attn (BT, N, P)."""
        bt, p, _ = patches.shape
        q = self.q_proj(self.query.expand(bt, -1, -1))
        k = self.k_proj(patches)
        v = self.v_proj(patches)
        logits = torch.matmul(q, k.transpose(-2, -1)).mul(self.scale)
        locality = str(getattr(self, "locality", "none")).lower()
        if locality in ("gauss", "hard_halo"):
            d2, side, cell = self._patch_query_geometry(p, patches.device, logits.dtype)
            if locality == "gauss":
                sigma = self.log_sigma.to(dtype=logits.dtype).exp().clamp_min(
                    float(getattr(self, "locality_sigma_min", 1.0))
                )
                logits = logits - d2.unsqueeze(0) / (2.0 * sigma.view(1, -1, 1).pow(2))
            else:
                # hard halo: keep window expanded by ``halo`` patches, else −∞
                halo = int(getattr(self, "halo", 1))
                g = self.spatial_grid
                allow = torch.zeros(
                    self.num_queries, p, device=patches.device, dtype=torch.bool
                )
                for i in range(self.num_queries):
                    r, c = divmod(i, g)
                    r0 = max(0, r * cell - halo)
                    r1 = min(side, (r + 1) * cell + halo)
                    c0 = max(0, c * cell - halo)
                    c1 = min(side, (c + 1) * cell + halo)
                    mask_2d = torch.zeros(side, side, device=patches.device, dtype=torch.bool)
                    mask_2d[r0:r1, c0:c1] = True
                    allow[i] = mask_2d.reshape(-1)
                logits = logits.masked_fill(~allow.unsqueeze(0), torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        slots = self.out_proj(torch.matmul(attn, v))
        return slots, attn


def _pixel_cell_diff(
    cur_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    spatial_grid: int = 4,
    assume_normalized: bool = True,
) -> torch.Tensor:
    """(B, N) mean |RGB| diff pooled to spatial_grid^2 cells."""
    while cur_pixels.ndim > 4:
        cur_pixels = cur_pixels[:, 0]
    while goal_pixels.ndim > 4:
        goal_pixels = goal_pixels[:, 0]
    if goal_pixels.size(0) == 1 and cur_pixels.size(0) > 1:
        goal_pixels = goal_pixels.expand(cur_pixels.size(0), -1, -1, -1)

    cur = cur_pixels.float()
    goal = goal_pixels.float()
    if assume_normalized:
        cur = denormalize_imagenet(cur)
        goal = denormalize_imagenet(goal)
    else:
        if cur.max() > 1.5:
            cur = cur / 255.0
            goal = goal / 255.0
        cur = cur.clamp(0.0, 1.0)
        goal = goal.clamp(0.0, 1.0)

    diff = (cur - goal).abs().mean(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(diff, (spatial_grid, spatial_grid))
    return pooled.flatten(1)  # (B, N)


def _otsu_threshold_1d(values: torch.Tensor) -> torch.Tensor:
    """Per-row Otsu threshold. values: (B, N) -> thr (B, 1)."""
    b, n = values.shape
    device = values.device
    thr = torch.empty(b, 1, device=device, dtype=values.dtype)
    for i in range(b):
        v = values[i]
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin < 1e-8:
            thr[i, 0] = v.mean()
            continue
        hist = torch.histc(v, bins=64, min=vmin, max=vmax)
        prob = hist / hist.sum().clamp_min(1e-8)
        omega = torch.cumsum(prob, dim=0)
        centers = torch.linspace(vmin, vmax, 64, device=device, dtype=values.dtype)
        mu = torch.cumsum(prob * centers, dim=0)
        mu_t = mu[-1]
        sigma = (mu_t * omega - mu).pow(2) / (omega * (1.0 - omega)).clamp_min(1e-8)
        thr[i, 0] = centers[int(sigma.argmax())]
    return thr


def soft_weights_from_diff(
    flat: torch.Tensor,
    tau: float | torch.Tensor = 0.08,
    temp: float = 0.02,
    eps: float = 0.1,
) -> torch.Tensor:
    """w_i = ε + (1-ε)·sigmoid((d_i-τ)/T). flat: (B, N)."""
    if not torch.is_tensor(tau):
        tau = flat.new_tensor(float(tau))
    if tau.ndim == 0:
        tau = tau.view(1, 1)
    elif tau.ndim == 1:
        tau = tau.view(-1, 1)
    t = max(float(temp), 1e-6)
    return float(eps) + (1.0 - float(eps)) * torch.sigmoid((flat - tau) / t)


def compute_m_task_mask(
    cur_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    spatial_grid: int = 4,
    threshold: float | str = 0.08,
    mode: str = "eps",
    topk: int = 4,
    assume_normalized: bool = True,
    mask_weight: str = "hard",
    soft_eps: float = 0.1,
    soft_temp: float = 0.02,
    tau_mode: str = "fixed",
    return_diff: bool = False,
):
    """M_task weights over spatial_grid^2 tokens from |cur - goal| pixel diff.

    Preferred: compute on [0, 1] RGB. If ``assume_normalized``, undo ImageNet
    normalize first. ``mode`` (hard path):
      - ``eps``: pool then ``diff > threshold`` (default threshold=0.08)
      - ``topk``: keep the ``topk`` highest-diff cells
      - ``mean``: legacy per-sample mean threshold
    ``tau_mode``: ``fixed`` / ``quantile`` (threshold is the τ) / ``otsu``.
    ``mask_weight``: ``hard`` → {0,1} with all-ones fallback; ``soft`` → sigmoid.
    Returns (B, N) float weights. If ``return_diff``, also returns flat diffs.
    """
    flat = _pixel_cell_diff(
        cur_pixels, goal_pixels, spatial_grid=spatial_grid, assume_normalized=assume_normalized
    )

    tau_mode = str(tau_mode).lower()
    if tau_mode == "otsu":
        thr = _otsu_threshold_1d(flat)
    elif tau_mode == "quantile":
        # caller passes the desired τ via ``threshold`` (already a quantile of d)
        thr = 0.08 if threshold is None else float(threshold)
    else:
        thr = 0.08 if threshold is None else float(threshold)

    if str(mask_weight).lower() == "soft":
        mask = soft_weights_from_diff(flat, tau=thr, temp=soft_temp, eps=soft_eps)
        return (mask, flat) if return_diff else mask

    mode = str(mode).lower()
    if mode == "topk":
        k = max(1, min(int(topk), flat.size(1)))
        _, idx = flat.topk(k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
    elif mode == "mean" or threshold == "mean":
        thr_m = flat.mean(dim=1, keepdim=True)
        mask = (flat > thr_m).float()
    else:
        if torch.is_tensor(thr):
            mask = (flat > thr).float()
        else:
            mask = (flat > float(thr)).float()

    empty = mask.sum(dim=1, keepdim=True) <= 0
    if empty.any():
        mask = torch.where(empty, torch.ones_like(mask), mask)
    return (mask, flat) if return_diff else mask


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        token_mode: str = "cls",
        num_spatial_tokens: int = 1,
        spatial_grid: int = 4,
        cost_mode: str = "dense_terminal",
        m_task_threshold: float | str = 0.08,
        m_task_mode: str = "eps",
        m_task_topk: int = 4,
        m_task_assume_normalized: bool = True,
        pool_mode: str = "avg",
        include_global_token: bool = False,
        pos_mode: str = "learned",
        n_lo_mode: str = "attn",
        encoder_hidden_dim: int | None = None,
        embed_dim: int = 192,
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
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.token_mode = token_mode
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.spatial_grid = int(spatial_grid)
        self.cost_mode = cost_mode
        self.m_task_threshold = m_task_threshold
        self.m_task_mode = m_task_mode
        self.m_task_topk = int(m_task_topk)
        self.m_task_assume_normalized = bool(m_task_assume_normalized)
        self.pool_mode = str(pool_mode)
        self.include_global_token = bool(include_global_token)
        self.pos_mode = str(pos_mode)
        # attn: centroid-quantized low nibble; fixed_center: lock to slot-cell center
        self.n_lo_mode = str(n_lo_mode)
        self.embed_dim = int(embed_dim)
        self.locality = str(locality)
        self.locality_sigma_init = float(locality_sigma_init)
        self.locality_sigma_min = float(locality_sigma_min)
        self.halo = int(halo)
        self.fine_mode = str(fine_mode)
        self.softargmax_temp = float(softargmax_temp)
        self.fine_fourier_freqs = int(fine_fourier_freqs)
        self.fine_conf_gate = bool(fine_conf_gate)
        self.pos_prop_enabled = bool(pos_prop_enabled)
        self.pos_prop_weight = float(pos_prop_weight)

        hidden = encoder_hidden_dim
        if hidden is None:
            hidden = getattr(getattr(encoder, "config", None), "hidden_size", None)
        self.query_pool = None
        if self.pool_mode == "query" and self.num_spatial_tokens > 1:
            self.query_pool = SpatialQueryPool(
                hidden_dim=int(hidden),
                num_queries=self.num_spatial_tokens,
                spatial_grid=self.spatial_grid,
                locality=self.locality,
                locality_sigma_init=self.locality_sigma_init,
                locality_sigma_min=self.locality_sigma_min,
                halo=self.halo,
            )
        self.pos_prop_head = None
        if self.pos_prop_enabled:
            self.pos_prop_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, 2),
            )

    @property
    def use_spatial_tokens(self) -> bool:
        token_mode = getattr(self, "token_mode", "cls")
        n = int(getattr(self, "num_spatial_tokens", 1))
        return token_mode in ("spatial16", "spatial") and n > 1

    @property
    def num_tokens(self) -> int:
        if not self.use_spatial_tokens:
            return 1
        return int(self.num_spatial_tokens) + int(
            getattr(self, "include_global_token", False)
        )

    def _pool_patches_to_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (BT, P, H) -> (BT, N, H) with N = spatial_grid**2."""
        bt, p, h = patch_tokens.shape
        side = int(p**0.5)
        assert side * side == p, f"patch count {p} is not a square grid"
        grid = rearrange(patch_tokens, "bt (h w) d -> bt d h w", h=side, w=side)
        g = int(getattr(self, "spatial_grid", 4))
        pooled = F.adaptive_avg_pool2d(grid, (g, g))
        return rearrange(pooled, "bt d gh gw -> bt (gh gw) d")

    def _dummy_nibble(self, bt: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_hi = torch.arange(n, device=device).view(1, n).expand(bt, -1)
        g = int(self.spatial_grid)
        center = (g // 2) * g + (g // 2)
        n_lo = torch.full((bt, n), center, device=device, dtype=torch.long)
        return n_hi, n_lo, n_hi.clone()

    def _fixed_center_n_lo(self, bt: int, n: int, device) -> torch.Tensor:
        g = int(self.spatial_grid)
        center = (g // 2) * g + (g // 2)
        return torch.full((bt, n), center, device=device, dtype=torch.long)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info['pixels']: (B, T, C, H, W)
        info['action']: (B, T, A) optional
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        # (B, T, C, H, W) -> (B*T, C, H, W)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        hidden = output.last_hidden_state

        if self.use_spatial_tokens:
            cls = hidden[:, :1]
            patch_tokens = hidden[:, 1:]
            attn = None
            pool_mode = getattr(self, "pool_mode", "avg")
            query_pool = getattr(self, "query_pool", None)
            include_global = getattr(self, "include_global_token", False)
            if pool_mode == "query" and query_pool is not None:
                slots, attn = query_pool(patch_tokens)
                n_hi, n_lo, n_hi_content, mu, conf = nibble_codes_from_attn(
                    attn,
                    spatial_grid=self.spatial_grid,
                    softargmax_temp=float(getattr(self, "softargmax_temp", 1.0)),
                )
            else:
                slots = self._pool_patches_to_tokens(patch_tokens)
                n_hi, n_lo, n_hi_content = self._dummy_nibble(
                    slots.size(0), slots.size(1), slots.device
                )
                g = int(self.spatial_grid)
                center = (g // 2) + 0.0
                mu = torch.full(
                    (slots.size(0), slots.size(1), 2),
                    center,
                    device=slots.device,
                    dtype=slots.dtype,
                )
                conf = torch.ones(
                    slots.size(0), slots.size(1), device=slots.device, dtype=slots.dtype
                )
            # Ablation: lock low nibble to cell center (structured slot PE only).
            if getattr(self, "n_lo_mode", "attn") == "fixed_center":
                n_lo = self._fixed_center_n_lo(
                    slots.size(0), slots.size(1), slots.device
                )

            if include_global:
                tokens = torch.cat([cls, slots], dim=1)
            else:
                tokens = slots

            n = tokens.size(1)
            emb = self.projector(rearrange(tokens, "bt n d -> (bt n) d"))
            emb = rearrange(emb, "(bt n) d -> bt n d", n=n)
            info["emb"] = rearrange(emb, "(b t) n d -> b t n d", b=b)
            # discrete spatial codes for S2 NibblePE / S3 CE (no grad through round)
            info["n_hi"] = rearrange(n_hi.detach(), "(b t) n -> b t n", b=b)
            info["n_lo"] = rearrange(n_lo.detach(), "(b t) n -> b t n", b=b)
            info["n_hi_content"] = rearrange(
                n_hi_content.detach(), "(b t) n -> b t n", b=b
            )
            info["mu"] = rearrange(mu, "(b t) n d -> b t n d", b=b)
            info["conf"] = rearrange(conf, "(b t) n -> b t n", b=b)
            if attn is not None:
                info["slot_attn"] = rearrange(attn, "(b t) n p -> b t n p", b=b)
        else:
            pixels_emb = hidden[:, 0]  # cls token
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, n_hi=None, n_lo=None, mu=None, conf=None):
        """Predict next state embedding
        emb: (B, T, D) or (B, T, N, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(
            emb, act_emb, n_hi=n_hi, n_lo=n_lo, mu=mu, conf=conf
        )
        if preds.ndim == 4:
            b, t, n, _ = preds.shape
            preds = self.pred_proj(rearrange(preds, "b t n d -> (b t n) d"))
            preds = rearrange(preds, "(b t n) d -> b t n d", b=b, t=t, n=n)
        else:
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_mu_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Regress continuous fine codes from slot embeddings. emb: (B,T,N,D)."""
        head = getattr(self, "pos_prop_head", None)
        if head is None:
            raise RuntimeError("pos_prop_head is not enabled")
        include_global = bool(getattr(self, "include_global_token", False))
        slots = emb[:, :, 1:] if include_global else emb
        return head(slots)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        info['pixels']: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is history + horizon
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # select sample dim S=0 from conditioning, encode (B, T, ...)
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb0 = _init["emb"].unsqueeze(1)
        expand_shape = (B, S) + emb0.shape[2:]
        emb = info["emb"] = emb0.expand(expand_shape)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        n_hi = n_lo = mu = conf = None
        if "n_hi" in _init and "n_lo" in _init:
            n_hi = rearrange(
                _init["n_hi"].unsqueeze(1).expand(B, S, *_init["n_hi"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()
            n_lo = rearrange(
                _init["n_lo"].unsqueeze(1).expand(B, S, *_init["n_lo"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()
        if "mu" in _init:
            mu = rearrange(
                _init["mu"].unsqueeze(1).expand(B, S, *_init["mu"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()
        if "conf" in _init:
            conf = rearrange(
                _init["conf"].unsqueeze(1).expand(B, S, *_init["conf"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()

        HS = history_size
        g = int(getattr(self, "spatial_grid", 4))
        use_pos_prop = bool(getattr(self, "pos_prop_enabled", False)) and getattr(
            self, "pos_prop_head", None
        ) is not None

        def _append_codes(pred_step: torch.Tensor):
            nonlocal n_hi, n_lo, mu, conf
            if n_hi is None:
                return
            n_hi = torch.cat([n_hi, n_hi[:, -1:]], dim=1)
            if use_pos_prop:
                mu_next = self.predict_mu_from_emb(pred_step)
                # clamp to cell coords [0, g)
                mu_next = mu_next.clamp(0.0, float(g) - 1e-3)
                cy = mu_next[..., 0]
                cx = mu_next[..., 1]
                n_lo_next = (
                    cy.round().long().clamp(0, g - 1) * g
                    + cx.round().long().clamp(0, g - 1)
                )
                n_lo = torch.cat([n_lo, n_lo_next], dim=1)
                if mu is not None:
                    mu = torch.cat([mu, mu_next], dim=1)
                if conf is not None:
                    conf = torch.cat([conf, conf[:, -1:]], dim=1)
            else:
                n_lo = torch.cat([n_lo, n_lo[:, -1:]], dim=1)
                if mu is not None:
                    mu = torch.cat([mu, mu[:, -1:]], dim=1)
                if conf is not None:
                    conf = torch.cat([conf, conf[:, -1:]], dim=1)

        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            hi_trunc = n_hi[:, -HS:] if n_hi is not None else None
            lo_trunc = n_lo[:, -HS:] if n_lo is not None else None
            mu_trunc = mu[:, -HS:] if mu is not None else None
            conf_trunc = conf[:, -HS:] if conf is not None else None
            pred_emb = self.predict(
                emb_trunc,
                act_trunc,
                n_hi=hi_trunc,
                n_lo=lo_trunc,
                mu=mu_trunc,
                conf=conf_trunc,
            )[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            _append_codes(pred_emb)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # predict last state
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        hi_trunc = n_hi[:, -HS:] if n_hi is not None else None
        lo_trunc = n_lo[:, -HS:] if n_lo is not None else None
        mu_trunc = mu[:, -HS:] if mu is not None else None
        conf_trunc = conf[:, -HS:] if conf is not None else None
        pred_emb = self.predict(
            emb_trunc,
            act_trunc,
            n_hi=hi_trunc,
            n_lo=lo_trunc,
            mu=mu_trunc,
            conf=conf_trunc,
        )[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def _pad_m_task(self, m_task: torch.Tensor, device, fill_cls: float = 1.0) -> torch.Tensor:
        mask = m_task.to(device).float()
        n_slots = int(self.num_spatial_tokens)
        if (
            getattr(self, "include_global_token", False)
            and mask.ndim >= 1
            and mask.size(-1) == n_slots
        ):
            ones_shape = list(mask.shape)
            ones_shape[-1] = 1
            ones = mask.new_full(ones_shape, float(fill_cls))
            mask = torch.cat([ones, mask], dim=-1)
        return mask

    def _effective_cost_lambda(self) -> float:
        lam = float(getattr(self, "cost_lambda", 1.0))
        anneal = str(getattr(self, "lambda_anneal", "off")).lower()
        if anneal in ("", "off", "none", "false", "0"):
            return lam
        # linear: λ_j = λ* · j / (J-1), j=0..J-1; reset each replan via cem loop
        j = int(getattr(self, "_cem_iter", 0))
        j_total = max(int(getattr(self, "_cem_n_steps", 1)), 1)
        if j_total <= 1:
            return lam
        return lam * float(j) / float(j_total - 1)

    def _norm_group_cost(self, cost: torch.Tensor, group: str) -> torch.Tensor:
        """Normalize a (B, S) cost group. cost_norm: none|calib|zscore."""
        mode = str(getattr(self, "cost_norm", "none")).lower()
        if mode in ("", "none"):
            return cost
        if mode == "zscore":
            mu = cost.mean(dim=1, keepdim=True)
            sd = cost.std(dim=1, keepdim=True).clamp_min(1e-6)
            return (cost - mu) / sd
        if mode == "calib":
            key = f"_cost_scale_{group}"
            scale = getattr(self, key, None)
            if scale is None:
                # collect medians over first few get_cost batches, then freeze
                buf_key = f"_cost_calib_buf_{group}"
                buf = getattr(self, buf_key, None)
                if buf is None:
                    buf = []
                    setattr(self, buf_key, buf)
                buf.append(cost.detach().flatten().median().item())
                need = int(getattr(self, "cost_calib_batches", 10))
                if len(buf) >= need:
                    med = float(sorted(buf)[len(buf) // 2])
                    setattr(self, key, max(med, 1e-6))
                    scale = getattr(self, key)
                else:
                    # provisional: per-batch median
                    return cost / cost.detach().median().clamp_min(1e-6)
            return cost / float(scale)
        return cost

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T', D) or (B, S, T', N, D)
        goal_emb = info_dict["goal_emb"]

        spatial = pred_emb.ndim == 5

        if spatial:
            # goal: (B, T_g, N, D) or (B, S, T_g, N, D) -> (B, S, 1, N, D)
            if goal_emb.ndim == 4:
                goal_last = goal_emb[:, None, -1:, :, :]
            else:
                goal_last = goal_emb[:, :, -1:, :, :]
            pred_last = pred_emb[:, :, -1:, :, :]
            goal_last = goal_last.expand_as(pred_last).detach()
            # per-token squared L2 over feature dim
            per_tok = (pred_last - goal_last).pow(2).sum(dim=-1)  # (B, S, 1, N)
            per_tok_sq = per_tok.squeeze(2)  # (B, S, N)

            m_task = info_dict.get("m_task")
            cost_mode = getattr(self, "cost_mode", "dense_terminal")
            has_global = bool(getattr(self, "include_global_token", False))
            n_slots = int(self.num_spatial_tokens)

            if cost_mode == "cls_terminal":
                # only global CLS (index 0); slots ignored
                if has_global and per_tok_sq.size(-1) >= 1:
                    cost = per_tok_sq[..., 0]
                else:
                    cost = per_tok_sq.mean(dim=-1)
                return self._norm_group_cost(cost, "cls")

            if cost_mode == "decoupled":
                # C = ĉ_cls + λ · Σ w_i ĉ_i / Σ w_i  (slots only in the sum)
                if has_global and per_tok_sq.size(-1) == n_slots + 1:
                    c_cls = per_tok_sq[..., 0]
                    c_slot = per_tok_sq[..., 1:]
                else:
                    c_cls = per_tok_sq.mean(dim=-1)
                    c_slot = per_tok_sq
                if m_task is not None:
                    w = m_task.to(per_tok_sq.device).float()
                    if w.size(-1) == n_slots + 1 and has_global:
                        w = w[..., 1:]
                    elif w.size(-1) != c_slot.size(-1):
                        w = torch.ones_like(c_slot[0:1, 0:1, :]).expand_as(c_slot)
                    else:
                        w = w[:, None, :].expand_as(c_slot)
                else:
                    w = torch.ones_like(c_slot)
                slot_num = (c_slot * w).sum(dim=-1)
                slot_den = w.sum(dim=-1).clamp_min(1e-6)
                c_slot_m = slot_num / slot_den
                c_cls_n = self._norm_group_cost(c_cls, "cls")
                c_slot_n = self._norm_group_cost(c_slot_m, "slot")
                cost = c_cls_n + self._effective_cost_lambda() * c_slot_n
            elif cost_mode == "m_task_terminal" and m_task is not None:
                # m_task: (B, N_slots) -> pad CLS=1; mean over weighted tokens
                mask = self._pad_m_task(m_task, per_tok.device)
                if mask.ndim == 2:
                    mask = mask[:, None, None, :]
                weighted = (per_tok * mask).sum(dim=(2, 3))  # (B, S)
                denom = mask.sum(dim=-1).squeeze(-1).squeeze(-1).clamp_min(1.0)  # (B,)
                cost = weighted / denom[:, None]
            else:
                # dense terminal: mean over spatial tokens (fair vs M_task)
                cost = per_tok.mean(dim=(2, 3))

            # optional token-CVaR over slot errors (third-cell coverage)
            cvar_q = float(getattr(self, "cvar_q", 0.0) or 0.0)
            cvar_beta = float(getattr(self, "cvar_beta", 0.0) or 0.0)
            if cvar_q > 0.0 and cvar_beta > 0.0:
                if has_global and per_tok_sq.size(-1) == n_slots + 1:
                    errs = per_tok_sq[..., 1:]
                else:
                    errs = per_tok_sq
                k = max(1, int(math.ceil(cvar_q * errs.size(-1))))
                topk_vals, _ = errs.topk(k, dim=-1, largest=True)
                cvar = topk_vals.mean(dim=-1)
                cost = cost + cvar_beta * self._norm_group_cost(cvar, "cvar")
            return cost

        # CLS dense terminal (baseline)
        if goal_emb.ndim == 3:
            goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)
        else:
            goal_emb = goal_emb[:, :, -1:, :].expand_as(pred_emb)

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def _build_slot_relevance(
        self,
        cur: torch.Tensor,
        goal_pix: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        slot_attn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, N_slots) relevance weights / diffs for mask construction."""
        src = str(getattr(self, "relevance_source", "pixel")).lower()
        g = int(getattr(self, "spatial_grid", 4))
        assume = bool(getattr(self, "m_task_assume_normalized", True))

        if src == "latent" and goal_emb is not None:
            cur_info = self.encode({"pixels": cur.unsqueeze(1)})
            cur_emb = cur_info["emb"]  # (B, 1, N, D) or (B, 1, D)
            if cur_emb.ndim == 4 and goal_emb.ndim == 4:
                # drop CLS if present
                c = cur_emb[:, -1]
                ge = goal_emb[:, -1]
                if (
                    getattr(self, "include_global_token", False)
                    and c.size(1) == int(self.num_spatial_tokens) + 1
                ):
                    c = c[:, 1:]
                    ge = ge[:, 1:]
                return (c - ge.detach()).pow(2).sum(dim=-1)  # (B, N)
            # fallback
            src = "pixel"

        if src == "attn_proj" and slot_attn is not None:
            # slot_attn: (B, T, N, P) or (B, N, P) — use last frame
            attn = slot_attn
            while attn.ndim > 3:
                attn = attn[:, -1]
            # patch-level |RGB| mean diff, pooled to attn patch grid
            while cur.ndim > 4:
                cur = cur[:, 0]
            while goal_pix.ndim > 4:
                goal_pix = goal_pix[:, 0]
            c = cur.float()
            gp = goal_pix.float()
            if assume:
                c = denormalize_imagenet(c)
                gp = denormalize_imagenet(gp)
            diff = (c - gp).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            p = attn.size(-1)
            side = int(math.sqrt(p))
            if side * side == p:
                diff_p = F.adaptive_avg_pool2d(diff, (side, side)).flatten(1)  # (B,P)
                return (attn * diff_p[:, None, :]).sum(dim=-1)  # (B, N)
            src = "pixel"

        return _pixel_cell_diff(cur, goal_pix, spatial_grid=g, assume_normalized=assume)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute cost of action candidates given goal and initial state.
        info tensors use shape (B, S, T, ...); action_candidates (B, S, T, A)
        """
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # take S=0 slice -> (B, T, ...)
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_") and k != "goal":
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("goal", None)
        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]

        cost_mode = getattr(self, "cost_mode", "dense_terminal")
        needs_mask = cost_mode in ("m_task_terminal", "decoupled")
        if self.use_spatial_tokens and needs_mask and "m_task" not in info_dict:
            cur = info_dict["pixels"][:, 0, -1]  # (B, C, H, W)
            goal_pix = info_dict["goal"][:, 0]
            while goal_pix.ndim > 4:
                goal_pix = goal_pix[:, 0]

            # optional: encode current frame for attn / latent relevance
            slot_attn = None
            src = str(getattr(self, "relevance_source", "pixel")).lower()
            if src in ("attn_proj", "latent"):
                cur_enc = self.encode({"pixels": cur.unsqueeze(1)})
                slot_attn = cur_enc.get("slot_attn")

            diffs = self._build_slot_relevance(
                cur, goal_pix, goal_emb=goal["emb"], slot_attn=slot_attn
            )

            mask_weight = str(getattr(self, "mask_weight", "hard")).lower()
            tau_mode = str(getattr(self, "tau_mode", "fixed")).lower()
            thr = getattr(self, "m_task_threshold", 0.08)
            soft_eps = float(getattr(self, "soft_eps", 0.1))
            soft_temp = float(getattr(self, "soft_temp", 0.02))

            if tau_mode == "otsu":
                tau = _otsu_threshold_1d(diffs)
            else:
                tau = 0.08 if thr is None else float(thr)

            if mask_weight == "soft":
                info_dict["m_task"] = soft_weights_from_diff(
                    diffs, tau=tau, temp=soft_temp, eps=soft_eps
                )
            elif str(getattr(self, "m_task_mode", "eps")).lower() == "topk":
                k = max(1, min(int(getattr(self, "m_task_topk", 4)), diffs.size(1)))
                _, idx = diffs.topk(k, dim=1)
                mask = torch.zeros_like(diffs)
                mask.scatter_(1, idx, 1.0)
                empty = mask.sum(dim=1, keepdim=True) <= 0
                if empty.any():
                    mask = torch.where(empty, torch.ones_like(mask), mask)
                info_dict["m_task"] = mask
            else:
                if torch.is_tensor(tau):
                    mask = (diffs > tau).float()
                else:
                    mask = (diffs > float(tau)).float()
                empty = mask.sum(dim=1, keepdim=True) <= 0
                if empty.any():
                    mask = torch.where(empty, torch.ones_like(mask), mask)
                info_dict["m_task"] = mask

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
