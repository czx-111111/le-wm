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
):
    """Quantize per-query attention into radix-16 nibble codes.

    attn: (BT, N, P) over a square patch grid.
    n_hi: slot id 0..N-1 (tokenization-aligned high nibble)
    n_lo: centroid inside that slot's 4x4 window, 0..15
    n_hi_content: coarse cell of the full-map centroid (aux target)
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
    prob = (flat / mass).view(bt, n, cell, cell)
    ys = torch.arange(cell, device=device, dtype=prob.dtype).view(1, 1, cell, 1)
    xs = torch.arange(cell, device=device, dtype=prob.dtype).view(1, 1, 1, cell)
    cy = (prob * ys).sum(dim=(2, 3))
    cx = (prob * xs).sum(dim=(2, 3))
    n_lo = (
        cy.round().long().clamp(0, cell - 1) * cell
        + cx.round().long().clamp(0, cell - 1)
    )

    fine_mass = attn_fine.flatten(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
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
    return n_hi, n_lo, n_hi_content


class SpatialQueryPool(nn.Module):
    """Single-head cross-attention from a 4x4 query grid onto ViT patches."""

    def __init__(self, hidden_dim: int, num_queries: int = 16, spatial_grid: int = 4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.spatial_grid = int(spatial_grid)
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, self.hidden_dim))
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale = self.hidden_dim**-0.5
        self.reset_parameters()

    def reset_parameters(self):
        pe = sincos_2d_pe(self.spatial_grid, self.spatial_grid, self.hidden_dim)
        with torch.no_grad():
            self.query.copy_(pe.unsqueeze(0))
            self.query.add_(0.02 * torch.randn_like(self.query))

    def forward(self, patches: torch.Tensor):
        """patches: (BT, P, H) -> slots (BT, N, H), attn (BT, N, P)."""
        bt = patches.size(0)
        q = self.q_proj(self.query.expand(bt, -1, -1))
        k = self.k_proj(patches)
        v = self.v_proj(patches)
        attn = torch.matmul(q, k.transpose(-2, -1)).mul_(self.scale).softmax(dim=-1)
        slots = self.out_proj(torch.matmul(attn, v))
        return slots, attn


def compute_m_task_mask(
    cur_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    spatial_grid: int = 4,
    threshold: float | str = 0.08,
    mode: str = "eps",
    topk: int = 4,
    assume_normalized: bool = True,
):
    """Binary M_task over spatial_grid^2 tokens from |cur - goal| pixel diff.

    Preferred: compute on [0, 1] RGB. If ``assume_normalized``, undo ImageNet
    normalize first. ``mode``:
      - ``eps``: pool then ``diff > threshold`` (default threshold=0.08)
      - ``topk``: keep the ``topk`` highest-diff cells
      - ``mean``: legacy per-sample mean threshold
    Returns (B, N) float mask in {0, 1}. All-ones fallback if empty.
    """
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
        # uint8 or [0,1] / [0,255]
        if cur.max() > 1.5:
            cur = cur / 255.0
            goal = goal / 255.0
        cur = cur.clamp(0.0, 1.0)
        goal = goal.clamp(0.0, 1.0)

    diff = (cur - goal).abs().mean(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(diff, (spatial_grid, spatial_grid))
    flat = pooled.flatten(1)  # (B, N)

    mode = str(mode).lower()
    if mode == "topk":
        k = max(1, min(int(topk), flat.size(1)))
        _, idx = flat.topk(k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
    elif mode == "mean" or threshold == "mean":
        thr = flat.mean(dim=1, keepdim=True)
        mask = (flat > thr).float()
    else:
        # eps (default): fixed threshold on [0, 1] RGB diff
        thr = 0.08 if threshold is None else float(threshold)
        mask = (flat > thr).float()

    empty = mask.sum(dim=1, keepdim=True) <= 0
    if empty.any():
        mask = torch.where(empty, torch.ones_like(mask), mask)
    return mask


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
        encoder_hidden_dim: int | None = None,
        embed_dim: int = 192,
        aux_nibble: bool = False,
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
        self.aux_nibble = bool(aux_nibble)
        self.embed_dim = int(embed_dim)

        hidden = encoder_hidden_dim
        if hidden is None:
            hidden = getattr(getattr(encoder, "config", None), "hidden_size", None)
        self.query_pool = None
        if self.pool_mode == "query" and self.num_spatial_tokens > 1:
            self.query_pool = SpatialQueryPool(
                hidden_dim=int(hidden),
                num_queries=self.num_spatial_tokens,
                spatial_grid=self.spatial_grid,
            )
        n_class = self.spatial_grid * self.spatial_grid
        if self.aux_nibble:
            self.nibble_hi_head = nn.Linear(self.embed_dim, n_class)
            self.nibble_lo_head = nn.Linear(self.embed_dim, n_class)
        else:
            self.nibble_hi_head = None
            self.nibble_lo_head = None

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
                n_hi, n_lo, n_hi_content = nibble_codes_from_attn(
                    attn, spatial_grid=self.spatial_grid
                )
            else:
                slots = self._pool_patches_to_tokens(patch_tokens)
                n_hi, n_lo, n_hi_content = self._dummy_nibble(
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
            info["n_hi"] = rearrange(n_hi, "(b t) n -> b t n", b=b)
            info["n_lo"] = rearrange(n_lo, "(b t) n -> b t n", b=b)
            info["n_hi_content"] = rearrange(n_hi_content, "(b t) n -> b t n", b=b)
            if attn is not None:
                info["slot_attn"] = rearrange(attn, "(b t) n p -> b t n p", b=b)
        else:
            pixels_emb = hidden[:, 0]  # cls token
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, n_hi=None, n_lo=None):
        """Predict next state embedding
        emb: (B, T, D) or (B, T, N, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb, n_hi=n_hi, n_lo=n_lo)
        if preds.ndim == 4:
            b, t, n, _ = preds.shape
            preds = self.pred_proj(rearrange(preds, "b t n d -> (b t n) d"))
            preds = rearrange(preds, "(b t n) d -> b t n d", b=b, t=t, n=n)
        else:
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def nibble_logits(self, pred_emb: torch.Tensor):
        """pred_emb: (B, T, N, D) -> hi/lo logits over spatial slots."""
        if self.nibble_hi_head is None or self.nibble_lo_head is None:
            return None, None
        slots = pred_emb
        if getattr(self, "include_global_token", False):
            slots = pred_emb[:, :, 1:, :]
        return self.nibble_hi_head(slots), self.nibble_lo_head(slots)

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

        n_hi = n_lo = None
        if "n_hi" in _init and "n_lo" in _init:
            n_hi = rearrange(
                _init["n_hi"].unsqueeze(1).expand(B, S, *_init["n_hi"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()
            n_lo = rearrange(
                _init["n_lo"].unsqueeze(1).expand(B, S, *_init["n_lo"].shape[1:]),
                "b s ... -> (b s) ...",
            ).clone()

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            hi_trunc = n_hi[:, -HS:] if n_hi is not None else None
            lo_trunc = n_lo[:, -HS:] if n_lo is not None else None
            pred_emb = self.predict(
                emb_trunc, act_trunc, n_hi=hi_trunc, n_lo=lo_trunc
            )[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            if n_hi is not None:
                n_hi = torch.cat([n_hi, n_hi[:, -1:]], dim=1)
                n_lo = torch.cat([n_lo, n_lo[:, -1:]], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # predict last state
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        hi_trunc = n_hi[:, -HS:] if n_hi is not None else None
        lo_trunc = n_lo[:, -HS:] if n_lo is not None else None
        pred_emb = self.predict(
            emb_trunc, act_trunc, n_hi=hi_trunc, n_lo=lo_trunc
        )[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def _pad_m_task(self, m_task: torch.Tensor, device) -> torch.Tensor:
        mask = m_task.to(device).float()
        n_slots = int(self.num_spatial_tokens)
        if (
            getattr(self, "include_global_token", False)
            and mask.ndim >= 1
            and mask.size(-1) == n_slots
        ):
            ones_shape = list(mask.shape)
            ones_shape[-1] = 1
            ones = torch.ones(*ones_shape, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([ones, mask], dim=-1)
        return mask

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

            m_task = info_dict.get("m_task")
            cost_mode = getattr(self, "cost_mode", "dense_terminal")
            if cost_mode == "m_task_terminal" and m_task is not None:
                # m_task: (B, N) -> (B, 1, 1, N); mean over masked tokens
                mask = self._pad_m_task(m_task, per_tok.device)
                if mask.ndim == 2:
                    mask = mask[:, None, None, :]
                weighted = (per_tok * mask).sum(dim=(2, 3))  # (B, S)
                denom = mask.sum(dim=-1).squeeze(-1).squeeze(-1).clamp_min(1.0)  # (B,)
                cost = weighted / denom[:, None]
            else:
                # dense terminal: mean over spatial tokens (fair vs M_task)
                cost = per_tok.mean(dim=(2, 3))
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

        if (
            self.use_spatial_tokens
            and getattr(self, "cost_mode", "dense_terminal") == "m_task_terminal"
            and "m_task" not in info_dict
        ):
            # history last frame vs goal (ImageNet-normalized in eval pipelines;
            # denormalize inside compute_m_task_mask by default)
            cur = info_dict["pixels"][:, 0, -1]  # (B, C, H, W)
            goal_pix = info_dict["goal"][:, 0]
            while goal_pix.ndim > 4:
                goal_pix = goal_pix[:, 0]
            info_dict["m_task"] = compute_m_task_mask(
                cur,
                goal_pix,
                spatial_grid=int(getattr(self, "spatial_grid", 4)),
                threshold=getattr(self, "m_task_threshold", 0.08),
                mode=getattr(self, "m_task_mode", "eps"),
                topk=int(getattr(self, "m_task_topk", 4)),
                assume_normalized=bool(
                    getattr(self, "m_task_assume_normalized", True)
                ),
            )

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
