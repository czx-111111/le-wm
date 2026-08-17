"""JEPA Implementation"""

from __future__ import annotations

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

    @property
    def use_spatial_tokens(self) -> bool:
        token_mode = getattr(self, "token_mode", "cls")
        n = int(getattr(self, "num_spatial_tokens", 1))
        return token_mode in ("spatial16", "spatial") and n > 1

    def _pool_patches_to_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (BT, P, H) -> (BT, N, H) with N = spatial_grid**2."""
        bt, p, h = patch_tokens.shape
        side = int(p**0.5)
        assert side * side == p, f"patch count {p} is not a square grid"
        grid = rearrange(patch_tokens, "bt (h w) d -> bt d h w", h=side, w=side)
        g = int(getattr(self, "spatial_grid", 4))
        pooled = F.adaptive_avg_pool2d(grid, (g, g))
        return rearrange(pooled, "bt d gh gw -> bt (gh gw) d")

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
            patch_tokens = hidden[:, 1:]  # drop CLS
            tokens = self._pool_patches_to_tokens(patch_tokens)  # (BT, N, H)
            n = tokens.size(1)
            emb = self.projector(rearrange(tokens, "bt n d -> (bt n) d"))
            emb = rearrange(emb, "(bt n) d -> bt n d", n=n)
            info["emb"] = rearrange(emb, "(b t) n d -> b t n d", b=b)
        else:
            pixels_emb = hidden[:, 0]  # cls token
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D) or (B, T, N, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        if preds.ndim == 4:
            b, t, n, _ = preds.shape
            preds = self.pred_proj(rearrange(preds, "b t n d -> (b t n) d"))
            preds = rearrange(preds, "(b t n) d -> b t n d", b=b, t=t, n=n)
        else:
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

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

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # predict last state
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

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
                mask = m_task.to(per_tok.device).float()
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
