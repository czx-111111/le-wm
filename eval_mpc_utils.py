"""Shared helpers for open-loop CEM vs MPC receding-horizon eval."""
from __future__ import annotations

import numpy as np
import torch


def as_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_hwc_uint8_compatible(frame: np.ndarray) -> np.ndarray:
    """Keep float/uint8 HWC; eval transforms will normalize."""
    frame = as_numpy(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def macro_end_indices(n_macros: int, frameskip: int, total_steps: int):
    idxs = []
    for m in range(n_macros):
        idx = (m + 1) * frameskip - 1
        idxs.append(min(idx, total_steps - 1))
    return idxs


def extract_macro_frames(obses: dict, n_macros: int, frameskip: int):
    """Pick end-of-macro visual/proprio from env.step_multiple outputs."""
    vis = as_numpy(obses["visual"])
    prop = as_numpy(obses["proprio"])
    t_total = vis.shape[0]
    pixels, proprios = [], []
    for idx in macro_end_indices(n_macros, frameskip, t_total):
        pixels.append(to_hwc_uint8_compatible(vis[idx]))
        p = prop[idx]
        proprios.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return pixels, proprios


def stack_history_clip(hist_pixels, hist_proprio, hist_act_raw, history_size, frameskip, action_dim):
    """Build clip dict for build_info_dict / normalize_concat_actions."""
    assert len(hist_pixels) == history_size
    assert len(hist_proprio) == history_size
    act = np.asarray(hist_act_raw, dtype=np.float32).reshape(-1, action_dim)
    need = history_size * frameskip
    if act.shape[0] < need:
        pad = np.zeros((need - act.shape[0], action_dim), dtype=np.float32)
        act = np.concatenate([pad, act], axis=0)
    elif act.shape[0] > need:
        act = act[-need:]
    return {
        "pixels": torch.from_numpy(np.stack(hist_pixels)),
        "proprio": torch.from_numpy(np.stack(hist_proprio).astype(np.float32)),
        "action_raw": act,
    }
