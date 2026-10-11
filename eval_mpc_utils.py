"""Shared helpers for open-loop CEM vs MPC receding-horizon eval."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PLAN_PROTOCOLS = ("lewm", "legacy")
EVAL_SPLITS = ("dev", "test", "all")

COST_MODE_CHOICES = (
    "dense_terminal",
    "m_task_terminal",
    "cls_terminal",
    "decoupled",
)


def add_cost_cli_args(parser):
    """Stage-1 planning cost CLI (defaults preserve r1/r3 behaviour)."""
    parser.add_argument(
        "--cost-mode",
        default="dense_terminal",
        choices=list(COST_MODE_CHOICES),
        help="规划 cost；C1/S1/S2 默认 m_task_terminal；阶段1 扫 cls_terminal/decoupled",
    )
    parser.add_argument("--cost-lambda", type=float, default=1.0, help="decoupled 槽相对 CLS 权重")
    parser.add_argument(
        "--cost-norm",
        default="none",
        choices=["none", "calib", "zscore"],
        help="组内尺度归一化；decoupled 默认建议 calib",
    )
    parser.add_argument(
        "--mask-weight",
        default="hard",
        choices=["hard", "soft"],
        help="hard=0/1 M_task；soft=ε+(1-ε)·sigmoid((d-τ)/T)",
    )
    parser.add_argument("--soft-eps", type=float, default=0.1)
    parser.add_argument("--soft-temp", type=float, default=0.02)
    parser.add_argument(
        "--tau-mode",
        default="fixed",
        choices=["fixed", "quantile", "otsu"],
        help="fixed 用 --m-task-threshold；quantile 也用该值作已标定 τ；otsu 每样本自动",
    )
    parser.add_argument(
        "--relevance-source",
        default="pixel",
        choices=["pixel", "attn_proj", "latent"],
        help="d_i 来源：像素格差 / 注意力投影 / 潜空间槽距",
    )
    parser.add_argument(
        "--cvar-q",
        type=float,
        default=0.0,
        help="token-CVaR 比例；0=关闭",
    )
    parser.add_argument("--cvar-beta", type=float, default=0.0, help="CVaR 系数；0=关闭")
    parser.add_argument(
        "--lambda-anneal",
        default="off",
        choices=["off", "linear"],
        help="CEM 迭代内 λ 从 0 线性升到 cost-lambda",
    )
    parser.add_argument("--m-task-threshold", type=float, default=0.08)
    parser.add_argument("--m-task-mode", default="eps", choices=["eps", "topk", "mean"])
    parser.add_argument("--m-task-topk", type=int, default=4)
    parser.add_argument(
        "--m-task-raw-pixels",
        action="store_true",
        help="若 info 像素未 ImageNet normalize，打开此开关",
    )


def apply_cost_attrs(model, args):
    """Write stage-1 cost knobs onto a loaded JEPA (getattr defaults keep old ckpts)."""
    model.cost_mode = args.cost_mode
    model.cost_lambda = float(getattr(args, "cost_lambda", 1.0))
    model.cost_norm = str(getattr(args, "cost_norm", "none"))
    model.mask_weight = str(getattr(args, "mask_weight", "hard"))
    model.soft_eps = float(getattr(args, "soft_eps", 0.1))
    model.soft_temp = float(getattr(args, "soft_temp", 0.02))
    model.tau_mode = str(getattr(args, "tau_mode", "fixed"))
    model.relevance_source = str(getattr(args, "relevance_source", "pixel"))
    model.cvar_q = float(getattr(args, "cvar_q", 0.0))
    model.cvar_beta = float(getattr(args, "cvar_beta", 0.0))
    model.lambda_anneal = str(getattr(args, "lambda_anneal", "off"))
    model.m_task_threshold = args.m_task_threshold
    model.m_task_mode = args.m_task_mode
    model.m_task_topk = args.m_task_topk
    model.m_task_assume_normalized = not bool(getattr(args, "m_task_raw_pixels", False))
    return model


def filter_split_episodes(ep_ids: np.ndarray, split: str) -> np.ndarray:
    """Disjoint episode halves of the *_eval.h5 file.

    ``test`` = even episode index (final numbers only), ``dev`` = odd index
    (all tuning / screening), ``all`` = no filter.
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    ep_ids = np.asarray(ep_ids)
    if split == "all":
        return ep_ids
    parity = 0 if split == "test" else 1
    return ep_ids[ep_ids % 2 == parity]


def plan_context_len(plan_protocol: str, history_size: int) -> int:
    """Frames fed to the planner.

    ``lewm``: current env frame only, as in official LeWM eval
    (``swm.PlanConfig.history_len`` defaults to 1) and DINO-WM ``obs_0``.
    ``legacy``: ``history_size`` frames with their actions pinned (r1 behaviour).
    """
    if plan_protocol not in PLAN_PROTOCOLS:
        raise ValueError(f"plan_protocol must be one of {PLAN_PROTOCOLS}, got {plan_protocol!r}")
    return 1 if plan_protocol == "lewm" else int(history_size)


def resolve_stats_path(dataset_path, stats_path=None) -> str:
    """Normalizer stats must come from the training split (what the model saw)."""
    if stats_path:
        return str(stats_path)
    p = Path(dataset_path)
    if p.name.endswith("_eval.h5"):
        cand = p.with_name(p.name[: -len("_eval.h5")] + "_train.h5")
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"cannot infer training-split stats file for {dataset_path}; pass --stats-dataset"
    )


@torch.no_grad()
def cem_lewm(
    model,
    info: dict,
    horizon: int,
    action_dim: int,
    num_samples: int,
    n_steps: int,
    topk: int,
    var_scale: float,
    device,
):
    """Same update rule as ``stable_worldmodel.solver.CEMSolver`` (no warm start).

    Candidates cover only the planning horizon, so the first planned action is
    paired with the current frame. Returns the final elite mean and the mean
    elite cost of the last iteration.
    """
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.full((horizon, action_dim), float(var_scale), device=device)
    k = min(int(topk), int(num_samples))
    elite_cost = float("nan")
    n_steps = int(n_steps)
    # expose CEM progress for λ annealing (reset each replan call)
    model._cem_n_steps = n_steps
    for j in range(n_steps):
        model._cem_iter = j
        noise = torch.randn(num_samples, horizon, action_dim, device=device)
        candidates = mean + std * noise
        candidates[0] = mean
        costs = model.get_cost(info, candidates.unsqueeze(0))[0]
        vals, idx = torch.topk(costs, k, largest=False)
        elite = candidates[idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0)
        elite_cost = float(vals.mean().item())
    model._cem_iter = 0
    return mean, elite_cost


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
