#!/usr/bin/env python3
"""
LeWM Wall 仿真评估（DINO-WM DotWall）

对齐训练配置：action_dim=2, frameskip=5 → 规划维数 10。
成功判据：||goal[:2] - cur[:2]||_2 < 4.5（与 DINO-WM WallEnvWrapper.eval_state 一致）。
评估条数：默认 50（对齐 plan_wall.yaml 的 n_evals=50）。
默认规划卡：物理 GPU 2。

默认 planner=mpc（滚动规划；Wall 消融最优 n_taken_actions=3, mpc_max_iters=8）。
可用 --planner openloop 复现旧开环 CEM。

用法:
  python eval_wall.py \\
    --dataset ./data/wall_eval.h5 \\
    --ckpt ./data/lewm_wall_v1_epoch_50_object.ckpt \\
    --plan-gpu 2 --num-eval 50 --planner mpc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _early_apply_gpu_argv():
    """在 import torch 之前锁定物理 GPU。"""
    argv = sys.argv[1:]
    plan_gpu = "2"
    for i, a in enumerate(argv):
        if a == "--plan-gpu" and i + 1 < len(argv):
            plan_gpu = argv[i + 1]
        elif a.startswith("--plan-gpu="):
            plan_gpu = a.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan_gpu)
    os.environ["PLAN_GPU"] = str(plan_gpu)


_early_apply_gpu_argv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import load_lewm_weights_into_model
from eval_mpc_utils import (
    as_numpy,
    extract_macro_frames,
    stack_history_clip,
    to_hwc_uint8_compatible,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def build_goal_lookup(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        episode_idx = f["ep_idx"][:]
        step_idx = f["step_idx"][:]
    lookup = {}
    for global_idx, (ep, step) in enumerate(zip(episode_idx, step_idx)):
        lookup[(int(ep), int(step))] = int(global_idx)
    return lookup


def sample_eval_points(
    h5_path: str,
    num_eval: int,
    goal_offset_steps: int,
    history_size: int,
    frameskip: int,
    seed: int = 42,
):
    """采样 (episode, start_step)，保证 history(带 frameskip) 与 goal 都在轨迹内。"""
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]

    # history 最后一帧: start + (H-1)*fs
    # history 动作覆盖到: start + H*fs - 1
    # goal: start + goal_offset
    need = max(goal_offset_steps, (history_size - 1) * frameskip, history_size * frameskip - 1)
    max_start = ep_len - need - 1
    valid_ep = np.nonzero(max_start >= 0)[0]
    if len(valid_ep) == 0:
        raise RuntimeError(
            f"没有足够长的 episode：需要至少 {need + 1} 帧 "
            f"(goal_offset={goal_offset_steps}, history={history_size}, frameskip={frameskip})"
        )

    g = np.random.default_rng(seed)
    replace = len(valid_ep) < num_eval
    chosen_ep = g.choice(valid_ep, size=num_eval, replace=replace)

    starts = []
    for ep_id in chosen_ep:
        hi = int(max_start[ep_id])
        starts.append(int(g.integers(0, hi + 1)))

    starts = np.asarray(starts, dtype=np.int64)
    print(
        f"[INFO] {len(valid_ep)} 个有效 episode, 采样 {num_eval} 组 "
        f"(replace={replace}, need_len>={need + 1})"
    )
    return chosen_ep.astype(np.int64), starts


def load_frame(h5_path: str, idx: int):
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][idx]
        action = f["action"][idx]
        proprio = f["proprio"][idx] if "proprio" in f else np.zeros(2, dtype=np.float32)
        state = f["state"][idx] if "state" in f else None
        door = float(f["door_location"][idx]) if "door_location" in f else None
        wall = float(f["wall_location"][idx]) if "wall_location" in f else None
    return {
        "pixels": pixels,
        "action": action,
        "proprio": proprio,
        "state": state,
        "door_location": door,
        "wall_location": wall,
    }


def load_history_with_frameskip(
    h5_path: str,
    lookup,
    episode_id: int,
    start_step: int,
    history_size: int,
    frameskip: int,
    action_dim: int,
):
    """加载 history 帧（每隔 frameskip），并把中间 raw action 拼成 (H, fs*D)。"""
    pixels_list, proprio_list = [], []
    raw_actions = []

    for t in range(history_size):
        step = int(start_step + t * frameskip)
        idx = lookup[(int(episode_id), step)]
        fr = load_frame(h5_path, idx)
        pixels_list.append(fr["pixels"])
        proprio_list.append(fr["proprio"])

    # raw actions covering [start, start + H*fs)
    for s in range(history_size * frameskip):
        idx = lookup[(int(episode_id), int(start_step + s))]
        fr = load_frame(h5_path, idx)
        act = np.asarray(fr["action"], dtype=np.float32).reshape(-1)
        if act.shape[0] != action_dim:
            raise ValueError(f"action_dim 期望 {action_dim}，实际 {act.shape[0]}")
        raw_actions.append(act)

    raw_actions = np.stack(raw_actions, axis=0)  # (H*fs, D)
    concat = raw_actions.reshape(history_size, frameskip * action_dim)

    return {
        "pixels": torch.from_numpy(np.stack(pixels_list)),
        "action_raw": raw_actions,  # for normalize-then-concat
        "action": torch.from_numpy(concat).float(),  # placeholder, overwritten after norm
        "proprio": torch.from_numpy(np.stack(proprio_list)).float(),
    }


def build_processors(h5_path: str, keys=("action", "proprio")):
    process = {}
    with h5py.File(h5_path, "r") as f:
        for col in keys:
            if col not in f:
                continue
            data = f[col][:]
            data = data[~np.isnan(data).any(axis=1)]
            scaler = preprocessing.StandardScaler().fit(data)
            process[col] = scaler
            if col != "action":
                process[f"goal_{col}"] = scaler
    return process


def normalize_concat_actions(
    raw_actions: np.ndarray,
    process: dict,
    history_size: int,
    frameskip: int,
    action_dim: int,
) -> torch.Tensor:
    """raw (H*fs, D) → normalize → (H, fs*D)."""
    flat = np.asarray(raw_actions, dtype=np.float32).reshape(-1, action_dim)
    if "action" in process:
        flat = process["action"].transform(flat)
    concat = flat.reshape(history_size, frameskip * action_dim)
    return torch.from_numpy(concat.astype(np.float32))


def denormalize_concat_actions(
    concat_actions: np.ndarray,
    process: dict,
    frameskip: int,
    action_dim: int,
) -> np.ndarray:
    """(horizon, fs*D) normalized → (horizon*fs, D) raw env actions."""
    flat = np.asarray(concat_actions, dtype=np.float32).reshape(-1, frameskip * action_dim)
    flat = flat.reshape(-1, action_dim)
    if "action" in process:
        flat = process["action"].inverse_transform(flat)
    return flat.astype(np.float32)


def apply_processors(sample: dict, process: dict):
    out = dict(sample)
    for col, proc in process.items():
        if col.startswith("goal_") or col == "action" or col not in out:
            continue
        arr = out[col]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        out[col] = torch.from_numpy(proc.transform(arr)).float()
    return out


def _to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform):
    clip = apply_processors(clip, process)

    pixels_np = (
        clip["pixels"].detach().cpu().numpy()
        if torch.is_tensor(clip["pixels"])
        else np.asarray(clip["pixels"])
    )
    frame_tensors = [transform["pixels"](_to_uint8_hwc(pixels_np[i])) for i in range(pixels_np.shape[0])]
    pixels = torch.stack(frame_tensors, dim=0).unsqueeze(0).unsqueeze(0)  # (B,S,T,C,H,W)

    action = torch.nan_to_num(clip["action"].float(), 0.0).unsqueeze(0).unsqueeze(0)
    info = {"pixels": pixels, "action": action}

    if "proprio" in clip:
        info["proprio"] = clip["proprio"].float().unsqueeze(0).unsqueeze(0)

    goal_img = transform["goal"](_to_uint8_hwc(goal_pixels))
    info["goal"] = goal_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    goal_prop = np.asarray(goal_proprio, dtype=np.float32).reshape(1, -1)
    if "goal_proprio" in process:
        goal_prop = process["goal_proprio"].transform(goal_prop)
    info["goal_proprio"] = torch.from_numpy(goal_prop).float().unsqueeze(0).unsqueeze(0)
    return info


# ---------------------------------------------------------------------------
# Model / CEM
# ---------------------------------------------------------------------------


@torch.no_grad()
def cem_plan(
    model,
    info,
    horizon: int,
    action_dim: int,
    num_samples: int = 300,
    n_steps: int = 10,
    topk: int = 30,
    var_scale: float = 1.0,
    device: str = "cuda",
):
    device = torch.device(device)
    info = {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in info.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize()

    history = info["pixels"].shape[2]
    total_steps = history + horizon

    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = info["action"][:, 0, :history]

    best_cost = None
    best_actions = None

    for step in range(n_steps):
        if step == 0 or (step + 1) % max(1, n_steps // 5) == 0 or step + 1 == n_steps:
            print(f"    CEM {step + 1}/{n_steps}", flush=True)
        noise = torch.randn(num_samples, total_steps, action_dim, device=device)
        candidates = mean + std * noise
        candidates[:, :history] = info["action"][0, 0, :history]
        costs = model.get_cost(info, candidates.unsqueeze(0))
        k = min(topk, num_samples)
        _, idx = torch.topk(costs[0], k, largest=False)
        elite = candidates[idx]
        mean = elite.mean(dim=0, keepdim=True)
        std = elite.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.05)
        mean[:, :history] = info["action"][:, 0, :history]

        min_cost, min_idx = costs[0].min(dim=0)
        if best_cost is None or min_cost.item() < best_cost:
            best_cost = min_cost.item()
            best_actions = candidates[min_idx.item()]

    return best_actions[history:], best_cost


def load_lewm_model(
    ckpt_path: str,
    action_dim: int,
    history_size: int = 3,
    embed_dim: int = 192,
    smoothed_dim: int = 10,
    encoder_scale: str = "tiny",
    patch_size: int = 14,
    img_size: int = 224,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
    predictor_emb_dropout: float = 0.0,
):
    # *_epoch_*_object.ckpt 是整模 JEPA；*_weights.ckpt 才是 Lightning state_dict
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        print(f"[INFO] loaded pickled JEPA object from {ckpt_path}")
    else:
        encoder = spt.backbone.utils.vit_hf(
            encoder_scale,
            patch_size=patch_size,
            image_size=img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size

        predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=predictor_mlp_dim,
            dim_head=predictor_dim_head,
            dropout=predictor_dropout,
            emb_dropout=predictor_emb_dropout,
        )

        # 与 train.py 一致：Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        # smoothed_dim 默认 10；wall effective=10 时完全对齐
        action_encoder = Embedder(
            input_dim=action_dim,
            smoothed_dim=smoothed_dim,
            emb_dim=embed_dim,
        )

        projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
        predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=predictor_proj,
        )
        # 把已 load 的 dict 再走统一解析（兼容带/不带 model. 前缀）
        if isinstance(checkpoint, dict):
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
                print(f"[WARN] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:8]}...")
        else:
            model = load_lewm_weights_into_model(model, ckpt_path)

    w = model.action_encoder.patch_embed.weight
    print(f"[INFO] action_encoder.patch_embed.weight = {tuple(w.shape)} (期望 {(smoothed_dim, action_dim, 1)})")

    model.eval()
    model.requires_grad_(False)
    if hasattr(model.encoder, "interpolate_pos_encoding"):
        model.encoder.interpolate_pos_encoding = True
    return model


# ---------------------------------------------------------------------------
# Wall env
# ---------------------------------------------------------------------------


def _import_wall_env_wrapper():
    """导入 WallEnv，避免执行 env/__init__.py（会强制加载 MuJoCo/pointmaze）。"""
    import types

    env_dir = ROOT / "env"
    # 强制用占位 package，跳过 env/__init__.py 里的 gym.register / pointmaze
    pkg = types.ModuleType("env")
    pkg.__path__ = [str(env_dir)]
    pkg.__file__ = str(env_dir / "__init__.py")
    pkg.__package__ = "env"
    sys.modules["env"] = pkg

    from env.wall.wall_env_wrapper import WallEnvWrapper, DEFAULT_CFG

    return WallEnvWrapper, DEFAULT_CFG


def make_wall_env(device: str = "cpu"):
    import copy

    try:
        WallEnvWrapper, DEFAULT_CFG = _import_wall_env_wrapper()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "导入 WallEnv 失败。请确认依赖齐全，且在仓库根目录运行。"
            f" 原始错误: {e}"
        ) from e

    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.device = device
    cfg.fix_wall = True
    env = WallEnvWrapper(
        rng=0,
        wall_config=cfg,
        fix_wall=True,
        cross_wall=False,
        device=device,
    )
    return env


def run_sim_episode(env, init_state, goal_state, door, wall, actions, seed: int):
    """执行 raw actions (T, 2)，返回 success / state_dist。"""
    env.update_env(
        {
            "fix_door_location": torch.tensor(float(door)),
            "fix_wall_location": torch.tensor(float(wall)),
        }
    )
    init = np.asarray(init_state, dtype=np.float32).reshape(-1)
    env.prepare(seed, init)

    act_list = []
    for a in np.asarray(actions, dtype=np.float32).reshape(-1, 2):
        act_list.append(torch.tensor(a, dtype=torch.float32))

    if not act_list:
        raise RuntimeError("空动作序列")

    _, _, _, infos = env.step_multiple(act_list)
    final_state = infos["state"][-1]
    if torch.is_tensor(final_state):
        final_state = final_state.detach().cpu().numpy()
    final_state = np.asarray(final_state, dtype=np.float32).reshape(-1)

    goal = np.asarray(goal_state, dtype=np.float32).reshape(-1)
    metrics = env.eval_state(goal, final_state)
    return {
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "final_state": final_state,
    }


def run_mpc_episode(
    model,
    env,
    *,
    init_clip: dict,
    init_state,
    goal_state,
    goal_pixels,
    goal_proprio,
    door,
    wall,
    process: dict,
    transform: dict,
    history_size: int,
    frameskip: int,
    action_dim: int,
    horizon: int,
    n_taken_actions: int,
    mpc_max_iters: int,
    cem_kwargs: dict,
    device: str,
    seed: int,
):
    """MPC: CEM plan → execute n_taken macros → replan from env feedback."""
    effective_act_dim = frameskip * action_dim
    env.update_env(
        {
            "fix_door_location": torch.tensor(float(door)),
            "fix_wall_location": torch.tensor(float(wall)),
        }
    )
    init = np.asarray(init_state, dtype=np.float32).reshape(-1)
    env.prepare(seed, init)

    pixels_np = as_numpy(init_clip["pixels"])
    proprio_np = as_numpy(init_clip["proprio"])
    hist_pixels = [to_hwc_uint8_compatible(pixels_np[i]) for i in range(history_size)]
    hist_proprio = [np.asarray(proprio_np[i], dtype=np.float32).reshape(-1) for i in range(history_size)]
    hist_act_raw = np.asarray(init_clip["action_raw"], dtype=np.float32).reshape(-1, action_dim)

    goal = np.asarray(goal_state, dtype=np.float32).reshape(-1)
    final_state = init.copy()
    plan_costs = []
    n_mpc_iters = 0
    success = False
    state_dist = float(env.eval_state(goal, final_state)["state_dist"])

    for it in range(mpc_max_iters):
        cur_metrics = env.eval_state(goal, final_state)
        state_dist = float(cur_metrics["state_dist"])
        if cur_metrics["success"]:
            success = True
            print(f"  [MPC] early success at iter={it}", flush=True)
            break

        clip = stack_history_clip(
            hist_pixels, hist_proprio, hist_act_raw, history_size, frameskip, action_dim
        )
        clip["action"] = normalize_concat_actions(
            clip["action_raw"], process, history_size, frameskip, action_dim
        )
        info = build_info_dict(clip, goal_pixels, goal_proprio, process, transform)
        planned_norm, plan_cost = cem_plan(
            model,
            info,
            horizon=horizon,
            action_dim=effective_act_dim,
            device=device,
            **cem_kwargs,
        )
        plan_costs.append(float(plan_cost))
        n_mpc_iters = it + 1

        planned = planned_norm.detach().cpu().numpy()
        n_take = min(int(n_taken_actions), int(planned.shape[0]))
        env_actions = denormalize_concat_actions(
            planned[:n_take], process, frameskip, action_dim
        )
        print(
            f"  [MPC {it + 1}/{mpc_max_iters}] cost={plan_cost:.4f} "
            f"take={n_take} raw_steps={env_actions.shape[0]} dist={state_dist:.4f}",
            flush=True,
        )

        act_list = [torch.tensor(a, dtype=torch.float32) for a in env_actions]
        obses, _, _, infos = env.step_multiple(act_list)

        new_pixels, new_proprio = extract_macro_frames(obses, n_take, frameskip)
        hist_act_raw = np.concatenate([hist_act_raw, env_actions], axis=0)
        need = history_size * frameskip
        hist_act_raw = hist_act_raw[-need:]

        # Prefer env macros for history; pad with previous if shorter than H
        hist_pixels = (hist_pixels + new_pixels)[-history_size:]
        hist_proprio = (hist_proprio + new_proprio)[-history_size:]

        final_state = as_numpy(infos["state"][-1]).astype(np.float32).reshape(-1)
        state_dist = float(env.eval_state(goal, final_state)["state_dist"])
        success = bool(env.eval_state(goal, final_state)["success"])
        if success:
            print(f"  [MPC] success after iter={it + 1}", flush=True)
            break

    return {
        "success": success,
        "state_dist": state_dist,
        "final_state": final_state,
        "plan_cost": float(np.mean(plan_costs)) if plan_costs else -1.0,
        "mpc_iters": n_mpc_iters,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def evaluate(args):
    effective_act_dim = args.frameskip * args.action_dim
    print("=" * 60)
    print("LeWM Wall 仿真评估")
    print("=" * 60)
    print(
        f"[INFO] PLAN_GPU={os.environ.get('PLAN_GPU')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"(进程内 cuda:0 = 该物理卡)"
    )
    print(
        f"[INFO] action_dim={args.action_dim} frameskip={args.frameskip} "
        f"effective_act_dim={effective_act_dim} history={args.history_size} "
        f"horizon={args.horizon} goal_offset={args.goal_offset} num_eval={args.num_eval}"
    )
    print(f"[INFO] planner={args.planner} n_taken={args.n_taken_actions} mpc_max_iters={args.mpc_max_iters}")
    print("[INFO] success: ||goal[:2]-cur[:2]|| < 4.5 （对齐 DINO-WM）")

    device = args.plan_device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        device = "cpu"

    print(f"[INFO] 加载模型: {args.ckpt}")
    model = load_lewm_model(
        args.ckpt,
        action_dim=effective_act_dim,
        history_size=args.history_size,
        embed_dim=args.embed_dim,
        smoothed_dim=args.smoothed_dim,
        encoder_scale=args.encoder_scale,
        patch_size=args.patch_size,
        img_size=args.img_size,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
    )
    model = model.to(device)

    process = build_processors(args.dataset)
    transform = {"pixels": img_transform(args.img_size), "goal": img_transform(args.img_size)}
    lookup = build_goal_lookup(args.dataset)
    eval_episodes, eval_starts = sample_eval_points(
        args.dataset,
        args.num_eval,
        args.goal_offset,
        args.history_size,
        args.frameskip,
        args.seed,
    )

    print("[INFO] 初始化 WallEnv...")
    env = make_wall_env(device="cpu")

    successes, dists, costs = [], [], []
    details = []
    t0 = time.time()

    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_starts)):
        print(f"\n[{i + 1}/{args.num_eval}] episode={ep_id} start={start_step}", flush=True)
        goal_step = int(start_step + args.goal_offset)
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]

        start_fr = load_frame(args.dataset, start_idx)
        goal_fr = load_frame(args.dataset, goal_idx)
        if start_fr["state"] is None or goal_fr["state"] is None:
            print("  [WARN] 缺少 state，跳过")
            continue
        if start_fr["door_location"] is None or start_fr["wall_location"] is None:
            print("  [WARN] 缺少 door/wall_location，跳过")
            continue

        clip = load_history_with_frameskip(
            args.dataset,
            lookup,
            ep_id,
            start_step,
            args.history_size,
            args.frameskip,
            args.action_dim,
        )
        clip["action"] = normalize_concat_actions(
            clip["action_raw"],
            process,
            args.history_size,
            args.frameskip,
            args.action_dim,
        )

        cem_kwargs = dict(
            num_samples=args.cem_num_samples,
            n_steps=args.cem_n_steps,
            topk=args.cem_topk,
            var_scale=args.cem_var_scale,
        )

        if args.planner == "mpc":
            sim = run_mpc_episode(
                model,
                env,
                init_clip=clip,
                init_state=start_fr["state"],
                goal_state=goal_fr["state"],
                goal_pixels=goal_fr["pixels"],
                goal_proprio=goal_fr["proprio"],
                door=start_fr["door_location"],
                wall=start_fr["wall_location"],
                process=process,
                transform=transform,
                history_size=args.history_size,
                frameskip=args.frameskip,
                action_dim=args.action_dim,
                horizon=args.horizon,
                n_taken_actions=args.n_taken_actions,
                mpc_max_iters=args.mpc_max_iters,
                cem_kwargs=cem_kwargs,
                device=device,
                seed=args.seed + i,
            )
            plan_cost = sim["plan_cost"]
            print(
                f"  door={start_fr['door_location']:.1f} wall={start_fr['wall_location']:.1f} "
                f"mpc_iters={sim['mpc_iters']}",
                flush=True,
            )
        else:
            info = build_info_dict(
                clip,
                goal_fr["pixels"],
                goal_fr["proprio"],
                process,
                transform,
            )
            planned_norm, plan_cost = cem_plan(
                model,
                info,
                horizon=args.horizon,
                action_dim=effective_act_dim,
                device=device,
                **cem_kwargs,
            )
            env_actions = denormalize_concat_actions(
                planned_norm.detach().cpu().numpy(),
                process,
                args.frameskip,
                args.action_dim,
            )
            print(
                f"  plan_cost={plan_cost:.4f} env_actions={env_actions.shape} "
                f"door={start_fr['door_location']:.1f} wall={start_fr['wall_location']:.1f}",
                flush=True,
            )
            sim = run_sim_episode(
                env,
                init_state=start_fr["state"],
                goal_state=goal_fr["state"],
                door=start_fr["door_location"],
                wall=start_fr["wall_location"],
                actions=env_actions,
                seed=args.seed + i,
            )

        successes.append(sim["success"])
        dists.append(sim["state_dist"])
        costs.append(float(plan_cost))
        detail = {
            "ep_id": int(ep_id),
            "start_step": int(start_step),
            "goal_step": goal_step,
            "success": sim["success"],
            "state_dist": sim["state_dist"],
            "plan_cost": float(plan_cost),
            "planner": args.planner,
        }
        if args.planner == "mpc":
            detail["mpc_iters"] = sim.get("mpc_iters", -1)
        details.append(detail)
        print(
            f"  state_dist={sim['state_dist']:.4f} success={sim['success']}",
            flush=True,
        )

    elapsed = time.time() - t0
    metrics = {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": 4.5,
        "mean_state_dist": float(np.mean(dists)) if dists else -1.0,
        "median_state_dist": float(np.median(dists)) if dists else -1.0,
        "mean_planning_cost": float(np.mean(costs)) if costs else -1.0,
        "elapsed_sec": float(elapsed),
        "action_dim": args.action_dim,
        "frameskip": args.frameskip,
        "effective_act_dim": effective_act_dim,
        "horizon": args.horizon,
        "goal_offset": args.goal_offset,
        "history_size": args.history_size,
        "planner": args.planner,
        "n_taken_actions": args.n_taken_actions if args.planner == "mpc" else None,
        "mpc_max_iters": args.mpc_max_iters if args.planner == "mpc" else None,
    }
    return metrics, details


def parse_args():
    p = argparse.ArgumentParser(description="LeWM Wall 仿真评估")
    p.add_argument("--dataset", default="./data/wall_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_wall_v1_weights.ckpt")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument(
        "--goal-offset",
        type=int,
        default=25,
        help="目标相对 start 的 raw 步数；默认 horizon*frameskip=25（对齐 DINO goal_H）",
    )
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="./outputs/wall_sim_results.txt")
    p.add_argument("--details-json", default="./outputs/wall_sim_details.json")

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)

    # 对齐 plan_wall.yaml: samples=300, opt_steps=10, topk=30
    p.add_argument("--cem-num-samples", type=int, default=300)
    p.add_argument("--cem-n-steps", type=int, default=10)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument(
        "--planner",
        choices=["mpc", "openloop"],
        default="mpc",
        help="mpc=滚动规划（默认）；openloop=旧开环 CEM",
    )
    p.add_argument(
        "--n-taken-actions",
        type=int,
        default=3,
        help="MPC 每轮执行的 macro action 数（Wall 消融最优为 3）",
    )
    p.add_argument(
        "--mpc-max-iters",
        type=int,
        default=8,
        help="MPC 最大重规划轮数；成功则提前停（Wall 消融最优为 8）",
    )
    p.add_argument(
        "--plan-gpu",
        type=str,
        default="2",
        help="规划用物理 GPU 编号（默认 2；在 import torch 前写入 CUDA_VISIBLE_DEVICES）",
    )
    p.add_argument("--plan-device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    metrics, details = evaluate(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("==== WALL SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    details_path = Path(args.details_json)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": details}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(
        f"success_rate={metrics['success_rate']:.1f}%  "
        f"mean_state_dist={metrics['mean_state_dist']:.4f}  "
        f"median_state_dist={metrics['median_state_dist']:.4f}"
    )
    print(f"结果写入: {out}")
    print(f"明细写入: {details_path}")


if __name__ == "__main__":
    main()
