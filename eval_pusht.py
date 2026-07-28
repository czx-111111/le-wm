#!/usr/bin/env python3
"""
LeWM PushT 仿真评估（DINO-WM pusht_noise）

对齐训练配置：action_dim=2, frameskip=5 → 规划维数 10。
成功判据：pos_diff(:4)<20 且 angle_diff<π/9（与 DINO-WM PushTWrapper.eval_state 一致）。
评估条数：默认 50（对齐 plan_pusht.yaml 的 n_evals=50）。
默认规划卡：物理 GPU 2。

用法:
  python eval_pusht.py \\
    --dataset ./data/pusht_noise_eval.h5 \\
    --ckpt ./data/lewm_pusht_noise_2k_weights.ckpt \\
    --plan-gpu 2 --num-eval 50
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
    # PushT 渲染走 pygame；无显示时用 dummy
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


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

    need = max(
        goal_offset_steps,
        (history_size - 1) * frameskip,
        history_size * frameskip - 1,
    )
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
        proprio = f["proprio"][idx] if "proprio" in f else np.zeros(4, dtype=np.float32)
        state = f["state"][idx] if "state" in f else None
        ep_id = int(f["ep_idx"][idx]) if "ep_idx" in f else 0
        shape = "T"
        if "ep_shape" in f:
            raw = f["ep_shape"][ep_id]
            if isinstance(raw, (bytes, np.bytes_)):
                shape = raw.decode("utf-8")
            else:
                shape = str(raw)
    return {
        "pixels": pixels,
        "action": action,
        "proprio": proprio,
        "state": state,
        "shape": shape,
        "ep_id": ep_id,
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
        "action_raw": raw_actions,
        "action": torch.from_numpy(concat).float(),
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
    """raw (H*fs, D) → normalize → (H, fs*D)。"""
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
    """(horizon, fs*D) normalized → (horizon*fs, D) raw env actions（已是 /100 尺度）。"""
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
    frame_tensors = [
        transform["pixels"](_to_uint8_hwc(pixels_np[i])) for i in range(pixels_np.shape[0])
    ]
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
    n_steps: int = 30,
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

    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=smoothed_dim,
        emb_dim=embed_dim,
    )

    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d
    )
    predictor_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    w = model.action_encoder.patch_embed.weight
    print(
        f"[INFO] action_encoder.patch_embed.weight = {tuple(w.shape)} "
        f"(期望 {(smoothed_dim, action_dim, 1)})"
    )

    model.eval()
    model.requires_grad_(False)
    if hasattr(model.encoder, "interpolate_pos_encoding"):
        model.encoder.interpolate_pos_encoding = True
    return model


# ---------------------------------------------------------------------------
# PushT env
# ---------------------------------------------------------------------------


def _import_pusht_wrapper():
    """导入 PushTWrapper，避免执行 env/__init__.py（会强制加载 MuJoCo/pointmaze）。"""
    import types

    env_dir = ROOT / "env"
    pkg = types.ModuleType("env")
    pkg.__path__ = [str(env_dir)]
    pkg.__file__ = str(env_dir / "__init__.py")
    pkg.__package__ = "env"
    sys.modules["env"] = pkg

    pusht_pkg = types.ModuleType("env.pusht")
    pusht_pkg.__path__ = [str(env_dir / "pusht")]
    pusht_pkg.__package__ = "env.pusht"
    sys.modules["env.pusht"] = pusht_pkg

    from env.pusht.pusht_wrapper import PushTWrapper

    return PushTWrapper


def make_pusht_env(with_velocity: bool = True, with_target: bool = True, render_size: int = 224):
    try:
        PushTWrapper = _import_pusht_wrapper()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "导入 PushTEnv 失败。请确认已安装 pygame/pymunk/shapely/opencv，"
            f"且在仓库根目录运行。原始错误: {e}"
        ) from e

    env = PushTWrapper(with_velocity=with_velocity, with_target=with_target)
    # 与训练图像尺度对齐（env 默认也可能是 224）
    if hasattr(env, "render_size"):
        env.render_size = render_size
    return env


def run_sim_episode(env, init_state, goal_state, actions, seed: int, shape: str = "T"):
    """执行 raw actions (T, 2)，返回 success / state_dist。

    注意：HDF5 中 action 已 /100；PushTEnv.step 会再 * action_scale(=100)。
    """
    env.update_env({"shape": shape})
    init = np.asarray(init_state, dtype=np.float32).reshape(-1)
    env.prepare(seed, init)

    act_arr = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
    if act_arr.shape[0] == 0:
        raise RuntimeError("空动作序列")

    act_list = [a for a in act_arr]
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def evaluate(args):
    effective_act_dim = args.frameskip * args.action_dim
    print("=" * 60)
    print("LeWM PushT 仿真评估")
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
    print("[INFO] success: pos_diff(:4)<20 且 |angle|<π/9 （对齐 DINO-WM）")

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

    print("[INFO] 初始化 PushTEnv...")
    env = make_pusht_env(
        with_velocity=not args.no_velocity,
        with_target=True,
        render_size=args.img_size,
    )

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
            print("  [WARN] 缺少 state，跳过（请用含 state 的 pusht_noise_eval.h5）")
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
            num_samples=args.cem_num_samples,
            n_steps=args.cem_n_steps,
            topk=args.cem_topk,
            var_scale=args.cem_var_scale,
            device=device,
        )
        env_actions = denormalize_concat_actions(
            planned_norm.detach().cpu().numpy(),
            process,
            args.frameskip,
            args.action_dim,
        )
        print(
            f"  plan_cost={plan_cost:.4f} env_actions={env_actions.shape} shape={start_fr['shape']}",
            flush=True,
        )

        sim = run_sim_episode(
            env,
            init_state=start_fr["state"],
            goal_state=goal_fr["state"],
            actions=env_actions,
            seed=args.seed + i,
            shape=start_fr["shape"],
        )
        successes.append(sim["success"])
        dists.append(sim["state_dist"])
        costs.append(float(plan_cost))
        details.append(
            {
                "ep_id": int(ep_id),
                "start_step": int(start_step),
                "goal_step": goal_step,
                "shape": start_fr["shape"],
                "success": sim["success"],
                "state_dist": sim["state_dist"],
                "plan_cost": float(plan_cost),
            }
        )
        print(
            f"  state_dist={sim['state_dist']:.4f} success={sim['success']}",
            flush=True,
        )

    elapsed = time.time() - t0
    metrics = {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_rule": "pos_diff(:4)<20 and angle_diff<pi/9",
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
        "with_velocity": not args.no_velocity,
    }
    return metrics, details


def parse_args():
    p = argparse.ArgumentParser(description="LeWM PushT 仿真评估")
    p.add_argument("--dataset", default="./data/pusht_noise_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_pusht_noise_2k_weights.ckpt")
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
    p.add_argument("--output", default="./outputs/pusht_sim_results.txt")
    p.add_argument("--details-json", default="./outputs/pusht_sim_details.json")

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)

    # 对齐 plan_pusht.yaml: samples=300, opt_steps=30, topk=30
    p.add_argument("--cem-num-samples", type=int, default=300)
    p.add_argument("--cem-n-steps", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument(
        "--plan-gpu",
        type=str,
        default="2",
        help="规划用物理 GPU 编号（默认 2；在 import torch 前写入 CUDA_VISIBLE_DEVICES）",
    )
    p.add_argument("--plan-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--no-velocity",
        action="store_true",
        help="关闭 velocity（默认开启，与训练 with_velocity 一致）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    metrics, details = evaluate(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("==== PUSHT SIMULATION EVALUATION ====\n")
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
