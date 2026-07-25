#!/usr/bin/env python3
"""
LeWM Rope 仿真评估（DINO-WM FlexEnv / PyFleX）

PyTorch CUDA 与 PyFleX 不能共用同一张物理 GPU。
默认: PLAN_GPU=4 做 CEM，EGL_GPU=5 做 Flex（子进程）。

用法:
  source setup_env.sh
  export PLAN_GPU=4 EGL_GPU=5

  # 一键（规划卡≠仿真卡）
  python eval_rope_1.py --dataset ./data/rope_eval.h5 --ckpt ./data/lewm_rope_weights.ckpt \\
      --plan-gpu 4 --egl-gpu 5 --num-eval 2 --plan-device cuda

  # 若仍 segfault，用两步法（父进程必须完全退出后再仿真）:
  python eval_rope_1.py --dataset ./data/rope_eval.h5 --ckpt ./data/lewm_rope_weights.ckpt \\
      --plan-gpu 4 --egl-gpu 5 --num-eval 2 --plan-device cuda \\
      --cem-num-samples 64 --cem-n-steps 10 --success-threshold 1.0 \\
      --plan-only --plans-out ./outputs/rope_plans.pkl
  python eval_rope_sim_worker.py --plans ./outputs/rope_plans.pkl \\
      --output ./outputs/rope_metrics.json --success-threshold 1.0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _early_apply_gpu_argv():
    """在 import torch 之前从命令行读取 GPU 参数。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--egl-gpu" and i + 1 < len(argv):
            os.environ["EGL_GPU"] = argv[i + 1]
        elif a.startswith("--egl-gpu="):
            os.environ["EGL_GPU"] = a.split("=", 1)[1]
        elif a == "--plan-gpu" and i + 1 < len(argv):
            os.environ["PLAN_GPU"] = argv[i + 1]
        elif a.startswith("--plan-gpu="):
            os.environ["PLAN_GPU"] = a.split("=", 1)[1]


_early_apply_gpu_argv()


def _setup_runtime_env():
    """只设环境变量 / 路径，不 init pyflex。
    规划用 PLAN_GPU，仿真用 EGL_GPU —— 必须是两张不同的物理卡。
    """
    pyflex_root = ROOT / "AdaptiGraph" / "PyFleX"
    build_dir = pyflex_root / "bindings" / "build"
    sdl_lib = pyflex_root / "external" / "SDL2-2.0.4" / "lib" / "x64"

    os.environ.setdefault("PYFLEX_HEADLESS", "1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility,graphics")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    egl_gpu = str(os.environ.get("EGL_GPU", "5"))
    # 默认规划卡 = EGL_GPU-1（绕开），避免与 Flex 同卡
    default_plan = str(max(0, int(egl_gpu) - 1))
    plan_gpu = str(os.environ.get("PLAN_GPU", default_plan))
    if plan_gpu == egl_gpu:
        print(
            f"[WARN] PLAN_GPU({plan_gpu})==EGL_GPU({egl_gpu})，同卡会导致子进程 pyflex.init segfault；"
            f"自动把 PLAN_GPU 改为 {default_plan if default_plan != egl_gpu else '0'}"
        )
        plan_gpu = default_plan if default_plan != egl_gpu else ("0" if egl_gpu != "0" else "1")

    os.environ["EGL_GPU"] = egl_gpu
    os.environ["PLAN_GPU"] = plan_gpu
    # 父进程（PyTorch 规划）只看见 PLAN_GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = plan_gpu

    os.environ["PYFLEXROOT"] = str(pyflex_root)
    os.environ["PYTHONPATH"] = str(build_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    extras = [str(sdl_lib), "/usr/lib/nvidia"]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(extras + ([ld] if ld else []))

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

    print(f"[INFO] PYFLEXROOT={pyflex_root}")
    print(f"[INFO] PLAN_GPU={plan_gpu} (PyTorch)  EGL_GPU={egl_gpu} (PyFleX)  — 必须不同")


_setup_runtime_env()

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def chamfer_distance(x, y):
    """x,y: [B,N,D] — 只用前 3 维."""
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1)
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1)
    dis = torch.norm(torch.add(x, -y), 2, dim=3)
    dis_xy = torch.mean(torch.min(dis, dim=2)[0])
    dis_yx = torch.mean(torch.min(dis, dim=1)[0])
    return dis_xy + dis_yx


def init_flex_env():
    """规划完成后再调用：init pyflex + FlexEnvWrapper."""
    print("[INFO] Phase-2: 初始化 PyFleX / FlexEnv（规划已结束）...", flush=True)
    import pyflex
    from deformable_env.FlexEnvWrapper import FlexEnvWrapper
    from deformable_env.src.sim.sim_env import flex_env as flex_env_mod

    if not flex_env_mod._PYFLEX_INITIALIZED:
        pyflex.set_screenWidth(224)
        pyflex.set_screenHeight(224)
        pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
        pyflex.set_light_fov(70.0)
        print(f"[INFO] pyflex.init(True) EGL_GPU={os.environ.get('EGL_GPU')} ...", flush=True)
        pyflex.init(True)
        flex_env_mod._PYFLEX_INITIALIZED = True
        print("[INFO] pyflex.init OK", flush=True)

    env = FlexEnvWrapper("rope")
    print("[INFO] FlexEnv OK", flush=True)
    return env


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


def sample_eval_points(h5_path: str, num_eval: int, goal_offset_steps: int, history_size: int, seed: int = 42):
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]

    max_start = ep_len - goal_offset_steps - 1
    valid_ep = np.nonzero((max_start >= 0) & (ep_len >= history_size))[0]
    if len(valid_ep) == 0:
        raise RuntimeError("没有足够长的 episode 可用于评估")

    g = np.random.default_rng(seed)
    replace = len(valid_ep) < num_eval
    chosen_ep = g.choice(valid_ep, size=num_eval, replace=replace)

    starts = []
    for ep_id in chosen_ep:
        hi = int(max_start[ep_id])
        starts.append(int(g.integers(0, hi + 1)))

    starts = np.asarray(starts, dtype=np.int64)
    print(f"[INFO] {len(valid_ep)} 个有效 episode, 采样 {num_eval} 组 (replace={replace})")
    return chosen_ep.astype(np.int64), starts


def load_frame(h5_path: str, idx: int):
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][idx]
        action = f["action"][idx]
        proprio = f["proprio"][idx] if "proprio" in f else np.zeros(1, dtype=np.float32)
        state = f["state"][idx] if "state" in f else None
    return pixels, action, proprio, state


def load_sequence(h5_path: str, lookup, episode_id: int, start_step: int, history_size: int):
    pixels_list, action_list, proprio_list = [], [], []
    for t in range(history_size):
        idx = lookup[(int(episode_id), int(start_step + t))]
        pixels, action, proprio, _ = load_frame(h5_path, idx)
        pixels_list.append(pixels)
        action_list.append(action)
        proprio_list.append(proprio)
    return {
        "pixels": torch.from_numpy(np.stack(pixels_list)),
        "action": torch.from_numpy(np.stack(action_list)).float(),
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


def apply_processors(sample: dict, process: dict):
    out = dict(sample)
    for col, proc in process.items():
        if col.startswith("goal_") or col not in out:
            continue
        arr = out[col]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        out[col] = torch.from_numpy(proc.transform(arr)).float()
    return out


def denormalize_actions(actions: np.ndarray, process: dict) -> np.ndarray:
    if "action" not in process:
        return actions
    flat = actions.reshape(-1, actions.shape[-1])
    denorm = process["action"].inverse_transform(flat)
    return denorm.reshape(actions.shape).astype(np.float32)


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device=None):
    """在 CPU 上构建 info；device 仅作兼容参数，默认不搬 CUDA（避免与 PyFlex 抢上下文时在此处 segfault）。"""
    print("    [dbg] apply_processors...", flush=True)
    clip = apply_processors(clip, process)

    print("    [dbg] transform pixels...", flush=True)
    pixels = clip["pixels"]
    if torch.is_tensor(pixels):
        pixels_np = pixels.detach().cpu().numpy()
    else:
        pixels_np = np.asarray(pixels)

    frame_tensors = []
    for i in range(pixels_np.shape[0]):
        frame = pixels_np[i]
        # 统一成 uint8 HWC，再交给 torchvision
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        t = transform["pixels"](frame)  # -> (C,H,W) float
        frame_tensors.append(t)
    pixels = torch.stack(frame_tensors, dim=0)  # (T,C,H,W)
    # (B,S,T,C,H,W) 留在 CPU
    pixels = pixels.unsqueeze(0).unsqueeze(0)
    print(f"    [dbg] pixels={tuple(pixels.shape)} dtype={pixels.dtype}", flush=True)

    action = torch.nan_to_num(clip["action"].float(), 0.0).unsqueeze(0).unsqueeze(0)
    info = {"pixels": pixels, "action": action}

    if "proprio" in clip:
        info["proprio"] = clip["proprio"].float().unsqueeze(0).unsqueeze(0)

    print("    [dbg] transform goal...", flush=True)
    goal_np = np.asarray(goal_pixels)
    if goal_np.dtype != np.uint8:
        if goal_np.max() <= 1.0:
            goal_np = (goal_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            goal_np = goal_np.clip(0, 255).astype(np.uint8)
    if goal_np.ndim == 3 and goal_np.shape[0] in (1, 3) and goal_np.shape[-1] not in (1, 3):
        goal_np = np.transpose(goal_np, (1, 2, 0))
    goal_img = transform["goal"](goal_np)  # (C,H,W)
    info["goal"] = goal_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (B,S,T=1,C,H,W)
    print(f"    [dbg] goal={tuple(info['goal'].shape)}", flush=True)

    goal_prop = np.asarray(goal_proprio, dtype=np.float32).reshape(1, -1)
    if "goal_proprio" in process:
        goal_prop = process["goal_proprio"].transform(goal_prop)
    info["goal_proprio"] = torch.from_numpy(goal_prop).float().unsqueeze(0).unsqueeze(0)
    print("    [dbg] build_info_dict done (CPU)", flush=True)
    return info


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
    # 将 info 搬到规划设备（build_info 已留在 CPU）
    print(f"    [dbg] move info -> {device}...", flush=True)
    info = {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in info.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("    [dbg] info on device OK", flush=True)

    history = info["pixels"].shape[2]
    total_steps = history + horizon

    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = info["action"][:, 0, :history]

    best_cost = None
    best_actions = None

    for step in range(n_steps):
        if step == 0 or (step + 1) % max(1, n_steps // 5) == 0 or step + 1 == n_steps:
            print(f"    [dbg] CEM iter {step+1}/{n_steps}...", flush=True)
        noise = torch.randn(num_samples, total_steps, action_dim, device=device)
        candidates = mean + std * noise
        candidates[:, :history] = info["action"][0, 0, :history]
        candidates_input = candidates.unsqueeze(0)

        costs = model.get_cost(info, candidates_input)
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
        mlp_scale=4,
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
        print(f"[WARN] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:8]}...")

    w = model.action_encoder.patch_embed.weight
    print(f"[INFO] action_encoder.patch_embed.weight = {tuple(w.shape)} (期望 {(smoothed_dim, action_dim, 1)})")

    model.eval()
    model.requires_grad_(False)
    if hasattr(model.encoder, "interpolate_pos_encoding"):
        model.encoder.interpolate_pos_encoding = True
    return model


def to_particle_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        if state.size % 4 != 0:
            raise ValueError(f"state 长度 {state.size} 不能 reshape 为 (N, 4)")
        state = state.reshape(-1, 4)
    return state


def plan_episodes(
    h5_path: str,
    model,
    process,
    transform,
    device,
    num_eval: int = 50,
    goal_offset_steps: int = 15,
    horizon: int = 5,
    history_size: int = 3,
    action_dim: int = 4,
    seed: int = 42,
    cem_num_samples: int = 300,
    cem_n_steps: int = 30,
    cem_topk: int = 30,
    cem_var_scale: float = 1.0,
):
    """Phase-1: 只做 CEM 规划，不碰 PyFleX。"""
    lookup = build_goal_lookup(h5_path)
    eval_episodes, eval_start_idx = sample_eval_points(
        h5_path, num_eval, goal_offset_steps, history_size, seed
    )

    plans = []
    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_start_idx)):
        print(f"\n[plan {i + 1}/{num_eval}] episode={ep_id} start={start_step}", flush=True)
        goal_step = int(start_step + goal_offset_steps)
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]

        _, _, _, init_state = load_frame(h5_path, start_idx)
        goal_pixels, _, goal_proprio, goal_state = load_frame(h5_path, goal_idx)
        if init_state is None or goal_state is None:
            print("  [WARN] 缺少 state，跳过")
            continue

        clip = load_sequence(h5_path, lookup, ep_id, start_step, history_size)
        info = build_info_dict(clip, goal_pixels, goal_proprio, process, transform)
        planned_norm, plan_cost = cem_plan(
            model,
            info,
            horizon=horizon,
            action_dim=action_dim,
            num_samples=cem_num_samples,
            n_steps=cem_n_steps,
            topk=cem_topk,
            var_scale=cem_var_scale,
            device=device,
        )
        planned_actions = denormalize_actions(planned_norm.detach().cpu().numpy(), process)
        print(f"  plan_cost={plan_cost:.4f}", flush=True)
        plans.append(
            {
                "ep_id": int(ep_id),
                "start_step": int(start_step),
                "seed": int(seed + i),
                "init_state": init_state,
                "goal_state": goal_state,
                "actions": planned_actions.astype(np.float32),
                "plan_cost": float(plan_cost),
            }
        )
    return plans


def execute_plans(env, plans, success_threshold: float = 1.0):
    """Phase-2: 在 Flex 仿真中执行已规划动作。"""
    successes = []
    chamfer_distances = []
    planning_costs = []

    for i, plan in enumerate(plans):
        print(
            f"\n[sim {i + 1}/{len(plans)}] episode={plan['ep_id']} start={plan['start_step']}",
            flush=True,
        )
        init_particles = to_particle_state(plan["init_state"]).astype(np.float32)
        env.prepare(plan["seed"], init_particles)

        infos = None
        for t, action in enumerate(plan["actions"]):
            _, _, _, infos = env.step_multiple([np.asarray(action, dtype=np.float32)])

        final_state = infos["state"][-1]
        goal_particles = to_particle_state(plan["goal_state"])
        final_np = np.asarray(final_state, dtype=np.float32)
        if final_np.ndim == 1:
            final_np = final_np.reshape(-1, 4)
        cd = chamfer_distance(
            torch.from_numpy(np.asarray(goal_particles, dtype=np.float32)[None, ...]),
            torch.from_numpy(final_np[None, ...]),
        ).item()
        success = cd < success_threshold
        successes.append(bool(success))
        chamfer_distances.append(cd)
        planning_costs.append(plan["plan_cost"])
        print(f"  chamfer={cd:.6f} success={success} (thr={success_threshold})", flush=True)

    cds = np.asarray(chamfer_distances, dtype=np.float64) if chamfer_distances else np.array([])
    return {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": success_threshold,
        "mean_chamfer_distance": float(cds.mean()) if cds.size else -1.0,
        "median_chamfer_distance": float(np.median(cds)) if cds.size else -1.0,
        "mean_planning_cost": float(np.mean(planning_costs)) if planning_costs else -1.0,
    }


def release_torch_cuda(model=None):
    """释放规划模型。"""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    print("[INFO] 已释放 PyTorch CUDA 缓存", flush=True)


def run_sim_subprocess(plans, success_threshold: float) -> dict:
    """在全新子进程中跑 Flex；子进程独占 EGL_GPU。"""
    import pickle
    import subprocess
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="rope_eval_"))
    plans_path = tmpdir / "plans.pkl"
    metrics_path = tmpdir / "metrics.json"
    with open(plans_path, "wb") as f:
        pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)

    worker = ROOT / "eval_rope_sim_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "--plans",
        str(plans_path),
        "--output",
        str(metrics_path),
        "--success-threshold",
        str(success_threshold),
    ]
    env = os.environ.copy()
    egl_gpu = os.environ.get("EGL_GPU", "5")
    env["EGL_GPU"] = egl_gpu
    # 子进程必须清除 CUDA_VISIBLE_DEVICES：EGL_GPU 是物理 EGL 设备下标，
    # 不能与 CUDA 可见卡重映射混用（否则 pyflex.init 常直接 segfault）
    env.pop("CUDA_VISIBLE_DEVICES", None)
    print(
        f"[INFO] Phase-2: 子进程仿真 EGL_GPU={egl_gpu} CUDA_VISIBLE_DEVICES=(unset)",
        flush=True,
    )
    print(f"[INFO] cmd: {' '.join(cmd)}", flush=True)
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(
            f"仿真子进程失败, exit={ret.returncode}。"
            f"若仍 segfault，请用两步法（父进程完全退出后再仿真）:\n"
            f"  python eval_rope_1.py --dataset ./data/rope_eval.h5 "
            f"--ckpt ./data/lewm_rope_weights.ckpt "
            f"--plan-gpu 4 --egl-gpu 5 --num-eval 2 --plan-device cuda "
            f"--plan-only --plans-out ./outputs/rope_plans.pkl\n"
            f"  python eval_rope_sim_worker.py --plans ./outputs/rope_plans.pkl "
            f"--output ./outputs/rope_metrics.json --success-threshold 1.0"
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(metrics: dict, results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        f.write("==== ROPE SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def parse_args():
    p = argparse.ArgumentParser(description="LeWM Rope 仿真评估")
    p.add_argument("--dataset", default="./data/rope_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_rope_weights.ckpt")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=15)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Chamfer Distance < thr 算成功；rope 粒子尺度下 CD 常在数个单位，0.05 过严",
    )
    p.add_argument("--output", default="./outputs/rope_sim_results.txt")

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)

    p.add_argument("--cem-num-samples", type=int, default=300)
    p.add_argument("--cem-n-steps", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument(
        "--plan-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="CEM 规划设备",
    )
    p.add_argument("--plan-gpu", type=str, default=None, help="规划用物理 GPU，默认 EGL_GPU-1")
    p.add_argument("--egl-gpu", type=str, default=None, help="Flex/EGL 用物理 GPU，默认环境变量或 5")
    p.add_argument("--plans-out", type=str, default=None, help="保存规划结果 pkl")
    p.add_argument("--plan-only", action="store_true", help="只规划并保存，不仿真")
    p.add_argument("--sim-from-plans", type=str, default=None, help="跳过规划，直接对已有 pkl 仿真")
    p.add_argument("--smoke-env", action="store_true", help="只测 Flex 环境初始化")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LeWM Rope 仿真评估（规划卡 ≠ Flex 卡；仿真在子进程）")
    print("=" * 60)
    print(
        f"[INFO] PLAN_GPU={os.environ.get('PLAN_GPU')} EGL_GPU={os.environ.get('EGL_GPU')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if args.smoke_env:
        import pickle
        import subprocess
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="rope_smoke_"))
        plans_path = tmp / "plans.pkl"
        metrics_path = tmp / "m.json"
        with open(plans_path, "wb") as f:
            pickle.dump([], f)
        env = os.environ.copy()
        env["EGL_GPU"] = env.get("EGL_GPU", "5")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [
            sys.executable,
            str(ROOT / "eval_rope_sim_worker.py"),
            "--plans",
            str(plans_path),
            "--output",
            str(metrics_path),
        ]
        print(
            f"[INFO] smoke-env via subprocess EGL_GPU={env['EGL_GPU']} "
            f"CUDA_VISIBLE_DEVICES=(unset)...",
            flush=True,
        )
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise RuntimeError("smoke-env 失败")
        print("[INFO] --smoke-env 通过，退出")
        return

    # 仅仿真已有规划
    if args.sim_from_plans:
        import pickle

        with open(args.sim_from_plans, "rb") as f:
            plans = pickle.load(f)
        print(f"[INFO] 从 {args.sim_from_plans} 加载 {len(plans)} 条规划", flush=True)
        metrics = run_sim_subprocess(plans, success_threshold=args.success_threshold)
        metrics["mode"] = "sim_from_plans"
        print("\n" + "=" * 60)
        print("结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        write_results(metrics, Path(args.output))
        print(f"\n已写入: {args.output}")
        return

    dataset_path = Path(args.dataset)
    ckpt_path = Path(args.ckpt)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重不存在: {ckpt_path}")

    effective_act_dim = args.frameskip * args.action_dim
    print(f"dataset: {dataset_path}")
    print(f"ckpt:    {ckpt_path}")
    print(f"num_eval={args.num_eval} goal_offset={args.goal_offset} horizon={args.horizon}")
    print(f"history={args.history_size} action_dim={effective_act_dim} success_thr={args.success_threshold}")

    process = build_processors(str(dataset_path), ("action", "proprio"))
    transform = {"pixels": img_transform(args.img_size), "goal": img_transform(args.img_size)}

    print("\n[INFO] Phase-1: 加载模型并 CEM 规划（无 PyFleX）...", flush=True)
    model = load_lewm_model(
        str(ckpt_path),
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
    if args.plan_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，plan-device 改用 cpu")
        args.plan_device = "cpu"
    device = torch.device(args.plan_device)
    model = model.to(device)
    print(f"[INFO] plan device={device}", flush=True)

    t0 = time.time()
    plans = plan_episodes(
        str(dataset_path),
        model,
        process,
        transform,
        device,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset,
        horizon=args.horizon,
        history_size=args.history_size,
        action_dim=effective_act_dim,
        seed=args.seed,
        cem_num_samples=args.cem_num_samples,
        cem_n_steps=args.cem_n_steps,
        cem_topk=args.cem_topk,
        cem_var_scale=args.cem_var_scale,
    )
    plan_time = time.time() - t0
    print(f"[INFO] 规划完成: {len(plans)} 条, {plan_time:.1f}s", flush=True)

    plans_out = args.plans_out or str(ROOT / "outputs" / "rope_plans.pkl")
    Path(plans_out).parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(plans_out, "wb") as f:
        pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] 规划已保存: {plans_out}", flush=True)

    release_torch_cuda(model)
    model = None

    if args.plan_only:
        print("[INFO] --plan-only：请另开命令跑仿真（父进程需完全退出）:")
        print(
            f"  python eval_rope_sim_worker.py --plans {plans_out} "
            f"--output {args.output}.metrics.json --success-threshold {args.success_threshold}"
        )
        return

    t1 = time.time()
    metrics = run_sim_subprocess(plans, success_threshold=args.success_threshold)
    sim_time = time.time() - t1
    metrics.update(
        {
            "mode": "sim_two_process",
            "plan_device": args.plan_device,
            "plan_gpu": os.environ.get("PLAN_GPU"),
            "egl_gpu": os.environ.get("EGL_GPU"),
            "plans_path": plans_out,
            "planning_time_sec": plan_time,
            "simulation_time_sec": sim_time,
            "evaluation_time_sec": plan_time + sim_time,
        }
    )

    print("\n" + "=" * 60)
    print("结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    out = Path(args.output)
    write_results(metrics, out)
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
