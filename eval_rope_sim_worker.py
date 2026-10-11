#!/usr/bin/env python3
"""
Flex 仿真执行子进程：故意不 import torch / JEPA，避免与 PyFleX 抢 GPU。

重要：不要设 CUDA_VISIBLE_DEVICES=<EGL_GPU>。
EGL_GPU 是 eglQueryDevicesEXT 的下标（物理卡序号），不受 CUDA_VISIBLE_DEVICES 重映射。
若 CUDA_VISIBLE_DEVICES=5 且 EGL_GPU=5，常见直接 segfault（你之前 smoke 能过时通常没限制可见卡）。

用法:
  unset CUDA_VISIBLE_DEVICES
  export EGL_GPU=5
  python eval_rope_sim_worker.py --plans ./outputs/rope_plans.pkl \\
      --output ./outputs/rope_metrics.json --success-threshold 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def setup_env():
    pyflex_root = ROOT / "AdaptiGraph" / "PyFleX"
    build_dir = pyflex_root / "bindings" / "build"
    sdl_lib = pyflex_root / "external" / "SDL2-2.0.4" / "lib" / "x64"

    os.environ.setdefault("PYFLEX_HEADLESS", "1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility,graphics")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    egl_gpu = os.environ.get("EGL_GPU", "5")
    os.environ["EGL_GPU"] = str(egl_gpu)

    # 关键：清除可见卡限制，让 EGL_GPU 与物理设备序号对齐
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    os.environ["PYFLEXROOT"] = str(pyflex_root)
    os.environ["PYTHONPATH"] = str(build_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    extras = [str(sdl_lib), "/usr/lib/nvidia"]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(extras + ([ld] if ld else []))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))


def chamfer_distance_np(x: np.ndarray, y: np.ndarray) -> float:
    """x,y: (N,D)/(M,D)，只用前 3 维。"""
    x = np.asarray(x, dtype=np.float64)[:, :3]
    y = np.asarray(y, dtype=np.float64)[:, :3]
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def to_particle_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        state = state.reshape(-1, 4)
    return state


def save_video(frames: np.ndarray, path: Path, fps: int = 5) -> None:
    """frames: (T,H,W,3) uint8 RGB。优先 imageio，否则逐帧 PNG。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)

    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), list(frames), fps=fps)
        print(f"[sim-worker] video -> {path}", flush=True)
        return
    except Exception as e:
        print(f"[sim-worker] imageio 写 mp4 失败 ({e})，改存 PNG 序列", flush=True)

    frame_dir = path.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        import matplotlib.pyplot as plt

        for t, fr in enumerate(frames):
            plt.imsave(str(frame_dir / f"frame_{t:04d}.png"), fr)
        print(f"[sim-worker] frames -> {frame_dir}/", flush=True)
        return

    for t, fr in enumerate(frames):
        Image.fromarray(fr).save(frame_dir / f"frame_{t:04d}.png")
    print(f"[sim-worker] frames -> {frame_dir}/", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="若指定，则把每条 episode 的仿真 RGB 存成 mp4（需 imageio）",
    )
    parser.add_argument("--video-fps", type=int, default=5)
    args = parser.parse_args()

    setup_env()
    print(
        f"[sim-worker] EGL_GPU={os.environ.get('EGL_GPU')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}",
        flush=True,
    )

    # 先 init pyflex，再 import FlexEnv（避免无关 import 干扰）
    import pyflex

    pyflex.set_screenWidth(224)
    pyflex.set_screenHeight(224)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.0)
    print("[sim-worker] pyflex.init(True)...", flush=True)
    pyflex.init(True)
    print("[sim-worker] pyflex.init OK", flush=True)

    from deformable_env.FlexEnvWrapper import FlexEnvWrapper
    from deformable_env.src.sim.sim_env import flex_env as flex_env_mod

    flex_env_mod._PYFLEX_INITIALIZED = True
    env = FlexEnvWrapper("rope")
    print("[sim-worker] FlexEnv OK", flush=True)

    with open(args.plans, "rb") as f:
        plans = pickle.load(f)

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    successes = []
    cds = []
    costs = []
    video_paths = []
    details = []
    for i, plan in enumerate(plans):
        print(
            f"[sim-worker {i+1}/{len(plans)}] ep={plan['ep_id']} start={plan['start_step']}",
            flush=True,
        )
        init_particles = to_particle_state(plan["init_state"])
        actions = np.asarray(plan["actions"], dtype=np.float32)

        if video_dir is not None:
            # rollout 返回含初始帧的 visual: (T+1, H, W, 3) RGB
            obses, states = env.rollout(plan["seed"], init_particles, actions)
            final_state = np.asarray(states[-1], dtype=np.float32)
            frames = np.asarray(obses["visual"])
            vpath = video_dir / (
                f"ep{plan['ep_id']:04d}_start{plan['start_step']:03d}.mp4"
            )
            save_video(frames, vpath, fps=args.video_fps)
            video_paths.append(str(vpath))
        else:
            env.prepare(plan["seed"], init_particles)
            infos = None
            for action in actions:
                _, _, _, infos = env.step_multiple([action])
            final_state = np.asarray(infos["state"][-1], dtype=np.float32)

        if final_state.ndim == 1:
            final_state = final_state.reshape(-1, 4)
        goal = to_particle_state(plan["goal_state"])
        cd = chamfer_distance_np(goal, final_state)
        ok = cd < args.success_threshold
        successes.append(bool(ok))
        cds.append(cd)
        costs.append(float(plan["plan_cost"]))
        details.append({"ep_id": int(plan["ep_id"]), "start_step": int(plan["start_step"]),
                        "chamfer": float(cd), "success": bool(ok), "plan_cost": float(plan["plan_cost"])})
        print(f"  chamfer={cd:.6f} success={ok}", flush=True)

    cds_arr = np.asarray(cds, dtype=np.float64)
    metrics = {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": args.success_threshold,
        "mean_chamfer_distance": float(cds_arr.mean()) if cds_arr.size else -1.0,
        "median_chamfer_distance": float(np.median(cds_arr)) if cds_arr.size else -1.0,
        "mean_planning_cost": float(np.mean(costs)) if costs else -1.0,
    }
    metrics["details"] = details
    if video_paths:
        metrics["videos"] = video_paths
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[sim-worker] wrote {args.output}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
