#!/usr/bin/env python3
"""
LeWM Granular 仿真评估（DINO-WM FlexEnv / PyFleX）

PyTorch CUDA 与 PyFleX 不能共用同一张物理 GPU。
可用卡: 3,4,5,6,7。默认 PLAN_GPU=3（CEM），EGL_GPU=4（Flex 子进程）。

用法:
  source setup_env.sh

  # 一键（规划卡 ≠ 仿真卡）
  python eval_granular.py \\
    --dataset ./data/granular_eval.h5 \\
    --ckpt ./data/lewm_granular_v1_weights.ckpt \\
    --plan-gpu 3 --egl-gpu 4 --num-eval 10 --plan-device cuda

  # 若子进程 segfault，用两步法（父进程完全退出后再仿真）:
  python eval_granular.py \\
    --dataset ./data/granular_eval.h5 \\
    --ckpt ./data/lewm_granular_v1_weights.ckpt \\
    --plan-gpu 3 --egl-gpu 4 --num-eval 10 --plan-device cuda \\
    --plan-only --plans-out ./outputs/granular_plans.pkl
  unset CUDA_VISIBLE_DEVICES
  export EGL_GPU=4
  python eval_granular_sim_worker.py \\
    --plans ./outputs/granular_plans.pkl \\
    --output ./outputs/granular_metrics.json --success-threshold 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 默认可用卡对：规划 3 / 仿真 4（必须不同）
_DEFAULT_PLAN_GPU = "3"
_DEFAULT_EGL_GPU = "4"


def _early_apply_gpu_argv():
    """在 import torch / eval_rope_1 之前锁定 GPU。"""
    argv = sys.argv[1:]
    plan_gpu = os.environ.get("PLAN_GPU", _DEFAULT_PLAN_GPU)
    egl_gpu = os.environ.get("EGL_GPU", _DEFAULT_EGL_GPU)
    for i, a in enumerate(argv):
        if a == "--egl-gpu" and i + 1 < len(argv):
            egl_gpu = argv[i + 1]
        elif a.startswith("--egl-gpu="):
            egl_gpu = a.split("=", 1)[1]
        elif a == "--plan-gpu" and i + 1 < len(argv):
            plan_gpu = argv[i + 1]
        elif a.startswith("--plan-gpu="):
            plan_gpu = a.split("=", 1)[1]

    if plan_gpu == egl_gpu:
        # 同卡会 segfault；自动换一张可用卡
        alts = [g for g in ("3", "4", "5", "6", "7") if g != egl_gpu]
        plan_gpu = alts[0] if alts else ("0" if egl_gpu != "0" else "1")
        print(
            f"[WARN] PLAN_GPU 与 EGL_GPU 相同，已自动将 PLAN_GPU 改为 {plan_gpu}",
            flush=True,
        )

    os.environ["PLAN_GPU"] = str(plan_gpu)
    os.environ["EGL_GPU"] = str(egl_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan_gpu)


_early_apply_gpu_argv()

# 复用 rope 评估里的 CEM / 模型加载逻辑（import 时会再跑一遍 GPU setup，沿用上面的环境变量）
from eval_rope_1 import (  # noqa: E402
    build_processors,
    img_transform,
    load_lewm_model,
    plan_episodes,
    release_torch_cuda,
)


def run_sim_subprocess(plans, success_threshold: float) -> dict:
    """在全新子进程中跑 Flex（granular）；子进程独占 EGL_GPU。"""
    import subprocess
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="granular_eval_"))
    plans_path = tmpdir / "plans.pkl"
    metrics_path = tmpdir / "metrics.json"
    with open(plans_path, "wb") as f:
        pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)

    worker = ROOT / "eval_granular_sim_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "--plans",
        str(plans_path),
        "--output",
        str(metrics_path),
        "--success-threshold",
        str(success_threshold),
        "--object-name",
        "granular",
    ]
    env = os.environ.copy()
    egl_gpu = os.environ.get("EGL_GPU", _DEFAULT_EGL_GPU)
    env["EGL_GPU"] = egl_gpu
    env.pop("CUDA_VISIBLE_DEVICES", None)
    print(
        f"[INFO] Phase-2: 子进程仿真 EGL_GPU={egl_gpu} CUDA_VISIBLE_DEVICES=(unset)",
        flush=True,
    )
    print(f"[INFO] cmd: {' '.join(cmd)}", flush=True)
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(
            f"仿真子进程失败, exit={ret.returncode}。\n"
            f"请用两步法（父进程完全退出后再仿真）:\n"
            f"  python eval_granular.py --plan-only --plans-out ./outputs/granular_plans.pkl ...\n"
            f"  unset CUDA_VISIBLE_DEVICES; export EGL_GPU={egl_gpu}\n"
            f"  python eval_granular_sim_worker.py --plans ./outputs/granular_plans.pkl "
            f"--output ./outputs/granular_metrics.json --success-threshold {success_threshold}"
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(metrics: dict, results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        f.write("==== GRANULAR SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def parse_args():
    p = argparse.ArgumentParser(description="LeWM Granular 仿真评估")
    p.add_argument("--dataset", default="./data/granular_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_granular_v1_weights.ckpt")
    p.add_argument("--num-eval", type=int, default=10)
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
        help="Chamfer Distance < thr 算成功",
    )
    p.add_argument("--output", default="./outputs/granular_sim_results.txt")

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
    p.add_argument(
        "--plan-gpu",
        type=str,
        default=_DEFAULT_PLAN_GPU,
        help=f"规划用物理 GPU（可用 3/4/5/6/7），默认 {_DEFAULT_PLAN_GPU}",
    )
    p.add_argument(
        "--egl-gpu",
        type=str,
        default=_DEFAULT_EGL_GPU,
        help=f"Flex/EGL 用物理 GPU（须与 plan-gpu 不同），默认 {_DEFAULT_EGL_GPU}",
    )
    p.add_argument("--plans-out", type=str, default=None, help="保存规划结果 pkl")
    p.add_argument("--plan-only", action="store_true", help="只规划并保存，不仿真")
    p.add_argument("--sim-from-plans", type=str, default=None, help="跳过规划，直接对已有 pkl 仿真")
    p.add_argument("--smoke-env", action="store_true", help="只测 Flex 环境初始化")
    return p.parse_args()


def main():
    import torch

    args = parse_args()

    # 再次对齐（parse 后的值）
    if args.plan_gpu == args.egl_gpu:
        raise SystemExit(
            f"PLAN_GPU 与 EGL_GPU 不能相同（当前都是 {args.plan_gpu}）。"
            f"可用卡: 3,4,5,6,7，例如 --plan-gpu 3 --egl-gpu 4"
        )
    os.environ["PLAN_GPU"] = str(args.plan_gpu)
    os.environ["EGL_GPU"] = str(args.egl_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.plan_gpu)

    print("=" * 60)
    print("LeWM Granular 仿真评估（规划卡 ≠ Flex 卡；仿真在子进程）")
    print("=" * 60)
    print(
        f"[INFO] PLAN_GPU={os.environ.get('PLAN_GPU')} EGL_GPU={os.environ.get('EGL_GPU')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if args.smoke_env:
        import subprocess
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="granular_smoke_"))
        plans_path = tmp / "plans.pkl"
        metrics_path = tmp / "m.json"
        with open(plans_path, "wb") as f:
            pickle.dump([], f)
        env = os.environ.copy()
        env["EGL_GPU"] = str(args.egl_gpu)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [
            sys.executable,
            str(ROOT / "eval_granular_sim_worker.py"),
            "--plans",
            str(plans_path),
            "--output",
            str(metrics_path),
            "--object-name",
            "granular",
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

    if args.sim_from_plans:
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
    print(
        f"history={args.history_size} action_dim={effective_act_dim} "
        f"success_thr={args.success_threshold}"
    )

    process = build_processors(str(dataset_path), ("action", "proprio"))
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }

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

    plans_out = args.plans_out or str(ROOT / "outputs" / "granular_plans.pkl")
    Path(plans_out).parent.mkdir(parents=True, exist_ok=True)
    with open(plans_out, "wb") as f:
        pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] 规划已保存: {plans_out}", flush=True)

    release_torch_cuda(model)
    model = None

    if args.plan_only:
        print("[INFO] --plan-only：请另开命令跑仿真（父进程需完全退出）:")
        print(
            f"  unset CUDA_VISIBLE_DEVICES; export EGL_GPU={args.egl_gpu}\n"
            f"  python eval_granular_sim_worker.py --plans {plans_out} "
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
