#!/usr/bin/env python3
"""
LeWM Granular 仿真评估（DINO-WM FlexEnv / PyFleX）

PyTorch CUDA 与 PyFleX 不能共用同一张物理 GPU。
可用卡: 3,4,5,6,7。默认 PLAN_GPU=3（CEM），EGL_GPU=4（Flex 子进程）。
协议：goal_offset 须与 horizon 对齐（默认均为 5）。

默认 planner=mpc（滚动规划；对齐 Wall 的 n_taken=3 / max_iters=8）。
--planner openloop 复现旧开环 CEM。

用法:
  source setup_env.sh

  # MPC（推荐；用最终轮 object / weights，不用 Ep50）
  python eval_granular.py \\
    --dataset ./data/granular_eval.h5 \\
    --ckpt ./data/lewm_granular_v1_epoch_100_object.ckpt \\
    --plan-gpu 3 --egl-gpu 4 --num-eval 50 \\
    --planner mpc --n-taken-actions 3 --mpc-max-iters 8 \\
    --goal-offset 5 --horizon 5 --plan-device cuda

  # 开环对照
  python eval_granular.py ... --planner openloop
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

_DEFAULT_PLAN_GPU = "0"
_DEFAULT_EGL_GPU = "1"


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
        alts = [g for g in ("0", "1") if g != egl_gpu]
        plan_gpu = alts[0] if alts else ("0" if egl_gpu != "0" else "1")
        print(
            f"[WARN] PLAN_GPU 与 EGL_GPU 相同，已自动将 PLAN_GPU 改为 {plan_gpu}",
            flush=True,
        )

    os.environ["PLAN_GPU"] = str(plan_gpu)
    os.environ["EGL_GPU"] = str(egl_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan_gpu)


_early_apply_gpu_argv()

from eval_mpc_utils import (  # noqa: E402
    EVAL_SPLITS,
    PLAN_PROTOCOLS,
    add_cost_cli_args,
    apply_cost_attrs,
    plan_context_len,
    resolve_stats_path,
    to_hwc_uint8_compatible,
)
from eval_rope_1 import (  # noqa: E402
    build_goal_lookup,
    build_info_dict,
    build_processors,
    cem_plan,
    denormalize_actions,
    img_transform,
    load_frame,
    load_lewm_model,
    load_sequence,
    plan_episodes,
    release_torch_cuda,
    sample_eval_points,
    to_particle_state,
)


def chamfer_distance_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)[:, :3]
    y = np.asarray(y, dtype=np.float64)[:, :3]
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def _atomic_write_pickle(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class FlexMpcClient:
    """与 eval_granular_mpc_worker.py 通信的常驻 Flex 客户端。"""

    def __init__(self, egl_gpu: str, object_name: str = "granular", timeout: float = 600.0):
        self.egl_gpu = str(egl_gpu)
        self.object_name = object_name
        self.timeout = timeout
        self.ipc = Path(tempfile.mkdtemp(prefix="granular_mpc_ipc_"))
        self.req_path = self.ipc / "request.pkl"
        self.resp_path = self.ipc / "response.pkl"
        self.ready_path = self.ipc / "ready.pkl"
        for p in (self.req_path, self.resp_path, self.ready_path):
            if p.exists():
                p.unlink()

        env = os.environ.copy()
        env["EGL_GPU"] = self.egl_gpu
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [
            sys.executable,
            str(ROOT / "eval_granular_mpc_worker.py"),
            "--ipc-dir",
            str(self.ipc),
            "--object-name",
            self.object_name,
        ]
        print(
            f"[INFO] 启动 MPC Flex worker EGL_GPU={self.egl_gpu} ipc={self.ipc}",
            flush=True,
        )
        self.proc = subprocess.Popen(cmd, env=env)
        self._wait_path(self.ready_path, label="worker ready")

    def _wait_path(self, path: Path, label: str = "file"):
        t0 = time.time()
        while not path.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"MPC worker 已退出 (code={self.proc.returncode})，等待 {label} 失败"
                )
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"等待 {label} 超时 ({self.timeout}s): {path}")
            time.sleep(0.02)

    def _call(self, req: dict) -> dict:
        if self.resp_path.exists():
            self.resp_path.unlink()
        _atomic_write_pickle(self.req_path, req)
        self._wait_path(self.resp_path, label=f"response for cmd={req.get('cmd')}")
        time.sleep(0.02)
        with open(self.resp_path, "rb") as f:
            resp = pickle.load(f)
        try:
            self.resp_path.unlink()
        except OSError:
            pass
        if not resp.get("ok", False):
            raise RuntimeError(f"MPC worker error: {resp.get('error')}")
        return resp

    def prepare(self, seed: int, init_state) -> dict:
        return self._call(
            {
                "cmd": "prepare",
                "seed": int(seed),
                "init_state": np.asarray(init_state, dtype=np.float32),
            }
        )

    def step(self, actions: np.ndarray) -> dict:
        return self._call(
            {"cmd": "step", "actions": np.asarray(actions, dtype=np.float32)}
        )

    def close(self):
        try:
            if self.proc.poll() is None:
                self._call({"cmd": "shutdown"})
                self.proc.wait(timeout=60)
        except Exception as e:
            print(f"[WARN] MPC worker shutdown: {e}", flush=True)
            if self.proc.poll() is None:
                self.proc.kill()


def run_sim_subprocess(plans, success_threshold: float) -> dict:
    """开环：一次性子进程跑完全部 plans。"""
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
        f"[INFO] Phase-2 openloop: 子进程仿真 EGL_GPU={egl_gpu}",
        flush=True,
    )
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"仿真子进程失败, exit={ret.returncode}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(metrics: dict, results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        f.write("==== GRANULAR SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            if k == "details":
                continue
            f.write(f"{k}: {v}\n")
    with results_path.with_name(results_path.stem + "_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _fit_proprio(prop: np.ndarray, ref_dim: int) -> np.ndarray:
    prop = np.asarray(prop, dtype=np.float32).reshape(-1)
    if prop.size == ref_dim:
        return prop
    if prop.size > ref_dim:
        return prop[:ref_dim]
    out = np.zeros(ref_dim, dtype=np.float32)
    out[: prop.size] = prop
    return out


def run_mpc_episodes(
    h5_path: str,
    model,
    process,
    transform,
    device,
    client: FlexMpcClient,
    *,
    num_eval: int,
    goal_offset_steps: int,
    horizon: int,
    history_size: int,
    action_dim: int,
    n_taken_actions: int,
    mpc_max_iters: int,
    success_threshold: float,
    seed: int,
    cem_kwargs: dict,
    plan_ctx: int,
    split: str = "dev",
) -> dict:
    """MPC：CEM → 执行 n_taken → 用仿真观测重规划。"""
    import torch

    lookup = build_goal_lookup(h5_path)
    eval_episodes, eval_start_idx = sample_eval_points(
        h5_path, num_eval, goal_offset_steps, history_size, seed, split=split
    )
    num_eval = len(eval_episodes)

    successes = []
    cds = []
    costs = []
    mpc_iters_list = []

    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_start_idx)):
        print(
            f"\n[mpc {i + 1}/{num_eval}] episode={ep_id} start={start_step}",
            flush=True,
        )
        torch.manual_seed(int(seed) + i)
        goal_step = int(start_step + goal_offset_steps)
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]

        _, _, _, init_state = load_frame(h5_path, start_idx)
        goal_pixels, _, goal_proprio, goal_state = load_frame(h5_path, goal_idx)
        if init_state is None or goal_state is None:
            print("  [WARN] 缺少 state，跳过")
            continue

        clip0 = load_sequence(h5_path, lookup, ep_id, start_step, plan_ctx)
        pixels_np = clip0["pixels"].numpy() if hasattr(clip0["pixels"], "numpy") else np.asarray(clip0["pixels"])
        proprio_np = (
            clip0["proprio"].numpy()
            if hasattr(clip0["proprio"], "numpy")
            else np.asarray(clip0["proprio"])
        )
        act_np = (
            clip0["action"].numpy()
            if hasattr(clip0["action"], "numpy")
            else np.asarray(clip0["action"])
        )

        hist_pixels = [to_hwc_uint8_compatible(pixels_np[t]) for t in range(plan_ctx)]
        hist_proprio = [np.asarray(proprio_np[t], dtype=np.float32).reshape(-1) for t in range(plan_ctx)]
        prop_dim = int(hist_proprio[0].size)
        hist_act = [np.asarray(act_np[t], dtype=np.float32).reshape(-1) for t in range(plan_ctx)]

        goal_particles = to_particle_state(goal_state)
        prep = client.prepare(seed=int(seed + i), init_state=init_state)
        final_state = to_particle_state(prep["state"])
        cd = chamfer_distance_np(goal_particles, final_state)
        plan_costs = []
        n_iters = 0
        success = cd < success_threshold

        for it in range(mpc_max_iters):
            cd = chamfer_distance_np(goal_particles, final_state)
            if cd < success_threshold:
                success = True
                print(f"  [MPC] early success at iter={it} cd={cd:.4f}", flush=True)
                break

            clip = {
                "pixels": np.stack(hist_pixels),
                "action": np.stack(hist_act).astype(np.float32),
                "proprio": np.stack(hist_proprio).astype(np.float32),
            }
            clip_t = {
                "pixels": torch.from_numpy(clip["pixels"]),
                "action": torch.from_numpy(clip["action"]).float(),
                "proprio": torch.from_numpy(clip["proprio"]).float(),
            }
            # build_info_dict 期望 pixels 可转 numpy；uint8 HWC ok
            info = build_info_dict(
                clip_t, goal_pixels, goal_proprio, process, transform
            )
            planned_norm, plan_cost = cem_plan(
                model,
                info,
                horizon=horizon,
                action_dim=action_dim,
                device=device,
                **cem_kwargs,
            )
            plan_costs.append(float(plan_cost))
            n_iters = it + 1

            planned = denormalize_actions(
                planned_norm.detach().cpu().numpy(), process
            )
            n_take = min(int(n_taken_actions), int(planned.shape[0]))
            take_actions = planned[:n_take]
            print(
                f"  [MPC {it + 1}/{mpc_max_iters}] cost={plan_cost:.4f} "
                f"take={n_take} cd={cd:.4f}",
                flush=True,
            )

            step_out = client.step(take_actions)
            visuals = np.asarray(step_out["visuals"])
            proprios = np.asarray(step_out["proprios"])
            final_state = to_particle_state(step_out["final_state"])

            new_pixels = []
            new_proprio = []
            for t in range(n_take):
                new_pixels.append(to_hwc_uint8_compatible(visuals[t]))
                if proprios.ndim == 1:
                    p = proprios
                else:
                    p = proprios[t]
                new_proprio.append(_fit_proprio(p, prop_dim))

            for t in range(n_take):
                hist_act.append(np.asarray(take_actions[t], dtype=np.float32).reshape(-1))
            hist_act = hist_act[-plan_ctx:]
            hist_pixels = (hist_pixels + new_pixels)[-plan_ctx:]
            hist_proprio = (hist_proprio + new_proprio)[-plan_ctx:]

            cd = chamfer_distance_np(goal_particles, final_state)
            success = cd < success_threshold
            if success:
                print(f"  [MPC] success after iter={it + 1} cd={cd:.4f}", flush=True)
                break

        successes.append(bool(success))
        cds.append(float(cd))
        costs.append(float(np.mean(plan_costs)) if plan_costs else -1.0)
        mpc_iters_list.append(int(n_iters))
        print(
            f"  done: success={success} cd={cd:.4f} mpc_iters={n_iters}",
            flush=True,
        )

    cds_arr = np.asarray(cds, dtype=np.float64) if cds else np.array([])
    return {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": success_threshold,
        "mean_chamfer_distance": float(cds_arr.mean()) if cds_arr.size else -1.0,
        "median_chamfer_distance": float(np.median(cds_arr)) if cds_arr.size else -1.0,
        "mean_planning_cost": float(np.mean(costs)) if costs else -1.0,
        "mean_mpc_iters": float(np.mean(mpc_iters_list)) if mpc_iters_list else -1.0,
        "object_name": "granular",
        "planner": "mpc",
        "n_taken_actions": n_taken_actions,
        "mpc_max_iters": mpc_max_iters,
    }


def parse_args():
    p = argparse.ArgumentParser(description="LeWM Granular 仿真评估")
    p.add_argument("--dataset", default="./data/granular_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_granular_v1_weights.ckpt")
    p.add_argument("--num-eval", type=int, default=0, help="0 = every episode of --split exactly once")
    p.add_argument(
        "--goal-offset",
        type=int,
        default=5,
        help="目标相对 start 的步数；须与 --horizon 对齐",
    )
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--success-threshold", type=float, default=1.0)
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
    p.add_argument("--token-mode", default="cls", choices=["cls", "spatial16", "spatial"])
    p.add_argument("--num-spatial-tokens", type=int, default=16)
    p.add_argument("--spatial-grid", type=int, default=4)
    p.add_argument("--pool-mode", default="avg", choices=["avg", "query"])
    p.add_argument("--include-global-token", action="store_true")
    p.add_argument("--pos-mode", default="learned", choices=["learned", "nibble"])
    p.add_argument(
        "--inject-every-layer",
        action="store_true",
        default=None,
        help="omit to keep the value stored in an *_object.ckpt",
    )
    p.add_argument(
        "--n-lo-mode",
        default=None,
        choices=["attn", "fixed_center"],
        help="attn=注意力质心细码；fixed_center=锁格子中心",
    )
    add_cost_cli_args(p)

    p.add_argument("--cem-num-samples", type=int, default=300)
    p.add_argument("--cem-n-steps", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument("--plan-protocol", default="lewm", choices=PLAN_PROTOCOLS)
    p.add_argument(
        "--split",
        default="dev",
        choices=EVAL_SPLITS,
        help="dev = odd eval episodes (tuning); test = even (final numbers only); all = both",
    )
    p.add_argument(
        "--stats-dataset",
        default=None,
        help="h5 used to fit action/proprio normalizers; default: *_train.h5 next to --dataset",
    )
    p.add_argument("--plan-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--plan-gpu", type=str, default=_DEFAULT_PLAN_GPU)
    p.add_argument("--egl-gpu", type=str, default=_DEFAULT_EGL_GPU)

    p.add_argument(
        "--planner",
        choices=["mpc", "openloop"],
        default="openloop",
        help="openloop=开环 CEM（r3 正式协议，默认）；mpc=滚动规划（仅消融）",
    )
    p.add_argument(
        "--n-taken-actions",
        type=int,
        default=3,
        help="MPC 每轮执行的 action 数（对齐 Wall 消融最优 3）",
    )
    p.add_argument(
        "--mpc-max-iters",
        type=int,
        default=8,
        help="MPC 最大重规划轮数",
    )

    p.add_argument("--plans-out", type=str, default=None)
    p.add_argument("--plan-only", action="store_true", help="仅 openloop：只规划不仿真")
    p.add_argument("--sim-from-plans", type=str, default=None)
    p.add_argument("--smoke-env", action="store_true")
    return p.parse_args()


def main():
    import torch

    args = parse_args()

    if args.plan_gpu == args.egl_gpu:
        raise SystemExit(
            f"PLAN_GPU 与 EGL_GPU 不能相同（当前都是 {args.plan_gpu}）。"
            f"例如 --plan-gpu 3 --egl-gpu 4"
        )
    os.environ["PLAN_GPU"] = str(args.plan_gpu)
    os.environ["EGL_GPU"] = str(args.egl_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.plan_gpu)

    print("=" * 60)
    print("LeWM Granular 仿真评估（规划卡 ≠ Flex 卡）")
    print("=" * 60)
    print(
        f"[INFO] PLAN_GPU={os.environ.get('PLAN_GPU')} EGL_GPU={os.environ.get('EGL_GPU')} "
        f"planner={args.planner}"
    )

    if args.smoke_env:
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
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise RuntimeError("smoke-env 失败")
        print("[INFO] --smoke-env 通过，退出")
        return

    if args.sim_from_plans:
        with open(args.sim_from_plans, "rb") as f:
            plans = pickle.load(f)
        metrics = run_sim_subprocess(plans, success_threshold=args.success_threshold)
        metrics["mode"] = "sim_from_plans"
        write_results(metrics, Path(args.output))
        print(f"已写入: {args.output}")
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
    print(
        f"num_eval={args.num_eval} goal_offset={args.goal_offset} horizon={args.horizon} "
        f"planner={args.planner} n_taken={args.n_taken_actions} "
        f"mpc_max_iters={args.mpc_max_iters}"
    )

    stats_path = resolve_stats_path(dataset_path, args.stats_dataset)
    print(f"[INFO] normalizer stats: {stats_path}  plan_protocol={args.plan_protocol}")
    process = build_processors(stats_path, ("action", "proprio"))
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }

    print("\n[INFO] 加载模型...", flush=True)
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
        token_mode=args.token_mode,
        num_spatial_tokens=args.num_spatial_tokens,
        spatial_grid=args.spatial_grid,
        pool_mode=args.pool_mode,
        include_global_token=args.include_global_token,
        pos_mode=args.pos_mode,
        inject_every_layer=args.inject_every_layer,
        n_lo_mode=args.n_lo_mode,
        cost_mode=args.cost_mode,
        m_task_threshold=args.m_task_threshold,
        m_task_mode=args.m_task_mode,
        m_task_topk=args.m_task_topk,
        m_task_assume_normalized=not args.m_task_raw_pixels,
    )
    apply_cost_attrs(model, args)
    if args.plan_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA unavailable (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}); "
            "pass a valid --plan-gpu or --plan-device cpu explicitly"
        )
    device = torch.device(args.plan_device)
    model = model.to(device)
    print(f"[INFO] plan device={device}", flush=True)

    cem_kwargs = dict(
        num_samples=args.cem_num_samples,
        n_steps=args.cem_n_steps,
        topk=args.cem_topk,
        var_scale=args.cem_var_scale,
        protocol=args.plan_protocol,
    )

    t0 = time.time()
    if args.planner == "mpc":
        if args.plan_only:
            raise SystemExit("MPC 模式不支持 --plan-only（需要 plan↔sim 交错）")
        client = FlexMpcClient(egl_gpu=str(args.egl_gpu), object_name="granular")
        try:
            metrics = run_mpc_episodes(
                str(dataset_path),
                model,
                process,
                transform,
                device,
                client,
                num_eval=args.num_eval,
                goal_offset_steps=args.goal_offset,
                horizon=args.horizon,
                history_size=args.history_size,
                action_dim=effective_act_dim,
                n_taken_actions=args.n_taken_actions,
                mpc_max_iters=args.mpc_max_iters,
                success_threshold=args.success_threshold,
                seed=args.seed,
                cem_kwargs=cem_kwargs,
                plan_ctx=plan_context_len(args.plan_protocol, args.history_size),
                split=args.split,
            )
        finally:
            client.close()
            release_torch_cuda(model)
        elapsed = time.time() - t0
        metrics.update(
            {
                "mode": "mpc_two_process",
                "plan_protocol": args.plan_protocol,
                "split": args.split,
                "stats_dataset": stats_path,
                "plan_device": args.plan_device,
                "plan_gpu": os.environ.get("PLAN_GPU"),
                "egl_gpu": os.environ.get("EGL_GPU"),
                "evaluation_time_sec": elapsed,
            }
        )
    else:
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
            plan_protocol=args.plan_protocol,
            split=args.split,
        )
        plan_time = time.time() - t0
        # Per-run default so concurrent evals never read each other's plans.
        plans_out = args.plans_out or str(Path(args.output).with_suffix("")) + "_plans.pkl"
        Path(plans_out).parent.mkdir(parents=True, exist_ok=True)
        with open(plans_out, "wb") as f:
            pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] 规划已保存: {plans_out}", flush=True)
        release_torch_cuda(model)
        model = None

        if args.plan_only:
            print("[INFO] --plan-only：请另开命令跑仿真")
            return

        t1 = time.time()
        metrics = run_sim_subprocess(plans, success_threshold=args.success_threshold)
        sim_time = time.time() - t1
        metrics.update(
            {
                "mode": "openloop_two_process",
                "plan_protocol": args.plan_protocol,
                "split": args.split,
                "stats_dataset": stats_path,
                "planner": "openloop",
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
