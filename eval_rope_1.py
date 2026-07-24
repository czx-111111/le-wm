#!/usr/bin/env python3
"""
LeWM Rope 仿真评估（DINO-WM FlexEnv）

:
  1. 从 convert_dino_rope_to_hdf5.py 生成的 HDF5 采样 (start, goal)
  2. 用 LeWM + CEM 在 latent 空间规划动作
  3. 在 PyFlex rope 仿真中执行，用 Chamfer Distance 判成功

:
  python eval_rope.py \\
    --dataset ./data/rope_eval.h5 \\
    --ckpt ./data/lewm_rope_weights.ckpt \\
    --num-eval 50 \\
    --goal-offset 15 \\
    --horizon 5 \\
    --success-threshold 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rope_project" / "dino_wm"))
sys.path.insert(0, str(ROOT / "deformable_env"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYFLEX_HEADLESS", "1")

import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from deformable_env.FlexEnvWrapper import FlexEnvWrapper, chamfer_distance


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
        ep_offset = f["ep_offset"][:]

    # 需要: start .. start+history-1 存在, 且 start+goal_offset 存在
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
    print(f"[INFO] {len(valid_ep)} 个有效 episode, 采样 {num_eval} 组评估点 (replace={replace})")
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


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device):
    clip = apply_processors(clip, process)

    pixels = clip["pixels"].float()
    # HDF5: (T, H, W, C) -> 逐帧 ImageNet normalize
    if pixels.ndim == 4 and pixels.shape[-1] in (1, 3):
        pixels = torch.stack([transform["pixels"](p) for p in pixels], dim=0)
    elif pixels.ndim == 4:
        pixels = transform["pixels"](pixels)
    # JEPA: (B, S, T, C, H, W)
    pixels = pixels.unsqueeze(0).unsqueeze(0).to(device)

    action = torch.nan_to_num(clip["action"], 0.0).unsqueeze(0).unsqueeze(0).to(device)
    info = {"pixels": pixels, "action": action}

    if "proprio" in clip:
        info["proprio"] = clip["proprio"].unsqueeze(0).unsqueeze(0).to(device)

    goal_img = torch.from_numpy(np.asarray(goal_pixels))
    # 必须带时间维 T=1, 否则 encode 会把 channel 当成 T
    goal_img = transform["goal"](goal_img).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    info["goal"] = goal_img  # (B, S, T=1, C, H, W)

    goal_prop = np.asarray(goal_proprio, dtype=np.float32).reshape(1, -1)
    if "goal_proprio" in process:
        goal_prop = process["goal_proprio"].transform(goal_prop)
    # (B, S, T=1, D)
    info["goal_proprio"] = (
        torch.from_numpy(goal_prop).float().unsqueeze(0).unsqueeze(0).to(device)
    )
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
    history = info["pixels"].shape[2]
    total_steps = history + horizon

    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = info["action"][:, 0, :history]

    best_cost = None
    best_actions = None

    for _ in range(n_steps):
        noise = torch.randn(num_samples, total_steps, action_dim, device=device)
        candidates = mean + std * noise  # (S, T, A)
        candidates[:, :history] = info["action"][0, 0, :history]
        candidates_input = candidates.unsqueeze(0)  # (B=1, S, T, A)

        costs = model.get_cost(info, candidates_input)  # (1, S)
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

    # 权重: patch_embed [10,4,1], embed 10->768->192  => input_dim=4, smoothed_dim=10, emb_dim=192
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

    # 核对 action_encoder 是否对上
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


def run_evaluation(
    h5_path: str,
    model,
    process,
    transform,
    device,
    env,
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
    success_threshold: float = 0.05,
):
    lookup = build_goal_lookup(h5_path)
    eval_episodes, eval_start_idx = sample_eval_points(
        h5_path, num_eval, goal_offset_steps, history_size, seed
    )

    successes = []
    chamfer_distances = []
    planning_costs = []
    start_time = time.time()

    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_start_idx)):
        print(f"\n[{i + 1}/{num_eval}] episode={ep_id} start={start_step}")

        goal_step = int(start_step + goal_offset_steps)
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]

        _, _, _, init_state = load_frame(h5_path, start_idx)
        goal_pixels, _, goal_proprio, goal_state = load_frame(h5_path, goal_idx)
        if init_state is None or goal_state is None:
            print("  [WARN] 缺少 state，跳过")
            continue

        clip = load_sequence(h5_path, lookup, ep_id, start_step, history_size)
        info = build_info_dict(clip, goal_pixels, goal_proprio, process, transform, device)

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
        planned_norm_np = planned_norm.detach().cpu().numpy()
        planned_actions = denormalize_actions(planned_norm_np, process)
        print(f"  plan_cost={plan_cost:.4f}")

        init_particles = to_particle_state(init_state)
        obs, _ = env.prepare(seed + i, init_particles)
        del obs

        for action in planned_actions:
            _, _, _, infos = env.step_multiple([action])

        final_state = infos["state"][-1]
        goal_particles = to_particle_state(goal_state)
        cd = chamfer_distance(
            torch.tensor([goal_particles], dtype=torch.float32),
            torch.tensor([final_state], dtype=torch.float32),
        ).item()
        success = cd < success_threshold

        successes.append(bool(success))
        chamfer_distances.append(cd)
        planning_costs.append(plan_cost)
        print(f"  chamfer={cd:.6f} success={success} (thr={success_threshold})")

    elapsed = time.time() - start_time
    cds = np.asarray(chamfer_distances, dtype=np.float64) if chamfer_distances else np.array([])
    metrics = {
        "mode": "sim",
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": success_threshold,
        "mean_chamfer_distance": float(cds.mean()) if cds.size else -1.0,
        "median_chamfer_distance": float(np.median(cds)) if cds.size else -1.0,
        "mean_planning_cost": float(np.mean(planning_costs)) if planning_costs else -1.0,
        "evaluation_time_sec": elapsed,
    }
    return metrics


def write_results(metrics: dict, results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        f.write("==== ROPE SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def parse_args():
    p = argparse.ArgumentParser(description="LeWM Rope 仿真评估")
    p.add_argument("--dataset", default="./data/rope_eval.h5", help="HDF5 路径 (convert_dino_rope_to_hdf5.py 输出)")
    p.add_argument("--ckpt", default="./data/lewm_rope_weights.ckpt", help="LeWM weights.ckpt")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=15, help="goal 相对 start 的步数")
    p.add_argument("--horizon", type=int, default=5, help="CEM 规划视野")
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=1, help="与训练一致; 权重显示有效 action_dim=4 => frameskip=1")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--success-threshold",
        type=float,
        default=0.05,
        help="Chamfer Distance < threshold 视为成功 (DINO-WM 原版 threshold=0 会导致永远失败)",
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
    return p.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    ckpt_path = Path(args.ckpt)

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重不存在: {ckpt_path}")

    effective_act_dim = args.frameskip * args.action_dim
    if effective_act_dim != 4:
        print(f"[WARN] effective_act_dim={effective_act_dim}, 但权重 patch_embed 为 [10,4,1]，请确认 --frameskip/--action-dim")

    print("=" * 60)
    print("LeWM Rope 仿真评估")
    print("=" * 60)
    print(f"dataset: {dataset_path}")
    print(f"ckpt:    {ckpt_path}")
    print(f"num_eval={args.num_eval} goal_offset={args.goal_offset} horizon={args.horizon}")
    print(f"history={args.history_size} action_dim={effective_act_dim} success_thr={args.success_threshold}")

    process = build_processors(str(dataset_path), ("action", "proprio"))
    transform = {"pixels": img_transform(args.img_size), "goal": img_transform(args.img_size)}

    print("\n[INFO] 加载模型...")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[INFO] device={device}")

    print("\n[INFO] 创建 Flex rope 仿真环境 (首次初始化可能较慢)...")
    # 不要在外部提前 pyflex.init，FlexEnv.__init__ 会自己初始化
    env = FlexEnvWrapper("rope")

    print("\n[INFO] 开始评估...")
    metrics = run_evaluation(
        str(dataset_path),
        model,
        process,
        transform,
        device,
        env,
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
        success_threshold=args.success_threshold,
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
