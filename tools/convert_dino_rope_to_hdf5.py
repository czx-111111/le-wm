#!/usr/bin/env python3
"""
DINO-WM rope 原始数据 -> LeWM HDF5（按 episode 划分，避免 train/eval 泄露）

训练与评估需要的 key 可以不同，但必须来自不同 episode：
  rope_train.h5  : pixels, action, proprio, ep_*   （训练）
  rope_eval.h5   : 同上 + state, step_idx          （仿真评估）

示例:
  python convert_dino_rope_to_hdf5.py \\
    --data-path /path/to/dino_rope_raw \\
    --out-dir ./data \\
    --eval-ratio 0.1 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _fix_1d(x: np.ndarray, last_dim: int | None = None) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    if last_dim is not None and x.shape[-1] != last_dim:
        return x.reshape(x.shape[0], -1)
    return x


def load_episodes(data_path: Path):
    """返回 list[dict]，每个 episode 含 pixels/action/proprio/state。"""
    ep_dirs = sorted(
        [d for d in os.listdir(data_path) if os.path.isdir(data_path / d)]
    )
    print(f"[INFO] 找到 {len(ep_dirs)} 个子目录")
    episodes = []

    for ep_dir in tqdm(ep_dirs, desc="load"):
        ep_path = data_path / ep_dir
        h5_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".h5")])
        if not h5_files:
            continue

        ep_pixels, ep_proprio, ep_action, ep_state = [], [], [], []
        for hf in h5_files:
            with h5py.File(ep_path / hf, "r") as f:
                pixels = f["observations/color/cam_0"][:]
                eef = _fix_1d(f["eef_states"][:])
                action = _fix_1d(f["action"][:])
                positions = f["positions"][:]
                state = positions.reshape(positions.shape[0], -1)

                ep_pixels.append(pixels)
                ep_proprio.append(eef)
                ep_action.append(action)
                ep_state.append(state)

        if not ep_pixels:
            continue

        episodes.append(
            {
                "pixels": np.concatenate(ep_pixels, axis=0),
                "proprio": np.concatenate(ep_proprio, axis=0),
                "action": np.concatenate(ep_action, axis=0),
                "state": np.concatenate(ep_state, axis=0),
            }
        )
    return episodes


def pack_episodes(episodes: list[dict], include_state: bool):
    all_pixels, all_proprio, all_actions = [], [], []
    all_states, all_ep_idx, all_step_idx, ep_lens = [], [], [], []

    for ep_id, ep in enumerate(episodes):
        T = ep["pixels"].shape[0]
        all_pixels.append(ep["pixels"])
        all_proprio.append(ep["proprio"])
        all_actions.append(ep["action"])
        all_ep_idx.append(np.full(T, ep_id, dtype=np.int64))
        all_step_idx.append(np.arange(T, dtype=np.int64))
        ep_lens.append(T)
        if include_state:
            all_states.append(ep["state"])

    ep_len = np.asarray(ep_lens, dtype=np.int64)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(np.int64)
    out = {
        "pixels": np.concatenate(all_pixels, axis=0),
        "proprio": np.concatenate(all_proprio, axis=0),
        "action": np.concatenate(all_actions, axis=0),
        "ep_idx": np.concatenate(all_ep_idx, axis=0),
        "step_idx": np.concatenate(all_step_idx, axis=0),
        "ep_len": ep_len,
        "ep_offset": ep_offset,
    }
    if include_state:
        out["state"] = np.concatenate(all_states, axis=0)
    return out


def save_h5(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
    print(f"[INFO] saved {path}")
    for k, v in data.items():
        print(f"  {k}: {np.asarray(v).shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="DINO-WM rope 原始目录")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="按 episode 划分的评估集比例（默认 10%）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-name",
        default="rope_train.h5",
        help="训练用 hdf5 文件名（无 state 也可，这里仍可带 state，训练 config 不读即可）",
    )
    parser.add_argument("--eval-name", default="rope_eval.h5")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    episodes = load_episodes(data_path)
    n = len(episodes)
    if n == 0:
        raise RuntimeError(f"未找到 episode: {data_path}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_eval = max(1, int(round(n * args.eval_ratio)))
    n_eval = min(n_eval, n - 1)  # 至少留 1 条给 train
    eval_ids = set(perm[:n_eval].tolist())
    train_eps = [episodes[i] for i in range(n) if i not in eval_ids]
    eval_eps = [episodes[i] for i in range(n) if i in eval_ids]

    print(
        f"[INFO] split: total={n} train={len(train_eps)} eval={len(eval_eps)} "
        f"(eval_ratio={args.eval_ratio}, seed={args.seed})"
    )

    # 训练：可不含 state（更小）；评估：必须含 state
    train_data = pack_episodes(train_eps, include_state=False)
    eval_data = pack_episodes(eval_eps, include_state=True)

    save_h5(out_dir / args.train_name, train_data)
    save_h5(out_dir / args.eval_name, eval_data)
    print("[INFO] train/eval episode 集合互斥，无泄露。")


if __name__ == "__main__":
    main()
