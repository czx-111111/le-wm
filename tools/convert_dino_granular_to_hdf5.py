#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO-WM granular 原始数据 -> LeWM HDF5（按 episode 划分，避免 train/eval 泄露）

训练与评估需要的 key 不同，且必须来自互斥 episode：
  granular_train.h5 : pixels, action, proprio, ep_idx, step_idx, ep_len, ep_offset
  granular_eval.h5  : 同上 + state（仿真评估需要 particle positions）

输入目录结构（与 rope 相同）:
  data/deformable/granular/
    000000/00.h5, 01.h5, ...
    000001/00.h5, ...

示例:
  python convert_dino_granular_to_hdf5.py \\
    --data-path ./data/deformable/granular \\
    --out-dir ./data \\
    --eval-ratio 0.1 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x


def _align_action(action: np.ndarray, T: int) -> np.ndarray:
    """每个 push 的 h5 通常只有 1 条 action、T 帧图像，广播到 (T, D)。"""
    action = _ensure_2d(action).astype(np.float32)
    if action.shape[0] == T:
        return action
    if action.shape[0] == 1:
        return np.repeat(action, T, axis=0)
    raise ValueError(f"action 长度 ({action.shape[0]}) 与帧数 T={T} 无法对齐")


def _align_len(x: np.ndarray, T: int, name: str) -> np.ndarray:
    x = _ensure_2d(x).astype(np.float32)
    if x.shape[0] == T:
        return x
    if x.shape[0] == 1 and T > 1:
        return np.repeat(x, T, axis=0)
    raise ValueError(f"{name} 长度 ({x.shape[0]}) != T={T}")


def list_episode_dirs(data_path: Path) -> list[str]:
    dirs = sorted(
        d
        for d in os.listdir(data_path)
        if (data_path / d).is_dir() and d.isdigit()
    )
    # 至少要有一个 .h5
    valid = []
    for d in dirs:
        h5s = [f for f in os.listdir(data_path / d) if f.endswith(".h5")]
        if h5s:
            valid.append(d)
    return valid


def load_episode(ep_path: Path, cam_id: str, include_state: bool) -> dict:
    h5_files = sorted(f for f in os.listdir(ep_path) if f.endswith(".h5"))
    if not h5_files:
        raise FileNotFoundError(f"无 h5: {ep_path}")

    ep_pixels, ep_proprio, ep_action, ep_state = [], [], [], []
    for hf in h5_files:
        with h5py.File(ep_path / hf, "r") as f:
            cam_key = f"observations/color/{cam_id}"
            pixels = np.asarray(f[cam_key][:], dtype=np.uint8)
            if pixels.ndim == 3:
                pixels = pixels[None, ...]
            T = pixels.shape[0]

            proprio = _align_len(f["eef_states"][:], T, "eef_states")
            action = _align_action(f["action"][:], T)

            ep_pixels.append(pixels)
            ep_proprio.append(proprio)
            ep_action.append(action)

            if include_state:
                if "positions" not in f:
                    raise KeyError(f"{ep_path / hf}: 缺少 positions（eval 需要 state）")
                state = _align_len(f["positions"][:], T, "positions")
                ep_state.append(state)

    out = {
        "pixels": np.concatenate(ep_pixels, axis=0),
        "proprio": np.concatenate(ep_proprio, axis=0),
        "action": np.concatenate(ep_action, axis=0),
    }
    T = out["pixels"].shape[0]
    assert out["proprio"].shape[0] == T and out["action"].shape[0] == T
    if include_state:
        out["state"] = np.concatenate(ep_state, axis=0)
        assert out["state"].shape[0] == T
    return out


def write_split_h5(
    out_path: Path,
    data_path: Path,
    ep_dirs: list[str],
    cam_id: str,
    include_state: bool,
    compression: str | None = None,
):
    """按 episode 流式写入，避免整表进内存。"""
    if not ep_dirs:
        raise RuntimeError(f"空 split，无法写入 {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_kwargs = {"compression": compression} if compression else {}

    # 用第一个 episode 定维度
    first = load_episode(data_path / ep_dirs[0], cam_id, include_state)
    _, H, W, C = first["pixels"].shape
    proprio_dim = first["proprio"].shape[1]
    action_dim = first["action"].shape[1]
    state_dim = first["state"].shape[1] if include_state else None

    ep_lens: list[int] = []
    ep_offsets: list[int] = []
    current_offset = 0

    with h5py.File(out_path, "w") as f_out:
        f_out.create_dataset(
            "pixels",
            shape=(0, H, W, C),
            maxshape=(None, H, W, C),
            dtype=np.uint8,
            chunks=(1, H, W, C),
            **ds_kwargs,
        )
        f_out.create_dataset(
            "proprio",
            shape=(0, proprio_dim),
            maxshape=(None, proprio_dim),
            dtype=np.float32,
            chunks=True,
            **ds_kwargs,
        )
        f_out.create_dataset(
            "action",
            shape=(0, action_dim),
            maxshape=(None, action_dim),
            dtype=np.float32,
            chunks=True,
            **ds_kwargs,
        )
        f_out.create_dataset(
            "ep_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
            **ds_kwargs,
        )
        f_out.create_dataset(
            "step_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
            **ds_kwargs,
        )
        if include_state:
            f_out.create_dataset(
                "state",
                shape=(0, state_dim),
                maxshape=(None, state_dim),
                dtype=np.float32,
                chunks=True,
                **ds_kwargs,
            )

        for local_ep_id, ep_dir in enumerate(tqdm(ep_dirs, desc=out_path.name)):
            ep = (
                first
                if local_ep_id == 0
                else load_episode(data_path / ep_dir, cam_id, include_state)
            )
            T = ep["pixels"].shape[0]
            if ep["pixels"].shape[1:] != (H, W, C):
                raise ValueError(
                    f"{ep_dir}: 图像尺寸 {ep['pixels'].shape[1:]} != {(H, W, C)}"
                )

            f_out["pixels"].resize((current_offset + T, H, W, C))
            f_out["pixels"][current_offset : current_offset + T] = ep["pixels"]

            f_out["proprio"].resize((current_offset + T, proprio_dim))
            f_out["proprio"][current_offset : current_offset + T] = ep["proprio"]

            f_out["action"].resize((current_offset + T, action_dim))
            f_out["action"][current_offset : current_offset + T] = ep["action"]

            f_out["ep_idx"].resize((current_offset + T,))
            f_out["ep_idx"][current_offset : current_offset + T] = local_ep_id

            f_out["step_idx"].resize((current_offset + T,))
            f_out["step_idx"][current_offset : current_offset + T] = np.arange(
                T, dtype=np.int64
            )

            if include_state:
                if ep["state"].shape[1] != state_dim:
                    raise ValueError(
                        f"{ep_dir}: state_dim {ep['state'].shape[1]} != {state_dim}"
                    )
                f_out["state"].resize((current_offset + T, state_dim))
                f_out["state"][current_offset : current_offset + T] = ep["state"]

            ep_lens.append(T)
            ep_offsets.append(current_offset)
            current_offset += T

        f_out.create_dataset("ep_len", data=np.asarray(ep_lens, dtype=np.int64))
        f_out.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        f_out.attrs["format"] = "stable-worldmodel-hdf5"
        f_out.attrs["source"] = "dino-wm-granular"
        f_out.attrs["cam_id"] = cam_id
        f_out.attrs["include_state"] = include_state
        f_out.attrs["n_episodes"] = len(ep_dirs)

    print(f"[INFO] saved {out_path}")
    print(f"  episodes={len(ep_dirs)} frames={current_offset}")
    with h5py.File(out_path, "r") as f:
        for k in f.keys():
            print(f"  {k}: {f[k].shape} {f[k].dtype}")


def split_episodes(
    ep_dirs: list[str], eval_ratio: float, seed: int
) -> tuple[list[str], list[str], list[int], list[int]]:
    n = len(ep_dirs)
    if n < 2:
        raise RuntimeError(f"至少需要 2 个 episode 才能划分 train/eval，当前 {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_eval = max(1, int(round(n * eval_ratio)))
    n_eval = min(n_eval, n - 1)  # 至少留 1 条给 train

    eval_idx = sorted(perm[:n_eval].tolist())
    eval_set = set(eval_idx)
    train_idx = [i for i in range(n) if i not in eval_set]

    train_dirs = [ep_dirs[i] for i in train_idx]
    eval_dirs = [ep_dirs[i] for i in eval_idx]
    return train_dirs, eval_dirs, train_idx, eval_idx


def main():
    parser = argparse.ArgumentParser(
        description="DINO-WM granular -> LeWM train/eval HDF5（episode 级划分，无泄露）"
    )
    parser.add_argument(
        "--data-path",
        default="./data/deformable/granular",
        help="DINO-WM granular 原始目录",
    )
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="按 episode 划分的评估集比例（默认 10%）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cam-id", default="cam_0")
    parser.add_argument(
        "--train-name",
        default="granular_train.h5",
        help="训练 hdf5（不含 state）",
    )
    parser.add_argument(
        "--eval-name",
        default="granular_eval.h5",
        help="评估 hdf5（含 state）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="只使用前 N 个 episode（调试）",
    )
    parser.add_argument(
        "--compression",
        default="",
        choices=["", "gzip", "lzf"],
        help="HDF5 压缩（默认不压缩）",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    ep_dirs = list_episode_dirs(data_path)
    if args.max_episodes is not None:
        ep_dirs = ep_dirs[: args.max_episodes]

    if not ep_dirs:
        raise FileNotFoundError(f"未找到 episode: {data_path}")

    train_dirs, eval_dirs, train_idx, eval_idx = split_episodes(
        ep_dirs, args.eval_ratio, args.seed
    )
    # 互斥检查
    assert set(train_dirs).isdisjoint(set(eval_dirs)), "train/eval episode 泄露！"

    print(f"[INFO] 数据目录: {data_path}")
    print(
        f"[INFO] split: total={len(ep_dirs)} train={len(train_dirs)} "
        f"eval={len(eval_dirs)} (eval_ratio={args.eval_ratio}, seed={args.seed})"
    )

    # 保存划分清单，便于复现 / 核查无泄露
    split_meta = {
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "cam_id": args.cam_id,
        "n_total": len(ep_dirs),
        "n_train": len(train_dirs),
        "n_eval": len(eval_dirs),
        "train_episodes": train_dirs,
        "eval_episodes": eval_dirs,
        "train_indices": train_idx,
        "eval_indices": eval_idx,
    }
    meta_path = out_dir / "granular_split.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] split manifest: {meta_path}")

    compression = args.compression or None

    print("[INFO] 写训练集（无 state）...")
    write_split_h5(
        out_dir / args.train_name,
        data_path,
        train_dirs,
        args.cam_id,
        include_state=False,
        compression=compression,
    )

    print("[INFO] 写评估集（含 state）...")
    write_split_h5(
        out_dir / args.eval_name,
        data_path,
        eval_dirs,
        args.cam_id,
        include_state=True,
        compression=compression,
    )

    print("[INFO] train/eval episode 集合互斥，无泄露。")
    print(
        f"[INFO] train keys: pixels, proprio, action, ep_idx, step_idx, ep_len, ep_offset"
    )
    print(
        f"[INFO] eval  keys: pixels, proprio, action, state, ep_idx, step_idx, ep_len, ep_offset"
    )


if __name__ == "__main__":
    main()
