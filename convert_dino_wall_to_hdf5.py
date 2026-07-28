#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO-WM wall_single 原始数据 -> LeWM HDF5（按 episode 划分，避免 train/eval 泄露）

训练与评估需要的 key 不同，且必须来自互斥 episode：
  wall_train.h5 : pixels, action, proprio, ep_idx, step_idx, ep_len, ep_offset
  wall_eval.h5  : 同上 + state, door_location, wall_location（仿真评估需要）

输入目录结构（DINO-WM wall_single）:
  data/wall_single/
    states.pth
    actions.pth
    door_locations.pth
    wall_locations.pth
    obses/
      episode_000.pth
      episode_001.pth
      ...

示例:
  python convert_dino_wall_to_hdf5.py \\
    --data-path ./data/wall_single \\
    --out-dir ./data \\
    --eval-ratio 0.1 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


def _torch_load(path: Path):
    """兼容新旧 PyTorch 的 torch.load。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x


def _pixels_to_hwc_uint8(pixels: np.ndarray) -> np.ndarray:
    """统一为 (T, H, W, C) uint8。支持 TCHW / THWC，以及 float[0,1] / uint8。"""
    x = np.asarray(pixels)
    if x.ndim == 3:
        # 单帧 (C,H,W) 或 (H,W,C)
        if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
        x = x[None, ...]
    elif x.ndim == 4:
        # (T,C,H,W) vs (T,H,W,C)
        if x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (0, 2, 3, 1))  # TCHW -> THWC
    else:
        raise ValueError(f"无法解析图像形状: {x.shape}")

    if np.issubdtype(x.dtype, np.floating):
        # 常见为 [0,1]；若已是 [0,255] 则不放大
        if float(np.nanmax(x)) <= 1.0 + 1e-3:
            x = np.clip(x * 255.0, 0, 255)
        else:
            x = np.clip(x, 0, 255)
        x = np.rint(x).astype(np.uint8)
    else:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def list_episode_indices(obs_dir: Path, n_meta: int) -> list[int]:
    """根据 obses/episode_*.pth 与 meta 长度取交集。"""
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"缺少图像目录: {obs_dir}")

    pat = re.compile(r"^episode_(\d+)\.pth$")
    found = []
    for p in obs_dir.iterdir():
        m = pat.match(p.name)
        if m:
            found.append(int(m.group(1)))
    found = sorted(found)
    if not found:
        raise FileNotFoundError(f"{obs_dir} 下没有 episode_*.pth")

    # 只保留 meta 范围内且文件存在的 episode
    valid = [i for i in found if i < n_meta]
    missing = [i for i in range(n_meta) if i not in set(found)]
    if missing:
        print(
            f"[WARN] meta 有 {n_meta} 条，但缺 {len(missing)} 个 episode 图像 "
            f"(例如 episode_{missing[0]:03d}.pth)；将跳过缺失项"
        )
    if not valid:
        raise RuntimeError("没有可用的 episode（meta 与 obses 无交集）")
    return valid


def load_meta(data_path: Path) -> dict[str, np.ndarray]:
    required = ["states.pth", "actions.pth", "door_locations.pth", "wall_locations.pth"]
    for name in required:
        if not (data_path / name).is_file():
            raise FileNotFoundError(f"缺少 {data_path / name}")

    states = _to_numpy(_torch_load(data_path / "states.pth"))
    actions = _to_numpy(_torch_load(data_path / "actions.pth"))
    door = _to_numpy(_torch_load(data_path / "door_locations.pth"))
    wall = _to_numpy(_torch_load(data_path / "wall_locations.pth"))

    if states.ndim != 3:
        raise ValueError(f"states 期望 (N,T,D)，实际 {states.shape}")
    if actions.ndim != 3:
        raise ValueError(f"actions 期望 (N,T,D)，实际 {actions.shape}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"states/actions episode 数不一致: {states.shape[0]} vs {actions.shape[0]}"
        )
    if states.shape[1] != actions.shape[1]:
        raise ValueError(
            f"states/actions 时间长度不一致: {states.shape[1]} vs {actions.shape[1]}"
        )

    # door/wall: (N,T) / (N,T,1) / (N,) 都兼容，统一成 (N,T)
    def _as_nt(x: np.ndarray, name: str, n: int, t: int) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            if x.shape[0] != n:
                raise ValueError(f"{name} 形状 {x.shape} 与 N={n} 不匹配")
            return np.repeat(x.reshape(n, 1), t, axis=1)
        if x.ndim == 2:
            if x.shape[0] != n:
                raise ValueError(f"{name} 形状 {x.shape} 与 N={n} 不匹配")
            if x.shape[1] == 1:
                return np.repeat(x, t, axis=1)
            if x.shape[1] != t:
                raise ValueError(f"{name} 形状 {x.shape} 与 T={t} 不匹配")
            return x
        if x.ndim == 3 and x.shape[-1] == 1:
            return _as_nt(x[..., 0], name, n, t)
        raise ValueError(f"{name} 无法解析形状: {x.shape}")

    n, t = states.shape[:2]
    door = _as_nt(door, "door_locations", n, t)
    wall = _as_nt(wall, "wall_locations", n, t)

    print(
        f"[INFO] meta: N={n} T={t} state_dim={states.shape[-1]} "
        f"action_dim={actions.shape[-1]}"
    )
    return {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "door_locations": door.astype(np.float32),
        "wall_locations": wall.astype(np.float32),
    }


def load_episode(
    data_path: Path,
    meta: dict[str, np.ndarray],
    ep_idx: int,
    include_state: bool,
) -> dict:
    """加载单个 episode；proprio 与 DINO-WM 一致，取 states。"""
    obs_path = data_path / "obses" / f"episode_{ep_idx:03d}.pth"
    if not obs_path.is_file():
        # 兼容超过 999 时可能的更宽编号
        candidates = list((data_path / "obses").glob(f"episode_{ep_idx}.pth")) + list(
            (data_path / "obses").glob(f"episode_{ep_idx:04d}.pth")
        )
        if not candidates:
            raise FileNotFoundError(obs_path)
        obs_path = candidates[0]

    pixels = _pixels_to_hwc_uint8(_to_numpy(_torch_load(obs_path)))
    T_img = pixels.shape[0]
    T_meta = meta["states"].shape[1]
    T = min(T_img, T_meta)
    if T_img != T_meta:
        print(
            f"[WARN] episode_{ep_idx:03d}: 图像 T={T_img} meta T={T_meta}，截断到 T={T}"
        )

    # DINO-WM: proprio = states.clone()，不做额外归一化（LeWM 训练时再 normalize）
    proprio = _ensure_2d(meta["states"][ep_idx, :T])
    action = _ensure_2d(meta["actions"][ep_idx, :T])
    door = meta["door_locations"][ep_idx, :T].astype(np.float32)
    wall = meta["wall_locations"][ep_idx, :T].astype(np.float32)

    out = {
        "pixels": pixels[:T],
        "proprio": proprio.astype(np.float32),
        "action": action.astype(np.float32),
        "door_location": door,
        "wall_location": wall,
    }
    if include_state:
        out["state"] = proprio.copy()
    return out


def write_split_h5(
    out_path: Path,
    data_path: Path,
    meta: dict[str, np.ndarray],
    ep_indices: list[int],
    include_state: bool,
    compression: str | None = None,
):
    """按 episode 流式写入，避免把全部图像载入内存。"""
    if not ep_indices:
        raise RuntimeError(f"空 split，无法写入 {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_kwargs = {"compression": compression} if compression else {}

    first = load_episode(data_path, meta, ep_indices[0], include_state)
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
        f_out.create_dataset(
            "door_location",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
            **ds_kwargs,
        )
        f_out.create_dataset(
            "wall_location",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
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

        for local_ep_id, src_ep_idx in enumerate(tqdm(ep_indices, desc=out_path.name)):
            ep = (
                first
                if local_ep_id == 0
                else load_episode(data_path, meta, src_ep_idx, include_state)
            )
            T = ep["pixels"].shape[0]
            if ep["pixels"].shape[1:] != (H, W, C):
                raise ValueError(
                    f"episode_{src_ep_idx:03d}: 图像尺寸 {ep['pixels'].shape[1:]} != {(H, W, C)}"
                )
            if ep["proprio"].shape[1] != proprio_dim:
                raise ValueError(
                    f"episode_{src_ep_idx:03d}: proprio_dim {ep['proprio'].shape[1]} != {proprio_dim}"
                )
            if ep["action"].shape[1] != action_dim:
                raise ValueError(
                    f"episode_{src_ep_idx:03d}: action_dim {ep['action'].shape[1]} != {action_dim}"
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

            f_out["door_location"].resize((current_offset + T,))
            f_out["door_location"][current_offset : current_offset + T] = ep[
                "door_location"
            ]

            f_out["wall_location"].resize((current_offset + T,))
            f_out["wall_location"][current_offset : current_offset + T] = ep[
                "wall_location"
            ]

            if include_state:
                if ep["state"].shape[1] != state_dim:
                    raise ValueError(
                        f"episode_{src_ep_idx:03d}: state_dim {ep['state'].shape[1]} != {state_dim}"
                    )
                f_out["state"].resize((current_offset + T, state_dim))
                f_out["state"][current_offset : current_offset + T] = ep["state"]

            ep_lens.append(T)
            ep_offsets.append(current_offset)
            current_offset += T

        f_out.create_dataset("ep_len", data=np.asarray(ep_lens, dtype=np.int64))
        f_out.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        f_out.attrs["format"] = "stable-worldmodel-hdf5"
        f_out.attrs["source"] = "dino-wm-wall_single"
        f_out.attrs["include_state"] = include_state
        f_out.attrs["n_episodes"] = len(ep_indices)
        f_out.attrs["action_dim"] = action_dim
        f_out.attrs["proprio_dim"] = proprio_dim

    print(f"[INFO] saved {out_path}")
    print(f"  episodes={len(ep_indices)} frames={current_offset}")
    with h5py.File(out_path, "r") as f:
        for k in f.keys():
            print(f"  {k}: {f[k].shape} {f[k].dtype}")


def split_episodes(
    ep_indices: list[int], eval_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    n = len(ep_indices)
    if n < 2:
        raise RuntimeError(f"至少需要 2 个 episode 才能划分 train/eval，当前 {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_eval = max(1, int(round(n * eval_ratio)))
    n_eval = min(n_eval, n - 1)

    eval_pos = sorted(perm[:n_eval].tolist())
    eval_set = set(eval_pos)
    train_pos = [i for i in range(n) if i not in eval_set]

    train_ids = [ep_indices[i] for i in train_pos]
    eval_ids = [ep_indices[i] for i in eval_pos]
    return train_ids, eval_ids, train_pos, eval_pos


def main():
    parser = argparse.ArgumentParser(
        description="DINO-WM wall_single -> LeWM train/eval HDF5（episode 级划分，无泄露）"
    )
    parser.add_argument(
        "--data-path",
        default="./data/wall_single",
        help="DINO-WM wall_single 原始目录",
    )
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
        default="wall_train.h5",
        help="训练 hdf5（不含 state）",
    )
    parser.add_argument(
        "--eval-name",
        default="wall_eval.h5",
        help="评估 hdf5（含 state）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="只使用前 N 个有效 episode（调试）",
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
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")

    meta = load_meta(data_path)
    ep_indices = list_episode_indices(data_path / "obses", meta["states"].shape[0])
    if args.max_episodes is not None:
        ep_indices = ep_indices[: args.max_episodes]

    train_ids, eval_ids, train_pos, eval_pos = split_episodes(
        ep_indices, args.eval_ratio, args.seed
    )
    assert set(train_ids).isdisjoint(set(eval_ids)), "train/eval episode 泄露！"

    print(f"[INFO] 数据目录: {data_path}")
    print(
        f"[INFO] split: total={len(ep_indices)} train={len(train_ids)} "
        f"eval={len(eval_ids)} (eval_ratio={args.eval_ratio}, seed={args.seed})"
    )

    split_meta = {
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "n_total": len(ep_indices),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "train_episodes": train_ids,
        "eval_episodes": eval_ids,
        "train_indices": train_pos,
        "eval_indices": eval_pos,
        "action_dim": int(meta["actions"].shape[-1]),
        "proprio_dim": int(meta["states"].shape[-1]),
        "note": "wall action_dim=2；训练 config 请设 wm.action_dim: 2",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "wall_split.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] split manifest: {meta_path}")

    compression = args.compression or None

    print("[INFO] 写训练集（无 state）...")
    write_split_h5(
        out_dir / args.train_name,
        data_path,
        meta,
        train_ids,
        include_state=False,
        compression=compression,
    )

    print("[INFO] 写评估集（含 state）...")
    write_split_h5(
        out_dir / args.eval_name,
        data_path,
        meta,
        eval_ids,
        include_state=True,
        compression=compression,
    )

    print("[INFO] train/eval episode 集合互斥，无泄露。")
    print(
        "[INFO] train keys: pixels, proprio, action, door_location, wall_location, "
        "ep_idx, step_idx, ep_len, ep_offset"
    )
    print(
        "[INFO] eval  keys: 同上 + state"
    )
    print(
        f"[INFO] 训练时请使用 action_dim={meta['actions'].shape[-1]} "
        f"(与 rope/granular 的 4 不同)"
    )


if __name__ == "__main__":
    main()
