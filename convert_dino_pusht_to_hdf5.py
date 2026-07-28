#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO-WM pusht_noise 原始数据 -> LeWM HDF5（按 episode 划分，避免 train/eval 泄露）

默认只取前 2000 条 train episode（训练时间更短；
DINO-WM 论文 PushT：1k≈0.48 / 5k≈0.72 / 全量≈0.92，2k 介于其间）。

训练与评估需要的 key 不同，且必须来自互斥 episode：
  pusht_noise_train_2k.h5 : pixels, action, proprio, ep_idx, step_idx, ep_len, ep_offset
  pusht_noise_eval.h5     : 同上 + state（仿真评估需要；val 通常很小，默认全量保留）

已有全量 HDF5 时，可用 --from-h5 直接抽子集（不必重解 mp4）：
  python convert_dino_pusht_to_hdf5.py \\
    --from-h5 ./data/pusht_noise_train.h5 \\
    --out-dir ./data \\
    --train-name pusht_noise_train_2k.h5 \\
    --max-episodes 2000

从原始数据转换：
  python convert_dino_pusht_to_hdf5.py \\
    --data-path ./data/pusht_noise \\
    --out-dir ./data
"""
from __future__ import annotations

import argparse
import json
import pickle
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
        if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
        x = x[None, ...]
    elif x.ndim == 4:
        if x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (0, 2, 3, 1))  # TCHW -> THWC
    else:
        raise ValueError(f"无法解析图像形状: {x.shape}")

    if np.issubdtype(x.dtype, np.floating):
        if float(np.nanmax(x)) <= 1.0 + 1e-3:
            x = np.clip(x * 255.0, 0, 255)
        else:
            x = np.clip(x, 0, 255)
        x = np.rint(x).astype(np.uint8)
    else:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _read_mp4(path: Path) -> np.ndarray:
    """读取 mp4 为 (T, H, W, C) uint8。优先 decord，其次 imageio / cv2。"""
    # 1) decord（与 DINO-WM 一致）
    try:
        import decord
        from decord import VideoReader

        decord.bridge.set_bridge("native")
        reader = VideoReader(str(path), num_threads=1)
        frames = reader.get_batch(list(range(len(reader))))
        return _pixels_to_hwc_uint8(_to_numpy(frames))
    except Exception:
        pass

    # 2) imageio
    try:
        import imageio.v3 as iio

        frames = iio.imread(path)
        return _pixels_to_hwc_uint8(frames)
    except Exception:
        pass

    # 3) opencv
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"视频无帧: {path}")
        return _pixels_to_hwc_uint8(np.stack(frames, axis=0))
    except Exception as e:
        raise RuntimeError(
            f"无法读取 {path}（已尝试 decord / imageio / cv2）: {e}"
        ) from e


def list_episode_indices(obs_dir: Path, n_meta: int) -> list[int]:
    """根据 obses/episode_*.mp4 与 meta 长度取交集。"""
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"缺少图像目录: {obs_dir}")

    pat = re.compile(r"^episode_(\d+)\.mp4$")
    found = []
    for p in obs_dir.iterdir():
        m = pat.match(p.name)
        if m:
            found.append(int(m.group(1)))
    found = sorted(found)
    if not found:
        raise FileNotFoundError(f"{obs_dir} 下没有 episode_*.mp4")

    valid = [i for i in found if i < n_meta]
    missing = [i for i in range(n_meta) if i not in set(found)]
    if missing:
        print(
            f"[WARN] meta 有 {n_meta} 条，但缺 {len(missing)} 个 episode 视频 "
            f"(例如 episode_{missing[0]:03d}.mp4)；将跳过缺失项"
        )
    if not valid:
        raise RuntimeError("没有可用的 episode（meta 与 obses 无交集）")
    return valid


def _resolve_obs_path(data_path: Path, ep_idx: int) -> Path:
    obs_path = data_path / "obses" / f"episode_{ep_idx:03d}.mp4"
    if obs_path.is_file():
        return obs_path
    candidates = list((data_path / "obses").glob(f"episode_{ep_idx}.mp4")) + list(
        (data_path / "obses").glob(f"episode_{ep_idx:04d}.mp4")
    )
    if not candidates:
        raise FileNotFoundError(obs_path)
    return candidates[0]


def load_meta(
    data_path: Path,
    *,
    relative: bool = True,
    action_scale: float = 100.0,
    with_velocity: bool = True,
) -> dict[str, np.ndarray]:
    action_name = "rel_actions.pth" if relative else "abs_actions.pth"
    required = ["states.pth", action_name, "seq_lengths.pkl"]
    if with_velocity:
        required.append("velocities.pth")
    for name in required:
        if not (data_path / name).is_file():
            raise FileNotFoundError(f"缺少 {data_path / name}")

    states = _to_numpy(_torch_load(data_path / "states.pth")).astype(np.float32)
    actions = _to_numpy(_torch_load(data_path / action_name)).astype(np.float32)
    actions = actions / float(action_scale)

    with open(data_path / "seq_lengths.pkl", "rb") as f:
        seq_lengths_raw = pickle.load(f)
    seq_lengths = np.asarray(seq_lengths_raw, dtype=np.int64)

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
    if seq_lengths.ndim != 1 or seq_lengths.shape[0] != states.shape[0]:
        raise ValueError(
            f"seq_lengths 期望 (N,)，实际 {seq_lengths.shape}，N={states.shape[0]}"
        )

    n, t_max = states.shape[:2]
    seq_lengths = np.clip(seq_lengths, 1, t_max)

    # proprio / state 构造与 DINO-WM PushTDataset 一致
    proprios = states[..., :2].copy()
    full_states = states.copy()
    if with_velocity:
        velocities = _to_numpy(_torch_load(data_path / "velocities.pth")).astype(
            np.float32
        )
        if velocities.shape[0] != n or velocities.shape[1] != t_max:
            raise ValueError(
                f"velocities 形状 {velocities.shape} 与 states {states.shape} 不匹配"
            )
        full_states = np.concatenate([full_states, velocities], axis=-1)
        proprios = np.concatenate([proprios, velocities], axis=-1)

    shapes_file = data_path / "shapes.pkl"
    if shapes_file.is_file():
        with open(shapes_file, "rb") as f:
            shapes = list(pickle.load(f))
        if len(shapes) < n:
            shapes = shapes + ["T"] * (n - len(shapes))
        shapes = shapes[:n]
    else:
        shapes = ["T"] * n

    print(
        f"[INFO] meta ({data_path.name}): N={n} T_max={t_max} "
        f"state_dim={full_states.shape[-1]} proprio_dim={proprios.shape[-1]} "
        f"action_dim={actions.shape[-1]} action_scale={action_scale} "
        f"relative={relative} with_velocity={with_velocity} "
        f"seq_len[min/mean/max]="
        f"{int(seq_lengths.min())}/{float(seq_lengths.mean()):.1f}/{int(seq_lengths.max())}"
    )
    return {
        "states": full_states,
        "proprios": proprios.astype(np.float32),
        "actions": actions.astype(np.float32),
        "seq_lengths": seq_lengths,
        "shapes": shapes,
    }


def load_episode(
    data_path: Path,
    meta: dict,
    ep_idx: int,
    include_state: bool,
) -> dict:
    """加载单个 episode；按 seq_lengths 截断。"""
    pixels = _read_mp4(_resolve_obs_path(data_path, ep_idx))

    T_img = pixels.shape[0]
    T_meta = int(meta["seq_lengths"][ep_idx])
    T_pad = meta["states"].shape[1]
    T = min(T_img, T_meta, T_pad)
    if T_img != T_meta:
        print(
            f"[WARN] episode_{ep_idx:03d}: 视频 T={T_img} seq_len={T_meta}，截断到 T={T}"
        )

    proprio = _ensure_2d(meta["proprios"][ep_idx, :T])
    action = _ensure_2d(meta["actions"][ep_idx, :T])

    out = {
        "pixels": pixels[:T],
        "proprio": proprio.astype(np.float32),
        "action": action.astype(np.float32),
        "shape": str(meta["shapes"][ep_idx]),
    }
    if include_state:
        out["state"] = _ensure_2d(meta["states"][ep_idx, :T]).astype(np.float32)
    return out


def write_split_h5(
    out_path: Path,
    data_path: Path,
    meta: dict,
    ep_indices: list[int],
    include_state: bool,
    compression: str | None = None,
):
    """按 episode 流式写入，避免把全部视频载入内存。"""
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
    ep_shapes: list[str] = []
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

            if include_state:
                if ep["state"].shape[1] != state_dim:
                    raise ValueError(
                        f"episode_{src_ep_idx:03d}: state_dim {ep['state'].shape[1]} != {state_dim}"
                    )
                f_out["state"].resize((current_offset + T, state_dim))
                f_out["state"][current_offset : current_offset + T] = ep["state"]

            ep_lens.append(T)
            ep_offsets.append(current_offset)
            ep_shapes.append(ep["shape"])
            current_offset += T

        f_out.create_dataset("ep_len", data=np.asarray(ep_lens, dtype=np.int64))
        f_out.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        # 每 episode 一个 shape 字符串（评估时可用）
        dt = h5py.string_dtype(encoding="utf-8")
        f_out.create_dataset("ep_shape", data=np.asarray(ep_shapes, dtype=object), dtype=dt)

        f_out.attrs["format"] = "stable-worldmodel-hdf5"
        f_out.attrs["source"] = "dino-wm-pusht_noise"
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


def subsample_existing_h5(
    src_path: Path,
    out_path: Path,
    max_episodes: int,
    compression: str | None = None,
):
    """从已有 LeWM HDF5 抽取前 max_episodes 个 episode（不重解视频）。"""
    if max_episodes <= 0:
        raise ValueError(f"max_episodes 必须 > 0，当前 {max_episodes}")
    if not src_path.is_file():
        raise FileNotFoundError(f"找不到源 HDF5: {src_path}")
    if src_path.resolve() == out_path.resolve():
        raise ValueError("输出路径不能与源文件相同")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_kwargs = {"compression": compression} if compression else {}

    with h5py.File(src_path, "r") as f_in:
        if "ep_len" not in f_in or "ep_offset" not in f_in:
            raise KeyError(f"{src_path} 缺少 ep_len / ep_offset")
        if "pixels" not in f_in:
            raise KeyError(f"{src_path} 缺少 pixels")

        ep_len = f_in["ep_len"][:]
        ep_offset = f_in["ep_offset"][:]
        n_src = int(ep_len.shape[0])
        n_keep = min(max_episodes, n_src)
        if n_keep < n_src:
            print(f"[INFO] 从 {n_src} 个 episode 中取前 {n_keep} 个 -> {out_path.name}")
        else:
            print(f"[INFO] 源仅有 {n_src} 个 episode，全部写出 -> {out_path.name}")

        end = int(ep_offset[n_keep - 1] + ep_len[n_keep - 1])
        n_frames_total = int(f_in["pixels"].shape[0])
        frame_keys = [
            k
            for k in f_in.keys()
            if k not in ("ep_len", "ep_offset", "ep_shape")
            and f_in[k].ndim >= 1
            and f_in[k].shape[0] == n_frames_total
        ]

        with h5py.File(out_path, "w") as f_out:
            for key in frame_keys:
                src_ds = f_in[key]
                f_out.create_dataset(
                    key,
                    shape=(end,) + tuple(src_ds.shape[1:]),
                    maxshape=(None,) + tuple(src_ds.shape[1:]),
                    dtype=src_ds.dtype,
                    chunks=src_ds.chunks,
                    **ds_kwargs,
                )
                for i in tqdm(range(n_keep), desc=f"copy {key}"):
                    s = int(ep_offset[i])
                    e = s + int(ep_len[i])
                    f_out[key][s:e] = src_ds[s:e]

            if "ep_idx" in f_out:
                for i in range(n_keep):
                    s = int(ep_offset[i])
                    e = s + int(ep_len[i])
                    f_out["ep_idx"][s:e] = i

            f_out.create_dataset("ep_len", data=ep_len[:n_keep].astype(np.int64))
            f_out.create_dataset("ep_offset", data=ep_offset[:n_keep].astype(np.int64))

            if "ep_shape" in f_in:
                f_out.create_dataset("ep_shape", data=f_in["ep_shape"][:n_keep])

            for attr_key, attr_val in f_in.attrs.items():
                f_out.attrs[attr_key] = attr_val
            f_out.attrs["n_episodes"] = n_keep
            f_out.attrs["subsampled_from"] = str(src_path)
            f_out.attrs["max_episodes"] = int(max_episodes)

    print(f"[INFO] saved {out_path} (episodes={n_keep}, frames={end})")
    with h5py.File(out_path, "r") as f:
        for k in f.keys():
            print(f"  {k}: {f[k].shape} {f[k].dtype}")
    return {"n_episodes": n_keep, "n_frames": end}


def _resolve_split_roots(data_path: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    返回 (train_root, eval_root, flat_root)。
    优先使用 DINO-WM 的 train/ + val/；否则当作扁平目录。
    """
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        return train_dir, val_dir, None
    if (data_path / "states.pth").is_file() and (data_path / "obses").is_dir():
        return None, None, data_path
    raise FileNotFoundError(
        f"无法识别数据布局: {data_path}\n"
        f"期望存在 train/+val/ 子目录，或扁平的 states.pth + obses/"
    )


def _convert_one_split(
    split_root: Path,
    out_path: Path,
    include_state: bool,
    *,
    relative: bool,
    action_scale: float,
    with_velocity: bool,
    max_episodes: int | None,
    compression: str | None,
) -> dict:
    meta = load_meta(
        split_root,
        relative=relative,
        action_scale=action_scale,
        with_velocity=with_velocity,
    )
    ep_indices = list_episode_indices(split_root / "obses", meta["states"].shape[0])
    if max_episodes is not None:
        ep_indices = ep_indices[:max_episodes]
    write_split_h5(
        out_path,
        split_root,
        meta,
        ep_indices,
        include_state=include_state,
        compression=compression,
    )
    return {
        "root": str(split_root),
        "episodes": ep_indices,
        "n_episodes": len(ep_indices),
        "action_dim": int(meta["actions"].shape[-1]),
        "proprio_dim": int(meta["proprios"].shape[-1]),
        "state_dim": int(meta["states"].shape[-1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DINO-WM pusht_noise -> LeWM train/eval HDF5（默认 train 取 2000 episode）"
    )
    parser.add_argument(
        "--data-path",
        default="./data/pusht_noise",
        help="DINO-WM pusht_noise 原始目录（含 train/val 或扁平结构）",
    )
    parser.add_argument(
        "--from-h5",
        default="",
        help="若指定，则从已有 train HDF5 抽子集（不必重解 mp4）",
    )
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="扁平目录时按 episode 划分的评估集比例（默认 10%%；若已有 train/val 则忽略）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-name",
        default="pusht_noise_train_2k.h5",
        help="训练 hdf5（默认 2k 子集名）",
    )
    parser.add_argument(
        "--eval-name",
        default="pusht_noise_eval.h5",
        help="评估 hdf5（含 state；train/val 布局下 eval 默认不截断）",
    )
    parser.add_argument(
        "--relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 rel_actions.pth（默认）或 --no-relative 使用 abs_actions.pth",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=100.0,
        help="与 DINO-WM 一致：actions /= action_scale（默认 100）",
    )
    parser.add_argument(
        "--with-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="拼接 agent 速度到 state/proprio（默认开启，对齐 DINO-WM）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=2000,
        help="train 最多使用的 episode 数（默认 2000；设为 0 表示不截断）",
    )
    parser.add_argument(
        "--compression",
        default="",
        choices=["", "gzip", "lzf"],
        help="HDF5 压缩（默认不压缩）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = args.compression or None
    # 0 表示不截断；其余正整数生效
    max_episodes = None if args.max_episodes == 0 else args.max_episodes

    # ---- 快速路径：从已有全量 HDF5 抽子集 ----
    if args.from_h5:
        if max_episodes is None:
            raise ValueError("--from-h5 需要正的 --max-episodes（默认 2000）")
        src = Path(args.from_h5)
        info = subsample_existing_h5(
            src,
            out_dir / args.train_name,
            max_episodes=max_episodes,
            compression=compression,
        )
        meta_path = out_dir / "pusht_noise_split_2k.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "subsample_from_h5",
                    "source": str(src),
                    "train_name": args.train_name,
                    "n_train": info["n_episodes"],
                    "n_frames": info["n_frames"],
                    "max_episodes": max_episodes,
                    "note": "仅截断 train；eval 请继续使用原 pusht_noise_eval.h5",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] split manifest: {meta_path}")
        print(
            f"[INFO] 训练配置请将 dataset.name 设为 {Path(args.train_name).stem}"
        )
        return

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")

    train_root, eval_root, flat_root = _resolve_split_roots(data_path)

    common_kwargs = dict(
        relative=args.relative,
        action_scale=args.action_scale,
        with_velocity=args.with_velocity,
        compression=compression,
    )

    if train_root is not None and eval_root is not None:
        print(f"[INFO] 检测到 DINO-WM train/val 布局: {data_path}")
        print(f"[INFO] 写训练集（无 state，max_episodes={max_episodes}）...")
        train_info = _convert_one_split(
            train_root,
            out_dir / args.train_name,
            include_state=False,
            max_episodes=max_episodes,
            **common_kwargs,
        )
        # eval 通常很小，默认全量保留，避免把评估集也截断
        print("[INFO] 写评估集（含 state，不截断）...")
        eval_info = _convert_one_split(
            eval_root,
            out_dir / args.eval_name,
            include_state=True,
            max_episodes=None,
            **common_kwargs,
        )
        split_meta = {
            "mode": "dino_train_val",
            "seed": args.seed,
            "eval_ratio": None,
            "max_episodes_train": max_episodes,
            "n_train": train_info["n_episodes"],
            "n_eval": eval_info["n_episodes"],
            "train_episodes": train_info["episodes"],
            "eval_episodes": eval_info["episodes"],
            "action_dim": train_info["action_dim"],
            "proprio_dim": train_info["proprio_dim"],
            "state_dim": eval_info["state_dim"],
            "relative": args.relative,
            "action_scale": args.action_scale,
            "with_velocity": args.with_velocity,
            "note": "train 默认最多 2000 episode；eval 全量；action_dim=2；frameskip=5",
        }
    else:
        assert flat_root is not None
        print(f"[INFO] 检测到扁平布局，按 episode 随机划分: {flat_root}")
        meta = load_meta(
            flat_root,
            relative=args.relative,
            action_scale=args.action_scale,
            with_velocity=args.with_velocity,
        )
        ep_indices = list_episode_indices(flat_root / "obses", meta["states"].shape[0])
        if max_episodes is not None:
            ep_indices = ep_indices[:max_episodes]

        train_ids, eval_ids, train_pos, eval_pos = split_episodes(
            ep_indices, args.eval_ratio, args.seed
        )
        assert set(train_ids).isdisjoint(set(eval_ids)), "train/eval episode 泄露！"

        print(
            f"[INFO] split: total={len(ep_indices)} train={len(train_ids)} "
            f"eval={len(eval_ids)} (eval_ratio={args.eval_ratio}, seed={args.seed})"
        )

        print("[INFO] 写训练集（无 state）...")
        write_split_h5(
            out_dir / args.train_name,
            flat_root,
            meta,
            train_ids,
            include_state=False,
            compression=compression,
        )
        print("[INFO] 写评估集（含 state）...")
        write_split_h5(
            out_dir / args.eval_name,
            flat_root,
            meta,
            eval_ids,
            include_state=True,
            compression=compression,
        )
        split_meta = {
            "mode": "flat_random_split",
            "seed": args.seed,
            "eval_ratio": args.eval_ratio,
            "max_episodes": max_episodes,
            "n_total": len(ep_indices),
            "n_train": len(train_ids),
            "n_eval": len(eval_ids),
            "train_episodes": train_ids,
            "eval_episodes": eval_ids,
            "train_indices": train_pos,
            "eval_indices": eval_pos,
            "action_dim": int(meta["actions"].shape[-1]),
            "proprio_dim": int(meta["proprios"].shape[-1]),
            "state_dim": int(meta["states"].shape[-1]),
            "relative": args.relative,
            "action_scale": args.action_scale,
            "with_velocity": args.with_velocity,
            "note": "默认最多 2000 episode 再划分；action_dim=2；建议 frameskip=5",
        }

    meta_path = out_dir / "pusht_noise_split.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] split manifest: {meta_path}")

    print("[INFO] train/eval episode 集合互斥，无泄露。")
    print(
        "[INFO] train keys: pixels, proprio, action, ep_idx, step_idx, ep_len, ep_offset, ep_shape"
    )
    print("[INFO] eval  keys: 同上 + state")
    print(
        f"[INFO] 训练时请使用 action_dim={split_meta['action_dim']}，"
        f"建议 frameskip=5；dataset.name={Path(args.train_name).stem}"
    )


if __name__ == "__main__":
    main()
