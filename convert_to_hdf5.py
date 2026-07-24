#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgiBotWorld2026 -> stable-worldmodel 兼容 HDF5 转换脚本
边处理边写入，避免内存溢出
"""

from __future__ import annotations

import argparse
import glob
import os
import h5py
import numpy as np
import pandas as pd

# ----------------------------- 默认配置 ----------------------------- #
DEFAULT_ROOT = "/mnt/data/wangjianfeng/qwh/czx/le-wm/data/task_5015_test"
DEFAULT_OUT_DIR = "/mnt/data/wangjianfeng/qwh/czx/le-wm/data"
DEFAULT_CAMERA = "observation.images.top_head"
SEED = 42
TRAIN_RATIO = 0.8

PROPRIO_COL = "observation.state"
ACTION_COL = "action"
EP_COL = "episode_index"
FRAME_COL = "frame_index"


# ----------------------------- 视频解码 ----------------------------- #
def _import_video_reader():
    """使用 PyAV 读取视频（支持 AV1 等编码）"""
    try:
        import av
        import numpy as np

        def read_video(path: str) -> np.ndarray:
            container = av.open(path)
            frames = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
            container.close()
            if not frames:
                return np.empty((0, 0, 0, 0), dtype=np.uint8)
            return np.stack(frames, axis=0).astype(np.uint8)

        return read_video, "pyav"
    except ImportError:
        pass

    try:
        import cv2

        def read_video(path: str) -> np.ndarray:
            cap = cv2.VideoCapture(path)
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
                return np.empty((0, 0, 0, 0), dtype=np.uint8)
            return np.stack(frames, axis=0).astype(np.uint8)

        return read_video, "opencv"
    except ImportError as exc:
        raise ImportError("需要安装 av 或 opencv-python") from exc


# ----------------------------- 数据读取 ----------------------------- #
def load_all_parquet(root: str) -> pd.DataFrame:
    patterns = [
        os.path.join(root, "data", "data", "chunk-*", "episode_*.parquet"),
        os.path.join(root, "data", "chunk-*", "episode_*.parquet"),
        os.path.join(root, "chunk-*", "episode_*.parquet"),
    ]
    
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            print(f"[INFO] 找到 parquet 文件: {pattern}")
            break
    
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件，搜索根目录: {root}")

    print(f"[INFO] 找到 {len(files)} 个 parquet 文件")
    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        if EP_COL not in df.columns:
            stem = os.path.basename(fp)
            ep_id = int("".join(ch for ch in stem if ch.isdigit()))
            df[EP_COL] = ep_id
        dfs.append(df)

    table = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 合并后: {len(table)} 行, 列: {list(table.columns)}")
    return table


def list_camera_keys(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "data", "videos", "chunk-*", "observation.images.*"),
        os.path.join(root, "videos", "chunk-*", "observation.images.*"),
    ]
    
    for pattern in patterns:
        dirs = sorted(glob.glob(pattern))
        if dirs:
            return [os.path.basename(d) for d in dirs]
    return []


def resolve_camera_key(root: str, preferred: str) -> str:
    cameras = list_camera_keys(root)
    if not cameras:
        raise FileNotFoundError("未找到相机目录")
    if preferred in cameras:
        return preferred
    print(f"[WARN] 未找到相机 '{preferred}', 可用: {cameras}")
    print(f"[WARN] 改用: {cameras[0]}")
    return cameras[0]


def episode_video_path(root: str, camera_key: str, ep_id: int) -> str:
    patterns = [
        os.path.join(root, "data", "videos", "chunk-*", camera_key, f"episode_{ep_id:06d}.mp4"),
        os.path.join(root, "videos", "chunk-*", camera_key, f"episode_{ep_id:06d}.mp4"),
    ]
    
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    
    raise FileNotFoundError(f"未找到 episode {ep_id} 的视频")


def split_episodes(episode_ids, train_ratio=TRAIN_RATIO, seed=SEED):
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(set(int(e) for e in episode_ids)))
    rng.shuffle(ids)
    n_train = int(round(len(ids) * train_ratio))
    train_ids = sorted(ids[:n_train].tolist())
    test_ids = sorted(ids[n_train:].tolist())
    return train_ids, test_ids


# ----------------------------- 边处理边写入 ----------------------------- #
def convert_episodes_to_h5(
    out_path: str,
    root: str,
    table: pd.DataFrame,
    episode_ids: list[int],
    camera_key: str,
    attrs: dict | None = None
):
    """边处理边写入 HDF5，避免内存溢出"""
    read_video, backend = _import_video_reader()
    print(f"[INFO] 视频解码后端: {backend}, 相机: {camera_key}")

    sub = table[table[EP_COL].isin(episode_ids)].copy()
    sub = sub.sort_values([EP_COL, FRAME_COL]).reset_index(drop=True)

    if len(episode_ids) == 0:
        print("[WARN] 没有 episode 需要处理")
        return

    # 先读取第一个 episode 获取维度信息
    first_ep = episode_ids[0]
    ep_df = sub[sub[EP_COL] == first_ep]
    n_rows = len(ep_df)
    
    video_path = episode_video_path(root, camera_key, first_ep)
    first_frames = read_video(video_path)
    
    if len(first_frames) != n_rows:
        n_use = min(len(first_frames), n_rows)
        first_frames = first_frames[:n_use]
        n_rows = n_use
    
    h, w, c = first_frames.shape[1], first_frames.shape[2], first_frames.shape[3]
    action_dim = len(ep_df[ACTION_COL].iloc[0])
    proprio_dim = len(ep_df[PROPRIO_COL].iloc[0])
    
    print(f"[INFO] 图像尺寸: {h}x{w}x{c}, action维度: {action_dim}, proprio维度: {proprio_dim}")

    with h5py.File(out_path, "w") as f:
        # 创建可扩展的数据集
        f.create_dataset("pixels", shape=(0, h, w, c), maxshape=(None, h, w, c),
                        dtype=np.uint8, compression="gzip", chunks=True)
        f.create_dataset("action", shape=(0, action_dim), maxshape=(None, action_dim),
                        dtype=np.float32, compression="gzip", chunks=True)
        f.create_dataset("proprio", shape=(0, proprio_dim), maxshape=(None, proprio_dim),
                        dtype=np.float32, compression="gzip", chunks=True)
        f.create_dataset("episode_idx", shape=(0,), maxshape=(None,),
                        dtype=np.int64, compression="gzip", chunks=True)
        f.create_dataset("step_idx", shape=(0,), maxshape=(None,),
                        dtype=np.int64, compression="gzip", chunks=True)

        ep_lens = []
        ep_offsets = []
        current_offset = 0

        for i, ep_id in enumerate(episode_ids):
            ep_df = sub[sub[EP_COL] == ep_id]
            if ep_df.empty:
                print(f"[WARN] episode {ep_id} 无数据, 跳过")
                continue

            n_rows = len(ep_df)
            try:
                video_path = episode_video_path(root, camera_key, ep_id)
                frames = read_video(video_path)
            except Exception as e:
                print(f"[WARN] episode {ep_id} 视频读取失败: {e}, 跳过")
                continue

            if len(frames) != n_rows:
                n_use = min(len(frames), n_rows)
                if n_use == 0:
                    print(f"[WARN] episode {ep_id} 无有效帧, 跳过")
                    continue
                print(f"[WARN] episode {ep_id}: 视频帧数({len(frames)}) != parquet行数({n_rows}), 取前 {n_use} 帧")
                frames = frames[:n_use]
                ep_df = ep_df.iloc[:n_use]
                n_rows = n_use

            action = np.stack([np.asarray(x, dtype=np.float32) for x in ep_df[ACTION_COL].tolist()])
            proprio = np.stack([np.asarray(x, dtype=np.float32) for x in ep_df[PROPRIO_COL].tolist()])
            episode_idx = np.full(n_rows, ep_id, dtype=np.int64)
            step_idx = ep_df[FRAME_COL].to_numpy(dtype=np.int64)

            # 追加到数据集
            f["pixels"].resize((current_offset + n_rows, h, w, c))
            f["pixels"][current_offset:current_offset + n_rows] = frames

            f["action"].resize((current_offset + n_rows, action_dim))
            f["action"][current_offset:current_offset + n_rows] = action

            f["proprio"].resize((current_offset + n_rows, proprio_dim))
            f["proprio"][current_offset:current_offset + n_rows] = proprio

            f["episode_idx"].resize((current_offset + n_rows,))
            f["episode_idx"][current_offset:current_offset + n_rows] = episode_idx

            f["step_idx"].resize((current_offset + n_rows,))
            f["step_idx"][current_offset:current_offset + n_rows] = step_idx

            ep_lens.append(n_rows)
            ep_offsets.append(current_offset)
            current_offset += n_rows

            if (i + 1) % 10 == 0 or i + 1 == len(episode_ids):
                print(f"[INFO] 已处理 {i + 1}/{len(episode_ids)} 个 episode")

        # 写入元数据
        f.create_dataset("ep_len", data=np.array(ep_lens, dtype=np.int32), compression="gzip")
        f.create_dataset("ep_offset", data=np.array(ep_offsets, dtype=np.int64), compression="gzip")

        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

    print(f"[INFO] 写入完成: {out_path}, 总帧数: {current_offset}")


def print_h5_structure(path: str):
    print(f"\n===== HDF5 结构: {path} =====")
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            ds = f[key]
            print(f"  {key:12s}  shape={ds.shape}  dtype={ds.dtype}")
        if f.attrs:
            print(f"  [attrs] {dict(f.attrs)}")


# ----------------------------- 主流程 ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="AgiBotWorld2026 -> SWM HDF5")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_h5 = os.path.join(args.out_dir, "train.h5")
    test_h5 = os.path.join(args.out_dir, "test.h5")

    table = load_all_parquet(args.root)
    camera_key = resolve_camera_key(args.root, args.camera)
    print(f"[INFO] 可用相机: {list_camera_keys(args.root)}")

    all_eps = table[EP_COL].unique()
    train_ids, test_ids = split_episodes(all_eps, args.train_ratio, args.seed)
    print(f"\n[INFO] 总 episode: {len(all_eps)}")
    print(f"[INFO] 训练集 episode: {len(train_ids)}")
    print(f"[INFO] 测试集 episode: {len(test_ids)}")

    attrs = {
        "format": "stable-worldmodel-hdf5",
        "camera_key": camera_key,
        "source_root": args.root,
        "split_seed": args.seed,
        "train_ratio": args.train_ratio,
    }

    print(f"\n[INFO] 构建训练集 ...")
    convert_episodes_to_h5(train_h5, args.root, table, train_ids, camera_key, {**attrs, "split": "train"})

    print(f"\n[INFO] 构建测试集 ...")
    convert_episodes_to_h5(test_h5, args.root, table, test_ids, camera_key, {**attrs, "split": "test"})

    print("\n========== 转换完成 ==========")
    print_h5_structure(train_h5)
    print_h5_structure(test_h5)


if __name__ == "__main__":
    main()
