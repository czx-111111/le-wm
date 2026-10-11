#!/usr/bin/env python3
"""
LeWM PointMaze 仿真评估（DINO-WM PointMaze）

对齐训练配置：action_dim=2, frameskip=5 → 规划维数 10。
成功判据：||goal[:2] - cur[:2]||_2 < 0.5（与 DINO-WM PointMazeWrapper.eval_state 一致）。
评估条数：默认 50（对齐 plan_point_maze.yaml 的 n_evals=50）。
默认规划卡：物理 GPU 0。

默认 planner=mpc（滚动规划）；--planner openloop 可复现旧开环 CEM。

用法:
  # 首次 / mujoco_py 有问题时:
  #   pip install 'cython==0.29.37'
  #   sudo apt-get install -y libglew-dev libglfw3-dev libosmesa6-dev
  #   python fix_mujoco_py_cython.py
  # 也可: source setup_env.sh  （会 export LD_LIBRARY_PATH）
  python eval_point_maze.py \\
    --dataset ./data/point_maze_eval.h5 \\
    --ckpt ./data/lewm_point_maze_v2_epoch_30_object.ckpt \\
    --plan-gpu 7 --num-eval 50 --planner mpc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _early_apply_gpu_argv():
    """在 import torch 之前锁定物理 GPU。"""
    argv = sys.argv[1:]
    plan_gpu = os.environ.get("PLAN_GPU", "0")
    for i, a in enumerate(argv):
        if a == "--plan-gpu" and i + 1 < len(argv):
            plan_gpu = argv[i + 1]
        elif a.startswith("--plan-gpu="):
            plan_gpu = a.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan_gpu)
    os.environ["PLAN_GPU"] = str(plan_gpu)


def _find_mujoco_root() -> Path | None:
    candidates = [
        ROOT / "mujoco210" / "mujoco210",
        ROOT / "mujoco210",
        Path.home() / ".mujoco" / "mujoco210",
    ]
    for c in candidates:
        if (c / "bin").is_dir():
            return c.resolve()
    return None


def _setup_mujoco():
    """指向仓库内 mujoco210；必要时 re-exec 使 LD_LIBRARY_PATH 生效。"""
    mujoco_root = _find_mujoco_root()
    if mujoco_root is None:
        raise FileNotFoundError(
            "找不到 MuJoCo 210。请确认仓库有 mujoco210/mujoco210/bin，"
            "或安装到 ~/.mujoco/mujoco210。\n"
            "也可: export MUJOCO_PY_MUJOCO_PATH=/path/to/mujoco210"
        )

    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(mujoco_root)
    # 无显示服务器时用 EGL 渲染
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    bin_dir = str(mujoco_root / "bin")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(os.pathsep) if p]
    # torch's pip CUDA libs must shadow the system ones (e.g. an older
    # /usr/lib/x86_64-linux-gnu/libnvJitLink.so.12 breaks libcusparse).
    nv_root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    nv_libs = sorted(str(d) for d in nv_root.glob("*/lib") if d.is_dir())
    extras = [bin_dir, *nv_libs, "/usr/lib/nvidia", "/usr/lib/x86_64-linux-gnu"]
    new_parts = []
    for p in extras + parts:
        if p and p not in new_parts:
            new_parts.append(p)
    need_reexec = bin_dir not in parts
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_parts)
    print(f"[INFO] MUJOCO_PY_MUJOCO_PATH={mujoco_root}")

    if need_reexec and os.environ.get("_LEWM_MUJOCO_REEXEC") != "1":
        os.environ["_LEWM_MUJOCO_REEXEC"] = "1"
        print(f"[INFO] re-exec：注入 LD_LIBRARY_PATH={bin_dir}")
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _preload_mujoco_libs():
    """绝对路径预加载 libglewegl 等，避免 dlopen 找不到。"""
    import ctypes

    mujoco_root = _find_mujoco_root()
    if mujoco_root is None:
        return
    bin_dir = mujoco_root / "bin"
    for name in ("libglewegl.so", "libglew.so", "libglfw.so.3", "libmujoco210.so"):
        lib = bin_dir / name
        if not lib.is_file():
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            print(f"[INFO] preload {lib.name}")
        except OSError as e:
            print(f"[WARN] preload {lib.name} 失败: {e}")


def _patch_mujoco_py_for_cython3():
    """修复 mujoco_py 与 Cython 的兼容问题。

    - 若 builder 已被误打上 legacy_implicit_noexcept，而当前 Cython 不支持 → 回退
    - Cython>=3：给 cymj.pyx 的 callback 加 noexcept
    - Cython<3：无需改 cymj.pyx
    """
    import importlib.util
    import re

    try:
        import Cython

        cython_ver = str(Cython.__version__)
        cython_major = int(cython_ver.split(".")[0])
    except Exception as e:
        print(f"[WARN] 无法检测 Cython: {e}")
        return

    print(f"[INFO] Cython={cython_ver}")

    try:
        spec = importlib.util.find_spec("mujoco_py")
    except Exception:
        return
    if spec is None or not spec.submodule_search_locations:
        return

    pkg = Path(list(spec.submodule_search_locations)[0])
    builder = pkg / "builder.py"
    cymj = pkg / "cymj.pyx"

    # 1) 回退可能不兼容的 builder 补丁（匹配多种空格写法）
    if builder.is_file():
        text = builder.read_text(encoding="utf-8")
        if "legacy_implicit_noexcept" in text:
            text2, n = re.subn(
                r"dist\.ext_modules\s*=\s*cythonize\(\s*\[self\.extension\]\s*,"
                r"\s*compiler_directives\s*=\s*\{['\"]legacy_implicit_noexcept['\"]\s*:\s*True\}\s*\)",
                "dist.ext_modules = cythonize([self.extension])",
                text,
                count=1,
            )
            if n:
                builder.write_text(text2, encoding="utf-8")
                print("[INFO] 已回退 builder.py 中的 legacy_implicit_noexcept 补丁")
            else:
                print(
                    "[WARN] builder.py 含 legacy_implicit_noexcept 但未能自动回退，"
                    "请手动改回: dist.ext_modules = cythonize([self.extension])"
                )

    if cython_major < 3:
        print("[INFO] Cython<3，使用原始 cymj.pyx 即可")
        return

    if not cymj.is_file():
        print(f"[WARN] 找不到 {cymj}，请执行: pip install 'cython==0.29.37'")
        return

    pyx = cymj.read_text(encoding="utf-8")
    new_pyx = pyx
    # 给两个 callback 加 noexcept（Cython3 必需）
    new_pyx = re.sub(
        r"cdef void c_warning_callback\(const char \*msg\)(?! noexcept) with gil:",
        "cdef void c_warning_callback(const char *msg) noexcept with gil:",
        new_pyx,
        count=1,
    )
    new_pyx = re.sub(
        r"cdef void c_error_callback\(const char \*msg\)(?! noexcept) with gil:",
        "cdef void c_error_callback(const char *msg) noexcept with gil:",
        new_pyx,
        count=1,
    )
    if new_pyx == pyx:
        if "noexcept with gil" in pyx:
            print(f"[INFO] cymj.pyx 已含 noexcept（兼容 Cython {cython_ver}）")
        else:
            print(
                f"[WARN] 未能修补 {cymj}。"
                "请执行: pip install 'cython==0.29.37'"
            )
        return

    cymj.write_text(new_pyx, encoding="utf-8")
    # 清理旧生成物，强制下次重新 cythonize
    for pattern in ("cymj.c", "cymj.cpp", "cymj*.so"):
        for p in pkg.glob(pattern):
            try:
                p.unlink()
                print(f"[INFO] 删除旧编译产物: {p.name}")
            except OSError:
                pass
    print(f"[INFO] 已修补 {cymj}（callback + noexcept，Cython {cython_ver}）")


_early_apply_gpu_argv()
_setup_mujoco()
_preload_mujoco_libs()
_patch_mujoco_py_for_cython3()

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
from utils import build_lewm, load_lewm_weights_into_model
from eval_mpc_utils import (
    EVAL_SPLITS,
    PLAN_PROTOCOLS,
    filter_split_episodes,
    as_numpy,
    add_cost_cli_args,
    apply_cost_attrs,
    cem_lewm,
    extract_macro_frames,
    plan_context_len,
    resolve_stats_path,
    stack_history_clip,
    to_hwc_uint8_compatible,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


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


def sample_eval_points(
    h5_path: str,
    num_eval: int,
    goal_offset_steps: int,
    history_size: int,
    frameskip: int,
    seed: int = 42,
    split: str = "dev",
):
    """采样 (episode, start_step)，保证 history(带 frameskip) 与 goal 都在轨迹内。"""
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]

    need = max(
        goal_offset_steps,
        (history_size - 1) * frameskip,
        history_size * frameskip - 1,
    )
    max_start = ep_len - need - 1
    valid_ep = filter_split_episodes(np.nonzero(max_start >= 0)[0], split)
    if len(valid_ep) == 0:
        raise RuntimeError(
            f"没有足够长的 episode：需要至少 {need + 1} 帧 "
            f"(goal_offset={goal_offset_steps}, history={history_size}, frameskip={frameskip})"
        )

    g = np.random.default_rng(seed)
    if num_eval <= 0:
        num_eval = len(valid_ep)
    replace = len(valid_ep) < num_eval
    chosen_ep = g.choice(valid_ep, size=num_eval, replace=replace)

    starts = []
    for ep_id in chosen_ep:
        hi = int(max_start[ep_id])
        starts.append(int(g.integers(0, hi + 1)))

    starts = np.asarray(starts, dtype=np.int64)
    print(
        f"[INFO] {len(valid_ep)} 个有效 episode, 采样 {num_eval} 组 "
        f"(replace={replace}, need_len>={need + 1})"
    )
    return chosen_ep.astype(np.int64), starts


def load_frame(h5_path: str, idx: int):
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][idx]
        action = f["action"][idx]
        proprio = f["proprio"][idx] if "proprio" in f else np.zeros(4, dtype=np.float32)
        state = f["state"][idx] if "state" in f else None
    return {
        "pixels": pixels,
        "action": action,
        "proprio": proprio,
        "state": state,
    }


def load_history_with_frameskip(
    h5_path: str,
    lookup,
    episode_id: int,
    start_step: int,
    history_size: int,
    frameskip: int,
    action_dim: int,
):
    """加载 history 帧（每隔 frameskip），并把中间 raw action 拼成 (H, fs*D)。"""
    pixels_list, proprio_list = [], []
    raw_actions = []

    for t in range(history_size):
        step = int(start_step + t * frameskip)
        idx = lookup[(int(episode_id), step)]
        fr = load_frame(h5_path, idx)
        pixels_list.append(fr["pixels"])
        proprio_list.append(fr["proprio"])

    for s in range(history_size * frameskip):
        idx = lookup[(int(episode_id), int(start_step + s))]
        fr = load_frame(h5_path, idx)
        act = np.asarray(fr["action"], dtype=np.float32).reshape(-1)
        if act.shape[0] != action_dim:
            raise ValueError(f"action_dim 期望 {action_dim}，实际 {act.shape[0]}")
        raw_actions.append(act)

    raw_actions = np.stack(raw_actions, axis=0)  # (H*fs, D)
    concat = raw_actions.reshape(history_size, frameskip * action_dim)

    return {
        "pixels": torch.from_numpy(np.stack(pixels_list)),
        "action_raw": raw_actions,
        "action": torch.from_numpy(concat).float(),
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


def normalize_concat_actions(
    raw_actions: np.ndarray,
    process: dict,
    history_size: int,
    frameskip: int,
    action_dim: int,
) -> torch.Tensor:
    """raw (H*fs, D) → normalize → (H, fs*D)."""
    flat = np.asarray(raw_actions, dtype=np.float32).reshape(-1, action_dim)
    if "action" in process:
        flat = process["action"].transform(flat)
    concat = flat.reshape(history_size, frameskip * action_dim)
    return torch.from_numpy(concat.astype(np.float32))


def denormalize_concat_actions(
    concat_actions: np.ndarray,
    process: dict,
    frameskip: int,
    action_dim: int,
) -> np.ndarray:
    """(horizon, fs*D) normalized → (horizon*fs, D) raw env actions."""
    flat = np.asarray(concat_actions, dtype=np.float32).reshape(-1, frameskip * action_dim)
    flat = flat.reshape(-1, action_dim)
    if "action" in process:
        flat = process["action"].inverse_transform(flat)
    return flat.astype(np.float32)


def apply_processors(sample: dict, process: dict):
    out = dict(sample)
    for col, proc in process.items():
        if col.startswith("goal_") or col == "action" or col not in out:
            continue
        arr = out[col]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        out[col] = torch.from_numpy(proc.transform(arr)).float()
    return out


def _to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def build_info_dict(clip, goal_pixels, goal_proprio, process, transform):
    clip = apply_processors(clip, process)

    pixels_np = (
        clip["pixels"].detach().cpu().numpy()
        if torch.is_tensor(clip["pixels"])
        else np.asarray(clip["pixels"])
    )
    frame_tensors = [
        transform["pixels"](_to_uint8_hwc(pixels_np[i])) for i in range(pixels_np.shape[0])
    ]
    pixels = torch.stack(frame_tensors, dim=0).unsqueeze(0).unsqueeze(0)  # (B,S,T,C,H,W)

    action = torch.nan_to_num(clip["action"].float(), 0.0).unsqueeze(0).unsqueeze(0)
    info = {"pixels": pixels, "action": action}

    if "proprio" in clip:
        info["proprio"] = clip["proprio"].float().unsqueeze(0).unsqueeze(0)

    goal_img = transform["goal"](_to_uint8_hwc(goal_pixels))
    info["goal"] = goal_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    goal_prop = np.asarray(goal_proprio, dtype=np.float32).reshape(1, -1)
    if "goal_proprio" in process:
        goal_prop = process["goal_proprio"].transform(goal_prop)
    info["goal_proprio"] = torch.from_numpy(goal_prop).float().unsqueeze(0).unsqueeze(0)
    return info


# ---------------------------------------------------------------------------
# Model / CEM
# ---------------------------------------------------------------------------


@torch.no_grad()
def cem_plan(
    model,
    info,
    horizon: int,
    action_dim: int,
    num_samples: int = 300,
    n_steps: int = 10,
    topk: int = 30,
    var_scale: float = 1.0,
    device: str = "cuda",
    protocol: str = "lewm",
):
    device = torch.device(device)
    info = {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in info.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize()

    if protocol == "lewm":
        return cem_lewm(
            model, info, horizon, action_dim, num_samples, n_steps, topk, var_scale, device
        )

    history = info["pixels"].shape[2]
    total_steps = history + horizon

    mean = torch.zeros(1, total_steps, action_dim, device=device)
    std = torch.ones(1, total_steps, action_dim, device=device) * var_scale
    mean[:, :history] = info["action"][:, 0, :history]

    best_cost = None
    best_actions = None

    for step in range(n_steps):
        if step == 0 or (step + 1) % max(1, n_steps // 5) == 0 or step + 1 == n_steps:
            print(f"    CEM {step + 1}/{n_steps}", flush=True)
        noise = torch.randn(num_samples, total_steps, action_dim, device=device)
        candidates = mean + std * noise
        candidates[:, :history] = info["action"][0, 0, :history]
        costs = model.get_cost(info, candidates.unsqueeze(0))
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
    token_mode: str = "cls",
    num_spatial_tokens: int = 1,
    spatial_grid: int = 4,
    cost_mode: str = "dense_terminal",
    m_task_threshold: float = 0.08,
    m_task_mode: str = "eps",
    m_task_topk: int = 4,
    m_task_assume_normalized: bool = True,
    pool_mode: str = "avg",
    include_global_token: bool = False,
    pos_mode: str = "learned",
    inject_every_layer: bool | None = None,
    n_lo_mode: str | None = None,
):
    # *_object.ckpt 是整模 JEPA；weights 才是 Lightning state_dict
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        print(f"[INFO] loaded pickled JEPA object from {ckpt_path}")
        model.cost_mode = cost_mode
        model.m_task_threshold = m_task_threshold
        model.m_task_mode = m_task_mode
        model.m_task_topk = m_task_topk
        model.m_task_assume_normalized = m_task_assume_normalized
        if not hasattr(model, "token_mode"):
            model.token_mode = "cls"
        if not hasattr(model, "num_spatial_tokens"):
            model.num_spatial_tokens = 1
        if not hasattr(model, "spatial_grid"):
            model.spatial_grid = 4
        if not hasattr(model, "pool_mode"):
            model.pool_mode = "avg"
        if not hasattr(model, "include_global_token"):
            model.include_global_token = False
        if not hasattr(model, "pos_mode"):
            model.pos_mode = "learned"
        # Architecture flags come from the pickled model unless explicitly overridden.
        if n_lo_mode is not None:
            model.n_lo_mode = n_lo_mode
            if hasattr(model, "predictor"):
                model.predictor.n_lo_mode = n_lo_mode
        if inject_every_layer is not None and hasattr(model, "predictor"):
            model.predictor.inject_every_layer = bool(inject_every_layer)
        if token_mode != "cls" and getattr(model, "token_mode", "cls") != token_mode:
            print(
                f"[WARN] ckpt token_mode={model.token_mode} "
                f"!= requested {token_mode}; using ckpt"
            )
    else:
        model = build_lewm(
            action_dim=action_dim,
            history_size=history_size,
            embed_dim=embed_dim,
            encoder_scale=encoder_scale,
            patch_size=patch_size,
            img_size=img_size,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            predictor_mlp_dim=predictor_mlp_dim,
            predictor_dim_head=predictor_dim_head,
            predictor_dropout=predictor_dropout,
            predictor_emb_dropout=predictor_emb_dropout,
            smoothed_dim=smoothed_dim,
            token_mode=token_mode,
            num_spatial_tokens=num_spatial_tokens,
            spatial_grid=spatial_grid,
            cost_mode=cost_mode,
            m_task_threshold=m_task_threshold,
            m_task_mode=m_task_mode,
            m_task_topk=m_task_topk,
            m_task_assume_normalized=m_task_assume_normalized,
            pool_mode=pool_mode,
            include_global_token=include_global_token,
            pos_mode=pos_mode,
            inject_every_layer=bool(inject_every_layer),
            n_lo_mode=n_lo_mode or "attn",
        )
        model = load_lewm_weights_into_model(model, ckpt_path)

    w = model.action_encoder.patch_embed.weight
    print(
        f"[INFO] action_encoder.patch_embed.weight = {tuple(w.shape)} "
        f"(期望 {(smoothed_dim, action_dim, 1)})"
    )
    print(
        f"[INFO] token_mode={getattr(model, 'token_mode', 'cls')} "
        f"cost_mode={getattr(model, 'cost_mode', 'dense_terminal')} "
        f"num_spatial_tokens={getattr(model, 'num_spatial_tokens', 1)} "
        f"pool_mode={getattr(model, 'pool_mode', 'avg')} "
        f"include_global={getattr(model, 'include_global_token', False)} "
        f"pos_mode={getattr(model, 'pos_mode', 'learned')} "
        f"n_lo_mode={getattr(model, 'n_lo_mode', 'attn')} "
        f"inject_every_layer={getattr(getattr(model, 'predictor', None), 'inject_every_layer', None)}"
    )

    model.eval()
    model.requires_grad_(False)
    if hasattr(model.encoder, "interpolate_pos_encoding"):
        model.encoder.interpolate_pos_encoding = True
    return model


# ---------------------------------------------------------------------------
# PointMaze env
# ---------------------------------------------------------------------------


def _make_gym_register_idempotent():
    """跳过已注册 / 非法 ID（如本仓库 env/__init__.py 里的 'pusht'）。"""
    import re
    import gym
    from gym.envs import registration as reg

    if getattr(reg, "_lewm_idempotent_register", False):
        return

    _orig = reg.register
    _id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")

    def _safe_register(id, **kwargs):
        if not _id_re.match(str(id)):
            return
        specs = getattr(reg.registry, "env_specs", None)
        if specs is not None and id in specs:
            return
        try:
            return _orig(id, **kwargs)
        except gym.error.Error as e:
            msg = str(e)
            if "Cannot re-register" in msg or "malformed environment ID" in msg:
                return
            raise

    reg.register = _safe_register
    reg._lewm_idempotent_register = True


def _import_point_maze_cls():
    """绕过 env/__init__.py 与 pointmaze/__init__.py 的 gym.register 副作用。"""
    import importlib.util
    import types

    pkg_dir = ROOT / "env" / "pointmaze"

    if "env" not in sys.modules or not getattr(sys.modules["env"], "__path__", None):
        env_pkg = types.ModuleType("env")
        env_pkg.__path__ = [str(ROOT / "env")]
        env_pkg.__package__ = "env"
        sys.modules["env"] = env_pkg

    if "env.pointmaze" not in sys.modules or not getattr(
        sys.modules["env.pointmaze"], "__path__", None
    ):
        pm_pkg = types.ModuleType("env.pointmaze")
        pm_pkg.__path__ = [str(pkg_dir)]
        pm_pkg.__package__ = "env.pointmaze"
        pm_pkg.__file__ = str(pkg_dir / "__init__.py")
        sys.modules["env.pointmaze"] = pm_pkg
        setattr(sys.modules["env"], "pointmaze", pm_pkg)

    def _load(modname: str, filename: str):
        existing = sys.modules.get(modname)
        if existing is not None and getattr(existing, "__file__", None):
            return existing
        path = pkg_dir / filename
        spec = importlib.util.spec_from_file_location(
            modname, path, submodule_search_locations=None
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "env.pointmaze"
        sys.modules[modname] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        short = filename[:-3]
        setattr(sys.modules["env.pointmaze"], short, mod)
        return mod

    _load("env.pointmaze.dynamic_mjc", "dynamic_mjc.py")
    maze_model = _load("env.pointmaze.maze_model", "maze_model.py")
    wrapper = _load("env.pointmaze.point_maze_wrapper", "point_maze_wrapper.py")
    return maze_model.U_MAZE, wrapper.PointMazeWrapper


def make_point_maze_env():
    _preload_mujoco_libs()
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    _make_gym_register_idempotent()
    try:
        U_MAZE, PointMazeWrapper = _import_point_maze_cls()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "导入 PointMazeEnv 失败。请确认已安装 gym/mujoco_py，且在仓库根目录运行。"
            f" 原始错误: {e}"
        ) from e

    # 与 env/__init__.py 中 point_maze 注册一致：U_MAZE + sparse
    env = PointMazeWrapper(
        maze_spec=U_MAZE,
        reward_type="sparse",
        reset_target=False,
    )
    return env


def run_sim_episode(env, init_state, goal_state, actions, seed: int):
    """执行 raw actions (T, 2)，返回 success / state_dist。"""
    init = np.asarray(init_state, dtype=np.float32).reshape(-1)
    env.prepare(seed, init)

    act_arr = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
    if act_arr.shape[0] == 0:
        raise RuntimeError("空动作序列")

    act_list = [a for a in act_arr]
    _, _, _, infos = env.step_multiple(act_list)
    final_state = infos["state"][-1]
    if torch.is_tensor(final_state):
        final_state = final_state.detach().cpu().numpy()
    final_state = np.asarray(final_state, dtype=np.float32).reshape(-1)

    goal = np.asarray(goal_state, dtype=np.float32).reshape(-1)
    metrics = env.eval_state(goal, final_state)
    return {
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "final_state": final_state,
    }


def run_mpc_episode(
    model,
    env,
    *,
    init_clip: dict,
    init_state,
    goal_state,
    goal_pixels,
    goal_proprio,
    process: dict,
    transform: dict,
    history_size: int,
    frameskip: int,
    action_dim: int,
    horizon: int,
    n_taken_actions: int,
    mpc_max_iters: int,
    cem_kwargs: dict,
    device: str,
    seed: int,
    plan_ctx: int,
):
    """MPC: CEM plan → execute n_taken macros → replan from env feedback."""
    effective_act_dim = frameskip * action_dim
    init = np.asarray(init_state, dtype=np.float32).reshape(-1)
    env.prepare(seed, init)

    pixels_np = as_numpy(init_clip["pixels"])
    proprio_np = as_numpy(init_clip["proprio"])
    hist_pixels = [to_hwc_uint8_compatible(pixels_np[i]) for i in range(plan_ctx)]
    hist_proprio = [np.asarray(proprio_np[i], dtype=np.float32).reshape(-1) for i in range(plan_ctx)]
    hist_act_raw = np.asarray(init_clip["action_raw"], dtype=np.float32).reshape(-1, action_dim)

    goal = np.asarray(goal_state, dtype=np.float32).reshape(-1)
    final_state = init.copy()
    plan_costs = []
    n_mpc_iters = 0
    success = False
    state_dist = float(env.eval_state(goal, final_state)["state_dist"])
    dist_history = [state_dist]

    for it in range(mpc_max_iters):
        cur_metrics = env.eval_state(goal, final_state)
        state_dist = float(cur_metrics["state_dist"])
        if cur_metrics["success"]:
            success = True
            print(f"  [MPC] early success at iter={it}", flush=True)
            break

        clip = stack_history_clip(
            hist_pixels, hist_proprio, hist_act_raw, plan_ctx, frameskip, action_dim
        )
        clip["action"] = normalize_concat_actions(
            clip["action_raw"], process, plan_ctx, frameskip, action_dim
        )
        info = build_info_dict(clip, goal_pixels, goal_proprio, process, transform)
        planned_norm, plan_cost = cem_plan(
            model,
            info,
            horizon=horizon,
            action_dim=effective_act_dim,
            device=device,
            **cem_kwargs,
        )
        plan_costs.append(float(plan_cost))
        n_mpc_iters = it + 1

        planned = planned_norm.detach().cpu().numpy()
        n_take = min(int(n_taken_actions), int(planned.shape[0]))
        env_actions = denormalize_concat_actions(
            planned[:n_take], process, frameskip, action_dim
        )
        print(
            f"  [MPC {it + 1}/{mpc_max_iters}] cost={plan_cost:.4f} "
            f"take={n_take} raw_steps={env_actions.shape[0]} dist={state_dist:.4f}",
            flush=True,
        )

        act_list = [a for a in np.asarray(env_actions, dtype=np.float32).reshape(-1, 2)]
        obses, _, _, infos = env.step_multiple(act_list)

        new_pixels, new_proprio = extract_macro_frames(obses, n_take, frameskip)
        hist_act_raw = np.concatenate([hist_act_raw, env_actions], axis=0)
        need = plan_ctx * frameskip
        hist_act_raw = hist_act_raw[-need:]
        hist_pixels = (hist_pixels + new_pixels)[-plan_ctx:]
        hist_proprio = (hist_proprio + new_proprio)[-plan_ctx:]

        final_state = as_numpy(infos["state"][-1]).astype(np.float32).reshape(-1)
        m = env.eval_state(goal, final_state)
        state_dist = float(m["state_dist"])
        dist_history.append(state_dist)
        success = bool(m["success"])
        if success:
            print(f"  [MPC] success after iter={it + 1}", flush=True)
            break

    return {
        "success": success,
        "state_dist": state_dist,
        "final_state": final_state,
        "plan_cost": float(np.mean(plan_costs)) if plan_costs else -1.0,
        "mpc_iters": n_mpc_iters,
        "dist_history": dist_history,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def evaluate(args):
    effective_act_dim = args.frameskip * args.action_dim
    print("=" * 60)
    print("LeWM PointMaze 仿真评估")
    print("=" * 60)
    print(
        f"[INFO] PLAN_GPU={os.environ.get('PLAN_GPU')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"(进程内 cuda:0 = 该物理卡)"
    )
    print(
        f"[INFO] action_dim={args.action_dim} frameskip={args.frameskip} "
        f"effective_act_dim={effective_act_dim} history={args.history_size} "
        f"horizon={args.horizon} goal_offset={args.goal_offset} num_eval={args.num_eval}"
    )
    print(
        f"[INFO] planner={args.planner} n_taken={args.n_taken_actions} "
        f"mpc_max_iters={args.mpc_max_iters}"
    )
    print("[INFO] success: ||goal[:2]-cur[:2]|| < 0.5 （对齐 DINO-WM）")

    device = args.plan_device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA unavailable (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}); "
            "pass a valid --plan-gpu or --plan-device cpu explicitly"
        )

    print(f"[INFO] 加载模型: {args.ckpt}")
    model = load_lewm_model(
        args.ckpt,
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
    model = model.to(device)

    stats_path = resolve_stats_path(args.dataset, args.stats_dataset)
    print(f"[INFO] normalizer stats: {stats_path}  plan_protocol={args.plan_protocol}")
    process = build_processors(stats_path)
    ctx = plan_context_len(args.plan_protocol, args.history_size)
    transform = {"pixels": img_transform(args.img_size), "goal": img_transform(args.img_size)}
    lookup = build_goal_lookup(args.dataset)
    eval_episodes, eval_starts = sample_eval_points(
        args.dataset,
        args.num_eval,
        args.goal_offset,
        args.history_size,
        args.frameskip,
        args.seed,
        split=args.split,
    )
    args.num_eval = len(eval_episodes)

    print("[INFO] 初始化 PointMazeEnv...")
    env = make_point_maze_env()

    successes, dists, costs = [], [], []
    details = []
    t0 = time.time()

    for i, (ep_id, start_step) in enumerate(zip(eval_episodes, eval_starts)):
        print(f"\n[{i + 1}/{args.num_eval}] episode={ep_id} start={start_step}", flush=True)
        torch.manual_seed(int(args.seed) + i)
        goal_step = int(start_step + args.goal_offset)
        start_idx = lookup[(int(ep_id), int(start_step))]
        goal_idx = lookup[(int(ep_id), goal_step)]

        start_fr = load_frame(args.dataset, start_idx)
        goal_fr = load_frame(args.dataset, goal_idx)
        if start_fr["state"] is None or goal_fr["state"] is None:
            print("  [WARN] 缺少 state，跳过")
            continue

        clip = load_history_with_frameskip(
            args.dataset,
            lookup,
            ep_id,
            start_step,
            ctx,
            args.frameskip,
            args.action_dim,
        )
        clip["action"] = normalize_concat_actions(
            clip["action_raw"],
            process,
            ctx,
            args.frameskip,
            args.action_dim,
        )

        cem_kwargs = dict(
            num_samples=args.cem_num_samples,
            n_steps=args.cem_n_steps,
            topk=args.cem_topk,
            var_scale=args.cem_var_scale,
            protocol=args.plan_protocol,
        )

        if args.planner == "mpc":
            sim = run_mpc_episode(
                model,
                env,
                init_clip=clip,
                init_state=start_fr["state"],
                goal_state=goal_fr["state"],
                goal_pixels=goal_fr["pixels"],
                goal_proprio=goal_fr["proprio"],
                process=process,
                transform=transform,
                history_size=args.history_size,
                plan_ctx=ctx,
                frameskip=args.frameskip,
                action_dim=args.action_dim,
                horizon=args.horizon,
                n_taken_actions=args.n_taken_actions,
                mpc_max_iters=args.mpc_max_iters,
                cem_kwargs=cem_kwargs,
                device=device,
                seed=args.seed + i,
            )
            plan_cost = sim["plan_cost"]
            print(f"  mpc_iters={sim['mpc_iters']}", flush=True)
        else:
            info = build_info_dict(
                clip,
                goal_fr["pixels"],
                goal_fr["proprio"],
                process,
                transform,
            )
            planned_norm, plan_cost = cem_plan(
                model,
                info,
                horizon=args.horizon,
                action_dim=effective_act_dim,
                device=device,
                **cem_kwargs,
            )
            env_actions = denormalize_concat_actions(
                planned_norm.detach().cpu().numpy(),
                process,
                args.frameskip,
                args.action_dim,
            )
            print(
                f"  plan_cost={plan_cost:.4f} env_actions={env_actions.shape}",
                flush=True,
            )
            sim = run_sim_episode(
                env,
                init_state=start_fr["state"],
                goal_state=goal_fr["state"],
                actions=env_actions,
                seed=args.seed + i,
            )

        successes.append(sim["success"])
        dists.append(sim["state_dist"])
        costs.append(float(plan_cost))
        detail = {
            "ep_id": int(ep_id),
            "start_step": int(start_step),
            "goal_step": goal_step,
            "success": sim["success"],
            "state_dist": sim["state_dist"],
            "plan_cost": float(plan_cost),
            "planner": args.planner,
        }
        if args.planner == "mpc":
            detail["mpc_iters"] = sim.get("mpc_iters", -1)
            detail["dist_history"] = sim.get("dist_history")
        details.append(detail)
        print(
            f"  state_dist={sim['state_dist']:.4f} success={sim['success']}",
            flush=True,
        )

    elapsed = time.time() - t0
    metrics = {
        "num_eval": len(successes),
        "success_rate": float(np.mean(successes) * 100.0) if successes else 0.0,
        "success_threshold": 0.5,
        "mean_state_dist": float(np.mean(dists)) if dists else -1.0,
        "median_state_dist": float(np.median(dists)) if dists else -1.0,
        "mean_planning_cost": float(np.mean(costs)) if costs else -1.0,
        "elapsed_sec": float(elapsed),
        "action_dim": args.action_dim,
        "frameskip": args.frameskip,
        "effective_act_dim": effective_act_dim,
        "horizon": args.horizon,
        "goal_offset": args.goal_offset,
        "history_size": args.history_size,
        "plan_protocol": args.plan_protocol,
        "split": args.split,
        "stats_dataset": stats_path,
        "planner": args.planner,
        "n_taken_actions": args.n_taken_actions if args.planner == "mpc" else None,
        "mpc_max_iters": args.mpc_max_iters if args.planner == "mpc" else None,
    }
    return metrics, details


def parse_args():
    p = argparse.ArgumentParser(description="LeWM PointMaze 仿真评估")
    p.add_argument("--dataset", default="./data/point_maze_eval.h5")
    p.add_argument("--ckpt", default="./data/lewm_point_maze_v2_weights.ckpt")
    p.add_argument("--num-eval", type=int, default=0, help="0 = every episode of --split exactly once")
    p.add_argument(
        "--goal-offset",
        type=int,
        default=25,
        help="目标相对 start 的 raw 步数；默认 horizon*frameskip=25（对齐 DINO goal_H）",
    )
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="./outputs/point_maze_sim_results.txt")
    p.add_argument("--details-json", default="./outputs/point_maze_sim_details.json")

    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--encoder-scale", default="tiny")
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)
    p.add_argument(
        "--token-mode",
        default="cls",
        choices=["cls", "spatial16", "spatial"],
        help="cls=baseline；spatial16=C1 16-token（weights ckpt 重建时用）",
    )
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

    # 对齐 plan_point_maze.yaml: samples=300, opt_steps=10, topk=30
    p.add_argument("--cem-num-samples", type=int, default=300)
    p.add_argument("--cem-n-steps", type=int, default=10)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-var-scale", type=float, default=1.0)
    p.add_argument(
        "--split",
        default="dev",
        choices=EVAL_SPLITS,
        help="dev = odd eval episodes (tuning); test = even (final numbers only); all = both",
    )
    p.add_argument(
        "--plan-protocol",
        default="lewm",
        choices=PLAN_PROTOCOLS,
        help="lewm: 1-frame context, CEM over horizon only, return elite mean (official LeWM/swm); "
        "legacy: r1 behaviour (history_size dataset frames, pinned history actions, best sample)",
    )
    p.add_argument(
        "--stats-dataset",
        default=None,
        help="h5 used to fit action/proprio normalizers; default: *_train.h5 next to --dataset",
    )
    p.add_argument(
        "--planner",
        choices=["mpc", "openloop"],
        default="mpc",
        help="mpc=滚动规划（默认）；openloop=旧开环 CEM",
    )
    p.add_argument(
        "--n-taken-actions",
        type=int,
        default=5,
        help="MPC 每轮执行的 macro action 数（对齐 DINO n_taken_actions=5）",
    )
    p.add_argument(
        "--mpc-max-iters",
        type=int,
        default=10,
        help="MPC 最大重规划轮数；成功则提前停（r2：10，主表读 success@2）",
    )
    p.add_argument(
        "--plan-gpu",
        type=str,
        default="0",
        help="规划用物理 GPU 编号（默认 0；在 import torch 前写入 CUDA_VISIBLE_DEVICES）",
    )
    p.add_argument("--plan-device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    metrics, details = evaluate(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("==== POINT_MAZE SIMULATION EVALUATION ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    details_path = Path(args.details_json)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": details}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(
        f"success_rate={metrics['success_rate']:.1f}%  "
        f"mean_state_dist={metrics['mean_state_dist']:.4f}  "
        f"median_state_dist={metrics['median_state_dist']:.4f}"
    )
    print(f"结果写入: {out}")
    print(f"明细写入: {details_path}")


if __name__ == "__main__":
    main()
