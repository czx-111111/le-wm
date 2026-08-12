#!/usr/bin/env python3
"""
Granular / Flex 常驻仿真 worker（MPC 用）。

故意不 import torch。父进程通过 ipc_dir 下的 request.pkl / response.pkl 通信。

协议:
  request: {"cmd": "prepare"|"step"|"shutdown", ...}
  response: {"ok": True/False, "error": str?, ...}

用法（一般由 eval_granular.py 拉起）:
  unset CUDA_VISIBLE_DEVICES
  export EGL_GPU=4
  python eval_granular_mpc_worker.py --ipc-dir /tmp/xxx --object-name granular
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
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

    egl_gpu = os.environ.get("EGL_GPU", "4")
    os.environ["EGL_GPU"] = str(egl_gpu)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    os.environ["PYFLEXROOT"] = str(pyflex_root)
    os.environ["PYTHONPATH"] = str(build_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    extras = [str(sdl_lib), "/usr/lib/nvidia"]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(extras + ([ld] if ld else []))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))


def to_particle_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        state = state.reshape(-1, 4)
    return state


def _atomic_write_pickle(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _wait_file(path: Path, timeout: float | None = None) -> bool:
    t0 = time.time()
    while not path.exists():
        if timeout is not None and (time.time() - t0) > timeout:
            return False
        time.sleep(0.01)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-dir", required=True)
    parser.add_argument("--object-name", default="granular")
    args = parser.parse_args()

    setup_env()
    ipc = Path(args.ipc_dir)
    ipc.mkdir(parents=True, exist_ok=True)
    req_path = ipc / "request.pkl"
    resp_path = ipc / "response.pkl"
    for p in (req_path, resp_path):
        if p.exists():
            p.unlink()

    print(
        f"[mpc-worker] object={args.object_name} EGL_GPU={os.environ.get('EGL_GPU')} "
        f"ipc={ipc}",
        flush=True,
    )

    import pyflex

    pyflex.set_screenWidth(224)
    pyflex.set_screenHeight(224)
    pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
    pyflex.set_light_fov(70.0)
    print("[mpc-worker] pyflex.init(True)...", flush=True)
    pyflex.init(True)
    print("[mpc-worker] pyflex.init OK", flush=True)

    from deformable_env.FlexEnvWrapper import FlexEnvWrapper
    from deformable_env.src.sim.sim_env import flex_env as flex_env_mod

    flex_env_mod._PYFLEX_INITIALIZED = True
    env = FlexEnvWrapper(args.object_name)
    print("[mpc-worker] FlexEnv OK; waiting for requests", flush=True)

    # ready flag for parent
    _atomic_write_pickle(ipc / "ready.pkl", {"ok": True})

    while True:
        if not _wait_file(req_path, timeout=None):
            continue
        # small delay to avoid reading partial write
        time.sleep(0.02)
        try:
            with open(req_path, "rb") as f:
                req = pickle.load(f)
        except Exception as e:
            _atomic_write_pickle(resp_path, {"ok": False, "error": f"bad request: {e}"})
            try:
                req_path.unlink()
            except OSError:
                pass
            continue

        try:
            req_path.unlink()
        except OSError:
            pass

        cmd = req.get("cmd")
        try:
            if cmd == "shutdown":
                _atomic_write_pickle(resp_path, {"ok": True})
                print("[mpc-worker] shutdown", flush=True)
                break

            if cmd == "prepare":
                init = to_particle_state(req["init_state"])
                seed = int(req["seed"])
                obs, state_dct = env.prepare(seed, init)
                vis = np.asarray(obs["visual"])
                prop = np.asarray(state_dct.get("proprio", obs["proprio"]), dtype=np.float32)
                st = np.asarray(state_dct["state"], dtype=np.float32)
                _atomic_write_pickle(
                    resp_path,
                    {
                        "ok": True,
                        "visual": vis,
                        "proprio": prop.reshape(-1),
                        "state": st,
                    },
                )
                continue

            if cmd == "step":
                actions = np.asarray(req["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                act_list = [actions[i] for i in range(actions.shape[0])]
                obses, _, _, infos = env.step_multiple(act_list)
                visuals = np.asarray(obses["visual"])  # (T,H,W,C)
                # prefer eef proprio from infos if present
                if "pos_agent" in infos:
                    proprios = np.asarray(infos["pos_agent"], dtype=np.float32)
                else:
                    proprios = np.asarray(obses["proprio"], dtype=np.float32)
                states = np.asarray(infos["state"], dtype=np.float32)
                final_state = states[-1]
                _atomic_write_pickle(
                    resp_path,
                    {
                        "ok": True,
                        "visuals": visuals,
                        "proprios": proprios,
                        "states": states,
                        "final_state": final_state,
                    },
                )
                continue

            _atomic_write_pickle(resp_path, {"ok": False, "error": f"unknown cmd={cmd}"})
        except Exception as e:
            _atomic_write_pickle(resp_path, {"ok": False, "error": str(e)})
            print(f"[mpc-worker] ERROR cmd={cmd}: {e}", flush=True)


if __name__ == "__main__":
    main()
