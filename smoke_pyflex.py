#!/usr/bin/env python3
"""最小 pyflex.init 冒烟测试（不 import torch / FlexEnv）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
build = ROOT / "AdaptiGraph" / "PyFleX" / "bindings" / "build"
sys.path.insert(0, str(build))

os.environ.setdefault("PYFLEX_HEADLESS", "1")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_GPU", "5")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import numpy as np
import pyflex

print(
    f"EGL_GPU={os.environ.get('EGL_GPU')} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')} "
    f"pyflex={pyflex.__file__}",
    flush=True,
)
pyflex.set_screenWidth(64)
pyflex.set_screenHeight(64)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.0)
print("pyflex.init(True)...", flush=True)
pyflex.init(True)
print("SUCCESS: pyflex.init OK", flush=True)
