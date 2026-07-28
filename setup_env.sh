#!/bin/bash
# 在 le-wm 根目录下 source 本脚本后再跑评估
#   source setup_env.sh
#   python eval_granular.py --smoke-env --egl-gpu 4
#
# 规划与 Flex 必须用不同物理 GPU:
#   PLAN_GPU -> PyTorch CEM
#   EGL_GPU  -> PyFleX 仿真

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MuJoCo / mujoco_py 系统依赖（point_maze 评估需要）:
#   sudo apt-get install -y libglew-dev libglfw3-dev libosmesa6-dev
#   pip install 'cython==0.29.37'
#   python fix_mujoco_py_cython.py
if [ -d "$ROOT/mujoco210/mujoco210/bin" ]; then
  export MUJOCO_PY_MUJOCO_PATH="$ROOT/mujoco210/mujoco210"
  export LD_LIBRARY_PATH="$ROOT/mujoco210/mujoco210/bin:${LD_LIBRARY_PATH}"
elif [ -d "$ROOT/mujoco210/bin" ]; then
  export MUJOCO_PY_MUJOCO_PATH="$ROOT/mujoco210"
  export LD_LIBRARY_PATH="$ROOT/mujoco210/bin:${LD_LIBRARY_PATH}"
elif [ -d "$HOME/.mujoco/mujoco210/bin" ]; then
  export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
  export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# PyFleX 在 AdaptiGraph/PyFleX
export PYFLEXROOT="$ROOT/AdaptiGraph/PyFleX"
export PYTHONPATH="${PYFLEXROOT}/bindings/build:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}"

# Docker headless + NVIDIA EGL
export PYFLEX_HEADLESS=1
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

# Flex/EGL 与规划默认不同卡（可按需 export 覆盖）
# 注意: CUDA_VISIBLE_DEVICES 只给「规划进程」用。
# 跑 eval_*_sim_worker / smoke 前请: unset CUDA_VISIBLE_DEVICES
export EGL_GPU="${EGL_GPU:-4}"
export PLAN_GPU="${PLAN_GPU:-3}"
export CUDA_VISIBLE_DEVICES="$PLAN_GPU"

# 避免 Mesa 抢 EGL
if [ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
  mv /usr/share/glvnd/egl_vendor.d/50_mesa.json \
     /usr/share/glvnd/egl_vendor.d/50_mesa.json.disabled 2>/dev/null || true
fi

echo "PYFLEXROOT=$PYFLEXROOT"
echo "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
echo "PLAN_GPU=$PLAN_GPU (torch)  EGL_GPU=$EGL_GPU (pyflex)  — 必须不同"

if ! ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia'; then
  echo "[WARN] 找不到 libEGL_nvidia —— 当前容器多半没开 graphics。"
  echo "       请重建容器: -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
fi

if python -c "import pyflex; print('pyflex OK:', pyflex.__file__)" 2>/dev/null; then
  :
else
  echo "[WARN] import pyflex 失败。请确认:"
  echo "  ls ${PYFLEXROOT}/bindings/build/pyflex*.so"
fi
