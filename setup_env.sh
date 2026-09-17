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
# 只加 /usr/lib/nvidia（EGL）；不要把 /usr/lib/x86_64-linux-gnu 插到前面，
# 否则会盖住 venv 里 torch 自带的 libnvJitLink，导致 cusparse 符号错误。
export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH}"
# 保证 torch 的 nvidia 轮子优先
if [ -d "$ROOT/.venv/lib/python3.10/site-packages/nvidia" ]; then
  for _nv in nvjitlink cusparse cublas cudnn cufft curand cusolver cuda_runtime cuda_nvrtc nccl; do
    _d="$ROOT/.venv/lib/python3.10/site-packages/nvidia/${_nv}/lib"
    if [ -d "$_d" ]; then
      export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH}"
    fi
  done
fi
unset _nv _d

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
# 本机仅有 GPU 0/1；多卡机器可 export 覆盖
export EGL_GPU="${EGL_GPU:-1}"
export PLAN_GPU="${PLAN_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$PLAN_GPU"

# 用户空间 sysroot（无 sudo apt 时从 .deb 解压的 GL/EGL/GLEW/OSMesa）
# 注意: 运行时不要把 sysroot/cuda9.2 插到 LD_LIBRARY_PATH 前面，会破坏 torch 的 NVIDIA 库
DEPS_ROOT="$(cd "$ROOT/.." && pwd)/deps"
SYSROOT="$DEPS_ROOT/sysroot"
if [ -d "$SYSROOT/usr/include" ]; then
  export CPATH="$SYSROOT/usr/include:${CPATH:-}"
  export C_INCLUDE_PATH="$SYSROOT/usr/include:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="$SYSROOT/usr/include:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="$SYSROOT/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
  export PKG_CONFIG_PATH="$SYSROOT/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
# mujoco_py / 本机编译用
export CFLAGS="-I${SYSROOT}/usr/include ${CFLAGS:-}"
export CXXFLAGS="-I${SYSROOT}/usr/include ${CXXFLAGS:-}"
export LDFLAGS="-L${SYSROOT}/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
if [ -x "$SYSROOT/usr/bin/patchelf" ]; then
  export PATH="$SYSROOT/usr/bin:$PATH"
fi
# PyFleX 的 CUDA9.2 cudart 已打进 RPATH；仅作兜底追加到末尾
if [ -d "$DEPS_ROOT/cuda-9.2/lib64" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$DEPS_ROOT/cuda-9.2/lib64"
fi

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
