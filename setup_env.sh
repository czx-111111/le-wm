#!/bin/bash
# 设置 MuJoCo 和 PyFleX 环境变量（仅对当前终端有效）

# MuJoCo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/mujoco210/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# PyFleX
export PYFLEXROOT=$(pwd)/PyFleX
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH

echo "✅ 环境变量已设置（MuJoCo + PyFleX for Python 3.10）"
