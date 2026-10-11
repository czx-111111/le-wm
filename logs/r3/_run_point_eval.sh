#!/usr/bin/env bash
# point CLS+S2 eval on PLAN_GPU (default 1). Health already passed.
set -euo pipefail
cd /data/czx/le-wm/first
export STABLEWM_HOME="${STABLEWM_HOME:-/data/czx/le-wm/first/data}"
export PLAN_GPU="${PLAN_GPU:-1}"
export EGL_GPU="${EGL_GPU:-0}"
# shellcheck disable=SC1091
source setup_env.sh >/dev/null
echo "[$(date +%H:%M:%S)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PLAN_GPU=${PLAN_GPU}"

CLS_CKPT=data/runs/r3/point_maze_cls_r3_s3072_20260930_213550/point_maze_cls_last_object.ckpt
S2_CKPT=data/runs/r3/point_maze_s2_r3_s3072_20260930_213550/point_maze_s2_last_object.ckpt

echo "[$(date +%H:%M:%S)] EVAL point CLS"
bash scripts/eval_r3.sh point_maze cls "$CLS_CKPT"
echo "[$(date +%H:%M:%S)] EVAL point S2"
bash scripts/eval_r3.sh point_maze s2 "$S2_CKPT"
echo "[$(date +%H:%M:%S)] DONE"
