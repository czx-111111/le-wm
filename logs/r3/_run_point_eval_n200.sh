#!/usr/bin/env bash
# point CLS+S2 eval, 200 episodes, split=all, PLAN_GPU=1
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
EXTRA=(--num-eval 200 --split all)

OUT_CLS=outputs/r3/point_maze_cls_point_maze_cls_r3_s3072_20260930_213550_all_n200_e42
OUT_S2=outputs/r3/point_maze_s2_point_maze_s2_r3_s3072_20260930_213550_all_n200_e42

echo "[$(date +%H:%M:%S)] EVAL point CLS n=200 split=all"
bash scripts/eval_r3.sh point_maze cls "$CLS_CKPT" \
  "${EXTRA[@]}" \
  --output "${OUT_CLS}.txt" --details-json "${OUT_CLS}_details.json"

echo "[$(date +%H:%M:%S)] EVAL point S2 n=200 split=all"
bash scripts/eval_r3.sh point_maze s2 "$S2_CKPT" \
  "${EXTRA[@]}" \
  --output "${OUT_S2}.txt" --details-json "${OUT_S2}_details.json"

echo "[$(date +%H:%M:%S)] DONE"
