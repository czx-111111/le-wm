#!/usr/bin/env bash
# Re-eval previous wall CLS+S2 (20261001_093335), 100 eps, GPU0 — after rope eval frees the card.
set -euo pipefail
cd /data/czx/le-wm/first
export STABLEWM_HOME="${STABLEWM_HOME:-/data/czx/le-wm/first/data}"
export PLAN_GPU=0
export EGL_GPU=1
mkdir -p logs/r3 outputs/r3

echo "[$(date +%H:%M:%S)] waiting for rope eval to leave GPU0..."
while pgrep -f 'eval_rope_1\.py' >/dev/null 2>&1; do
  sleep 30
done
# brief settle
sleep 5
# shellcheck disable=SC1091
source setup_env.sh >/dev/null
echo "[$(date +%H:%M:%S)] GPU0 free; PLAN=$PLAN_GPU CUDA=$CUDA_VISIBLE_DEVICES"

CLS_CKPT=data/runs/r3/wall_cls_r3_s3072_20261001_093335/wall_cls_last_object.ckpt
S2_CKPT=data/runs/r3/wall_s2_r3_s3072_20261001_093335/wall_s2_last_object.ckpt
OUT_CLS=outputs/r3/wall_cls_wall_cls_r3_s3072_20261001_093335_all_n100_e42
OUT_S2=outputs/r3/wall_s2_wall_s2_r3_s3072_20261001_093335_all_n100_e42
EXTRA=(--num-eval 100 --split all)

echo "[$(date +%H:%M:%S)] EVAL wall CLS n=100"
bash scripts/eval_r3.sh wall cls "$CLS_CKPT" \
  "${EXTRA[@]}" --output "${OUT_CLS}.txt" --details-json "${OUT_CLS}_details.json"

echo "[$(date +%H:%M:%S)] EVAL wall S2 n=100"
bash scripts/eval_r3.sh wall s2 "$S2_CKPT" \
  "${EXTRA[@]}" --output "${OUT_S2}.txt" --details-json "${OUT_S2}_details.json"

echo "[$(date +%H:%M:%S)] DONE"
echo '=== CLS ==='
grep -E 'success_rate|num_eval' "${OUT_CLS}.txt" || true
echo '=== S2 ==='
grep -E 'success_rate|num_eval' "${OUT_S2}.txt" || true
