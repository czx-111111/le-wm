#!/usr/bin/env bash
# wall CLS+S2: health → eval (split=all, all valid eps), dual-GPU
set -euo pipefail
cd /data/czx/le-wm/first
export STABLEWM_HOME="${STABLEWM_HOME:-/data/czx/le-wm/first/data}"
mkdir -p logs/r3 outputs/r3

CLS_CKPT=data/runs/r3/wall_cls_r3_s3072_20261001_093335/wall_cls_last_object.ckpt
S2_CKPT=data/runs/r3/wall_s2_r3_s3072_20261001_093335/wall_s2_last_object.ckpt

echo "[$(date +%H:%M:%S)] HEALTH wall CLS+S2"
# shellcheck disable=SC1091
source setup_env.sh >/dev/null
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/repr_health.py --env wall \
  --ckpt "$CLS_CKPT" --ckpt "$S2_CKPT" --n-eps 30

OUT_CLS=outputs/r3/wall_cls_wall_cls_r3_s3072_20261001_093335_all_e42
OUT_S2=outputs/r3/wall_s2_wall_s2_r3_s3072_20261001_093335_all_e42
EXTRA=(--num-eval 0 --split all)

echo "[$(date +%H:%M:%S)] EVAL wall CLS on GPU0 + S2 on GPU1 (parallel)"
PLAN_GPU=0 bash scripts/eval_r3.sh wall cls "$CLS_CKPT" \
  "${EXTRA[@]}" --output "${OUT_CLS}.txt" --details-json "${OUT_CLS}_details.json" \
  > logs/r3/eval_wall_cls.log 2>&1 &
PID_CLS=$!

PLAN_GPU=1 bash scripts/eval_r3.sh wall s2 "$S2_CKPT" \
  "${EXTRA[@]}" --output "${OUT_S2}.txt" --details-json "${OUT_S2}_details.json" \
  > logs/r3/eval_wall_s2.log 2>&1 &
PID_S2=$!

echo "CLS_PID=$PID_CLS S2_PID=$PID_S2"
wait $PID_CLS; EC1=$?
wait $PID_S2; EC2=$?
echo "[$(date +%H:%M:%S)] CLS exit=$EC1 S2 exit=$EC2"
echo '=== CLS summary ==='
grep -E 'success_rate|有效 episode|Error|Traceback' logs/r3/eval_wall_cls.log | tail -10
tail -n 5 "${OUT_CLS}.txt" 2>/dev/null || true
echo '=== S2 summary ==='
grep -E 'success_rate|有效 episode|Error|Traceback' logs/r3/eval_wall_s2.log | tail -10
tail -n 5 "${OUT_S2}.txt" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] DONE"
exit $(( EC1 != 0 || EC2 != 0 ? 1 : 0 ))
