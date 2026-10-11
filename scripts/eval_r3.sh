#!/usr/bin/env bash
# 正式 r3 评估：scripts/eval_r3.sh <env> <method> <ckpt> [额外 CLI...]
# 例: scripts/eval_r3.sh wall s2 data/runs/r3/wall_s2_s3072_.../wall_s2_last_object.ckpt
#     scripts/eval_r3.sh rope cls $CK --split test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/setup_env.sh" >/dev/null

ENV="${1:?usage: $0 <wall|point_maze|rope|granular> <cls|c1|s1|s2> <ckpt> [...]}"
METHOD="${2:?}"
CKPT="${3:?}"
shift 3

export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT/data}"
mkdir -p "$ROOT/outputs/r3"

# 阶段1 cost*（docs/standard_r3.md §4.2.1）：C1/S1/S2 正式规划用
COST_STAR=(
  --cost-mode decoupled --cost-norm calib --cost-lambda 2
  --mask-weight hard --tau-mode fixed --m-task-threshold 0.08
  --relevance-source pixel --cvar-q 0.1 --cvar-beta 0.5 --lambda-anneal off
)

case "$METHOD" in
  cls)
    M=(--token-mode cls --cost-mode dense_terminal)
    ;;
  c1)
    M=(--token-mode spatial16 --pool-mode avg "${COST_STAR[@]}")
    ;;
  s1)
    M=(--token-mode spatial16 --pool-mode query --include-global-token \
       --pos-mode learned "${COST_STAR[@]}")
    ;;
  s2)
    M=(--token-mode spatial16 --pool-mode query --include-global-token \
       --pos-mode nibble --inject-every-layer --n-lo-mode attn \
       "${COST_STAR[@]}")
    ;;
  *) echo "unknown method: $METHOD" >&2; exit 1;;
esac

TAG="${ENV}_${METHOD}_$(basename "$(dirname "$CKPT")")_dev_e42"
OUT="$ROOT/outputs/r3/$TAG"

# plan-gpu = 物理卡号（eval_*.py 会据此设置 CUDA_VISIBLE_DEVICES）
COMMON=(--ckpt "$CKPT" --split dev --num-eval 0 --seed 42 --plan-protocol lewm --plan-gpu "${PLAN_GPU:-0}")

case "$ENV" in
  wall)
    exec .venv/bin/python eval_wall.py \
      --dataset "$STABLEWM_HOME/wall_eval.h5" \
      --goal-offset 25 --horizon 5 --planner mpc \
      --n-taken-actions 5 --mpc-max-iters 10 \
      --cem-num-samples 300 --cem-topk 30 --cem-n-steps 10 \
      "${COMMON[@]}" "${M[@]}" \
      --output "${OUT}.txt" --details-json "${OUT}_details.json" "$@"
    ;;
  point_maze)
    exec .venv/bin/python eval_point_maze.py \
      --dataset "$STABLEWM_HOME/point_maze_eval.h5" \
      --goal-offset 25 --horizon 5 --planner mpc \
      --n-taken-actions 5 --mpc-max-iters 10 \
      --cem-num-samples 300 --cem-topk 30 --cem-n-steps 10 \
      "${COMMON[@]}" "${M[@]}" \
      --output "${OUT}.txt" --details-json "${OUT}_details.json" "$@"
    ;;
  rope)
    exec .venv/bin/python eval_rope_1.py \
      --dataset "$STABLEWM_HOME/rope_eval.h5" \
      --goal-offset 5 --horizon 5 --history-size 3 --action-dim 4 \
      --cem-num-samples 300 --cem-topk 30 --cem-n-steps 30 \
      --egl-gpu "${EGL_GPU:-1}" \
      "${COMMON[@]}" "${M[@]}" \
      --output "${OUT}.txt" "$@"
    ;;
  granular)
    exec .venv/bin/python eval_granular.py \
      --dataset "$STABLEWM_HOME/granular_eval.h5" \
      --goal-offset 5 --horizon 5 --planner openloop \
      --cem-num-samples 300 --cem-topk 30 --cem-n-steps 30 \
      --egl-gpu "${EGL_GPU:-1}" \
      "${COMMON[@]}" "${M[@]}" \
      --output "${OUT}.txt" "$@"
    ;;
  *) echo "unknown env: $ENV" >&2; exit 1;;
esac
