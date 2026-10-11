#!/usr/bin/env bash
# 正式 r3 训练：scripts/train_r3.sh <env> <method> [额外 hydra 覆盖...]
# 例: scripts/train_r3.sh rope s2
#     scripts/train_r3.sh wall cls seed=1234 trainer.max_epochs=50
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# 加载本地 wandb 等密钥（.env 已 gitignore）
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env"
  set +a
fi

ENV="${1:?usage: $0 <wall|point_maze|rope|granular> <cls|c1|s1|s2> [hydra overrides...]}"
METHOD="${2:?usage: $0 <env> <cls|c1|s1|s2> [...]}"
shift 2

case "$ENV" in wall|point_maze|rope|granular) ;; *)
  echo "unknown env: $ENV" >&2; exit 1;;
esac
case "$METHOD" in cls|c1|s1|s2) ;; *)
  echo "unknown method: $METHOD" >&2; exit 1;;
esac

export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# wandb：环境变量优先；与 recipe 里 entity/project 对齐
if [[ -n "${WANDB_ENTITY:-}" || -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA=()
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA+=(wandb.config.entity="$WANDB_ENTITY")
  [[ -n "${WANDB_PROJECT:-}" ]] && EXTRA+=(wandb.config.project="$WANDB_PROJECT")
else
  EXTRA=()
fi

CFG="r3/${ENV}_${METHOD}"
echo "[train_r3] STABLEWM_HOME=$STABLEWM_HOME"
echo "[train_r3] config=$CFG $*"
exec .venv/bin/python train.py --config-name="$CFG" "${EXTRA[@]}" "$@"
