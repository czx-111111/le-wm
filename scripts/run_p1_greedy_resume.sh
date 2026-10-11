#!/usr/bin/env bash
# Resume stage-1 greedy cost sweep from the interrupted step.
# Does NOT re-seed cost* / baselines. Appends to greedy.log.
#
#   START_STEP=3 bash scripts/run_p1_greedy_resume.sh   # default
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/setup_env.sh" >/dev/null 2>&1 || true
export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT/data}"
mkdir -p "$ROOT/outputs/r3/p1_cost"
PY="$ROOT/.venv/bin/python"
PLAN_GPU="${PLAN_GPU:-0}"
EGL_GPU="${EGL_GPU:-1}"
LOG="$ROOT/outputs/r3/p1_cost/greedy.log"
START_STEP="${START_STEP:-3}"

exec >>"$LOG" 2>&1
echo "======== $(date -Is) RESUME greedy p1 from step ${START_STEP} ========"
"$PY" - <<'PY'
import json
from pathlib import Path
st = json.loads(Path("outputs/r3/p1_cost/sweep_state.json").read_text())
print("current cost*:", json.dumps(st["cost*"], ensure_ascii=False))
for h in st.get("history", []):
    print(" history:", h)
PY

run_step() {
  local step="$1"
  echo "======== $(date -Is) step $step ========"
  PYTHONPATH="$ROOT" "$PY" scripts/sweep_cost_p1.py \
    --step "$step" \
    --methods s2 \
    --envs wall,point_maze,rope,granular \
    --plan-gpu "$PLAN_GPU" \
    --egl-gpu "$EGL_GPU" \
    --commit-best
}

case "$START_STEP" in
  3)  STEPS=(3 31 4 5 6 7) ;;
  31) STEPS=(31 4 5 6 7) ;;
  4)  STEPS=(4 5 6 7) ;;
  5)  STEPS=(5 6 7) ;;
  6)  STEPS=(6 7) ;;
  7)  STEPS=(7) ;;
  *)
    echo "unknown START_STEP=$START_STEP" >&2
    exit 1
    ;;
esac

for s in "${STEPS[@]}"; do
  run_step "$s"
done

echo "======== $(date -Is) DONE (resume) ========"
"$PY" - <<'PY'
import json
from pathlib import Path
st = json.loads(Path("outputs/r3/p1_cost/sweep_state.json").read_text())
print("final cost*:", json.dumps(st["cost*"], indent=2, ensure_ascii=False))
for h in st.get("history", []):
    print(h)
PY
