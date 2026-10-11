#!/usr/bin/env bash
# Greedy stage-1 cost sweep on r3 S2 last (option A).
# Logs: outputs/r3/p1_cost/greedy.log
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

exec >>"$LOG" 2>&1
echo "======== $(date -Is) start greedy p1 cost sweep ========"

# Seed step-0 baseline from existing r3 S2/CLS evals into sweep_state.json
PYTHONPATH="$ROOT" "$PY" - <<'PY'
import json, shutil, sys
from pathlib import Path
ROOT = Path("/data/czx/le-wm/first")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/r3/p1_cost"
OUT.mkdir(parents=True, exist_ok=True)
state_path = OUT / "sweep_state.json"
baseline = {
    "cost_mode": "m_task_terminal",
    "cost_lambda": 1.0,
    "cost_norm": "none",
    "mask_weight": "hard",
    "soft_eps": 0.1,
    "soft_temp": 0.02,
    "tau_mode": "fixed",
    "m_task_threshold": 0.08,
    "relevance_source": "pixel",
    "cvar_q": 0.0,
    "cvar_beta": 0.0,
    "lambda_anneal": "off",
}
mapping = {
    "wall": (
        "outputs/r3/wall_s2_nt5i10_dev_e42.txt",
        "outputs/r3/wall_s2_nt5i10_dev_e42_details.json",
    ),
    "point_maze": (
        "outputs/r3/point_maze_s2_point_maze_s2_r3_s3072_20260930_213550_dev_e42.txt",
        "outputs/r3/point_maze_s2_point_maze_s2_r3_s3072_20260930_213550_dev_e42_details.json",
    ),
    "rope": (
        "outputs/r3/rope_s2_rope_s2_r3_s3072_20260930_140415_dev_e42.txt",
        "outputs/r3/rope_s2_rope_s2_r3_s3072_20260930_140415_dev_e42_metrics.json",
    ),
    "granular": (
        "outputs/r3/granular_s2_granular_s2_r3_s3072_20260930_161244_dev_e42.txt",
        "outputs/r3/granular_s2_granular_s2_r3_s3072_20260930_161244_dev_e42_metrics.json",
    ),
}
from scripts.sweep_cost_p1 import tag_for, parse_metric

results = []
for env, (txt, side) in mapping.items():
    tag = tag_for(env, "s2", baseline)
    dst_txt = OUT / f"{tag}.txt"
    if not dst_txt.is_file():
        shutil.copy2(ROOT / txt, dst_txt)
    if env in ("wall", "point_maze"):
        dst_d = OUT / f"{tag}_details.json"
        if not dst_d.is_file():
            shutil.copy2(ROOT / side, dst_d)
    else:
        dst_m = OUT / f"{tag}_metrics.json"
        if not dst_m.is_file():
            shutil.copy2(ROOT / side, dst_m)
    m = parse_metric(env, OUT / tag)
    m.update({"env": env, "method": "s2", "extras": baseline, "seeded": True})
    results.append(m)
    print(env, m.get("success_at_2"), m.get("mean_chamfer"), m.get("score"), m.get("ok"))

cls_map = {
    "wall": ("outputs/r3/wall_cls_nt5i10_dev_e42.txt", "outputs/r3/wall_cls_nt5i10_dev_e42_details.json"),
    "point_maze": (
        "outputs/r3/point_maze_cls_point_maze_cls_r3_s3072_20260930_213550_dev_e42.txt",
        "outputs/r3/point_maze_cls_point_maze_cls_r3_s3072_20260930_213550_dev_e42_details.json",
    ),
    "rope": (
        "outputs/r3/rope_cls_rope_cls_r3_s3072_20260930_140415_dev_e42.txt",
        "outputs/r3/rope_cls_rope_cls_r3_s3072_20260930_140415_dev_e42_metrics.json",
    ),
    "granular": (
        "outputs/r3/granular_cls_granular_cls_r3_s3072_20260930_161244_dev_e42.txt",
        "outputs/r3/granular_cls_granular_cls_r3_s3072_20260930_161244_dev_e42_metrics.json",
    ),
}
cls_base = dict(baseline, cost_mode="dense_terminal")
for env, (txt, side) in cls_map.items():
    tag = tag_for(env, "cls", cls_base)
    dst_txt = OUT / f"{tag}.txt"
    if not dst_txt.is_file():
        shutil.copy2(ROOT / txt, dst_txt)
    if env in ("wall", "point_maze"):
        dst_d = OUT / f"{tag}_details.json"
        if not dst_d.is_file():
            shutil.copy2(ROOT / side, dst_d)
    else:
        dst_m = OUT / f"{tag}_metrics.json"
        if not dst_m.is_file():
            shutil.copy2(ROOT / side, dst_m)
    m = parse_metric(env, OUT / tag)
    m.update({"env": env, "method": "cls", "extras": cls_base, "seeded": True})
    results.append(m)
    print("cls", env, m.get("success_at_2"), m.get("mean_chamfer"), m.get("ok"))

state = {
    "cost*": baseline,
    "results": results,
    "history": [{"step": 0, "note": "seeded from existing r3 S2 m_task / CLS dense"}],
}
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
print("seeded", state_path)
PY

run_step() {
  local step="$1"
  echo "======== $(date -Is) step $step ========"
  "$PY" scripts/sweep_cost_p1.py \
    --step "$step" \
    --methods s2 \
    --envs wall,point_maze,rope,granular \
    --plan-gpu "$PLAN_GPU" \
    --egl-gpu "$EGL_GPU" \
    --commit-best
}

# Greedy order (plan §阶段1)
run_step 1    # cls_terminal
run_step 2    # decoupled+calib λ
run_step 3    # soft ε
run_step 31   # soft T
run_step 4    # tau-mode
run_step 5    # relevance-source
run_step 6    # CVaR
run_step 7    # λ anneal

echo "======== $(date -Is) DONE ========"
"$PY" - <<'PY'
import json
from pathlib import Path
st = json.loads(Path("outputs/r3/p1_cost/sweep_state.json").read_text())
print("final cost*:", json.dumps(st["cost*"], indent=2, ensure_ascii=False))
print("history steps:", len(st.get("history", [])))
PY
