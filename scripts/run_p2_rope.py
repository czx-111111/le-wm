#!/usr/bin/env python3
"""Stage-2 C2 gated sweep on rope only (train → cost* eval → keep if better).

Resumable via outputs/r3/p2_loss/state.json. One change at a time on top of
the current best (starts from r3 S2 baseline).

Usage:
  source setup_env.sh
  export STABLEWM_HOME=$PWD/data
  .venv/bin/python scripts/run_p2_rope.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r3" / "p2_loss"
STATE_PATH = OUT / "state.json"
RUNS = ROOT / "data" / "runs" / "r3"
PY = ROOT / ".venv" / "bin" / "python"

BASELINE_METRICS = (
    ROOT
    / "outputs/r3/p1_cost/rope_s2_p1_costmode-decoupled_costlambda-2.0_costnorm-calib_"
    "maskweight-hard_softeps-0.1_softtemp-0.02_taumode-fixed_mtaskthreshold-0.08_"
    "relevancesource-pixel_cvarq-0.1_cvarbeta-0.5_metrics.json"
)
# Same ckpt family used for stage-1 cost* on rope
BASELINE_CKPT = (
    ROOT / "data/runs/r3/rope_s2_r3_s3072_20261001_232404/rope_s2_last_object.ckpt"
)

COST_STAR = [
    "--cost-mode", "decoupled",
    "--cost-norm", "calib",
    "--cost-lambda", "2",
    "--mask-weight", "hard",
    "--tau-mode", "fixed",
    "--m-task-threshold", "0.08",
    "--relevance-source", "pixel",
    "--cvar-q", "0.1",
    "--cvar-beta", "0.5",
    "--lambda-anneal", "off",
]
S2_TOKENS = [
    "--token-mode", "spatial16",
    "--pool-mode", "query",
    "--include-global-token",
    "--pos-mode", "nibble",
    "--inject-every-layer",
    "--n-lo-mode", "attn",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {
        "best": {
            "name": "baseline_r3_s2",
            "ckpt": str(BASELINE_CKPT),
            "metrics": str(BASELINE_METRICS),
            "overrides": [],
            "mean_chamfer": None,
        },
        "done": {},
        "history": [],
    }


def save_state(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def boot_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load_details(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {(e["ep_id"], e["start_step"]): e for e in d["details"]}


def mean_chamfer(path: Path) -> float:
    d = json.loads(Path(path).read_text())
    return float(d["mean_chamfer_distance"])


def paired_better(cand_metrics: Path, base_metrics: Path) -> tuple[bool, dict]:
    """True iff cand significantly better (lower Chamfer): paired CI entirely < 0."""
    base = load_details(base_metrics)
    cand = load_details(cand_metrics)
    keys = sorted(set(base) & set(cand))
    diff = np.array([cand[k]["chamfer"] - base[k]["chamfer"] for k in keys])
    lo, hi = boot_ci(diff)
    info = {
        "n": len(keys),
        "mean_diff": float(diff.mean()),
        "ci": [lo, hi],
        "cand_mean": mean_chamfer(cand_metrics),
        "base_mean": mean_chamfer(base_metrics),
        "significant_better": bool(hi < 0),
        "significant_worse": bool(lo > 0),
    }
    return info["significant_better"], info


def find_run_dir(output_model_name: str) -> Path | None:
    """Newest r3 run dir whose name starts with output_model_name_r3_."""
    prefix = f"{output_model_name}_r3_"
    cands = sorted(
        [p for p in RUNS.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def wait_last_ckpt(output_model_name: str, timeout_s: float = 5 * 3600) -> Path:
    """Wait until *_last_object.ckpt appears (training finished)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        run = find_run_dir(output_model_name)
        if run is not None:
            last = run / f"{output_model_name}_last_object.ckpt"
            # Also require train log to show completion if present
            if last.is_file() and last.stat().st_size > 1_000_000:
                # Avoid racing mid-write: size stable 5s
                sz = last.stat().st_size
                time.sleep(5)
                if last.stat().st_size == sz:
                    return last
        time.sleep(30)
    raise TimeoutError(f"timed out waiting for last ckpt of {output_model_name}")


def _training_in_flight(name: str) -> bool:
    """True if a live train.py process looks like this output_model_name."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return False
    return any(
        "train.py" in line and f"output_model_name={name}" in line and "run_p2_rope" not in line
        for line in out.splitlines()
    )


def train(name: str, overrides: list[str]) -> Path:
    """Train if last ckpt missing; return last_object path."""
    existing = find_run_dir(name)
    last = (existing / f"{name}_last_object.ckpt") if existing else None
    if last is not None and last.is_file():
        log(f"train skip (exists): {last}")
        return last

    # Resume / wait for an already-running job with this name
    if existing is not None and (last is None or not last.is_file()) and _training_in_flight(name):
        log(f"train wait (in-flight): {name}")
        return wait_last_ckpt(name)

    log_path = OUT / f"train_{name}.log"
    cmd = [
        str(PY),
        str(ROOT / "train.py"),
        "--config-name=r3/rope_s2",
        f"output_model_name={name}",
        *overrides,
    ]
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(ROOT / "data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log(f"train start: {' '.join(cmd)}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"train failed rc={rc}: see {log_path}")
    return wait_last_ckpt(name)


def eval_cost_star(name: str, ckpt: Path) -> Path:
    metrics = OUT / f"{name}_dev_e42_metrics.json"
    if metrics.is_file():
        log(f"eval skip (exists): {metrics}")
        return metrics

    out_txt = OUT / f"{name}_dev_e42.txt"
    log_path = OUT / f"eval_{name}.log"
    cmd = [
        str(PY),
        str(ROOT / "eval_rope_1.py"),
        "--ckpt", str(ckpt),
        "--dataset", "data/rope_eval.h5",
        "--split", "dev",
        "--num-eval", "0",
        "--seed", "42",
        "--goal-offset", "5",
        "--horizon", "5",
        "--history-size", "3",
        "--action-dim", "4",
        "--cem-num-samples", "300",
        "--cem-topk", "30",
        "--cem-n-steps", "30",
        "--plan-protocol", "lewm",
        "--plan-gpu", "0",
        "--egl-gpu", "1",
        *S2_TOKENS,
        *COST_STAR,
        "--output", str(out_txt),
    ]
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(ROOT / "data")
    # setup_env.sh must already be sourced by the parent shell
    log(f"eval start: {name}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    rc = proc.wait()
    if rc != 0 or not metrics.is_file():
        # eval_rope_1 writes metrics next to --output stem
        alt = out_txt.with_name(out_txt.stem + "_metrics.json")
        if alt.is_file() and not metrics.is_file():
            alt.rename(metrics)
        if rc != 0 or not metrics.is_file():
            raise RuntimeError(f"eval failed rc={rc}: see {log_path}")
    return metrics


def maybe_accept(state: dict, name: str, ckpt: Path, metrics: Path, overrides: list[str]) -> bool:
    better, info = paired_better(metrics, Path(state["best"]["metrics"]))
    state["history"].append({"name": name, "info": info, "overrides": overrides})
    log(
        f"compare {name}: mean={info['cand_mean']:.4f} vs best={info['base_mean']:.4f} "
        f"Δ={info['mean_diff']:+.4f} CI=[{info['ci'][0]:+.4f},{info['ci'][1]:+.4f}] "
        f"{'ACCEPT' if better else 'reject'}"
    )
    if better:
        state["best"] = {
            "name": name,
            "ckpt": str(ckpt),
            "metrics": str(metrics),
            "overrides": overrides,
            "mean_chamfer": info["cand_mean"],
        }
        save_state(state)
        return True
    save_state(state)
    return False


def run_variant(state: dict, name: str, new_overrides: list[str], base_overrides: list[str] | None = None) -> bool:
    """Train+eval one variant. Returns whether it became the new best."""
    if name in state["done"]:
        log(f"variant already done: {name}")
        # Still allow accept if it was better but state.best wasn't updated (resume)
        info = state["done"][name]
        return state["best"]["name"] == name

    overrides = list(base_overrides if base_overrides is not None else state["best"]["overrides"])
    overrides = overrides + new_overrides
    # de-dup keep last
    merged: dict[str, str] = {}
    flags = []
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            merged[k] = v
        else:
            flags.append(o)
    overrides = [f"{k}={v}" for k, v in merged.items()] + flags

    ckpt = train(name, overrides)
    metrics = eval_cost_star(name, ckpt)
    accepted = maybe_accept(state, name, ckpt, metrics, overrides)
    state["done"][name] = {
        "ckpt": str(ckpt),
        "metrics": str(metrics),
        "overrides": overrides,
        "accepted": accepted,
    }
    save_state(state)
    return accepted


def main() -> int:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASELINE_METRICS.is_file():
        log(f"MISSING baseline metrics: {BASELINE_METRICS}")
        return 1
    if not BASELINE_CKPT.is_file():
        log(f"MISSING baseline ckpt: {BASELINE_CKPT}")
        return 1

    state = load_state()
    if state["best"]["mean_chamfer"] is None:
        state["best"]["mean_chamfer"] = mean_chamfer(BASELINE_METRICS)
    save_state(state)
    log(f"best start: {state['best']['name']} chamfer={state['best']['mean_chamfer']:.4f}")

    # --- If the manually launched tb_group train is still running, wait for it ---
    if "rope_s2_tb_group" not in state["done"]:
        run = find_run_dir("rope_s2_tb_group")
        last = (run / "rope_s2_tb_group_last_object.ckpt") if run else None
        if run and (last is None or not last.is_file()):
            log("waiting for in-flight rope_s2_tb_group training…")
            wait_last_ckpt("rope_s2_tb_group")

    # 1–2: mode sweep from baseline (empty overrides)
    accepted_mode = None
    for name, ov in [
        ("rope_s2_tb_group", ["loss.token_balance.mode=group", "loss.token_balance.alpha=0.5"]),
        ("rope_s2_tb_varnorm", ["loss.token_balance.mode=group_varnorm", "loss.token_balance.alpha=0.5"]),
    ]:
        # compare modes against baseline, not against each other greedily for accept —
        # but we still use greedy best: first that beats current best wins, second can beat first.
        ok = run_variant(state, name, ov, base_overrides=[])
        if ok:
            accepted_mode = name

    # 3: alpha sweep only if a mode beat baseline
    mode_beat_baseline = state["best"]["name"] in ("rope_s2_tb_group", "rope_s2_tb_varnorm") or any(
        state["done"].get(n, {}).get("accepted") for n in ("rope_s2_tb_group", "rope_s2_tb_varnorm")
    )
    # More precise: best overrides contain token_balance.mode group*
    best_ovs = state["best"]["overrides"]
    has_mode = any("token_balance.mode=group" in o for o in best_ovs)
    if has_mode:
        mode_val = next(o.split("=", 1)[1] for o in best_ovs if o.startswith("loss.token_balance.mode="))
        for alpha in (0.25, 0.75):
            tag = "varnorm" if mode_val == "group_varnorm" else "group"
            name = f"rope_s2_tb_{tag}_a{str(alpha).replace('.', '')}"
            run_variant(
                state,
                name,
                [
                    f"loss.token_balance.mode={mode_val}",
                    f"loss.token_balance.alpha={alpha}",
                ],
                base_overrides=[],  # only the balance knobs; SIGReg/rollout not yet
            )
    else:
        log("skip alpha sweep (no token_balance mode beat baseline)")

    # 4: grouped SIGReg on current best
    for slot_w in (0.09, 0.03):
        tag = str(slot_w).replace(".", "")
        name = f"rope_s2_{state['best']['name'].replace('rope_s2_', '')}_sigreg{tag}"
        # shorten names
        name = f"rope_s2_sigreg_sw{tag}"
        # stack on best; if best already has other sigreg, replace via merge
        run_variant(
            state,
            name,
            [
                "loss.sigreg.grouped=true",
                "loss.sigreg.cls_weight=0.09",
                f"loss.sigreg.slot_weight={slot_w}",
            ],
        )

    # 5: rollout_k on current best
    for k in (2, 3):
        name = f"rope_s2_rollout_k{k}"
        run_variant(state, name, [f"loss.rollout_k={k}", "loss.rollout_weight=1.0"])

    log(f"STAGE2 DONE. best={state['best']['name']} chamfer={state['best']['mean_chamfer']}")
    log(f"overrides={state['best']['overrides']}")
    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL: {e!r}")
        raise
