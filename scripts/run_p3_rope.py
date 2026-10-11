#!/usr/bin/env python3
"""Stage-3 C3 gated sweep on rope only (baseline r3 S2 + cost*).

Order:
  1. gauss σ_min ∈ {1, 2}
  2. continuous softargmax_temp ∈ {0.5, 1.0}  (on current best)
  3. pos_prop weight ∈ {0.01, 0.1}            (on current best)

Usage:
  source setup_env.sh
  export STABLEWM_HOME=$PWD/data
  .venv/bin/python -u scripts/run_p3_rope.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r3" / "p3_repr"
STATE_PATH = OUT / "state.json"
RUNS = ROOT / "data" / "runs" / "r3"
PY = ROOT / ".venv" / "bin" / "python"

BASELINE_METRICS = (
    ROOT
    / "outputs/r3/p1_cost/rope_s2_p1_costmode-decoupled_costlambda-2.0_costnorm-calib_"
    "maskweight-hard_softeps-0.1_softtemp-0.02_taumode-fixed_mtaskthreshold-0.08_"
    "relevancesource-pixel_cvarq-0.1_cvarbeta-0.5_metrics.json"
)
BASELINE_CKPT = (
    ROOT / "data/runs/r3/rope_s2_r3_s3072_20261001_232404/rope_s2_last_object.ckpt"
)

COST_STAR = [
    "--cost-mode", "decoupled", "--cost-norm", "calib", "--cost-lambda", "2",
    "--mask-weight", "hard", "--tau-mode", "fixed", "--m-task-threshold", "0.08",
    "--relevance-source", "pixel", "--cvar-q", "0.1", "--cvar-beta", "0.5",
    "--lambda-anneal", "off",
]
S2_TOKENS = [
    "--token-mode", "spatial16", "--pool-mode", "query", "--include-global-token",
    "--pos-mode", "nibble", "--inject-every-layer", "--n-lo-mode", "attn",
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
    return float(json.loads(Path(path).read_text())["mean_chamfer_distance"])


def paired_better(cand_metrics: Path, base_metrics: Path):
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
    prefix = f"{output_model_name}_r3_"
    cands = sorted(
        [p for p in RUNS.glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def wait_last_ckpt(output_model_name: str, timeout_s: float = 5 * 3600) -> Path:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        run = find_run_dir(output_model_name)
        if run is not None:
            last = run / f"{output_model_name}_last_object.ckpt"
            if last.is_file() and last.stat().st_size > 1_000_000:
                sz = last.stat().st_size
                time.sleep(5)
                if last.stat().st_size == sz:
                    return last
        time.sleep(30)
    raise TimeoutError(f"timed out waiting for last ckpt of {output_model_name}")


def _training_in_flight(name: str) -> bool:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return False
    return any(
        "train.py" in line and f"output_model_name={name}" in line and "run_p3_rope" not in line
        for line in out.splitlines()
    )


def train(name: str, overrides: list[str]) -> Path:
    existing = find_run_dir(name)
    last = (existing / f"{name}_last_object.ckpt") if existing else None
    if last is not None and last.is_file():
        log(f"train skip (exists): {last}")
        return last
    if existing is not None and (last is None or not last.is_file()) and _training_in_flight(name):
        log(f"train wait (in-flight): {name}")
        return wait_last_ckpt(name)

    log_path = OUT / f"train_{name}.log"
    cmd = [
        str(PY), str(ROOT / "train.py"), "--config-name=r3/rope_s2",
        f"output_model_name={name}", *overrides,
    ]
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(ROOT / "data")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log(f"train start: {' '.join(cmd)}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
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
        str(PY), str(ROOT / "eval_rope_1.py"),
        "--ckpt", str(ckpt), "--dataset", "data/rope_eval.h5",
        "--split", "dev", "--num-eval", "0", "--seed", "42",
        "--goal-offset", "5", "--horizon", "5", "--history-size", "3",
        "--action-dim", "4", "--cem-num-samples", "300", "--cem-topk", "30",
        "--cem-n-steps", "30", "--plan-protocol", "lewm",
        "--plan-gpu", "0", "--egl-gpu", "1",
        *S2_TOKENS, *COST_STAR, "--output", str(out_txt),
    ]
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(ROOT / "data")
    log(f"eval start: {name}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    rc = proc.wait()
    alt = out_txt.with_name(out_txt.stem + "_metrics.json")
    if alt.is_file() and not metrics.is_file():
        alt.rename(metrics)
    if rc != 0 or not metrics.is_file():
        raise RuntimeError(f"eval failed rc={rc}: see {log_path}")
    return metrics


def maybe_accept(state, name, ckpt, metrics, overrides) -> bool:
    better, info = paired_better(metrics, Path(state["best"]["metrics"]))
    state["history"].append({"name": name, "info": info, "overrides": overrides})
    log(
        f"compare {name}: mean={info['cand_mean']:.4f} vs best={info['base_mean']:.4f} "
        f"Δ={info['mean_diff']:+.4f} CI=[{info['ci'][0]:+.4f},{info['ci'][1]:+.4f}] "
        f"{'ACCEPT' if better else 'reject'}"
    )
    if better:
        state["best"] = {
            "name": name, "ckpt": str(ckpt), "metrics": str(metrics),
            "overrides": overrides, "mean_chamfer": info["cand_mean"],
        }
    save_state(state)
    return better


def run_variant(state, name, new_overrides, base_overrides=None) -> bool:
    if name in state["done"]:
        log(f"variant already done: {name}")
        return state["best"]["name"] == name
    overrides = list(base_overrides if base_overrides is not None else state["best"]["overrides"])
    overrides = overrides + new_overrides
    merged: dict[str, str] = {}
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            merged[k] = v
    overrides = [f"{k}={v}" for k, v in merged.items()]
    ckpt = train(name, overrides)
    metrics = eval_cost_star(name, ckpt)
    accepted = maybe_accept(state, name, ckpt, metrics, overrides)
    state["done"][name] = {
        "ckpt": str(ckpt), "metrics": str(metrics),
        "overrides": overrides, "accepted": accepted,
    }
    save_state(state)
    return accepted


def main() -> int:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASELINE_METRICS.is_file() or not BASELINE_CKPT.is_file():
        log("MISSING baseline metrics/ckpt")
        return 1

    state = load_state()
    if state["best"]["mean_chamfer"] is None:
        state["best"]["mean_chamfer"] = mean_chamfer(BASELINE_METRICS)
    save_state(state)
    log(f"best start: {state['best']['name']} chamfer={state['best']['mean_chamfer']:.4f}")

    # 1. gauss σ_min sweep from baseline
    for smin in (1.0, 2.0):
        tag = str(smin).replace(".", "")
        run_variant(
            state,
            f"rope_s2_loc_gauss_smin{tag}",
            [
                "wm.locality=gauss",
                "wm.locality_sigma_init=2.0",
                f"wm.locality_sigma_min={smin}",
            ],
            base_overrides=[],
        )

    # 2. continuous fine on current best
    for temp in (0.5, 1.0):
        tag = str(temp).replace(".", "")
        run_variant(
            state,
            f"rope_s2_fine_cont_t{tag}",
            [
                "wm.fine_mode=continuous",
                f"wm.softargmax_temp={temp}",
                "wm.fine_fourier_freqs=8",
                "wm.fine_conf_gate=true",
            ],
        )

    # 3. pos_prop on current best
    for w in (0.01, 0.1):
        tag = str(w).replace(".", "")
        run_variant(
            state,
            f"rope_s2_posprop_w{tag}",
            [
                "wm.pos_prop.enabled=true",
                f"wm.pos_prop.weight={w}",
            ],
        )

    log(f"STAGE3 DONE. best={state['best']['name']} chamfer={state['best']['mean_chamfer']}")
    log(f"overrides={state['best']['overrides']}")
    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL: {e!r}")
        raise
