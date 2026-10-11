#!/usr/bin/env python3
"""Stage-1 greedy cost sweep (dev only). See plan p1-sweep.

Default targets: r3 S2 last ckpts (plan primary). Pass --methods cls,s1 to override.
Greedy order:
  1. cls_terminal
  2. decoupled+calib, sweep λ
  3. soft: ε then T
  4. tau-mode
  5. relevance-source
  6. CVaR q×β
  7. λ anneal

Usage:
  .venv/bin/python scripts/sweep_cost_p1.py --step 0          # baseline m_task / dense
  .venv/bin/python scripts/sweep_cost_p1.py --step 1          # cls_terminal
  .venv/bin/python scripts/sweep_cost_p1.py --step 2 --lambda-values 0.25,0.5,1,2
  .venv/bin/python scripts/sweep_cost_p1.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "r3" / "p1_cost"
STATE_PATH = OUT_DIR / "sweep_state.json"

# r3 single-card last ckpts (seed 3072)
CKPTS = {
    "wall": {
        "cls": "data/runs/r3/wall_cls_r3_s3072_20261001_093335/wall_cls_last_object.ckpt",
        "s2": "data/runs/r3/wall_s2_r3_s3072_20261001_093335/wall_s2_last_object.ckpt",
        # r1 last (=epoch_100); no r3 S1 yet
        "s1": "data/lewm_wall_s1_20260917_104917/lewm_wall_s1_epoch_100_object.ckpt",
    },
    "point_maze": {
        "cls": "data/runs/r3/point_maze_cls_r3_s3072_20260930_213550/point_maze_cls_last_object.ckpt",
        "s2": "data/runs/r3/point_maze_s2_r3_s3072_20260930_213550/point_maze_s2_last_object.ckpt",
        "s1": "data/lewm_point_maze_s1_20260920_193429/lewm_point_maze_s1_best_ep8_object.ckpt",
    },
    "rope": {
        "cls": "data/runs/r3/rope_cls_r3_s3072_20260930_140415/rope_cls_last_object.ckpt",
        "s2": "data/runs/r3/rope_s2_r3_s3072_20260930_140415/rope_s2_last_object.ckpt",
        "s1": "data/lewm_rope_s1_20260916_201520/lewm_rope_s1_epoch_100_object.ckpt",
    },
    "granular": {
        "cls": "data/runs/r3/granular_cls_r3_s3072_20260930_161244/granular_cls_last_object.ckpt",
        "s2": "data/runs/r3/granular_s2_r3_s3072_20260930_161244/granular_s2_last_object.ckpt",
        "s1": "data/lewm_granular_s1_20260917_083958/lewm_granular_s1_epoch_100_object.ckpt",
    },
}

ENVS = ("wall", "point_maze", "rope", "granular")


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {
        "cost*": {
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
        },
        "results": [],
        "history": [],
    }


def save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def tag_for(env: str, method: str, extras: dict) -> str:
    parts = [env, method, "p1"]
    for k in (
        "cost_mode",
        "cost_lambda",
        "cost_norm",
        "mask_weight",
        "soft_eps",
        "soft_temp",
        "tau_mode",
        "m_task_threshold",
        "relevance_source",
        "cvar_q",
        "cvar_beta",
        "lambda_anneal",
    ):
        if k not in extras:
            continue
        v = extras[k]
        if k in ("cvar_q", "cvar_beta") and float(v) == 0.0:
            continue
        if k == "lambda_anneal" and str(v) in ("off", "none"):
            continue
        if k == "cost_norm" and str(v) == "none" and extras.get("cost_mode") != "decoupled":
            continue
        parts.append(f"{k.replace('_', '')}-{v}")
    return "_".join(str(p) for p in parts)


def build_cmd(env: str, method: str, ckpt: str, extras: dict, plan_gpu: str, egl_gpu: str) -> list[str]:
    tag = tag_for(env, method, extras)
    out = OUT_DIR / tag
    py = str(ROOT / ".venv" / "bin" / "python")
    common = [
        "--ckpt",
        ckpt,
        "--split",
        "dev",
        "--num-eval",
        "0",
        "--seed",
        "42",
        "--plan-protocol",
        "lewm",
        "--plan-gpu",
        plan_gpu,
        "--cost-mode",
        str(extras["cost_mode"]),
        "--cost-lambda",
        str(extras.get("cost_lambda", 1.0)),
        "--cost-norm",
        str(extras.get("cost_norm", "none")),
        "--mask-weight",
        str(extras.get("mask_weight", "hard")),
        "--soft-eps",
        str(extras.get("soft_eps", 0.1)),
        "--soft-temp",
        str(extras.get("soft_temp", 0.02)),
        "--tau-mode",
        str(extras.get("tau_mode", "fixed")),
        "--m-task-threshold",
        str(extras.get("m_task_threshold", 0.08)),
        "--relevance-source",
        str(extras.get("relevance_source", "pixel")),
        "--cvar-q",
        str(extras.get("cvar_q", 0.0)),
        "--cvar-beta",
        str(extras.get("cvar_beta", 0.0)),
        "--lambda-anneal",
        str(extras.get("lambda_anneal", "off")),
        "--output",
        str(out) + ".txt",
    ]
    if method == "cls":
        method_args = ["--token-mode", "cls", "--cost-mode", "dense_terminal"]
        # CLS ignores most knobs; still allow override if extras force a mode
        if extras["cost_mode"] != "dense_terminal":
            method_args = ["--token-mode", "cls", "--cost-mode", extras["cost_mode"]]
    elif method == "s1":
        method_args = [
            "--token-mode",
            "spatial16",
            "--pool-mode",
            "query",
            "--include-global-token",
            "--pos-mode",
            "learned",
        ]
    else:  # s2
        method_args = [
            "--token-mode",
            "spatial16",
            "--pool-mode",
            "query",
            "--include-global-token",
            "--pos-mode",
            "nibble",
            "--inject-every-layer",
            "--n-lo-mode",
            "attn",
        ]

    if env == "wall":
        script = "eval_wall.py"
        env_args = [
            "--dataset",
            "data/wall_eval.h5",
            "--goal-offset",
            "25",
            "--horizon",
            "5",
            "--planner",
            "mpc",
            "--n-taken-actions",
            "5",
            "--mpc-max-iters",
            "10",
            "--cem-num-samples",
            "300",
            "--cem-topk",
            "30",
            "--cem-n-steps",
            "10",
            "--details-json",
            str(out) + "_details.json",
        ]
    elif env == "point_maze":
        script = "eval_point_maze.py"
        env_args = [
            "--dataset",
            "data/point_maze_eval.h5",
            "--goal-offset",
            "25",
            "--horizon",
            "5",
            "--planner",
            "mpc",
            "--n-taken-actions",
            "5",
            "--mpc-max-iters",
            "10",
            "--cem-num-samples",
            "300",
            "--cem-topk",
            "30",
            "--cem-n-steps",
            "10",
            "--details-json",
            str(out) + "_details.json",
        ]
    elif env == "rope":
        script = "eval_rope_1.py"
        env_args = [
            "--dataset",
            "data/rope_eval.h5",
            "--goal-offset",
            "5",
            "--horizon",
            "5",
            "--history-size",
            "3",
            "--action-dim",
            "4",
            "--cem-num-samples",
            "300",
            "--cem-topk",
            "30",
            "--cem-n-steps",
            "30",
            "--egl-gpu",
            egl_gpu,
        ]
    else:
        script = "eval_granular.py"
        env_args = [
            "--dataset",
            "data/granular_eval.h5",
            "--goal-offset",
            "5",
            "--horizon",
            "5",
            "--planner",
            "openloop",
            "--cem-num-samples",
            "300",
            "--cem-topk",
            "30",
            "--cem-n-steps",
            "30",
            "--egl-gpu",
            egl_gpu,
        ]

    # method_args may set cost-mode; put extras after so CLI wins for non-cls
    cmd = [py, script] + env_args + common + method_args
    # re-apply cost-mode from extras at end (wins over method_args for s1/s2)
    if method != "cls" or extras["cost_mode"] != "dense_terminal":
        cmd += ["--cost-mode", str(extras["cost_mode"])]
    return cmd


def _out_paths(out_base: Path) -> tuple[Path, Path, Path, Path]:
    """Avoid Path.with_suffix — tags contain dots like mtaskthreshold-0.08."""
    base = str(out_base)
    return (
        Path(base + ".txt"),
        Path(base + "_details.json"),
        Path(base + "_metrics.json"),
        Path(base + ".log"),
    )


def parse_metric(env: str, out_base: Path) -> dict:
    """Return comparable score: higher better for wall/point, lower better for rope/granular."""
    txt, details, metrics, _log = _out_paths(out_base)
    result = {"path": str(txt), "ok": False}
    if env in ("wall", "point_maze"):
        if details.is_file():
            data = json.loads(details.read_text())
            eps = data.get("details") or data.get("episodes") or []
            # r3 / mpc_budget_curve: success@k = success and mpc_iters <= k
            succ2, succ10 = [], []
            for ep in eps:
                if not isinstance(ep, dict):
                    continue
                ok = bool(ep.get("success"))
                iters = int(ep.get("mpc_iters", 99))
                succ2.append(float(ok and iters <= 2))
                succ10.append(float(ok and iters <= 10))
            if succ2:
                result["success_at_2"] = sum(succ2) / len(succ2)
                result["success_at_10"] = sum(succ10) / len(succ10)
                result["n"] = len(succ2)
                result["score"] = result["success_at_2"]
                result["higher_better"] = True
                result["ok"] = True
                return result
        if txt.is_file():
            import re

            for line in txt.read_text().splitlines():
                if "success" in line.lower() and "%" in line:
                    m = re.search(r"([0-9.]+)\s*%", line)
                    if m:
                        result["success"] = float(m.group(1)) / 100.0
                        result["score"] = result["success"]
                        result["higher_better"] = True
                        result["ok"] = True
                        return result
    else:
        mj = metrics if metrics.is_file() else Path(str(out_base) + "_metrics.json")
        if mj.is_file():
            data = json.loads(mj.read_text())
            chamfer = data.get("mean_chamfer_distance", data.get("mean_chamfer"))
            if chamfer is None and "details" in data:
                cds = [d["chamfer"] for d in data["details"] if "chamfer" in d]
                chamfer = sum(cds) / len(cds) if cds else None
            if chamfer is not None:
                result["mean_chamfer"] = float(chamfer)
                result["success_rate"] = data.get("success_rate")
                result["score"] = -float(chamfer)  # higher better internally
                result["higher_better"] = False
                result["ok"] = True
                return result
        if txt.is_file():
            import re

            for line in txt.read_text().splitlines():
                if "mean_chamfer" in line.lower():
                    m = re.search(r"([0-9.]+)", line.split(":")[-1])
                    if m:
                        result["mean_chamfer"] = float(m.group(1))
                        result["score"] = -float(m.group(1))
                        result["higher_better"] = False
                        result["ok"] = True
                        return result
    return result


def run_one(env: str, method: str, extras: dict, plan_gpu: str, egl_gpu: str, dry: bool) -> dict:
    ckpt = CKPTS[env][method]
    if not (ROOT / ckpt).is_file():
        return {"ok": False, "error": f"missing ckpt {ckpt}", "env": env, "method": method}
    out_base = OUT_DIR / tag_for(env, method, extras)
    txt_p, det_p, met_p, log_path = _out_paths(out_base)
    # skip if already done
    done = False
    if env in ("wall", "point_maze"):
        done = txt_p.is_file() and det_p.is_file()
    else:
        done = txt_p.is_file() and met_p.is_file()
    if done:
        metric = parse_metric(env, out_base)
        metric.update({"env": env, "method": method, "extras": extras, "skipped": True})
        return metric

    cmd = build_cmd(env, method, ckpt, extras, plan_gpu, egl_gpu)
    print(f"\n=== {env}/{method} ===", flush=True)
    print(" ".join(cmd), flush=True)
    if dry:
        return {"ok": True, "dry_run": True, "cmd": cmd, "env": env, "method": method}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # drop incomplete artifacts from a killed run
    if env in ("rope", "granular") and txt_p.is_file() and not met_p.is_file():
        txt_p.unlink(missing_ok=True)
    env_vars = os.environ.copy()
    env_vars["STABLEWM_HOME"] = str(ROOT / "data")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env_vars,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    metric = parse_metric(env, out_base)
    metric.update(
        {
            "env": env,
            "method": method,
            "extras": extras,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "log": str(log_path),
        }
    )
    if proc.returncode != 0:
        metric["ok"] = False
        metric["error"] = f"exit {proc.returncode}"
    print(json.dumps({k: metric[k] for k in metric if k != "raw"}, ensure_ascii=False), flush=True)
    return metric


def configs_for_step(step: int, base: dict) -> list[dict]:
    """Generate candidate extras dicts for a greedy step (on top of current cost*)."""
    b = dict(base)
    if step == 0:
        # baseline: CLS dense; S1/S2 m_task
        return [dict(b, cost_mode="m_task_terminal", cost_norm="none", mask_weight="hard")]
    if step == 1:
        return [dict(b, cost_mode="cls_terminal", cost_norm="none")]
    if step == 2:
        return [
            dict(b, cost_mode="decoupled", cost_norm="calib", cost_lambda=lam)
            for lam in (0.25, 0.5, 1.0, 2.0)
        ]
    if step == 3:
        # soft: first eps, then T (caller may run 3a/3b)
        return [
            dict(b, mask_weight="soft", soft_eps=e, soft_temp=b.get("soft_temp", 0.02))
            for e in (0.0, 0.1, 0.25)
        ]
    if step == 31:
        return [
            dict(b, mask_weight="soft", soft_eps=b.get("soft_eps", 0.1), soft_temp=t)
            for t in (0.01, 0.02, 0.04)
        ]
    if step == 4:
        out = [
            dict(b, tau_mode="fixed", m_task_threshold=0.08),
            dict(b, tau_mode="quantile", m_task_threshold=0.05),  # placeholder q≈0.5; refine offline
            dict(b, tau_mode="quantile", m_task_threshold=0.10),  # placeholder q≈0.75
            dict(b, tau_mode="otsu"),
        ]
        return out
    if step == 5:
        return [dict(b, relevance_source=s) for s in ("pixel", "attn_proj", "latent")]
    if step == 6:
        return [
            dict(b, cvar_q=q, cvar_beta=beta)
            for q in (0.1, 0.25)
            for beta in (0.5, 1.0)
        ]
    if step == 7:
        return [dict(b, lambda_anneal="linear")]
    raise ValueError(f"unknown step {step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, help="0=baseline … 7=λ anneal; 31=soft T")
    ap.add_argument("--methods", default="s2", help="comma: s2 / cls / s1")
    ap.add_argument("--envs", default="wall,point_maze,rope,granular")
    ap.add_argument("--plan-gpu", default="0")
    ap.add_argument("--egl-gpu", default="1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit-best", action="store_true", help="write winning extras into cost*")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    state = load_state()
    base = dict(state["cost*"])
    candidates = configs_for_step(args.step, base)

    all_runs = []
    for extras in candidates:
        for method in methods:
            # CLS baseline uses dense; skip meaningless multi-token-only knobs unless step0/1
            if method == "cls" and extras.get("cost_mode") not in (
                "dense_terminal",
                "cls_terminal",
                "m_task_terminal",
            ):
                # still allow decoupled etc. — criterion falls back for N=1
                pass
            if method == "cls" and args.step == 0:
                extras_m = dict(extras, cost_mode="dense_terminal")
            else:
                extras_m = extras
            for env in envs:
                r = run_one(env, method, extras_m, args.plan_gpu, args.egl_gpu, args.dry_run)
                all_runs.append(r)
                state["results"].append(r)
                save_state(state)

    # pick best config by majority of envs (for multi-token methods)
    if args.commit_best and not args.dry_run and "s2" in methods:
        # baseline scores from current cost*
        base_scores = {}
        for r in state["results"]:
            if r.get("method") != "s2" or not r.get("ok") or "score" not in r:
                continue
            if json.dumps(r.get("extras", {}), sort_keys=True) == json.dumps(base, sort_keys=True):
                base_scores[r["env"]] = r["score"]

        by_key: dict[str, list] = {}
        for r in all_runs:
            if r.get("method") != "s2" or not r.get("ok") or "score" not in r:
                continue
            key = json.dumps(r["extras"], sort_keys=True)
            by_key.setdefault(key, []).append(r)

        best_key, best_mean, best_wins = None, -1e9, -1
        for key, runs in by_key.items():
            # skip if identical to current base
            if key == json.dumps(base, sort_keys=True):
                continue
            scores = {x["env"]: x["score"] for x in runs}
            if len(scores) < 3:
                continue
            mean = sum(scores.values()) / len(scores)
            wins = 0
            losses = 0
            for env, sc in scores.items():
                bsc = base_scores.get(env)
                if bsc is None:
                    continue
                if sc > bsc + 1e-9:
                    wins += 1
                elif sc < bsc - 1e-9:
                    losses += 1
            # prefer more wins; tie-break by mean score
            rank = (wins, mean)
            if best_key is None or rank > (best_wins, best_mean):
                best_key, best_mean, best_wins = key, mean, wins

        if best_key is not None and best_wins >= 3:
            winner = json.loads(best_key)
            state["history"].append(
                {"step": args.step, "from": base, "to": winner, "wins": best_wins, "mean": best_mean}
            )
            state["cost*"] = winner
            save_state(state)
            print(f"[cost*] UPDATED (wins={best_wins}): {winner}", flush=True)
        else:
            state["history"].append(
                {
                    "step": args.step,
                    "kept": base,
                    "best_wins": best_wins,
                    "best_mean": best_mean,
                    "note": "no effective upgrade (≥3 envs)",
                }
            )
            save_state(state)
            print(f"[cost*] KEEP (best_wins={best_wins}): {base}", flush=True)

    summary = OUT_DIR / f"step{args.step}_summary.json"
    summary.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
