#!/usr/bin/env python3
"""
Unified CLI for plotting and analyzing planning experiment results.

Synthesizes functionality from the plotting/ folder: goal_H sweeps, baselines/ablations
stats, time-vs-success curves, export/filter/extract utilities, action-dim sweeps,
loss landscape 3D plots, noise/gamma and temp-config sweeps, and task-specific analysis.

Usage:
    python visualize_results.py <subcommand> [options]
    python visualize_results.py --help
    python visualize_results.py goal_h_sweep --help

For ablations directories, use: baselines_stats --base_dir <path_to_ablations>
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.experiment_results import (
    load_logs_json,
    load_logs_jsonl,
    extract_success_and_time,
    calculate_confidence_interval,
    detect_folder_type,
    load_experiment_results,
    load_baselines_results,
    load_ablations_results,
)

# Plotting deps (optional per subcommand)
def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _import_plotly():
    import plotly.graph_objects as go
    return go


def _require_dir(path, name="base_dir"):
    """Return Path if directory exists, else print error and return None."""
    p = Path(path)
    if not p.exists():
        print(f"Error: {name} does not exist: {p}", file=sys.stderr)
        return None
    return p


# Unified accessors
def read_logs_json(path):
    return load_logs_json(path)


def read_logs_jsonl(path):
    return load_logs_jsonl(path)


def get_success_and_time(entry):
    return extract_success_and_time(entry)


def conf_interval(values, confidence=0.95):
    return calculate_confidence_interval(values, confidence)


def get_folder_type(base_dir):
    return detect_folder_type(base_dir)


def load_results(base_dir, max_time=None):
    """Load experiment results (baselines or ablations)."""
    return load_experiment_results(base_dir, max_time=max_time)


def _collect_baselines_with_seeds(base_dir, envs=None, planners=None, goal_Hs=None):
    """Walk baselines layout and return list of (env, planner, goal_H, seed, success, time). Used by export_success_times."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    out = []
    for env_dir in sorted(base_path.iterdir()):
        if not env_dir.is_dir():
            continue
        if envs is not None and env_dir.name not in envs:
            continue
        for planner_dir in sorted(env_dir.iterdir()):
            if not planner_dir.is_dir():
                continue
            if planners is not None and planner_dir.name not in planners:
                continue
            for goal_h_dir in sorted(planner_dir.iterdir()):
                if not goal_h_dir.is_dir() or not goal_h_dir.name.startswith("goal_H_"):
                    continue
                try:
                    goal_h = int(goal_h_dir.name.replace("goal_H_", ""))
                except ValueError:
                    continue
                if goal_Hs is not None and goal_h not in goal_Hs:
                    continue
                for seed_dir in sorted(goal_h_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    logs_file = seed_dir / "logs.json"
                    if not logs_file.exists():
                        continue
                    try:
                        data = read_logs_json(logs_file)
                        success, time_val = get_success_and_time(data)
                        if success is not None and time_val is not None:
                            out.append((env_dir.name, planner_dir.name, goal_h, seed_dir.name, success, time_val))
                    except Exception:
                        continue
    return out


def wald_ci(F_t, n):
    """Wald CI for cumulative distribution: 1.96 * sqrt(F*(1-F)/n)."""
    return 1.96 * np.sqrt(F_t * (1 - F_t) / n) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# goal_h_sweep
# ---------------------------------------------------------------------------

def cmd_goal_h_sweep(args):
    """Plot success rate (and optionally completion time) vs goal_H per env/planner."""
    plt = _import_matplotlib()
    base_dir = _require_dir(args.base_dir)
    if base_dir is None:
        return 1
    envs = args.envs.split(",") if args.envs else None
    planners = args.planners.split(",") if args.planners else None
    goal_Hs = [int(x) for x in args.goal_Hs.split(",")] if args.goal_Hs else None
    results = load_results(base_dir)
    if not results:
        print("No data to plot.", file=sys.stderr)
        return 1
    is_ablations = any(planner is None for e in results for planner in results[e])
    if is_ablations:
        print("goal_h_sweep expects baselines layout (env/planner/goal_H/seed).", file=sys.stderr)
        return 1
    if envs is not None:
        results = {e: results[e] for e in results if e in envs}
    if planners is not None:
        for e in list(results):
            results[e] = {p: results[e][p] for p in results[e] if p in planners}
    if goal_Hs is not None:
        for e in list(results):
            for p in list(results[e]):
                results[e][p] = {gh: results[e][p][gh] for gh in results[e][p] if gh in goal_Hs}
    if args.condition_on_success:
        for e in list(results):
            for p in list(results[e]):
                for gh in list(results[e][p]):
                    results[e][p][gh] = [(s, t) for s, t in results[e][p][gh] if s and s >= 0.5]
    for e in list(results):
        for p in list(results[e]):
            if not results[e][p]:
                del results[e][p]
        if not results[e]:
            del results[e]
    if not results:
        print("No data to plot.", file=sys.stderr)
        return 1
    planner_list = planners or sorted(set(p for e in results for p in results[e]))
    env_list = sorted(results.keys())
    n_envs = len(env_list)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 6), squeeze=False)
    axes = axes.flatten()
    # Planner color palette; the "admm" key corresponds to the GRASP planner baseline.
    colors = {"admm": "blue", "latco": "green", "cem": "red"}
    for idx, env in enumerate(env_list):
        ax = axes[idx]
        for planner in planner_list:
            if planner not in results[env]:
                continue
            goal_Hs_sorted = sorted(results[env][planner].keys())
            means = []
            cis = []
            for gh in goal_Hs_sorted:
                vals = [s for s, t in results[env][planner][gh]]
                if vals:
                    means.append(np.mean(vals))
                    cis.append(conf_interval(vals))
                else:
                    means.append(np.nan)
                    cis.append(0.0)
            means = np.array(means)
            cis = np.array(cis)
            valid = ~np.isnan(means)
            if np.any(valid):
                ax.errorbar(
                    np.array(goal_Hs_sorted)[valid], means[valid], yerr=cis[valid],
                    label=planner, color=colors.get(planner, "gray"), capsize=4, marker="o"
                )
        ax.set_xlabel("Goal H")
        ax.set_ylabel("Success Rate")
        ax.set_title(env)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "goal_h_sweep.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    return 0


# ---------------------------------------------------------------------------
# baselines_stats
# ---------------------------------------------------------------------------

def cmd_baselines_stats(args):
    """Print summary statistics and optionally LaTeX tables for baselines/ablations."""
    base_dir = _require_dir(args.base_dir)
    if base_dir is None:
        return 1
    results = load_results(base_dir, max_time=args.max_time)
    if not results:
        print("No results found.", file=sys.stderr)
        return 1
    goal_H_str = args.goal_H
    goal_H_values = [int(x.strip()) for x in goal_H_str.split(",")] if goal_H_str else [2, 4, 6]
    is_ablations = any(planner is None for env in results for planner in results[env])
    if is_ablations:
        print("\n--- Ablations summary ---")
        for method in sorted(results.keys()):
            data = results[method][None][None]
            if not data:
                continue
            success_rates = [s for s, t in data]
            times = [t for s, t in data]
            mean_sr = np.mean(success_rates) * 100
            ci_sr = conf_interval(success_rates) * 100
            med_t = np.median(times) if times else float("nan")
            print(f"  {method}: success={mean_sr:.1f}% ± {ci_sr:.1f}%, median_time={med_t:.2f}s, n={len(data)}")
        if args.latex:
            print("\n\\begin{tabular}{lrrr}")
            print("Method & Success (\\%) & Median Time & N \\\\")
            for method in sorted(results.keys()):
                data = results[method][None][None]
                if not data:
                    continue
                mean_sr = np.mean([s for s, t in data]) * 100
                ci_sr = conf_interval([s for s, t in data]) * 100
                med_t = np.median([t for s, t in data])
                print(f"{method} & ${mean_sr:.1f} \\pm {ci_sr:.1f}$ & {med_t:.2f} & {len(data)} \\\\")
            print("\\end{tabular}")
        return 0
    env_order = sorted(results.keys())
    planner_order = sorted(set(p for e in results for p in results[e] if p is not None))
    print("\n--- Baselines summary ---")
    print(f"{'Env':<12} {'Planner':<20} {'GoalH':<6} {'N':<6} {'Success%':<10} {'±CI':<8} {'MedTime':<10}")
    print("-" * 80)
    for env in env_order:
        for planner in planner_order:
            if planner not in results[env]:
                continue
            for goal_H in goal_H_values:
                if goal_H not in results[env][planner]:
                    continue
                data = results[env][planner][goal_H]
                if not data:
                    continue
                sr = [s for s, t in data]
                times = [t for s, t in data]
                mean_sr = np.mean(sr) * 100
                ci_sr = conf_interval(sr) * 100
                med_t = np.median(times)
                print(f"{env:<12} {planner:<20} {goal_H:<6} {len(data):<6} {mean_sr:<10.1f} {ci_sr:<8.1f} {med_t:<10.2f}")
    if args.detailed:
        print("\n--- Detailed breakdown ---")
        for env in env_order:
            for planner in planner_order:
                if planner not in results[env]:
                    continue
                for goal_H in sorted(results[env][planner].keys()):
                    data = results[env][planner][goal_H]
                    if not data:
                        continue
                    sr = [s for s, t in data]
                    times = [t for s, t in data]
                    succ_times = [t for s, t in data if s > 0]
                    print(f"  {env} / {planner} / goal_H={goal_H}: n={len(data)}, mean_success={np.mean(sr):.3f}, "
                          f"median_time={np.median(times):.2f}s, median_success_time={np.median(succ_times) if succ_times else float('nan'):.2f}s")
    if args.latex:
        print("\n--- LaTeX success rate table (excerpt) ---")
        n_cols = len(env_order) * len(goal_H_values)
        print("\\begin{tabular}{l" + "r" * n_cols + "}")
        header = ["Planner"]
        for env in env_order:
            for goal_H in goal_H_values:
                header.append(f"{env}_{goal_H}")
        print(" & ".join(header) + " \\\\")
        print("\\hline")
        for planner in planner_order:
            row = [planner]
            for env in env_order:
                for goal_H in goal_H_values:
                    if env in results and planner in results[env] and goal_H in results[env][planner]:
                        data = results[env][planner][goal_H]
                        if data:
                            row.append(f"{np.mean([s for s, t in data]) * 100:.1f}")
                        else:
                            row.append("--")
                    else:
                        row.append("--")
            print(" & ".join(str(x) for x in row) + " \\\\")
        print("\\end{tabular}")
    return 0


# ---------------------------------------------------------------------------
# time_vs_success
# ---------------------------------------------------------------------------

def cmd_time_vs_success(args):
    """Plot cumulative success rate over time per env/planner."""
    plt = _import_matplotlib()
    base_dir = _require_dir(args.base_dir)
    if base_dir is None:
        return 1
    results = load_results(base_dir)
    if not results:
        print("No data to plot.", file=sys.stderr)
        return 1
    envs = args.envs.split(",") if args.envs else None
    planners = args.planners.split(",") if args.planners else None
    goal_Hs = [int(x) for x in args.goal_Hs.split(",")] if args.goal_Hs else None
    results_list = []
    is_ablations = any(planner is None for e in results for planner in results[e])
    if is_ablations:
        for method in results:
            for (s, t) in results[method][None][None]:
                if s is not None and t is not None:
                    results_list.append({"model": "ablations", "planner": method, "success_rate": s, "total_measured_time": t})
    else:
        for env in results:
            if envs is not None and env not in envs:
                continue
            for planner in results[env]:
                if planner is None:
                    continue
                if planners is not None and planner not in planners:
                    continue
                for goal_h in results[env][planner]:
                    if goal_Hs is not None and goal_h not in goal_Hs:
                        continue
                    for (s, t) in results[env][planner][goal_h]:
                        if s is not None and t is not None:
                            results_list.append({"model": env, "planner": planner, "success_rate": s, "total_measured_time": t})
    if not results_list:
        print("No data to plot.", file=sys.stderr)
        return 1
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results_list:
        grouped[r["model"]][r["planner"]].append({"success": r["success_rate"], "time": r["total_measured_time"]})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    planners_ord = ["latco", "cem", "admm"] if not is_ablations else sorted(set(p for m in grouped for p in grouped[m]))
    for model in sorted(grouped.keys()):
        fig, ax = plt.subplots(figsize=(8, 6))
        for planner in planners_ord:
            if planner not in grouped[model]:
                continue
            data = sorted(grouped[model][planner], key=lambda x: x["time"])
            n = len(data)
            times = [0] + [d["time"] for d in data]
            cum = [0] + [sum(d["success"] for d in data[:i]) / n for i in range(1, n + 1)]
            ax.plot(times, cum, label=planner, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Success Rate")
        ax.set_title(model)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        out_file = out_dir / f"cumulative_success_vs_time_{model}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_file}")
    return 0


# ---------------------------------------------------------------------------
# export_success_times
# ---------------------------------------------------------------------------

def cmd_export_success_times(args):
    """Export success times for env/planner/goal_H, or --max_time for a planner."""
    base_dir = _require_dir(args.base_dir)
    if base_dir is None:
        return 1
    if args.max_time:
        rows = _collect_baselines_with_seeds(
            base_dir, envs=[args.env] if args.env else None, planners=[args.planner], goal_Hs=None
        )
        success_rows = [(e, g, s, t) for e, p, g, s, success, t in rows if success > 0]
        if not success_rows:
            print("No successful runs found.")
            return 0
        total_runs = len(rows)
        max_entry = max(success_rows, key=lambda x: x[3])
        print(f"# Planner: {args.planner}, total runs: {total_runs}, successful: {len(success_rows)}")
        print(f"# Max success time: {max_entry[3]:.4f}s (env={max_entry[0]}, goal_H={max_entry[1]}, seed={max_entry[2]})")
        print(max_entry[3])
        if args.output:
            out_data = {"planner": args.planner, "max_time": max_entry[3], "all_times": [{"env": e, "goal_H": g, "seed": s, "time": t} for e, g, s, t in success_rows]}
            with open(args.output, "w") as f:
                json.dump(out_data, f, indent=2)
        return 0
    if not args.env or args.goal_H is None:
        print("Error: --env and --goal_H are required unless --max_time is used.", file=sys.stderr)
        return 1
    rows = _collect_baselines_with_seeds(
        base_dir, envs=[args.env], planners=[args.planner], goal_Hs=[args.goal_H]
    )
    success_times = [(seed, t) for env, planner, goal_h, seed, success, t in rows if success > 0]
    total_runs = len(rows)
    out_lines = []
    if args.format == "plain":
        out_lines = [str(t) for _, t in success_times]
    elif args.format == "json":
        out_data = {"environment": args.env, "planner": args.planner, "goal_H": args.goal_H, "total_runs": total_runs, "times": [t for _, t in success_times]}
        out_lines = [json.dumps(out_data, indent=2)]
    elif args.format == "csv":
        out_lines = ["seed,time"] + [f"{s},{t}" for s, t in success_times]
    out = "\n".join(out_lines)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Wrote {args.output}")
    else:
        print(f"# env={args.env} planner={args.planner} goal_H={args.goal_H} total_runs={total_runs} n_success={len(success_times)}")
        print(out)
    return 0


# ---------------------------------------------------------------------------
# extract_final_logs
# ---------------------------------------------------------------------------

def cmd_extract_final_logs(args):
    """Extract final JSON entry from each logs.json (JSONL) and write to output dir or tar."""
    exp_path = _require_dir(args.experiment_dir, "experiment_dir")
    if exp_path is None:
        return 1
    logs_files = list(exp_path.rglob("logs.json"))
    if not logs_files:
        print("No logs.json files found.")
        return 0
    if args.output_tar:
        tmpdir = tempfile.mkdtemp()
        try:
            for logs_file in logs_files:
                rel = logs_file.relative_to(exp_path)
                target = Path(tmpdir) / exp_path.name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                content = Path(logs_file).read_text().strip()
                final_json = None
                for line in reversed(content.splitlines()):
                    line = line.strip()
                    if line:
                        try:
                            final_json = json.dumps(json.loads(line))
                            break
                        except json.JSONDecodeError:
                            continue
                if final_json:
                    target.write_text(final_json + "\n")
            with tarfile.open(args.output_tar, "w:gz") as tar:
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        fp = Path(root) / f
                        tar.add(fp, arcname=fp.relative_to(tmpdir))
            print(f"Created {args.output_tar} ({len(logs_files)} files)")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        out_dir = Path(args.output_dir or (exp_path.parent / (exp_path.name + "_final_logs")))
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for logs_file in logs_files:
            rel = logs_file.relative_to(exp_path)
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            content = Path(logs_file).read_text().strip()
            final_json = None
            for line in reversed(content.splitlines()):
                line = line.strip()
                if line:
                    try:
                        final_json = json.dumps(json.loads(line))
                        break
                    except json.JSONDecodeError:
                        continue
            if final_json:
                target.write_text(final_json + "\n")
                count += 1
        print(f"Wrote {count} files to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# filter_success
# ---------------------------------------------------------------------------

def cmd_filter_success(args):
    """Print comma-separated list of seed subdirs where success_rate meets threshold."""
    base_dir = _require_dir(args.directory, "directory")
    if base_dir is None:
        return 1
    threshold = float(args.threshold)
    key = args.metric_key
    successful = []
    for subdir in sorted(base_dir.iterdir(), key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else 0)):
        if not subdir.is_dir():
            continue
        logs_file = subdir / "logs.json"
        if not logs_file.exists():
            continue
        try:
            data = read_logs_json(logs_file)
            if data is None:
                continue
            val = data.get(key)
            if val is not None and float(val) >= threshold:
                successful.append(subdir.name)
        except (json.JSONDecodeError, TypeError):
            continue
    print(", ".join(successful) if successful else "No successful runs found")
    return 0


# ---------------------------------------------------------------------------
# action_dim_sweep
# ---------------------------------------------------------------------------

def cmd_action_dim_sweep(args):
    """Plot metric vs action_dim. Layout: base_dir/{planner}/action_dim_{value}/{seed}/logs.json."""
    plt = _import_matplotlib()
    base_path = _require_dir(args.base_dir)
    if base_path is None:
        return 1
    results = defaultdict(lambda: defaultdict(list))
    for planner_dir in base_path.iterdir():
        if not planner_dir.is_dir():
            continue
        planner_name = planner_dir.name
        for ad_dir in planner_dir.iterdir():
            if not ad_dir.is_dir() or not ad_dir.name.startswith("action_dim_"):
                continue
            try:
                action_dim = int(ad_dir.name.split("_")[-1])
            except ValueError:
                continue
            for seed_dir in ad_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                logs_file = seed_dir / "logs.json"
                if not logs_file.exists():
                    continue
                try:
                    data = read_logs_json(logs_file)
                    if data is None:
                        continue
                    val = data.get(args.metric)
                    if val is not None:
                        results[planner_name][action_dim].append(float(val))
                except Exception:
                    continue
    if not results:
        print("No data to plot.", file=sys.stderr)
        return 1
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in sorted(results.keys()):
        adims = sorted(results[planner].keys())
        means = [np.mean(results[planner][ad]) for ad in adims]
        cis = [conf_interval(results[planner][ad]) for ad in adims]
        ax.errorbar(adims, means, yerr=cis, label=planner, capsize=4, marker="o")
    ax.set_xlabel("Action dimension")
    ax.set_ylabel(args.metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "action_dim_sweep.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    return 0


# ---------------------------------------------------------------------------
# loss_landscape
# ---------------------------------------------------------------------------

def cmd_loss_landscape(args):
    """Create 3D surface plot from a .npy file."""
    go = _import_plotly()
    npy_path = Path(args.npy_path)
    if not npy_path.exists():
        print(f"Error: file not found: {npy_path}", file=sys.stderr)
        return 1
    data = np.load(npy_path)
    if data.ndim != 2:
        print(f"Error: expected 2D array, got shape {data.shape}", file=sys.stderr)
        return 1
    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    fig = go.Figure(data=[go.Surface(z=data, x=X, y=Y, colorscale=args.colorscale)])
    fig.update_layout(
        title=args.title or f"Loss landscape - {npy_path.stem}",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Value"),
        width=800, height=600,
    )
    out_path = args.output_html or str(npy_path.with_suffix(".html"))
    fig.write_html(out_path)
    print(f"Saved {out_path}")
    return 0


# ---------------------------------------------------------------------------
# noise_gamma_sweep (layout: base_dir/{model}/admm/{seed}/logs.json, seed maps to noise_grad x gamma_action)
# ---------------------------------------------------------------------------

def cmd_noise_gamma_sweep(args):
    """Analyze GRASP noise_grad x gamma_action sweep; output heatmaps and CSV."""
    plt = _import_matplotlib()
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print("noise_gamma_sweep requires pandas and seaborn.", file=sys.stderr)
        return 1
    base_path = _require_dir(args.base_dir)
    if base_path is None:
        return 1
    noise_vals = [float(x) for x in (args.noise_vals or "0.01,0.1,1.0").split(",")]
    gamma_vals = [float(x) for x in (args.gamma_vals or "0.001,0.01,0.1,1.0").split(",")]
    param_combos = [(n, g) for n in noise_vals for g in gamma_vals]
    seed_to_params = {}
    for i in range(600):
        idx = i // 50
        seed_to_params[i] = param_combos[idx] if idx < len(param_combos) else param_combos[-1]
    models = args.models.split(",") if args.models else ["point_maze", "wall_single", "pusht"]
    rows = []
    for model in models:
        # Results are stored under {model}/admm, which corresponds to the GRASP planner.
        admm_dir = base_path / model / "admm"
        if not admm_dir.exists():
            continue
        for seed_dir in admm_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            logs_file = seed_dir / "logs.json"
            if not logs_file.exists():
                continue
            try:
                data = read_logs_json(logs_file)
                sr = data.get("final_eval/success_rate") if data else None
                if sr is not None:
                    seed = int(seed_dir.name)
                    params = seed_to_params.get(seed, (None, None))
                    rows.append({"model": model, "seed": seed, "noise_grad": params[0], "gamma_action": params[1], "success_rate": sr})
            except Exception:
                continue
    if not rows:
        print("No data found.", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "noise_gamma_sweep.csv", index=False)
    print(f"Saved CSV with {len(df)} rows to {out_dir / 'noise_gamma_sweep.csv'}")
    for model in df["model"].unique():
        pivot = df[df["model"] == model].groupby(["noise_grad", "gamma_action"])["success_rate"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", ax=ax, cmap="viridis")
        ax.set_title(f"{model} - mean success rate")
        plt.savefig(out_dir / f"noise_gamma_heatmap_{model}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_dir / f'noise_gamma_heatmap_{model}.png'}")
    return 0


# ---------------------------------------------------------------------------
# temp_config_sweep (layout: base_dir/{model}/{planner}/{seed}/logs.json)
# ---------------------------------------------------------------------------

def cmd_temp_config_sweep(args):
    """Compare planners (GRASP, latco, gd_no_noise, cem) across models.

    """
    plt = _import_matplotlib()
    base_path = _require_dir(args.base_dir)
    if base_path is None:
        return 1
    models = args.models.split(",") if args.models else ["point_maze", "wall_single", "pusht"]
    planners = args.planners.split(",") if args.planners else ["admm", "latco", "gd_no_noise", "cem"]
    results = []
    for model in models:
        for planner in planners:
            planner_dir = base_path / model / planner
            if not planner_dir.exists():
                continue
            for seed_dir in planner_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                logs_file = seed_dir / "logs.json"
                if not logs_file.exists():
                    continue
                try:
                    data = read_logs_json(logs_file)
                    sr = data.get("final_eval/success_rate") if data else None
                    if sr is not None:
                        results.append({"model": model, "planner": planner, "seed": int(seed_dir.name), "success_rate": sr})
                except Exception:
                    continue
    if not results:
        print("No data found.", file=sys.stderr)
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in planners:
        vals = [r["success_rate"] for r in results if r["planner"] == planner]
        if vals:
            ax.bar(planner, np.mean(vals), yerr=conf_interval(vals), capsize=4, label=planner)
    ax.set_ylabel("Success rate")
    ax.set_title("Temp config sweep (all models)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.savefig(out_dir / "temp_config_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'temp_config_sweep.png'}")
    return 0


# ---------------------------------------------------------------------------
# task_specific (JSONL logs, convergence + final metrics, optional CSV)
# ---------------------------------------------------------------------------

def cmd_task_specific(args):
    """Task-specific analysis: final + convergence metrics from JSONL logs, output CSV."""
    base_path = _require_dir(args.base_dir)
    if base_path is None:
        return 1
    planners = args.planners.split(",") if args.planners else ["cem", "latco", "gd_no_noise", "admm"]
    all_rows = []
    for planner in planners:
        planner_dir = base_path / planner
        if not planner_dir.exists():
            continue
        for run_dir in planner_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.isdigit():
                continue
            logs_file = run_dir / "logs.json"
            if not logs_file.exists():
                continue
            entries = read_logs_jsonl(logs_file)
            if not entries:
                continue
            final = None
            for e in reversed(entries):
                if "final_eval/success_rate" in e:
                    final = e
                    break
            if not final:
                final = entries[-1]
            row = {"planner": planner, "run_id": run_dir.name}
            row["final_success_rate"] = final.get("final_eval/success_rate") or final.get("plan_0/success_rate")
            row["total_measured_time"] = final.get("total_measured_time")
            convergence = []
            for e in entries:
                if "plan_0/success_rate" in e and "total_measured_time" in e:
                    convergence.append({"step": e.get("step", 0), "time": e["total_measured_time"], "success_rate": e["plan_0/success_rate"]})
            if convergence:
                first_succ = next((c for c in convergence if c["success_rate"] >= 1.0), None)
                row["first_success_time"] = first_succ["time"] if first_succ else None
                row["first_success_step"] = first_succ["step"] if first_succ else None
            else:
                row["first_success_time"] = row["first_success_step"] = None
            all_rows.append(row)
    if not all_rows:
        print("No data found.", file=sys.stderr)
        return 1
    try:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        out_csv = Path(args.output_csv or (str(base_path) + "_task_analysis.csv"))
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} ({len(df)} rows)")
    except ImportError:
        import csv
        out_csv = Path(args.output_csv or (str(base_path) + "_task_analysis.csv"))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f"Saved {out_csv} ({len(all_rows)} rows)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize and analyze planning experiment results")
    parser.add_argument("--base_dir", default=str(root / "baselines"), help="Base directory for results (default: ./baselines)")
    sub = parser.add_subparsers(dest="command", required=True)

    def a(name, **kwargs):
        return (name, kwargs)

    subcmd_specs = [
        ("goal_h_sweep", "Plot success rate vs goal_H per env/planner", cmd_goal_h_sweep, [
            a("--base_dir", default=str(root / "baselines")),
            a("--envs", default=None, help="Comma-separated env names"),
            a("--planners", default=None, help="Comma-separated planner names"),
            a("--goal_Hs", default=None, help="Comma-separated goal_H values"),
            a("--condition_on_success", action="store_true"),
            a("--output_dir", default=str(root / "analysis_outputs" / "goal_h_sweep")),
        ]),
        ("baselines_stats", "Print summary stats and optional LaTeX for baselines/ablations", cmd_baselines_stats, [
            a("--base_dir", default=str(root / "baselines")),
            a("--goal_H", default="2,4,6", help="Comma-separated goal_H for tables"),
            a("--max_time", type=float, default=None),
            a("--detailed", action="store_true"),
            a("--latex", action="store_true"),
        ]),
        ("time_vs_success", "Plot cumulative success over time", cmd_time_vs_success, [
            a("--base_dir", default=str(root / "baselines")),
            a("--envs", default=None),
            a("--planners", default=None),
            a("--goal_Hs", default=None),
            a("--output_dir", default=str(root / "analysis_outputs" / "time_vs_success")),
        ]),
        ("export_success_times", "Export success times (or --max_time for planner)", cmd_export_success_times, [
            a("--base_dir", default=str(root / "baselines")),
            a("--env", default=None),
            a("--goal_H", type=int, default=None),
            a("--planner", required=True),
            a("--output", default=None),
            a("--format", choices=["plain", "json", "csv"], default="plain"),
            a("--max_time", action="store_true"),
        ]),
        ("extract_final_logs", "Extract final JSON from each logs.json (JSONL), write dir or tar", cmd_extract_final_logs, [
            a("--experiment_dir", required=True),
            a("--output_dir", default=None),
            a("--output_tar", default=None),
        ]),
        ("filter_success", "List seed subdirs where success_rate >= threshold", cmd_filter_success, [
            a("directory", help="Directory containing numbered subdirs with logs.json"),
            a("--metric_key", default="final_eval/success_rate"),
            a("--threshold", default="1.0"),
        ]),
        ("action_dim_sweep", "Plot metric vs action_dim (layout: base_dir/planner/action_dim_N/seed/logs.json)", cmd_action_dim_sweep, [
            a("--base_dir", required=True),
            a("--metric", default="final_eval/success_rate"),
            a("--output_dir", default=str(root / "analysis_outputs" / "action_dim_sweep")),
        ]),
        ("loss_landscape", "3D surface plot from .npy file", cmd_loss_landscape, [
            a("npy_path", help="Path to .npy file"),
            a("--output_html", default=None),
            a("--title", default=None),
            a("--colorscale", default="viridis"),
        ]),
        ("noise_gamma_sweep", "Noise x gamma_action sweep heatmaps + CSV", cmd_noise_gamma_sweep, [
            a("--base_dir", required=True),
            a("--models", default=None),
            a("--noise_vals", default=None),
            a("--gamma_vals", default=None),
            a("--output_dir", default=str(root / "analysis_outputs" / "noise_gamma_sweep")),
        ]),
        ("temp_config_sweep", "Compare planners across models", cmd_temp_config_sweep, [
            a("--base_dir", required=True),
            a("--models", default=None),
            a("--planners", default=None),
            a("--output_dir", default=str(root / "analysis_outputs" / "temp_config_sweep")),
        ]),
        ("task_specific", "Task-specific metrics from JSONL logs, output CSV", cmd_task_specific, [
            a("--base_dir", required=True),
            a("--planners", default=None),
            a("--output_csv", default=None),
        ]),
    ]
    for name, help_text, func, args_list in subcmd_specs:
        p = sub.add_parser(name, help=help_text)
        for name_kw in args_list:
            arg_name = name_kw[0]
            kwargs = name_kw[1]
            p.add_argument(arg_name, **kwargs)
        p.set_defaults(func=func)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
