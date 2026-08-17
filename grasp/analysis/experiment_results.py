"""
Utilities for loading and aggregating experiment results from plan.py runs.

This module is designed to work with the directory layouts used in this repo:

- Baselines under ``baselines/``::

      baselines/{env}/{planner}/goal_H_{value}/{seed}/logs.json

- Ablations under an ``ablations``-style root::

      ablations/{method}/{seed}/logs.json

In both cases, ``logs.json`` files may be either:

- single-line JSON (pre-extracted final entry), or
- multi-line JSONL (raw ``plan.py`` output with intermediate + final entries).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_logs_json(file_path):
    """Load a logs.json file and return the final evaluation entry as a dict.

    Handles both single-line JSON (pre-extracted baselines) and multi-line
    JSONL (raw plan.py output). For JSONL, returns the last entry that
    contains ``final_eval/`` keys, falling back to the absolute last entry.
    """
    with open(file_path, "r") as f:
        content = f.read().strip()

    if not content:
        return None

    # Fast path: single-line JSON (baselines format)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Multi-line JSONL: parse line-by-line, return last final_eval entry
    last_final = None
    last_any = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            last_any = entry
            if any(k.startswith("final_eval/") for k in entry):
                last_final = entry
        except json.JSONDecodeError:
            continue

    return last_final if last_final is not None else last_any


def load_logs_jsonl(file_path):
    """Load a JSONL logs file and return all entries as a list of dicts.

    Useful when intermediate planner logs (``plan_0/`` prefix keys) are needed
    in addition to the final evaluation.
    """
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_success_and_time(logs_data):
    """Extract success indicator and total measured time from a log entry.

    Checks multiple key formats produced by different planners:
      - ``final_eval/success_rate``
      - ``final_eval/success``
      - ``plan_0/success_rate``
      - ``plan_0/success``

    Returns:
        (success, time) tuple where *success* is 0 or 1 (or None) and
        *time* is seconds (or None).
    """
    if not logs_data:
        return None, None

    success = None
    for key in (
        "final_eval/success_rate",
        "final_eval/success",
        "plan_0/success_rate",
        "plan_0/success",
    ):
        if key in logs_data:
            success = float(logs_data[key])
            break

    time_val = logs_data.get("total_measured_time")
    return success, time_val


def calculate_confidence_interval(values, confidence=0.95):
    """Compute a normal-approximation confidence interval half-width.

    Returns ``z * std / sqrt(n)`` where *z* is 1.96 for 95% confidence.
    """
    if len(values) < 2:
        return 0.0
    z = 1.96 if confidence == 0.95 else __import__("scipy.stats", fromlist=["norm"]).norm.ppf((1 + confidence) / 2)
    return z * np.std(values) / np.sqrt(len(values))


def detect_folder_type(base_dir):
    """Detect whether *base_dir* uses the baselines or ablations layout.

    Checks the folder name first, then inspects the directory tree structure.
    """
    base_path = Path(base_dir)

    if "ablations" in base_path.name.lower():
        return "ablations"
    if "baselines" in base_path.name.lower():
        return "baselines"

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        for inner in subdir.iterdir():
            if inner.is_dir() and inner.name.isdigit():
                if (inner / "logs.json").exists():
                    return "ablations"
            if inner.is_dir() and inner.name.startswith("goal_H_"):
                return "baselines"
        break

    return "baselines"


def load_experiment_results(base_dir, max_time=None):
    """Load all experiment results, auto-detecting directory layout.

    Args:
        base_dir: Root directory containing experiment results.
        max_time: If set, successes with ``total_measured_time > max_time``
            are reclassified as failures.

    Returns:
        Nested ``defaultdict``: ``results[env][planner][goal_H]`` is a list of
        ``(success, time)`` tuples for baselines, or
        ``results[method][None][None]`` for ablations.
    """
    folder_type = detect_folder_type(base_dir)
    if folder_type == "ablations":
        return load_ablations_results(base_dir, max_time=max_time)
    return load_baselines_results(base_dir, max_time=max_time)


def load_baselines_results(base_dir, max_time=None):
    """Walk a baselines directory tree and collect (success, time) tuples.

    Expected structure::

        base_dir/{env}/{planner}/goal_H_{value}/{seed}/logs.json
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base_path = Path(base_dir)

    if not base_path.exists():
        return results

    for env_dir in sorted(base_path.iterdir()):
        if not env_dir.is_dir():
            continue
        for planner_dir in sorted(env_dir.iterdir()):
            if not planner_dir.is_dir():
                continue
            for goal_h_dir in sorted(planner_dir.iterdir()):
                if not goal_h_dir.is_dir() or not goal_h_dir.name.startswith("goal_H_"):
                    continue
                try:
                    goal_h = int(goal_h_dir.name.replace("goal_H_", ""))
                except ValueError:
                    continue
                for seed_dir in sorted(goal_h_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    logs_file = seed_dir / "logs.json"
                    if not logs_file.exists():
                        continue
                    try:
                        data = load_logs_json(logs_file)
                        success, time_val = extract_success_and_time(data)
                        if success is not None and time_val is not None:
                            if max_time is not None and success > 0 and time_val > max_time:
                                success = 0
                            results[env_dir.name][planner_dir.name][goal_h].append(
                                (success, time_val)
                            )
                    except Exception:
                        continue

    return results


def load_ablations_results(base_dir, max_time=None):
    """Walk an ablations directory tree and collect (success, time) tuples.

    Expected structure::

        base_dir/{method}/{seed}/logs.json

    Results are keyed as ``results[method][None][None]`` for compatibility
    with the baselines format.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base_path = Path(base_dir)

    if not base_path.exists():
        return results

    for method_dir in sorted(base_path.iterdir()):
        if not method_dir.is_dir():
            continue
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            logs_file = seed_dir / "logs.json"
            if not logs_file.exists():
                continue
            try:
                data = load_logs_json(logs_file)
                success, time_val = extract_success_and_time(data)
                if success is not None and time_val is not None:
                    if max_time is not None and success > 0 and time_val > max_time:
                        success = 0
                    results[method_dir.name][None][None].append((success, time_val))
            except Exception:
                continue

    return results
