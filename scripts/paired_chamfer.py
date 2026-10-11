"""Paired comparison of rope/granular eval results (r3 *_metrics.json with details).

The first file is the baseline; every other file is compared with it on the
episodes both contain (matched by ep_id + start_step).

    python scripts/paired_chamfer.py outputs/r3/rope_cls_..._metrics.json outputs/r3/rope_s2_..._metrics.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load(path):
    d = json.load(open(path))
    if "details" not in d:
        raise SystemExit(f"{path}: no per-episode details (re-run with the r3 eval scripts)")
    return {(e["ep_id"], e["start_step"]): e for e in d["details"]}


def boot_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+")
    args = p.parse_args()
    runs = [(path, load(path)) for path in args.metrics]
    for path, d in runs:
        c = np.array([e["chamfer"] for e in d.values()])
        lo, hi = boot_ci(c)
        s = sum(e["success"] for e in d.values())
        print(f"n={len(c):3d} chamfer mean={c.mean():.3f} [{lo:.3f},{hi:.3f}] median={np.median(c):.3f} "
              f"success={s}/{len(c)}  {path}")
    base_path, base = runs[0]
    for path, d in runs[1:]:
        keys = sorted(set(base) & set(d))
        diff = np.array([d[k]["chamfer"] - base[k]["chamfer"] for k in keys])
        lo, hi = boot_ci(diff)
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print(f"[paired n={len(keys)}] {path} - base: mean diff={diff.mean():+.3f} "
              f"95% CI [{lo:+.3f},{hi:+.3f}] {sig} (negative = better than base)")


if __name__ == "__main__":
    main()
