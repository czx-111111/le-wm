"""success@k curve from wall / point_maze MPC details JSON.

Run eval once with a large cap (e.g. ``--n-taken-actions 5 --mpc-max-iters 10``);
per-episode torch seeding makes the first k rounds identical to a run with
``--mpc-max-iters k``, so success@k here equals that run's success rate.

    python scripts/mpc_budget_curve.py outputs/wall_cls_details_r2.json outputs/wall_s2_details_r2.json \
        --frameskip 5 --n-taken 5 --ks 1 2 4 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    out = {}
    for e in d["details"]:
        if e.get("planner") != "mpc":
            raise SystemExit(f"{path}: planner={e.get('planner')}, need mpc")
        out[(int(e["ep_id"]), int(e["start_step"]))] = e
    return out


def success_at(e: dict, k: int) -> bool:
    return bool(e["success"]) and int(e["mpc_iters"]) <= k


def dist_at(e: dict, k: int) -> float | None:
    h = e.get("dist_history")
    if not h:
        return None
    return float(h[min(k, len(h) - 1)])


def median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("details", nargs="+")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--n-taken", type=int, default=5)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 10])
    args = p.parse_args()

    runs = [(Path(x).stem, load(x)) for x in args.details]
    keys = set(runs[0][1])
    for name, r in runs[1:]:
        if set(r) != keys:
            print(f"[WARN] {name}: episode set differs from {runs[0][0]}; pairing on intersection")
            keys &= set(r)
    keys = sorted(keys)
    n = len(keys)

    for name, r in runs:
        eps = [r[k] for k in keys]
        cap = max(int(e["mpc_iters"]) for e in eps)
        early = sum(int(e["mpc_iters"]) == 0 and bool(e["success"]) for e in eps)
        print(f"\n== {name}  n={n}  early_success={early}  max_rounds_used={cap}")
        print(f"{'k':>3} {'env_steps':>9} {'success%':>9} {'wilson95':>15} {'mean_d':>7} {'med_d':>7}")
        for k in args.ks:
            s = sum(success_at(e, k) for e in eps)
            lo, hi = wilson(s, n)
            ds = [d for d in (dist_at(e, k) for e in eps) if d is not None]
            md = f"{sum(ds) / len(ds):7.3f}" if ds else "      -"
            mdd = f"{median(ds):7.3f}" if ds else "      -"
            print(f"{k:>3} {k * args.n_taken * args.frameskip:>9} {100 * s / n:>8.1f}% "
                  f"[{100 * lo:5.1f},{100 * hi:5.1f}] {md} {mdd}")

    if len(runs) >= 2:
        base_name, base = runs[0]
        for name, r in runs[1:]:
            print(f"\n== paired: {name} vs {base_name} (McNemar exact)")
            for k in args.ks:
                b = sum(success_at(r[x], k) and not success_at(base[x], k) for x in keys)
                c = sum(success_at(base[x], k) and not success_at(r[x], k) for x in keys)
                print(f"  k={k:>2}: +{b} / -{c}  diff={100 * (b - c) / n:+.1f}pp  p={mcnemar_exact(b, c):.3f}")


if __name__ == "__main__":
    main()
