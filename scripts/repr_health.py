"""Representation health check: run before any planning eval.

For each sampled eval episode, encode frames goal-W .. goal and measure the
Spearman correlation between remaining steps (goal - t) and the latent
distance ||z_t - z_goal||^2. The terminal planning cost relies on exactly this
ordering; rho ~ 0 means planning is random regardless of planner settings.

Also reports cross-frame vs cross-token spread (collapse check).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/repr_health.py --env point_maze \
        --ckpt data/<run>/<name>_last_object.ckpt [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WINDOW = {"wall": 25, "point_maze": 25, "rope": 20, "granular": 20}
# Per-env rho floor. Rope pixel-oracle Spearman is only ~0.27, so 0.5 is
# unreachable; use a lower bar and still require non-collapse (frame spread).
RHO_MIN = {"wall": 0.5, "point_maze": 0.5, "granular": 0.45, "rope": 0.25}
SPREAD_MIN = 0.05

TF = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    transforms.Resize(size=224),
])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def encode(model, frames, device):
    out = []
    for i in range(0, len(frames), 64):
        x = torch.stack([TF(f) for f in frames[i:i + 64]]).unsqueeze(1).to(device)
        e = model.encode({"pixels": x})["emb"][:, 0].float().cpu()
        out.append(e if e.ndim == 3 else e[:, None])
    return torch.cat(out).numpy()  # (F, N, D)


def check(ckpt, env, n_eps, seed, device):
    model = torch.load(ckpt, map_location="cpu", weights_only=False).to(device).eval()
    w = WINDOW[env]
    with h5py.File(ROOT / "data" / f"{env}_eval.h5", "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        rng = np.random.default_rng(seed)
        eps = rng.choice(len(ep_len), size=min(n_eps, len(ep_len)), replace=False)
        rho_g, rho_a, all_emb = [], [], []
        for ep in eps:
            L = int(ep_len[ep])
            goal = int(rng.integers(min(w, L - 1), L))
            idx = np.arange(goal - min(w, goal), goal + 1)
            e = encode(model, f["pixels"][ep_off[ep] + idx], device)
            remain = goal - idx
            rho_g.append(spearman(remain, ((e[:, 0] - e[-1, 0]) ** 2).sum(-1)))
            rho_a.append(spearman(remain, ((e - e[-1]) ** 2).mean((1, 2))))
            all_emb.append(e)
    E = np.concatenate(all_emb)
    frame_std = float(E.std(0).mean())
    token_std = float(E.mean(0).std(0).mean()) if E.shape[1] > 1 else 0.0
    res = dict(
        ckpt=str(ckpt),
        env=env,
        n_tokens=int(E.shape[1]),
        rho_global=round(float(np.mean(rho_g)), 3),
        rho_all_tokens=round(float(np.mean(rho_a)), 3),
        frame_std=round(frame_std, 4),
        token_std=round(token_std, 4),
    )
    rho_used = res["rho_all_tokens"] if E.shape[1] > 1 else res["rho_global"]
    rho_min = float(RHO_MIN[env])
    res["rho_min"] = rho_min
    res["healthy"] = bool(rho_used >= rho_min and frame_std >= SPREAD_MIN * max(1.0, token_std))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(WINDOW))
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--n-eps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    rows = []
    for c in args.ckpt:
        r = check(c, args.env, args.n_eps, args.seed, args.device)
        rows.append(r)
        flag = "OK " if r["healthy"] else "BAD"
        print(f"[{flag}] rho_global={r['rho_global']:+.3f} rho_all={r['rho_all_tokens']:+.3f} "
              f"frame_std={r['frame_std']:.4f} token_std={r['token_std']:.4f} "
              f"rho_min={r['rho_min']:.2f}  {c}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
