#!/usr/bin/env python3
"""Re-estimate BatchNorm running stats on training pixels, then (optionally) health-check.

Planning / repr_health run in eval mode and use BN running stats. If those stats
were skewed, eval embeddings can temporally collapse even when train-mode still
shows frame diversity. This script:

  1. resets BN running_mean / running_var
  2. runs encode() in train() mode over train-set frames (batch 128)
  3. saves <stem>_bnreest_object.ckpt next to the original
  4. optionally runs scripts.repr_health on before/after

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reest_bn.py \\
      --env rope --ckpt data/runs/r3/.../rope_cls_last_object.ckpt --health
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TF = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    transforms.Resize(size=224),
])


def bn_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]


def snapshot_bn(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    out = []
    for m in bn_modules(model):
        out.append((m.running_mean.detach().cpu().clone(),
                    m.running_var.detach().cpu().clone(),
                    int(m.num_batches_tracked.item()) if m.num_batches_tracked is not None else -1))
    return out


def reestimate(model: nn.Module, env: str, device: str, n_batches: int, batch_size: int, seed: int):
    bns = bn_modules(model)
    if not bns:
        raise RuntimeError("no BatchNorm modules found")
    # Cumulative average over the re-estimation pass (momentum=None).
    old_mom = []
    for m in bns:
        old_mom.append(m.momentum)
        m.reset_running_stats()
        m.momentum = None

    train_path = ROOT / "data" / f"{env}_train.h5"
    if not train_path.is_file():
        train_path = ROOT / "data" / "datasets" / f"{env}_train.h5"
    rng = np.random.default_rng(seed)
    model.train()
    model.to(device)

    with h5py.File(train_path, "r") as f:
        n = int(f["pixels"].shape[0])
        print(f"[reest] {train_path}  n_frames={n}  BN layers={len(bns)}  "
              f"batches={n_batches}×{batch_size}", flush=True)
        with torch.no_grad():
            for bi in range(n_batches):
                idx = rng.choice(n, size=batch_size, replace=False)
                # h5 fancy index wants sorted unique for speed; restore order after
                order = np.argsort(idx)
                pix = f["pixels"][np.sort(idx)]
                pix = pix[np.argsort(order)]
                x = torch.stack([TF(p) for p in pix]).unsqueeze(1).to(device)  # (B,1,C,H,W)
                _ = model.encode({"pixels": x})
                if (bi + 1) % 10 == 0 or bi == 0:
                    m0 = bns[0]
                    print(f"  batch {bi+1}/{n_batches}  "
                          f"running_mean.std={m0.running_mean.std().item():.4f}  "
                          f"tracked={int(m0.num_batches_tracked)}", flush=True)

    for m, mom in zip(bns, old_mom):
        m.momentum = mom
    model.eval()


def within_traj_frame_std(model, env, device, n_eps=10, w=20, seed=0) -> float:
    """Quick collapse probe used before/after (eval mode)."""
    model.eval()
    with h5py.File(ROOT / "data" / f"{env}_eval.h5", "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        rng = np.random.default_rng(seed)
        eps = rng.choice(len(ep_len), size=min(n_eps, len(ep_len)), replace=False)
        stds = []
        with torch.no_grad():
            for ep in eps:
                L = int(ep_len[ep])
                goal = int(rng.integers(min(w, L - 1), L))
                idx = np.arange(goal - min(w, goal), goal + 1)
                frames = f["pixels"][ep_off[ep] + idx]
                x = torch.stack([TF(fr) for fr in frames]).unsqueeze(1).to(device)
                e = model.encode({"pixels": x})["emb"][:, 0].float().cpu().numpy()
                if e.ndim == 2:
                    e = e[:, None]
                stds.append(float(e.std(0).mean()))
    return float(np.mean(stds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["rope", "granular", "wall", "point_maze"])
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--n-batches", type=int, default=100, help="128×100 ≈ 12.8k frames")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--health", action="store_true", help="run repr_health before/after")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from scripts.repr_health import check  # noqa: WPS433

    rows = []
    for ckpt in args.ckpt:
        ckpt = Path(ckpt)
        print(f"\n=== {ckpt} ===", flush=True)
        model = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.to(args.device)
        before_bn = snapshot_bn(model)
        before_std = within_traj_frame_std(model, args.env, args.device)
        print(f"[before] within-traj frame_std={before_std:.4f}  BN layers={len(before_bn)}", flush=True)

        health_before = None
        if args.health:
            health_before = check(str(ckpt), args.env, 50, 0, args.device)
            flag = "OK " if health_before["healthy"] else "BAD"
            print(f"[before health] [{flag}] rho={health_before['rho_all_tokens'] if health_before['n_tokens']>1 else health_before['rho_global']} "
                  f"frame_std={health_before['frame_std']}", flush=True)

        reestimate(model, args.env, args.device, args.n_batches, args.batch_size, args.seed)
        after_std = within_traj_frame_std(model, args.env, args.device)
        after_bn = snapshot_bn(model)
        mean_shift = float(np.mean([
            (a[0] - b[0]).abs().mean().item() for a, b in zip(after_bn, before_bn)
        ]))
        print(f"[after ] within-traj frame_std={after_std:.4f}  "
              f"running_mean |Δ|={mean_shift:.4f}", flush=True)

        out = ckpt.with_name(ckpt.stem.replace("_object", "") + "_bnreest_object.ckpt")
        if out.name == ckpt.name:
            out = ckpt.with_name(ckpt.stem + "_bnreest.ckpt")
        # keep naming: foo_last_object.ckpt -> foo_last_bnreest_object.ckpt
        stem = ckpt.name
        if stem.endswith("_object.ckpt"):
            out = ckpt.with_name(stem[:-len("_object.ckpt")] + "_bnreest_object.ckpt")
        torch.save(model.cpu(), out)
        print(f"[saved] {out}", flush=True)

        health_after = None
        if args.health:
            health_after = check(str(out), args.env, 50, 0, args.device)
            flag = "OK " if health_after["healthy"] else "BAD"
            rho = health_after["rho_all_tokens"] if health_after["n_tokens"] > 1 else health_after["rho_global"]
            print(f"[after  health] [{flag}] rho={rho} frame_std={health_after['frame_std']} "
                  f"token_std={health_after['token_std']}", flush=True)

        rows.append({
            "ckpt": str(ckpt),
            "out": str(out),
            "before_within_frame_std": round(before_std, 4),
            "after_within_frame_std": round(after_std, 4),
            "bn_running_mean_abs_delta": round(mean_shift, 4),
            "health_before": health_before,
            "health_after": health_after,
        })

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
