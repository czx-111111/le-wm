"""Peak GPU memory of one LeWM training step (fwd + bwd + AdamW) per micro-batch.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_train_memory.py --mode s2 --batches 16 32 48 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SIGReg  # noqa: E402
from utils import build_lewm  # noqa: E402

MODES = {
    "cls": dict(token_mode="cls"),
    "c1": dict(token_mode="spatial16", num_spatial_tokens=16, pool_mode="avg"),
    "s1": dict(
        token_mode="spatial16",
        num_spatial_tokens=16,
        pool_mode="query",
        include_global_token=True,
    ),
    "s2": dict(
        token_mode="spatial16",
        num_spatial_tokens=16,
        pool_mode="query",
        include_global_token=True,
        pos_mode="nibble",
        inject_every_layer=True,
        n_lo_mode="attn",
        use_relative_bias=False,
        abs_mode="shared",
    ),
}


def step(model, sigreg, opt, b, t, act_dim, ctx_len, device, grad_ckpt):
    pixels = torch.randn(b, t, 3, 224, 224, device=device)
    action = torch.randn(b, t, act_dim, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.encode({"pixels": pixels, "action": action})
        emb, act_emb = out["emb"], out["act_emb"]
        n_hi, n_lo = out.get("n_hi"), out.get("n_lo")
        pred = model.predict(
            emb[:, :ctx_len],
            act_emb[:, :ctx_len],
            n_hi=n_hi[:, :ctx_len] if n_hi is not None else None,
            n_lo=n_lo[:, :ctx_len] if n_lo is not None else None,
        )
        pred_loss = (pred - emb[:, 1:]).pow(2).mean()
    if emb.ndim == 4:
        sig_in = rearrange(emb, "b t n d -> t (b n) d")
    else:
        sig_in = emb.transpose(0, 1)
    with torch.autocast("cuda", enabled=False):
        sig = sigreg(sig_in.float())
    loss = pred_loss + 0.05 * sig
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), default="s2")
    p.add_argument("--batches", type=int, nargs="+", default=[16, 32, 48, 64])
    p.add_argument("--act-dim", type=int, default=10)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--timed-steps", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda")
    model = build_lewm(
        action_dim=args.act_dim, history_size=args.history, **MODES[args.mode]
    ).to(device)
    model.train()
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    print(
        f"mode={args.mode} attn_impl="
        f"{getattr(model.encoder.config, '_attn_implementation', '?')} "
        f"grad_ckpt={args.grad_ckpt}"
    )
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    t = args.history + 1
    free, total = torch.cuda.mem_get_info()
    print(f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB")
    for b in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(2):
                step(model, sigreg, opt, b, t, args.act_dim, args.history, device, args.grad_ckpt)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            start = time.perf_counter()
            for _ in range(args.timed_steps):
                step(model, sigreg, opt, b, t, args.act_dim, args.history, device, args.grad_ckpt)
            torch.cuda.synchronize()
            per_sample_ms = (time.perf_counter() - start) / (args.timed_steps * b) * 1e3
            print(
                f"batch={b:4d} peak_alloc={peak:6.2f}GiB peak_reserved={reserved:6.2f}GiB "
                f"{per_sample_ms:.2f}ms/sample"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"batch={b:4d} OOM")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
