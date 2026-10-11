import json, math, re, random
from pathlib import Path

O = Path("/data/czx/le-wm/first/outputs")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_det(name):
    d = json.load(open(O / name))["details"]
    return {(e["ep_id"], e["start_step"]): e for e in d}


def succ_table(env, files):
    print(f"\n==== {env} ====")
    dets = {k: load_det(v) for k, v in files.items()}
    keys0 = list(next(iter(dets.values())).keys())
    for k, d in dets.items():
        same = set(d.keys()) == set(keys0)
        n = len(d)
        s = sum(e["success"] for e in d.values())
        early = sum(e.get("mpc_iters", 1) == 0 for e in d.values())
        lo, hi = wilson(s, n)
        print(f"{k:28s} n={n} succ={s:3d}  95%CI=[{lo:4.1f},{hi:4.1f}]  early={early:2d} same_eps={same}")
    return dets


def paired(dets, a, b):
    A, B = dets[a], dets[b]
    ks = [k for k in A if k in B]
    b_ = sum(A[k]["success"] and not B[k]["success"] for k in ks)
    c_ = sum((not A[k]["success"]) and B[k]["success"] for k in ks)
    both = sum(A[k]["success"] == B[k]["success"] for k in ks)
    print(f"  {a:>24s} vs {b:<24s}: A-only={b_:2d} B-only={c_:2d} agree={both}/{len(ks)}  McNemar p={mcnemar_exact(b_, c_):.3f}")


wall = succ_table("wall (g25, MPC 2/10)", {
    "S2 old100 best": "wall_s2_details_n100_g25_t2i10.json",
    "S2 old100 ep100": "wall_s2_details_n100_g25_t2i10_ep100.json",
    "S2 new150 last": "wall_s2_details_n100_g25_t2i10_ep150.json",
    "S2 new150 best(ep147)": "wall_s2_details_n100_g25_t2i10_ep150_best.json",
    "S2 old100 best dense": "wall_s2_details_n100_g25_t2i10_dense.json",
    "CLS best": "wall_cls_details_n100_g25_t2i10.json",
    "CLS ep100": "wall_cls_details_n100_g25_t2i10_ep100.json",
})
for a, b in [("S2 old100 ep100", "S2 new150 last"), ("S2 new150 last", "S2 new150 best(ep147)"),
             ("S2 old100 best", "S2 old100 ep100"), ("CLS best", "CLS ep100"), ("S2 new150 last", "CLS best")]:
    paired(wall, a, b)

pm = {
    "S2 old ep3": "point_maze_s2_details_n100_g25_t5i10_ep3.json",
    "S2 old ep10": "point_maze_s2_details_n100_g25_t5i10_ep10.json",
    "S2 old ep20": "point_maze_s2_details_n100_g25_t5i10_ep20.json",
    "S2 old ep30": "point_maze_s2_details_n100_g25_t5i10_ep30.json",
    "S2 old ep40": "point_maze_s2_details_n100_g25_t5i10_ep40.json",
    "S2 old ep45": "point_maze_s2_details_n100_g25_t5i10_ep45.json",
    "S2 old ep50": "point_maze_s2_details_n100_g25_t5i10_ep50.json",
    "S2 old ep59": "point_maze_s2_details_n100_g25_t5i10_ep59.json",
    "S2 old ep60": "point_maze_s2_details_n100_g25_t5i10_ep60.json",
    "S2 new150 ep100": "point_maze_s2_details_n100_g25_t5i10_ep100.json",
    "S2 new150 last": "point_maze_s2_details_n100_g25_t5i10_ep150.json",
    "S2 new150 best(ep2)": "point_maze_s2_details_n100_g25_t5i10_ep150_best.json",
    "CLS best": "point_maze_cls_details_n100_g25_t5i10.json",
    "CLS ep100": "point_maze_cls_details_n100_g25_t5i10_ep100.json",
}
point = succ_table("point (g25, MPC 5/10)", pm)
for a, b in [("S2 old ep59", "S2 old ep60"), ("S2 old ep50", "S2 old ep59"), ("S2 new150 ep100", "S2 new150 last"),
             ("S2 old ep60", "S2 new150 last"), ("CLS best", "CLS ep100")]:
    paired(point, a, b)


def chamfers(name):
    txt = (O / name).read_text()
    eps = re.findall(r"\[sim-worker \d+/\d+\] ep=(\d+) start=(\d+)\s*\n\s*chamfer=([\d.]+)", txt)
    return {(int(e), int(s)): float(c) for e, s, c in eps}


def boot_ci(xs, B=10000, seed=0):
    r = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def cham_table(env, files):
    print(f"\n==== {env} (Chamfer, lower better) ====")
    C = {k: chamfers(v) for k, v in files.items()}
    for k, d in C.items():
        xs = list(d.values())
        lo, hi = boot_ci(xs)
        s = sum(x < 1.0 for x in xs)
        med = sorted(xs)[len(xs) // 2]
        print(f"{k:24s} n={len(xs)} mean={sum(xs)/len(xs):.3f} CI=[{lo:.2f},{hi:.2f}] succ={s}/{len(xs)} eps={sorted(d)[:3]}...")
    return C


def paired_cham(C, a, b):
    A, B = C[a], C[b]
    ks = [k for k in A if k in B]
    diff = [B[k] - A[k] for k in ks]
    lo, hi = boot_ci(diff)
    print(f"  {a:>20s} -> {b:<20s}: n={len(ks)} mean diff={sum(diff)/len(diff):+.3f}  95%CI=[{lo:+.2f},{hi:+.2f}]")


rope = cham_table("rope", {
    "S2 old100 best": "rope_s2_norel_sim_gpu0.log",
    "S2 old100 ep100": "rope_s2_norel_eval_n10_ep100.log",
    "S2 new150 last": "rope_s2_norel_eval_n10_ep150.log",
    "S2 new150 best(ep28)": "rope_s2_norel_eval_n10_ep150_best.log",
    "S2 old best dense": "rope_s2_norel_eval_n10_dense.log",
    "CLS best": "rope_cls_eval10.log",
    "CLS ep100": "rope_cls_eval_n10_ep100_retry.log",
})
for a, b in [("S2 old100 ep100", "S2 new150 last"), ("S2 new150 last", "S2 new150 best(ep28)"),
             ("CLS best", "CLS ep100"), ("CLS best", "S2 new150 last")]:
    paired_cham(rope, a, b)

gran = cham_table("granular", {
    "S2 old100 best": "granular_s2_norel_eval10.log",
    "S2 old100 ep100": "granular_s2_norel_eval_n10_ep100.log",
    "S2 new150 last": "granular_s2_norel_eval_n10_ep150.log",
    "S2 new150 best(ep4)": "granular_s2_norel_eval_n10_ep150_best.log",
    "S2 old best dense": "granular_s2_norel_eval_n10_dense.log",
    "CLS best": "granular_cls_eval10.log",
    "CLS ep100": "granular_cls_eval_n10_ep100_retry.log",
})
for a, b in [("S2 old100 ep100", "S2 new150 last"), ("S2 new150 last", "S2 new150 best(ep4)"),
             ("CLS best", "CLS ep100"), ("CLS best", "S2 new150 last")]:
    paired_cham(gran, a, b)
