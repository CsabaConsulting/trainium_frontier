"""Offline (CPU, no device) check of the BLOCK-layout causal-conv algorithm.

causal_conv_math.py did this for the one-row-per-partition layout. That layout
could not be made to fit SBUF (see the docstring of causal_conv_dev2.py), so the
kernel now uses [P, m, C] tiles: partition p holds m consecutive time rows, the
tile is one contiguous HBM block, and a causal shift of s rows is just starting
the load s*C elements earlier.

The whole correctness risk of that change is index arithmetic: which tile rows
fall outside [0, T) and therefore have to be memset, and whether the rectangle
decomposition of the in-range row span (_rects, imported from the kernel module
so this tests the REAL code) covers the tile exactly once. So this script

  1. brute-forces _rects over many (lo, hi, P, m): every rectangle contiguous,
     no overlaps, no gaps, exact cover;
  2. rebuilds tiles exactly the way _load_rows does (memset rects, then copy
     rects) and asserts the result equals the mathematical definition
     tile[p, jj] = seq[start + p*m + jj] with zeros outside the sequence -- with
     NaN-filled tiles so any untouched element is caught;
  3. simulates fwd / dx / dw on those tiles, in the kernel's dtypes and
     accumulation order, and compares against plain torch autograd on
     train.py's CausalConv.forward.

Run: python research/scratch/causal_conv_block_math.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from causal_conv_dev2 import _pick_m, _rects  # the kernel's own helpers

P = 128
FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------- 1. _rects

def rects_ok(P_, m):
    """Every [lo, hi) must be covered exactly once by contiguous rectangles."""
    R = P_ * m
    for lo in range(0, R + 1):
        for hi in (lo, lo + 1, lo + m, lo + m + 1, R - 1, R):
            if hi < lo or hi > R:
                continue
            seen = torch.zeros(R, dtype=torch.int32)
            for (p0, p1, j0, j1) in _rects(lo, hi, P_, m):
                if not (0 <= p0 < p1 <= P_ and 0 <= j0 < j1 <= m):
                    return f"bad rect {(p0, p1, j0, j1)} for [{lo},{hi}) m={m}"
                if p1 - p0 > 1 and (j0, j1) != (0, m):
                    return f"non-contiguous rect {(p0, p1, j0, j1)}"
                for p in range(p0, p1):
                    seen[p * m + j0:p * m + j1] += 1
            want = torch.zeros(R, dtype=torch.int32)
            want[lo:hi] = 1
            if not torch.equal(seen, want):
                return f"cover mismatch for [{lo},{hi}) m={m}: {_rects(lo, hi, P_, m)}"
    return None


for _m in (1, 2, 4, 8, 16):
    err = rects_ok(8, _m)          # small P, exhaustive over lo/hi
    check(f"_rects exact cover, P=8 m={_m}", err is None, err or "")
check("_rects exact cover, P=128 m=16", rects_ok(128, 16) is None)


# ------------------------------------------------- 2. tile build == definition

def build_tile(seq, start, m, C, dt):
    """Mirror of _load_rows: memset the out-of-range rectangles, copy the rest."""
    T = seq.shape[0]
    R = P * m
    tile = torch.full((P, m, C), float("nan"), dtype=torch.float32)
    lo, hi = max(0, -start), min(R, T - start)
    for (p0, p1, j0, j1) in _rects(0, lo, P, m) + _rects(hi, R, P, m):
        tile[p0:p1, j0:j1, :] = 0.0
    for (p0, p1, j0, j1) in _rects(lo, hi, P, m):
        r0 = start + p0 * m + j0
        n = (p1 - p0) * (j1 - j0)          # rects are contiguous, so this is exact
        tile[p0:p1, j0:j1, :] = seq[r0:r0 + n].reshape(p1 - p0, j1 - j0, C).float()
    assert not torch.isnan(tile).any(), f"tile has untouched elements, start={start}"
    return tile.to(dt)


def tile_ref(seq, start, m, C, dt):
    T = seq.shape[0]
    out = torch.zeros(P, m, C, dtype=dt)
    for p in range(P):
        for jj in range(m):
            r = start + p * m + jj
            if 0 <= r < T:
                out[p, jj] = seq[r]
    return out


torch.manual_seed(0)
for (T_, C_, m_) in ((2048, 8, 16), (256, 4, 2), (128, 4, 1), (512, 4, 4)):
    seq = torch.randn(T_, C_).to(torch.bfloat16)
    ok = True
    for start in (0, -1, -3, -7, 1, 3, 7, T_ - P * m_, T_ - P * m_ + 3):
        if start + P * m_ > T_ + P * m_:
            continue
        ok = ok and torch.equal(build_tile(seq, start, m_, C_, torch.bfloat16),
                                tile_ref(seq, start, m_, C_, torch.bfloat16))
    check(f"tile build == definition (T={T_} C={C_} m={m_})", ok)


# --------------------------------------------------------- 3. fwd / dx / dw

def eager_ref(x, w):
    """train.py CausalConv.forward, verbatim."""
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (w.shape[-1] - 1, 0))
    y = F.conv1d(xt, w.to(dtype=x.dtype), groups=x.shape[-1])
    return y.transpose(1, 2)


def sim(x, g, wt, m):
    """Kernel simulation. x/g [B, T, C], wt [K, C]; returns y, dx, dw[K, C]."""
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    R = P * m
    y = torch.empty_like(x)
    dx = torch.empty_like(x)
    dw = torch.zeros(K, C, dtype=torch.float32)     # PSUM accumulator, fp32
    for b in range(B):
        for base in range(0, T, R):
            # --- fwd
            acc = torch.zeros(P, m, C, dtype=torch.float32)
            for k in range(K):
                xs = build_tile(x[b], base - (K - 1 - k), m, C, dt)
                acc += xs.float() * wt[k].float().expand(P, m, C)
            y[b, base:base + R] = acc.to(dt).reshape(R, C)
            # --- dx
            acc = torch.zeros(P, m, C, dtype=torch.float32)
            for a in range(K):
                ga = build_tile(g[b], base + a, m, C, dt)
                acc += ga.float() * wt[K - 1 - a].float().expand(P, m, C)
            dx[b, base:base + R] = acc.to(dt).reshape(R, C)
            # --- dw (the tap product is rounded to dt: it feeds a bf16 PE matmul)
            g0 = build_tile(g[b], base, m, C, dt)
            for k in range(K):
                xs = build_tile(x[b], base - (K - 1 - k), m, C, dt)
                pr = (g0 * xs).to(dt)
                dw[k] += pr.float().sum(dim=(0, 1))
    return y, dx, dw


def rel_l2(got, ref):
    return (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)


def case(name, B, T, C, K, dt, scale=1.0, gscale=1.0, seed=0, dw_tol=6e-3):
    torch.manual_seed(seed)
    x = (torch.randn(B, T, C) * scale).to(dt)
    w = torch.randn(C, 1, K) * 0.5
    g = (torch.randn(B, T, C) * gscale).to(dt)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    eager_ref(xr, wr).backward(g)
    xg = x.float().clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    eager_ref(xg, wg).backward(g.float())
    dw_gold = wg.grad.detach()

    wt = w.view(C, K).t().contiguous().to(dt)
    m = _pick_m(T, C)
    ys, dxs, dws = sim(x, g, wt, m)
    dws_param = dws.to(dt).float().t().reshape(C, 1, K)

    yref = eager_ref(x.clone().requires_grad_(True), w.clone()).detach()
    print(f"{name}  B={B} T={T} C={C} K={K} {dt} m={m}")
    print(f"   y  rel_l2={rel_l2(ys, yref):.3e}   dx rel_l2={rel_l2(dxs, xr.grad):.3e}")
    if dt == torch.bfloat16:
        check(f"{name}: y bit-exact vs eager", bool((ys == yref).all()))
        check(f"{name}: dx bit-exact vs eager", bool((dxs == xr.grad.detach()).all()))
    else:
        check(f"{name}: y rel_l2 <= 1e-6", rel_l2(ys, yref) <= 1e-6)
        check(f"{name}: dx rel_l2 <= 1e-6", rel_l2(dxs, xr.grad.detach()) <= 1e-6)
    e_s = rel_l2(dws_param, dw_gold)
    e_e = rel_l2(wr.grad.detach(), dw_gold)
    print(f"   dw vs fp32 gold: kernel={e_s:.3e}  eager({dt})={e_e:.3e}")
    check(f"{name}: dw rel_l2 vs fp32 gold <= {dw_tol:.0e}", e_s <= dw_tol, f"{e_s:.3e}")
    if dt == torch.bfloat16:
        check(f"{name}: dw no worse than 2.5x eager's own bf16 error",
              e_s <= 2.5 * max(e_e, 1e-12), f"{e_s:.3e} vs {e_e:.3e}")


print("=" * 78)
print("depthwise causal conv1d, BLOCK layout -- offline algorithm check (CPU only)")
print("=" * 78)
case("fp32 exact-math", 1, 256, 64, 4, torch.float32, dw_tol=1e-5)
case("REAL SHAPE bf16", 1, 2048, 640, 4, torch.bfloat16)
case("B=3 (batch boundary)", 3, 256, 64, 4, torch.bfloat16)
case("T==128 (m == 1)", 2, 128, 64, 4, torch.bfloat16)
case("K=1", 1, 256, 64, 1, torch.bfloat16)
case("K=2", 1, 256, 64, 2, torch.bfloat16)
case("K=8 (pad wider than m)", 1, 256, 64, 8, torch.bfloat16)
case("large x (x*40)", 1, 512, 128, 4, torch.bfloat16, scale=40.0, gscale=5.0, seed=3)
case("tiny x (x*1e-3)", 1, 512, 128, 4, torch.bfloat16, scale=1e-3, gscale=1e-3, seed=4)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
