"""Depthwise causal conv1d NKI kernel, take 2: block layout.

Take 1 (causal_conv_dev.py) put ONE time-row per partition, i.e. a [128, C] tile,
and looped 16 times over T=2048. That structure cannot be made to fit SBUF:

  * a Python `range` tile loop is unrolled at trace time, and `nl.affine_range`
    behaves identically (measured: the fused backward fails with the *same*
    allocation id, _mem_279_276, either way -- the NKI docs do say all range
    iterators have identical effect in this version);
  * so all 16 tiles' scratch buffers are allocated at once. The forward
    (~19 KB/partition/tile) squeaked under the limit; the backward did not, and
    even a 4-buffer dx (7.7 KB/partition/tile, 123 KB total) failed on its very
    LAST allocation -- consistent with each [128, 640] tile being rounded up to a
    2 KB-granular slot (1280 -> 2048, 2560 -> 4096), which turns 123 KB into
    exactly 192 KB = the whole SBUF.

Take 2 removes the tile loop instead of fighting it. A tile is [P, m, C]: the
128 partitions each hold m CONSECUTIVE time rows, so the tile covers P*m rows
and, because x is [B, T, C] row-major, the tile is one fully contiguous HBM
block. The consequence that matters:

    shifting the tile by `s` time rows == starting the same contiguous load
    s*C elements earlier.

Every partition then automatically picks up the rows it needs from what would
have been its neighbour's rows -- there is no cross-partition data movement, no
transpose, and no per-tap tile bookkeeping. With m = 16 the whole T = 2048
sequence is ONE tile, so the trace has no tile loop at all and SBUF holds one
block's buffers (123 KB) instead of sixteen tiles' worth.

Zero-padding (the thing that makes the op causal) falls out of the same
arithmetic: a tap's tile covers sequence rows [start, start + P*m); the rows
outside [0, T) are exactly the ones to memset, and _rects() splits that row
range into at most three rectangles of the [P, m, C] tile, each of which is
contiguous in both HBM and SBUF. Nothing is special-cased on "first"/"last"
tile.

dw still needs a reduction along the partition axis, which no vector op can do,
so it runs on the (otherwise idle) PE: stationary = a [P, <=128] channel chunk
of the tap product, moving = ones [P, 1], dst = [<=128, 1] in PSUM, accumulated
over the m row-groups and over blocks by repeated nc_matmul into the same PSUM
region. That comes out with channels on the partition axis, i.e. dw is produced
directly in the [C, K] layout the eager parameter wants, and only C*K numbers
(80 bytes/partition) have to survive.

Run:  python research/scratch/causal_conv_dev2.py [--quick] [--split]
"""
import argparse
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
# Largest fp32 tile the block layout may use, per partition. Two fp32 tiles
# (accumulator + product) plus four bf16 tiles have to coexist inside SBUF's
# 192 KB/partition, with room for the taps.
_MAX_FP32_TILE = 48 * 1024


def _pick_m(T, C, P=_P):
    """Time rows per partition. m must divide T // P so that P*m tiles the
    sequence exactly, and m*C*4 bytes must stay a reasonable fp32 tile."""
    nt = T // P
    m = 1
    for cand in (1, 2, 4, 8, 16, 32):
        if nt % cand == 0 and cand <= nt and cand * C * 4 <= _MAX_FP32_TILE \
                and cand * C <= 32767:
            m = cand
    return m


def _rects(lo, hi, P, m):
    """Split the tile-row range [lo, hi) of a [P, m, C] tile into at most three
    (p0, p1, j0, j1) rectangles, each contiguous in both HBM and SBUF: a partial
    first partition, a run of whole partitions, a partial last partition."""
    if hi <= lo:
        return []
    p0, j0 = lo // m, lo % m
    p1, j1 = (hi - 1) // m, (hi - 1) % m + 1
    if p0 == p1:
        return [(p0, p0 + 1, j0, j1)]
    out = []
    if j0:
        out.append((p0, p0 + 1, j0, m))
        p0 += 1
    if j1 != m:
        out.append((p1, p1 + 1, 0, j1))
    else:
        p1 += 1
    if p0 < p1:
        out.append((p0, p1, 0, m))
    return out


def _load_rows(dst, src, b, T, start, P, m, C):
    """dst [P, m, C] <- sequence b's rows [start, start + P*m), zero outside
    [0, T). One contiguous DMA in the common case; the boundary blocks add a
    memset and at most two more DMAs."""
    R = P * m
    lo = max(0, -start)                 # tile rows before the sequence begins
    hi = min(R, T - start)              # first tile row past its end
    for (p0, p1, j0, j1) in _rects(0, lo, P, m) + _rects(hi, R, P, m):
        nisa.memset(dst=dst[p0:p1, j0:j1, 0:C], value=0.0)
    for (p0, p1, j0, j1) in _rects(lo, hi, P, m):
        off = (b * T + start + p0 * m + j0) * C
        nisa.dma_copy(dst=dst[p0:p1, j0:j1, 0:C],
                      src=src.ap(pattern=[[m * C, p1 - p0], [C, j1 - j0], [1, C]],
                                 offset=off))


def _store_rows(dst, src, b, T, start, P, m, C):
    """sequence b's rows [start, start + P*m) <- src [P, m, C]."""
    nisa.dma_copy(dst=dst.ap(pattern=[[m * C, P], [C, m], [1, C]],
                             offset=(b * T + start) * C),
                  src=src)


def _taps(wt, K, C, P, dt):
    """wt [K, C] -> [P, K, C] SBUF: stride 0 over the partition dim, so the DMA
    re-reads the same K*C elements for every partition."""
    wb = nl.ndarray((P, K, C), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb, src=wt.ap(pattern=[[0, P], [C, K], [1, C]]))
    return wb


def _tap_view(wb, k, K, C, P, m):
    """Tap k as a [P, m, C] operand: broadcast along partitions (stride K*C is
    the real per-partition stride) and along the m row-groups (stride 0)."""
    return wb.ap(pattern=[[K * C, P], [0, m], [1, C]], offset=k * C)


@nki.jit
def _causal_conv_fwd(x, wt):
    """y[b, t, c] = sum_k wt[k, c] * x[b, t - (K-1) + k, c]  (zero before t = 0).

    x [B, T, C] with T % 128 == 0, wt [K, C], both in x.dtype. Returns [B, T, C].
    """
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    P = _P
    m = _pick_m(T, C, P)
    R = P * m
    out = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, P, dt)
    for b in range(B):
        for bl in range(T // R):
            base = bl * R
            acc = nl.ndarray((P, m, C), dtype=nl.float32, buffer=nl.sbuf)
            ob = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
            for k in range(K):
                xs = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
                prod = nl.ndarray((P, m, C), dtype=nl.float32, buffer=nl.sbuf)
                _load_rows(xs, x, b, T, base - (K - 1 - k), P, m, C)
                # fp32 accumulator: the only rounding to dt is on the final add,
                # which is what makes y bit-exact against eager, not merely close.
                nisa.tensor_tensor(dst=(acc if k == 0 else prod), data1=xs,
                                   data2=_tap_view(wb, k, K, C, P, m), op=nl.multiply)
                if k > 0:
                    nisa.tensor_tensor(dst=(ob if k == K - 1 else acc), data1=acc,
                                       data2=prod, op=nl.add)
            if K == 1:
                nisa.tensor_copy(dst=ob, src=acc)
            _store_rows(out, ob, b, T, base, P, m, C)
    return out


def _dx_block(g, wb, dx, b, T, base, K, C, P, m, dt):
    """dx[n] = sum_a wt[K-1-a] * g[n + a] for one block; returns the a == 0 g
    tile, which the dw half needs anyway."""
    acc = nl.ndarray((P, m, C), dtype=nl.float32, buffer=nl.sbuf)
    ob = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
    g0 = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
    for a in range(K):
        ga = g0 if a == 0 else nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
        prod = nl.ndarray((P, m, C), dtype=nl.float32, buffer=nl.sbuf)
        _load_rows(ga, g, b, T, base + a, P, m, C)
        nisa.tensor_tensor(dst=(acc if a == 0 else prod), data1=ga,
                           data2=_tap_view(wb, K - 1 - a, K, C, P, m), op=nl.multiply)
        if a > 0:
            nisa.tensor_tensor(dst=(ob if a == K - 1 else acc), data1=acc,
                               data2=prod, op=nl.add)
    if K == 1:
        nisa.tensor_copy(dst=ob, src=acc)
    _store_rows(dx, ob, b, T, base, P, m, C)
    return g0


def _dw_block(x, g0, ones, ps, b, T, base, K, C, P, m, cs, NG, dt):
    """Add one block's contribution to dw[k, c] = sum_n g[n, c]*x[n-(K-1)+k, c].
    The sum over the 128 time partitions runs on the PE; the sum over the m row
    groups and over blocks is PSUM accumulation into the same region."""
    for k in range(K):
        xs = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
        pr = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
        _load_rows(xs, x, b, T, base - (K - 1 - k), P, m, C)
        nisa.tensor_tensor(dst=pr, data1=g0, data2=xs, op=nl.multiply)
        for jj in range(m):
            for jg in range(NG):
                # dst = stationary.T @ moving: sums over partitions (time) and
                # lands with the channel index on the partition axis.
                nisa.nc_matmul(dst=ps[0:cs, jg, k:k + 1],
                               stationary=pr[0:P, jj, jg * cs:(jg + 1) * cs],
                               moving=ones, is_moving_onezero=True)


@nki.jit
def _causal_conv_bwd(x, g, wt):
    """dx and dw of _causal_conv_fwd.

      dx[b, n, c] = sum_a wt[K-1-a, c] * g[b, n + a, c]   (g zero past t = T-1)
      dw[k, c]    = sum_{b,n} g[b, n, c] * x[b, n - (K-1) + k, c]

    Returns (dx [B, T, C] in x.dtype, dw [C, K] in wt.dtype -- already in the
    eager parameter's channel-major layout).
    """
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    P = _P
    m = _pick_m(T, C, P)
    R = P * m
    cs = min(P, C)
    NG = C // cs
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    dw = nl.ndarray((C, K), dtype=wt.dtype, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, P, dt)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps = nl.ndarray((cs, NG, K), dtype=nl.float32, buffer=nl.psum)
    for b in range(B):
        for bl in range(T // R):
            base = bl * R
            g0 = _dx_block(g, wb, dx, b, T, base, K, C, P, m, dt)
            _dw_block(x, g0, ones, ps, b, T, base, K, C, P, m, cs, NG, dt)
    sb = nl.ndarray((cs, NG, K), dtype=wt.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    for jg in range(NG):
        nisa.dma_copy(dst=dw[jg * cs:(jg + 1) * cs, 0:K], src=sb[0:cs, jg, 0:K])
    return dx, dw


@nki.jit
def _causal_conv_dx(g, wt):
    """dx only, for the --split fallback (two smaller kernels instead of one)."""
    B, T, C = g.shape
    K = wt.shape[0]
    dt = g.dtype
    P = _P
    m = _pick_m(T, C, P)
    R = P * m
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, P, dt)
    for b in range(B):
        for bl in range(T // R):
            _dx_block(g, wb, dx, b, T, bl * R, K, C, P, m, dt)
    return dx


@nki.jit
def _causal_conv_dw(x, g, K):
    """dw only, for the --split fallback."""
    B, T, C = x.shape
    dt = x.dtype
    P = _P
    m = _pick_m(T, C, P)
    R = P * m
    cs = min(P, C)
    NG = C // cs
    dw = nl.ndarray((C, K), dtype=dt, buffer=nl.shared_hbm)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps = nl.ndarray((cs, NG, K), dtype=nl.float32, buffer=nl.psum)
    for b in range(B):
        for bl in range(T // R):
            base = bl * R
            g0 = nl.ndarray((P, m, C), dtype=dt, buffer=nl.sbuf)
            _load_rows(g0, g, b, T, base, P, m, C)
            _dw_block(x, g0, ones, ps, b, T, base, K, C, P, m, cs, NG, dt)
    sb = nl.ndarray((cs, NG, K), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    for jg in range(NG):
        nisa.dma_copy(dst=dw[jg * cs:(jg + 1) * cs, 0:K], src=sb[0:cs, jg, 0:K])
    return dw


_SPLIT = False


class _CausalConvNKI(torch.autograd.Function):
    """Differentiable wrapper: x [B, T, C], wt [K, C] (both contiguous)."""

    @staticmethod
    def forward(ctx, x, wt):
        y = _causal_conv_fwd(x, wt)
        ctx.save_for_backward(x, wt)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, wt = ctx.saved_tensors
        g = grad_out.contiguous()
        if _SPLIT:
            dx = _causal_conv_dx(g, wt)
            dw_ck = _causal_conv_dw(x, g, wt.shape[0])
        else:
            dx, dw_ck = _causal_conv_bwd(x, g, wt)
        return dx, dw_ck.t()          # [C, K] -> [K, C], a view; no extra kernel


# -----------------------------------------------------------------------------
# reference + checks

def eager_ref(x, w):
    """train.py CausalConv.forward, verbatim."""
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (w.shape[-1] - 1, 0))
    y = F.conv1d(xt, w.to(dtype=x.dtype), groups=x.shape[-1])
    return y.transpose(1, 2)


FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def rel_l2(got, ref):
    return (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)


def report(tag, got, ref):
    d = (got.float() - ref.float()).abs()
    ex = (got.float() == ref.float()).float().mean().item() * 100
    cs = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"   {tag:3s}: max_abs={d.max().item():.3e}  bit_exact={ex:6.2f}%  "
          f"rel_l2={rel_l2(got, ref):.3e}  cos={cs:.9f}")


def case(name, B, T, C, K, dt, scale=1.0, gscale=1.0, seed=0, dw_tol=6e-3):
    torch.manual_seed(seed)
    x = (torch.randn(B, T, C) * scale).to(dt)
    w = torch.randn(C, 1, K) * 0.5           # fp32 param, like train.py
    g = (torch.randn(B, T, C) * gscale).to(dt)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    yr = eager_ref(xr, wr)
    yr.backward(g)

    # fp32 gold for dw: eager's own bf16 dw is not a reference (F.conv1d returns
    # the weight grad in the input dtype), so judge against the fp32 chain and
    # report how far eager itself lands from it.
    xg = x.float().clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    eager_ref(xg, wg).backward(g.float())
    dw_gold = wg.grad.detach()

    xk = x.to("neuron").clone().requires_grad_(True)
    wk = w.view(C, K).t().contiguous().to(dt).to("neuron").clone().requires_grad_(True)
    yk = _CausalConvNKI.apply(xk, wk)
    yk.backward(g.to("neuron"))
    got_y = yk.detach().cpu()
    got_dx = xk.grad.detach().cpu()
    # wk.grad is [K, C] (the wrapper transposed the kernel's [C, K]) -> param layout
    got_dw = wk.grad.detach().cpu().float().t().reshape(C, 1, K)

    print(f"{name}  B={B} T={T} C={C} K={K} {dt}  m={_pick_m(T, C)}")
    report("y", got_y, yr.detach())
    report("dx", got_dx, xr.grad.detach())
    report("dw", got_dw, wr.grad.detach())
    if dt == torch.bfloat16:
        check(f"{name}: y bit-exact vs eager", bool((got_y == yr.detach()).all()))
        check(f"{name}: dx bit-exact vs eager", bool((got_dx == xr.grad.detach()).all()))
    else:
        check(f"{name}: y rel_l2 <= 1e-6", rel_l2(got_y, yr.detach()) <= 1e-6,
              f"{rel_l2(got_y, yr.detach()):.3e}")
        check(f"{name}: dx rel_l2 <= 1e-6", rel_l2(got_dx, xr.grad.detach()) <= 1e-6,
              f"{rel_l2(got_dx, xr.grad.detach()):.3e}")
    e_k = rel_l2(got_dw, dw_gold)
    e_e = rel_l2(wr.grad.detach(), dw_gold)
    print(f"   dw vs fp32 gold: kernel={e_k:.3e}  eager({dt})={e_e:.3e}")
    check(f"{name}: dw rel_l2 vs fp32 gold <= {dw_tol:.0e}", e_k <= dw_tol, f"{e_k:.3e}")
    if dt == torch.bfloat16:
        check(f"{name}: dw no worse than 2.5x eager's own bf16 error",
              e_k <= 2.5 * max(e_e, 1e-12), f"{e_k:.3e} vs {e_e:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="real shape only")
    ap.add_argument("--split", action="store_true", help="two bwd kernels instead of one")
    args = ap.parse_args()
    _SPLIT = args.split
    print(f"backward: {'split dx/dw kernels' if _SPLIT else 'single fused kernel'}")

    case("REAL SHAPE bf16", 1, 2048, 640, 4, torch.bfloat16)
    if not args.quick:
        case("fp32 exact-math", 1, 256, 64, 4, torch.float32, dw_tol=1e-5)
        case("B=3 (batch boundary)", 3, 256, 64, 4, torch.bfloat16)
        case("T==128 (single row per partition)", 2, 128, 64, 4, torch.bfloat16)
        case("K=1", 1, 256, 64, 1, torch.bfloat16)
        case("K=2", 1, 256, 64, 2, torch.bfloat16)
        case("K=8 (pad wider than m)", 1, 256, 64, 8, torch.bfloat16)
        case("large x (x*40)", 1, 512, 128, 4, torch.bfloat16, scale=40.0, gscale=5.0, seed=3)
        case("tiny x (x*1e-3)", 1, 512, 128, 4, torch.bfloat16, scale=1e-3, gscale=1e-3, seed=4)

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)
