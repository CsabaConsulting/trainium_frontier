"""Dev harness for the fused depthwise-causal-conv1d NKI kernel (fwd + bwd).

Stage 2 (stage 1 was research/scratch/causal_conv_math.py, which validated the
derivation and the exact tiling/zero-padding structure on CPU with no device).
Here the kernel itself is written and compiled, and its output is compared to a
CPU eager reference (`CausalConv.forward` verbatim, under plain autograd).

Layout choice (measured constraints, not taste):
  * time on the 128 partitions, channels on the free axis. HBM x is [B, T, C]
    row-major, so every tile load is one contiguous [128, C] block -- no
    transposes anywhere, and the causal shift is just a row offset in the DMA.
  * the taps are the only thing that has to be broadcast along partitions (they
    are constant in t, varying in c). That is one DMA with a stride-0 partition
    pattern, done once per kernel invocation.
  * dw needs a reduction ALONG partitions, which no vector op can do, so each
    tile's tap products are reduced on the (otherwise idle) PE: stationary =
    the product tile's channel chunk [128, <=128], moving = ones [128, 1], so
    dst = [<=128, 1] in PSUM. That comes out with channels on the partition
    axis, i.e. dw is produced directly in the [C, K] layout the eager parameter
    wants, and only [C, K] numbers (80 bytes/partition) have to survive across
    tiles.

Loop structure (this is the part that took the measurements, see conv_probe.py):
  * the tile loop MUST be nl.affine_range. A plain Python `range` is unrolled at
    trace time and every tile gets its own SBUF allocation -- 16 tiles x a
    handful of [128, 640] tiles exhausts SBUF ("COMPILATION FAILED: Memory
    Location: {inst__I-4-0:_mem_139_139}@SB<0,0>(128x1280)"). Curiously the
    forward squeaked under the limit while dx did not, and a bisect (dx3 in
    conv_probe.py) showed the deciding factor is that dx reads g *ahead* of the
    current tile while the forward reads x *behind* it; unrolled read-ahead
    costs more live SBUF. affine_range removes the issue for both.
  * so the loop body must be index-uniform: the two tiles that need
    zero-padding (the FIRST tile of a sequence, whose x taps run off the start,
    and the LAST tile, whose g taps run off the end) are emitted explicitly
    outside the loop, which then covers only interior tiles.
  * buffers are allocated INSIDE the body. Hoisting them out makes every write a
    fresh SSA value that is never freed ("_mem_4_4_SsaClone61").

Run:  python research/scratch/causal_conv_dev.py [--quick]
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


def _load_taps(wt, K, C, P, dt):
    """Replicate wt [K, C] to every one of the P partitions -> [P, K, C] SBUF.
    Stride 0 over the partition dim: the DMA re-reads the same K*C HBM elements
    for each partition."""
    wb = nl.ndarray((P, K, C), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb, src=wt.ap(pattern=[[0, P], [C, K], [1, C]]))
    return wb


def _fwd_tile(x, wb, out, b, t0, K, C, P, dt, first):
    """One 128-row output tile of y[n] = sum_k wt[k] * x[n - (K-1) + k]."""
    acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
    ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
    for k in range(K):
        sh = K - 1 - k                        # this tap reads sh rows back
        xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
        prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        if first and sh > 0:
            nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
            nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
        else:
            nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
        # fp32 accumulator: the only rounding to dt is on the final add, which is
        # what makes y bit-exact against eager rather than merely close.
        nisa.tensor_tensor(dst=(acc if k == 0 else prod), data1=xs,
                           data2=wb[0:P, k, 0:C], op=nl.multiply)
        if k > 0:
            nisa.tensor_tensor(dst=(ob if k == K - 1 else acc), data1=acc,
                               data2=prod, op=nl.add)
    if K == 1:
        nisa.tensor_copy(dst=ob, src=acc)
    nisa.dma_copy(dst=out[b, t0:t0 + P, 0:C], src=ob)


@nki.jit
def _causal_conv_fwd(x, wt):
    """y[b, t, c] = sum_k wt[k, c] * x[b, t - (K-1) + k, c]  (zero before t=0).

    x [B, T, C] with T % 128 == 0, wt [K, C], both in x.dtype. Returns [B, T, C].
    """
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    P = _P
    NT = T // P
    out = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _load_taps(wt, K, C, P, dt)
    for b in range(B):
        _fwd_tile(x, wb, out, b, 0, K, C, P, dt, True)      # boundary tile
        if NT > 1:
            for i in nl.affine_range(NT - 1):
                _fwd_tile(x, wb, out, b, (i + 1) * P, K, C, P, dt, False)
    return out


def _dx_tile(g, wb, dx, b, t0, T, K, C, P, dt, last):
    """One 128-row tile of dx[n] = sum_a wt[K-1-a] * g[n + a]; returns the a == 0
    g tile (which the dw half of the kernel needs anyway)."""
    acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
    ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
    g0 = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
    for a in range(K):
        ga = g0 if a == 0 else nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
        prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        if last and a > 0:
            nisa.memset(dst=ga[P - a:P, 0:C], value=0.0)
            nisa.dma_copy(dst=ga[0:P - a, 0:C], src=g[b, t0 + a:t0 + P, 0:C])
        else:
            nisa.dma_copy(dst=ga, src=g[b, t0 + a:t0 + a + P, 0:C])
        nisa.tensor_tensor(dst=(acc if a == 0 else prod), data1=ga,
                           data2=wb[0:P, K - 1 - a, 0:C], op=nl.multiply)
        if a > 0:
            nisa.tensor_tensor(dst=(ob if a == K - 1 else acc), data1=acc,
                               data2=prod, op=nl.add)
    if K == 1:
        nisa.tensor_copy(dst=ob, src=acc)
    nisa.dma_copy(dst=dx[b, t0:t0 + P, 0:C], src=ob)
    return g0


def _dw_tile(x, g0, ones, dwa, b, t0, K, C, P, NCH, dt, first):
    """This tile's contribution to dw[k, c] = sum_n g[n, c] * x[n - (K-1) + k, c],
    reduced along the 128 time partitions on the PE and added into dwa."""
    ps = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.psum)
    for k in range(K):
        sh = K - 1 - k
        xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
        pr = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
        if first and sh > 0:
            nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
            nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
        else:
            nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
        nisa.tensor_tensor(dst=pr, data1=g0, data2=xs, op=nl.multiply)
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            # dst = stationary.T @ moving : sum over partitions (time) of pr,
            # landing with the channel index on the partition axis.
            nisa.nc_matmul(dst=ps[0:csz, j, k:k + 1], stationary=pr[0:P, c0:c0 + csz],
                           moving=ones, is_moving_onezero=True)
    nisa.tensor_tensor(dst=dwa, data1=dwa, data2=ps, op=nl.add)


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
    NT = T // P
    NCH = (C + P - 1) // P
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    dw = nl.ndarray((C, K), dtype=wt.dtype, buffer=nl.shared_hbm)
    wb = _load_taps(wt, K, C, P, dt)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    dwa = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dwa, value=0.0)

    for b in range(B):
        # dx's taps run off the END of the sequence on the last tile, dw's (and
        # the forward's) run off the START on the first tile, so both boundary
        # tiles are emitted explicitly and the affine loop covers the interior.
        g0 = _dx_tile(g, wb, dx, b, 0, T, K, C, P, dt, NT == 1)
        _dw_tile(x, g0, ones, dwa, b, 0, K, C, P, NCH, dt, True)
        if NT > 1:
            if NT > 2:
                for i in nl.affine_range(NT - 2):
                    t0 = (i + 1) * P
                    gi = _dx_tile(g, wb, dx, b, t0, T, K, C, P, dt, False)
                    _dw_tile(x, gi, ones, dwa, b, t0, K, C, P, NCH, dt, False)
            tl = (NT - 1) * P
            gl = _dx_tile(g, wb, dx, b, tl, T, K, C, P, dt, True)
            _dw_tile(x, gl, ones, dwa, b, tl, K, C, P, NCH, dt, False)

    sb = nl.ndarray((P, NCH, K), dtype=wt.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=dwa)
    for j in range(NCH):
        c0 = j * P
        csz = min(P, C - c0)
        nisa.dma_copy(dst=dw[c0:c0 + csz, 0:K], src=sb[0:csz, j, 0:K])
    return dx, dw


class _CausalConvNKI(torch.autograd.Function):
    """Differentiable wrapper: x [B, T, C], wt [K, C] (both bf16, contiguous)."""

    @staticmethod
    def forward(ctx, x, wt):
        y = _causal_conv_fwd(x, wt)
        ctx.save_for_backward(x, wt)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, wt = ctx.saved_tensors
        dx, dw_ck = _causal_conv_bwd(x, grad_out.contiguous(), wt)
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

    print(f"{name}  B={B} T={T} C={C} K={K} {dt}")
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
    args = ap.parse_args()

    case("REAL SHAPE bf16", 1, 2048, 640, 4, torch.bfloat16)
    if not args.quick:
        case("fp32 exact-math", 1, 256, 64, 4, torch.float32, dw_tol=1e-5)
        case("B=3 (batch boundary)", 3, 256, 64, 4, torch.bfloat16)
        case("T==128 (single tile)", 2, 128, 64, 4, torch.bfloat16)
        case("T==256 (no interior tiles)", 1, 256, 64, 4, torch.bfloat16)
        case("K=1", 1, 256, 64, 1, torch.bfloat16)
        case("K=2", 1, 256, 64, 2, torch.bfloat16)
        case("K=8", 1, 256, 64, 8, torch.bfloat16)
        case("large x (x*40)", 1, 512, 128, 4, torch.bfloat16, scale=40.0, gscale=5.0, seed=3)
        case("tiny x (x*1e-3)", 1, 512, 128, 4, torch.bfloat16, scale=1e-3, gscale=1e-3, seed=4)

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)
