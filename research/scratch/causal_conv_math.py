"""Offline (CPU, no device) check of the depthwise-causal-conv1d kernel algorithm.

Stage 1 of the same discipline used for the rope+QK-norm and softcap+CE kernels:
derive the backward from scratch, then verify the *algorithm* -- including the
exact tiling and zero-padding structure the NKI kernel will use -- against plain
torch autograd, before writing a single line of NKI.

What the kernel will do (x is [N, C] with N = B*T rows in (b, t) order, taps
wt is [K, C], one 128-row partition tile at a time, C on the free axis):

  fwd   y[n, c]  = sum_k   wt[k, c]     * x[n - (K-1) + k, c]
  dx    dx[n, c] = sum_a   wt[K-1-a, c] * g[n + a, c]
  dw    dw[k, c] = sum_n   g[n, c]      * x[n - (K-1) + k, c]

with x/g treated as zero outside their own sequence row (that zero-padding is
what makes the op causal, and it is per-sequence, not per-tile: a tap must never
read the tail of the previous batch row).

`sim_*` below mirror the kernel's loop structure literally -- the 128-row tile
loop, the per-tap shifted loads, which tiles need a memset+partial load, the
fp32 accumulator, the bf16 rounding of the tap products in the dw reduction --
so that a mismatch here is a mismatch in the algorithm rather than in NKI syntax.

Run: python research/scratch/causal_conv_math.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

P = 128  # NeuronCore SBUF partition limit; the kernel's row-tile height

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# -----------------------------------------------------------------------------
# The reference: train.py's CausalConv.forward, verbatim, under autograd.

def eager_ref(x, w):
    """x [B, T, C], w [C, 1, K] -- exactly train.py:247-251."""
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (w.shape[-1] - 1, 0))
    y = F.conv1d(xt, w.to(dtype=x.dtype), groups=x.shape[-1])
    return y.transpose(1, 2)


# -----------------------------------------------------------------------------
# Simulation of the NKI kernel, tile-for-tile.

def sim_fwd(x2, wt, T):
    """x2 [N, C], wt [K, C]; returns y [N, C] in x2.dtype."""
    N, C = x2.shape
    K = wt.shape[0]
    dt = x2.dtype
    y = torch.empty_like(x2)
    for b in range(N // T):
        for i in range(T // P):
            t0 = i * P
            r0 = b * T + t0
            acc = torch.zeros(P, C, dtype=torch.float32)
            for k in range(K):
                sh = K - 1 - k                      # this tap reads sh rows back
                xk = torch.zeros(P, C, dtype=dt)
                if t0 == 0 and sh > 0:
                    # only the first tile of a sequence can run off the start;
                    # exactly `sh` leading rows are zero pad
                    xk[sh:P] = x2[r0:r0 + P - sh]
                else:
                    xk[:] = x2[r0 - sh:r0 - sh + P]
                # one Vector-engine tensor_tensor per tap, fp32 accumulator
                acc += (xk.float() * wt[k].float().expand(P, C))
            y[r0:r0 + P] = acc.to(dt)
    return y


def sim_dx(g2, wt, T):
    """dx = anti-causal conv of g with reversed taps; g zero past sequence end."""
    N, C = g2.shape
    K = wt.shape[0]
    dt = g2.dtype
    dx = torch.empty_like(g2)
    for b in range(N // T):
        for i in range(T // P):
            t0 = i * P
            r0 = b * T + t0
            acc = torch.zeros(P, C, dtype=torch.float32)
            for a in range(K):
                ga = torch.zeros(P, C, dtype=dt)
                if t0 == T - P and a > 0:
                    # only the last tile of a sequence can run off the end
                    ga[0:P - a] = g2[r0 + a:b * T + T]
                else:
                    ga[:] = g2[r0 + a:r0 + a + P]
                acc += (ga.float() * wt[K - 1 - a].float().expand(P, C))
            dx[r0:r0 + P] = acc.to(dt)
    return dx


def sim_dw(x2, g2, T, K):
    """dw [K, C]: elementwise tap product in the input dtype (it feeds a bf16
    PE matmul in the kernel), summed over the 128 partitions in fp32 PSUM."""
    N, C = x2.shape
    dt = x2.dtype
    dw = torch.zeros(K, C, dtype=torch.float32)   # PSUM accumulator, fp32
    for b in range(N // T):
        for i in range(T // P):
            t0 = i * P
            r0 = b * T + t0
            g0 = g2[r0:r0 + P]
            for k in range(K):
                sh = K - 1 - k
                xk = torch.zeros(P, C, dtype=dt)
                if t0 == 0 and sh > 0:
                    xk[sh:P] = x2[r0:r0 + P - sh]
                else:
                    xk[:] = x2[r0 - sh:r0 - sh + P]
                prod = (g0 * xk).to(dt)           # product rounded to dt (bf16)
                dw[k] += prod.float().sum(dim=0)  # ones-vector matmul into PSUM
    return dw


# -----------------------------------------------------------------------------

ULP = {torch.bfloat16: 2.0 ** -8, torch.float32: 2.0 ** -24}


def report(tag, got, ref, dt):
    got32, ref32 = got.float(), ref.float()
    d = (got32 - ref32).abs()
    floor = max(ref32.abs().max().item(), 1e-30) * 1e-3
    u = (d / (torch.clamp(ref32.abs(), min=floor) * ULP[dt])).max().item()
    exact = (got32 == ref32).float().mean().item() * 100
    rel_l2 = d.norm().item() / max(ref32.norm().item(), 1e-30)
    cs = F.cosine_similarity(got32.flatten(), ref32.flatten(), dim=0).item()
    print(f"   {tag:3s}: max_abs={d.max().item():.3e}  max_ulp={u:7.3f}  "
          f"bit_exact={exact:6.2f}%  rel_l2={rel_l2:.3e}  cos={cs:.9f}")
    return u, exact


def rel_l2(got, ref):
    return (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)


def case(name, B, T, C, K, dt, scale=1.0, gscale=1.0, seed=0, dw_tol=6e-3):
    torch.manual_seed(seed)
    x = (torch.randn(B, T, C) * scale).to(dt)
    w = (torch.randn(C, 1, K) * 0.5)                 # fp32 param, like train.py
    g = (torch.randn(B, T, C) * gscale).to(dt)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    yr = eager_ref(xr, wr)
    yr.backward(g)

    # fp32 gold for dw: the same eager chain in full fp32. Eager's own bf16 dw is
    # NOT a gold reference (F.conv1d's weight-grad comes out in the input dtype),
    # so the kernel's dw is judged against fp64-ish truth and compared to how far
    # the eager bf16 path itself lands from it.
    xg = x.float().clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    eager_ref(xg, wg).backward(g.float())
    dw_gold = wg.grad.detach()

    wt = w.view(C, K).t().contiguous().to(dt)        # [K, C], what the kernel takes
    x2 = x.reshape(B * T, C)
    g2 = g.reshape(B * T, C)
    y_sim = sim_fwd(x2, wt, T).reshape(B, T, C)
    dx_sim = sim_dx(g2, wt, T).reshape(B, T, C)
    dw_sim = sim_dw(x2, g2, T, K)                    # [K, C] fp32
    dw_sim_param = dw_sim.to(dt).float().t().reshape(C, 1, K)  # bf16 like eager's, then param layout

    print(f"{name}  B={B} T={T} C={C} K={K} {dt}")
    report("y", y_sim, yr.detach(), dt)
    report("dx", dx_sim, xr.grad.detach(), dt)
    report("dw", dw_sim_param, wr.grad.detach(), dt)
    # At bf16 (the dtype that ships) y and dx must be BIT-EXACT vs eager: both are
    # sums of the same K products in the same fp32 accumulator, rounded to bf16
    # once at the end, so summation order cannot show through -- there is nothing
    # to trade off. In fp32 the accumulator has no headroom over the operands, so
    # F.conv1d's own summation order shows up in the last ULP; there the bound is
    # rel_l2 instead (the fp32 case exists to check the *derivation*, not rounding).
    if dt == torch.bfloat16:
        check(f"{name}: y bit-exact vs eager", bool((y_sim == yr.detach()).all()))
        check(f"{name}: dx bit-exact vs eager", bool((dx_sim == xr.grad.detach()).all()))
    else:
        check(f"{name}: y rel_l2 <= 1e-6", rel_l2(y_sim, yr.detach()) <= 1e-6,
              f"{rel_l2(y_sim, yr.detach()):.3e}")
        check(f"{name}: dx rel_l2 <= 1e-6", rel_l2(dx_sim, xr.grad.detach()) <= 1e-6,
              f"{rel_l2(dx_sim, xr.grad.detach()):.3e}")
    # dw is judged against the fp32 gold, alongside eager's own bf16 dw.
    e_sim = rel_l2(dw_sim_param, dw_gold)
    e_eager = rel_l2(wr.grad.detach(), dw_gold)
    e_sim_f32 = rel_l2(dw_sim.float().t().reshape(C, 1, K), dw_gold)
    print(f"   dw vs fp32 gold: kernel(bf16 out)={e_sim:.3e}  eager(bf16)={e_eager:.3e}  "
          f"kernel(pre-bf16-round)={e_sim_f32:.3e}")
    check(f"{name}: dw rel_l2 vs fp32 gold <= {dw_tol:.0e}", e_sim <= dw_tol, f"{e_sim:.3e}")
    if dt == torch.bfloat16:
        check(f"{name}: dw no worse than 2x eager's own bf16 error",
              e_sim <= 2.0 * max(e_eager, 1e-12), f"{e_sim:.3e} vs {e_eager:.3e}")


print("=" * 78)
print("depthwise causal conv1d -- offline algorithm/derivation check (CPU only)")
print("=" * 78)

# fp32 first: isolates the derivation from any dtype rounding.
case("fp32 exact-math check", 1, 256, 64, 4, torch.float32, dw_tol=1e-5)
# the real locked shape
case("REAL SHAPE bf16", 1, 2048, 640, 4, torch.bfloat16)
# multi-sequence: catches a tap that reads across a batch-row boundary
case("B=3 (batch boundary)", 3, 256, 64, 4, torch.bfloat16)
# T == P exactly: first tile and last tile are the same tile (both boundaries)
case("T==128 (single tile)", 2, 128, 64, 4, torch.bfloat16)
# other kernel widths, including K=1 (degenerate: pure per-channel scale)
case("K=1", 1, 256, 64, 1, torch.bfloat16)
case("K=2", 1, 256, 64, 2, torch.bfloat16)
case("K=8", 1, 256, 64, 8, torch.bfloat16)
# magnitudes
case("large x (x*40)", 1, 512, 128, 4, torch.bfloat16, scale=40.0, gscale=5.0, seed=3)
case("tiny x (x*1e-3)", 1, 512, 128, 4, torch.bfloat16, scale=1e-3, gscale=1e-3, seed=4)
# zero weight -- the state train.py actually ships in (weight is never trained)
print("zero-weight (the shipped state): y must be exactly 0")
torch.manual_seed(9)
x = (torch.randn(1, 256, 64)).to(torch.bfloat16)
wt0 = torch.zeros(4, 64, dtype=torch.bfloat16)
y0 = sim_fwd(x.reshape(-1, 64), wt0, 256)
check("zero weight -> y exactly 0", bool((y0 == 0).all()))

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
