"""Which backward structure actually compiles? One variant per process.

Measured, in order:
  * buffers allocated INSIDE the unrolled tile loop are fine -- the allocator
    sees each tile die at the end of the iteration and reuses the space (the
    forward compiles this way);
  * buffers HOISTED out of the loop and written once per iteration are NOT --
    every write becomes a fresh SSA value that is never freed, and the allocator
    runs out: "COMPILATION FAILED: Memory Location {..._SsaClone61}@SB<0,0>
    (128x1280)" for a plain [128, 640] bf16 tile written 64 times;
  * therefore the only buffer that may persist across tiles is the dw
    accumulator (it has to), so it must be TINY: the tap products are reduced
    along the partition axis on the PE *per tile* into PSUM ([C, K] worth of
    numbers = 80 bytes/partition) and only that is accumulated in SBUF.

Each variant is compiled in its own process: after a compilation failure the
runtime stream is poisoned and subsequent kernels report the *previous*
kernel's error (observed: dw_only reporting bwd_vb's neff cache key).

Run: python research/scratch/conv_probe.py <fwd|dx|dw|bwd>
"""
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

P = 128


def _taps(wt, K, C, dt):
    wb = nl.ndarray((P, K, C), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb, src=wt.ap(pattern=[[0, P], [C, K], [1, C]]))
    return wb


@nki.jit
def fwd(x, wt):
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    out = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, dt)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for k in range(K):
                sh = K - 1 - k
                xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                if t0 == 0 and sh > 0:
                    nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
                    nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
                else:
                    nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
                nisa.tensor_tensor(dst=(acc if k == 0 else prod), data1=xs,
                                   data2=wb[0:P, k, 0:C], op=nl.multiply)
                if k > 0:
                    nisa.tensor_tensor(dst=(ob if k == K - 1 else acc), data1=acc,
                                       data2=prod, op=nl.add)
            if K == 1:
                nisa.tensor_copy(dst=ob, src=acc)
            nisa.dma_copy(dst=out[b, t0:t0 + P, 0:C], src=ob)
    return out


@nki.jit
def dx_only(g, wt):
    B, T, C = g.shape
    K = wt.shape[0]
    dt = g.dtype
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, dt)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for a in range(K):
                ga = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                if t0 == T - P and a > 0:
                    nisa.memset(dst=ga[P - a:P, 0:C], value=0.0)
                    nisa.dma_copy(dst=ga[0:P - a, 0:C], src=g[b, t0 + a:T, 0:C])
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
    return dx


@nki.jit
def dw_only(x, g, K):
    B, T, C = x.shape
    dt = x.dtype
    NCH = (C + P - 1) // P
    dw = nl.ndarray((C, K), dtype=dt, buffer=nl.shared_hbm)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    dwa = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dwa, value=0.0)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            g0 = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=g0, src=g[b, t0:t0 + P, 0:C])
            ps = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.psum)
            for k in range(K):
                sh = K - 1 - k
                xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                pr = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                if t0 == 0 and sh > 0:
                    nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
                    nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
                else:
                    nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
                nisa.tensor_tensor(dst=pr, data1=g0, data2=xs, op=nl.multiply)
                for j in range(NCH):
                    c0 = j * P
                    csz = min(P, C - c0)
                    nisa.nc_matmul(dst=ps[0:csz, j, k:k + 1],
                                   stationary=pr[0:P, c0:c0 + csz],
                                   moving=ones, is_moving_onezero=True)
            nisa.tensor_tensor(dst=dwa, data1=dwa, data2=ps, op=nl.add)
    sb = nl.ndarray((P, NCH, K), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=dwa)
    for j in range(NCH):
        c0 = j * P
        csz = min(P, C - c0)
        nisa.dma_copy(dst=dw[c0:c0 + csz, 0:K], src=sb[0:csz, j, 0:K])
    return dw


@nki.jit
def red(a, mode):
    """Micro-probe: column sums of a [128, C] SBUF tile via the PE (the only unit
    that can reduce along partitions). out[c] = sum_p a[p, c].

    mode 0 = dst is a slice of a [P, NCH, K]-shaped psum tile, is_moving_onezero
    mode 1 = same, without is_moving_onezero
    mode 2 = dst is a dedicated [csz, 1] psum tile
    mode 3 = mode 2 with everything in fp32
    mode 4 = TWO matmuls per output slot (the tile split in half along
             partitions) into the SAME psum region: does a repeated nc_matmul to
             one region accumulate (the "accumulation group" the block-layout dw
             relies on) or overwrite? Correct iff it accumulates.
    mode 5 = four matmuls per slot, quarters, same question
    """
    Pin, C = a.shape
    NCH = (C + P - 1) // P
    out = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dt = nl.float32 if mode == 3 else a.dtype
    av = nl.ndarray((Pin, C), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=av, src=a) if mode != 3 else None
    if mode == 3:
        tmp = nl.ndarray((Pin, C), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tmp, src=a)
        nisa.tensor_copy(dst=av, src=tmp)
    ones = nl.ndarray((Pin, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    if mode in (0, 1):
        ps = nl.ndarray((P, NCH, 1), dtype=nl.float32, buffer=nl.psum)
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            nisa.nc_matmul(dst=ps[0:csz, j, 0:1], stationary=av[0:Pin, c0:c0 + csz],
                           moving=ones, is_moving_onezero=(mode == 0))
        sb = nl.ndarray((P, NCH, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb, src=ps)
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            nisa.dma_copy(dst=out[c0:c0 + csz, 0:1], src=sb[0:csz, j, 0:1])
    elif mode in (4, 5):
        nsp = 2 if mode == 4 else 4          # split the contraction into nsp parts
        h = Pin // nsp
        ps = nl.ndarray((P, NCH, 1), dtype=nl.float32, buffer=nl.psum)
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            for s in range(nsp):
                nisa.nc_matmul(dst=ps[0:csz, j, 0:1],
                               stationary=av[s * h:(s + 1) * h, c0:c0 + csz],
                               moving=ones[0:h, 0:1])
        sb = nl.ndarray((P, NCH, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb, src=ps)
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            nisa.dma_copy(dst=out[c0:c0 + csz, 0:1], src=sb[0:csz, j, 0:1])
    else:
        for j in range(NCH):
            c0 = j * P
            csz = min(P, C - c0)
            ps = nl.ndarray((csz, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=av[0:Pin, c0:c0 + csz], moving=ones)
            sb = nl.ndarray((csz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sb, src=ps)
            nisa.dma_copy(dst=out[c0:c0 + csz, 0:1], src=sb)
    return out


@nki.jit
def dx3(g, wt, mode):
    """dx bisect. mode is a bitmask:
       bit 0 -> read g BACKWARD (fwd's addressing, boundary on the first tile)
                instead of forward (mathematically wrong; compile probe only)
       bit 1 -> taps in order instead of reversed
    Everything else is byte-for-byte the working forward kernel's structure."""
    B, T, C = g.shape
    K = wt.shape[0]
    dt = g.dtype
    back = bool(mode & 1)
    inorder = bool(mode & 2)
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, dt)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for a in range(K):
                ga = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                if back:
                    if t0 == 0 and a > 0:
                        nisa.memset(dst=ga[0:a, 0:C], value=0.0)
                        nisa.dma_copy(dst=ga[a:P, 0:C], src=g[b, 0:P - a, 0:C])
                    else:
                        nisa.dma_copy(dst=ga, src=g[b, t0 - a:t0 - a + P, 0:C])
                else:
                    if t0 == T - P and a > 0:
                        nisa.memset(dst=ga[P - a:P, 0:C], value=0.0)
                        nisa.dma_copy(dst=ga[0:P - a, 0:C], src=g[b, t0 + a:T, 0:C])
                    else:
                        nisa.dma_copy(dst=ga, src=g[b, t0 + a:t0 + a + P, 0:C])
                tap = a if inorder else K - 1 - a
                nisa.tensor_tensor(dst=(acc if a == 0 else prod), data1=ga,
                                   data2=wb[0:P, tap, 0:C], op=nl.multiply)
                if a > 0:
                    nisa.tensor_tensor(dst=(ob if a == K - 1 else acc), data1=acc,
                                       data2=prod, op=nl.add)
            if K == 1:
                nisa.tensor_copy(dst=ob, src=acc)
            nisa.dma_copy(dst=dx[b, t0:t0 + P, 0:C], src=ob)
    return dx


@nki.jit
def dx2(g, wt):
    """dx with the minimum possible number of live tiles: one fp32 accumulator,
    one fp32 product, one bf16 shifted-g tile, one bf16 output tile per tile
    iteration (4 buffers instead of 10)."""
    B, T, C = g.shape
    K = wt.shape[0]
    dt = g.dtype
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, dt)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            ga = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for a in range(K):
                if t0 == T - P and a > 0:
                    nisa.memset(dst=ga[P - a:P, 0:C], value=0.0)
                    nisa.dma_copy(dst=ga[0:P - a, 0:C], src=g[b, t0 + a:T, 0:C])
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
    return dx


@nki.jit
def dw2(x, g, K):
    """dw accumulated entirely in PSUM: every tile's partition-reduction matmul
    targets the SAME psum region, so the tiles form one accumulation group and
    there is no per-tile SBUF add at all."""
    B, T, C = x.shape
    dt = x.dtype
    NCH = (C + P - 1) // P
    dw = nl.ndarray((C, K), dtype=dt, buffer=nl.shared_hbm)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.psum)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            g0 = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=g0, src=g[b, t0:t0 + P, 0:C])
            xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            pr = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for k in range(K):
                sh = K - 1 - k
                if t0 == 0 and sh > 0:
                    nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
                    nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
                else:
                    nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
                nisa.tensor_tensor(dst=pr, data1=g0, data2=xs, op=nl.multiply)
                for j in range(NCH):
                    c0 = j * P
                    csz = min(P, C - c0)
                    nisa.nc_matmul(dst=ps[0:csz, j, k:k + 1],
                                   stationary=pr[0:P, c0:c0 + csz],
                                   moving=ones, is_moving_onezero=True)
    sb = nl.ndarray((P, NCH, K), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    for j in range(NCH):
        c0 = j * P
        csz = min(P, C - c0)
        nisa.dma_copy(dst=dw[c0:c0 + csz, 0:K], src=sb[0:csz, j, 0:K])
    return dw


@nki.jit
def bwd(x, g, wt):
    """fused dx + dw."""
    B, T, C = x.shape
    K = wt.shape[0]
    dt = x.dtype
    NCH = (C + P - 1) // P
    dx = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
    dw = nl.ndarray((C, K), dtype=wt.dtype, buffer=nl.shared_hbm)
    wb = _taps(wt, K, C, dt)
    ones = nl.ndarray((P, 1), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    dwa = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dwa, value=0.0)
    for b in range(B):
        for i in range(T // P):
            t0 = i * P
            acc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            ob = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            g0 = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
            for a in range(K):
                ga = g0 if a == 0 else nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                prod = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                if t0 == T - P and a > 0:
                    nisa.memset(dst=ga[P - a:P, 0:C], value=0.0)
                    nisa.dma_copy(dst=ga[0:P - a, 0:C], src=g[b, t0 + a:T, 0:C])
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

            ps = nl.ndarray((P, NCH, K), dtype=nl.float32, buffer=nl.psum)
            for k in range(K):
                sh = K - 1 - k
                xs = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                pr = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
                if t0 == 0 and sh > 0:
                    nisa.memset(dst=xs[0:sh, 0:C], value=0.0)
                    nisa.dma_copy(dst=xs[sh:P, 0:C], src=x[b, 0:P - sh, 0:C])
                else:
                    nisa.dma_copy(dst=xs, src=x[b, t0 - sh:t0 - sh + P, 0:C])
                nisa.tensor_tensor(dst=pr, data1=g0, data2=xs, op=nl.multiply)
                for j in range(NCH):
                    c0 = j * P
                    csz = min(P, C - c0)
                    nisa.nc_matmul(dst=ps[0:csz, j, k:k + 1],
                                   stationary=pr[0:P, c0:c0 + csz],
                                   moving=ones, is_moving_onezero=True)
            nisa.tensor_tensor(dst=dwa, data1=dwa, data2=ps, op=nl.add)
    sb = nl.ndarray((P, NCH, K), dtype=wt.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=dwa)
    for j in range(NCH):
        c0 = j * P
        csz = min(P, C - c0)
        nisa.dma_copy(dst=dw[c0:c0 + csz, 0:K], src=sb[0:csz, j, 0:K])
    return dx, dw


# ---------------------------------------------------------------- runner

def eager_ref(x, w):
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (w.shape[-1] - 1, 0))
    y = F.conv1d(xt, w.to(dtype=x.dtype), groups=x.shape[-1])
    return y.transpose(1, 2)


def rel_l2(got, ref):
    return (got.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "bwd"
    # optional "B,T,C,K" second arg: the small single-tile shapes isolate the
    # per-tile math from anything that only shows up when tiles accumulate.
    B, T, C, K = (int(v) for v in sys.argv[2].split(",")) if len(sys.argv) > 2 \
        else (1, 2048, 640, 4)
    print(f"shape B={B} T={T} C={C} K={K}  variant={which}")
    DT = torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(B, T, C).to(DT)
    w = torch.randn(C, 1, K) * 0.5
    g = torch.randn(B, T, C).to(DT)
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    yr = eager_ref(xr, wr)
    yr.backward(g)
    xg = x.float().clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    eager_ref(xg, wg).backward(g.float())
    gold_dw = wg.grad.detach()

    xd, gd = x.to("neuron"), g.to("neuron")
    wd = w.view(C, K).t().contiguous().to(DT).to("neuron")

    def rep_dx(dx, ref):
        ex = (dx.float() == ref.float()).float().mean().item() * 100
        print(f"   dx/y: bit_exact={ex:6.2f}%  rel_l2={rel_l2(dx, ref):.3e}")

    def rep_dw(dw):
        p = dw.float().t().reshape(C, 1, K)
        print(f"   dw: rel_l2 vs fp32 gold={rel_l2(p, gold_dw):.3e}  "
              f"(eager bf16 itself = {rel_l2(wr.grad.detach(), gold_dw):.3e})")
        # a constant kernel/gold ratio per tap means a counting bug (psum
        # accumulating tiles more than once); noise means a layout bug.
        for k in range(K):
            r = p[:, 0, k] / gold_dw[:, 0, k].clamp(min=1e-6)
            print(f"      tap {k}: ratio median={r.median().item():+.4f} "
                  f"|kernel|={p[:, 0, k].abs().mean():.3e} "
                  f"|gold|={gold_dw[:, 0, k].abs().mean():.3e}")

    if which == "red":
        a = torch.arange(P * C, dtype=torch.float32).reshape(P, C) % 7.0 - 3.0
        ab = a.to(DT)
        ref = ab.float().sum(dim=0).reshape(C, 1)
        for m in range(6):
            try:
                got = red(ab.to("neuron"), m).cpu()
                bad = (got.float() - ref).abs() > 1e-3
                print(f"   mode {m}: rel_l2={rel_l2(got, ref):.3e}  "
                      f"wrong={int(bad.sum())}/{C}  got[:4]={got.flatten()[:4].tolist()} "
                      f"ref[:4]={ref.flatten()[:4].tolist()}")
            except Exception as e:  # noqa: BLE001
                print(f"   mode {m}: FAILED {str(e).splitlines()[0][:120]}")
        sys.exit(0)
    if which.startswith("dx3-"):
        rep_dx(dx3(gd, wd, int(which.split("-")[1])).cpu(), xr.grad.detach())
    elif which == "fwd":
        rep_dx(fwd(xd, wd).cpu(), yr.detach())
    elif which == "dx":
        rep_dx(dx_only(gd, wd).cpu(), xr.grad.detach())
    elif which == "dx2":
        rep_dx(dx2(gd, wd).cpu(), xr.grad.detach())
    elif which == "dw":
        rep_dw(dw_only(xd, gd, K).cpu())
    elif which == "dw2":
        rep_dw(dw2(xd, gd, K).cpu())
    else:
        dxo, dwo = bwd(xd, gd, wd)
        rep_dx(dxo.cpu(), xr.grad.detach())
        rep_dw(dwo.cpu())
    print(f"[OK] {which} compiled and ran")
