"""Dev + numerical-equivalence harness for the fused rope+RMSNorm NKI kernel.

Math
----
forward:   z = rope(x, cos, sin);  out = z * rsqrt(mean(z^2, -1) + eps)
backward:  dz = r * (g - (p * r^2 / D) * z),  r = rsqrt(mean(z^2)+eps), p = sum(g*z)/1
           dx = rope^T(dz)   (rope is an orthogonal rotation per (x1,x2) pair)

rope:      z1 = x1*cos + x2*sin ;  z2 = x2*cos - x1*sin
rope^T:    dx1 = dz1*cos - dz2*sin ; dx2 = dz1*sin + dz2*cos
"""
import os
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
import torch.nn.functional as F
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_RMS_EPS = 1.1920928955078125e-07  # torch F.rms_norm default eps == float32 eps (measured)


@nki.jit
def _rope_norm_fwd(x, cos, sin):
    """out[m,h,:] = rms_norm(rope(x[m,h,:], cos[t,:], sin[t,:])), t = m % T.

    x:   [BT, H, D] (BT % 128 == 0)
    cos: [T, D//2]  (T % 128 == 0)

    Rounding is matched to the eager chain: the rope products/sums are rounded to
    x.dtype (as `x1*cos + x2*sin` would be in bf16), then the RMS statistic and the
    final scaling are computed in fp32 (as `F.rms_norm` upcasts internally).
    """
    BT, H, D = x.shape
    T = cos.shape[0]
    Dh = D // 2
    dt = x.dtype
    BC = [[H, _P], [1, H], [0, D]]  # [P,H,1] -> [P,H,D] stride-0 broadcast
    out = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)

    for i in nl.affine_range(BT // _P):
        r0 = i * _P
        t0 = (i * _P) % T

        xb = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x[r0:r0 + _P])
        cb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=cos.ap(pattern=[[Dh, _P], [0, H], [1, Dh]], offset=t0 * Dh))
        sb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb, src=sin.ap(pattern=[[Dh, _P], [0, H], [1, Dh]], offset=t0 * Dh))

        x1, x2 = xb[0:_P, 0:H, 0:Dh], xb[0:_P, 0:H, Dh:D]
        zb = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        tb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        z1, z2 = zb[0:_P, 0:H, 0:Dh], zb[0:_P, 0:H, Dh:D]

        # rope, rounded to dt at each step like the eager elementwise chain
        nisa.tensor_tensor(dst=z1, data1=x1, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=x2, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=z1, data1=z1, data2=tb, op=nl.add)
        nisa.tensor_tensor(dst=z2, data1=x2, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=x1, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=z2, data1=z2, data2=tb, op=nl.subtract)

        # rms over the last dim, fp32
        zf = nl.ndarray((_P, H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=zf, src=zb)
        sq = nl.ndarray((_P, H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=zf, data2=zf, op=nl.multiply)
        ss = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=(2,))
        ms = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D,
                           op1=nl.add, operand1=_RMS_EPS)
        rr = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr, op=nl.rsqrt, data=ms)

        ob = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ob, data1=zf, data2=rr.ap(pattern=BC), op=nl.multiply)
        nisa.dma_copy(dst=out[r0:r0 + _P], src=ob)

    return out


@nki.jit
def _rope_norm_bwd(x, g, cos, sin):
    """Gradient of _rope_norm_fwd w.r.t. x.

    dz = r * (g - (<g,z> * r^2 / D) * z),  r = rsqrt(mean(z^2)+eps)  [fp32]
    dx = rope^T(dz):  dx1 = dz1*cos - dz2*sin,  dx2 = dz1*sin + dz2*cos  [x.dtype]
    """
    BT, H, D = x.shape
    T = cos.shape[0]
    Dh = D // 2
    dt = x.dtype
    BC = [[H, _P], [1, H], [0, D]]
    dx = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)

    for i in nl.affine_range(BT // _P):
        r0 = i * _P
        t0 = (i * _P) % T

        xb = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x[r0:r0 + _P])
        gf = nl.ndarray((_P, H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gf, src=g[r0:r0 + _P])
        cb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=cos.ap(pattern=[[Dh, _P], [0, H], [1, Dh]], offset=t0 * Dh))
        sb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb, src=sin.ap(pattern=[[Dh, _P], [0, H], [1, Dh]], offset=t0 * Dh))

        x1, x2 = xb[0:_P, 0:H, 0:Dh], xb[0:_P, 0:H, Dh:D]
        zb = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        tb = nl.ndarray((_P, H, Dh), dtype=dt, buffer=nl.sbuf)
        z1, z2 = zb[0:_P, 0:H, 0:Dh], zb[0:_P, 0:H, Dh:D]

        # recompute z = rope(x), same rounding as forward
        nisa.tensor_tensor(dst=z1, data1=x1, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=x2, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=z1, data1=z1, data2=tb, op=nl.add)
        nisa.tensor_tensor(dst=z2, data1=x2, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=x1, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=z2, data1=z2, data2=tb, op=nl.subtract)

        zf = nl.ndarray((_P, H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=zf, src=zb)
        acc = nl.ndarray((_P, H, D), dtype=nl.float32, buffer=nl.sbuf)

        # r = rsqrt(mean(z^2) + eps)
        nisa.tensor_tensor(dst=acc, data1=zf, data2=zf, op=nl.multiply)
        ss = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=acc, axis=(2,))
        ms = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D,
                           op1=nl.add, operand1=_RMS_EPS)
        rr = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr, op=nl.rsqrt, data=ms)

        # q = <g,z> * r^2 / D
        nisa.tensor_tensor(dst=acc, data1=gf, data2=zf, op=nl.multiply)
        pp = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=pp, op=nl.add, data=acc, axis=(2,))
        r2 = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=r2, data=ms)                      # r^2 = 1/(mean+eps)
        qq = nl.ndarray((_P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=qq, data=pp, op0=nl.multiply, operand0=1.0 / D)
        nisa.tensor_tensor(dst=qq, data1=qq, data2=r2, op=nl.multiply)

        # dz = r * (g - q*z), fp32, then rounded to dt (eager's rms_norm bwd emits dt)
        nisa.tensor_tensor(dst=acc, data1=zf, data2=qq.ap(pattern=BC), op=nl.multiply)
        nisa.tensor_tensor(dst=acc, data1=gf, data2=acc, op=nl.subtract)
        dzb = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dzb, data1=acc, data2=rr.ap(pattern=BC), op=nl.multiply)

        # dx = rope^T(dz), rounded to dt at each step
        d1, d2 = dzb[0:_P, 0:H, 0:Dh], dzb[0:_P, 0:H, Dh:D]
        ob = nl.ndarray((_P, H, D), dtype=dt, buffer=nl.sbuf)
        o1, o2 = ob[0:_P, 0:H, 0:Dh], ob[0:_P, 0:H, Dh:D]
        nisa.tensor_tensor(dst=o1, data1=d1, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=d2, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=o1, data1=o1, data2=tb, op=nl.subtract)
        nisa.tensor_tensor(dst=o2, data1=d2, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tb, data1=d1, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=o2, data1=o2, data2=tb, op=nl.add)

        nisa.dma_copy(dst=dx[r0:r0 + _P], src=ob)

    return dx


# ---------------------------------------------------------------------------- eager reference
def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def eager_ref(x, cos, sin):
    return F.rms_norm(apply_rotary_emb(x, cos, sin), (x.size(-1),))


class RopeNormNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        B, T, H, D = x.shape
        x3 = x.reshape(B * T, H, D)
        c2 = cos.reshape(-1, D // 2)
        s2 = sin.reshape(-1, D // 2)
        y = _rope_norm_fwd(x3, c2, s2).reshape(B, T, H, D)
        ctx.save_for_backward(x3, c2, s2)
        ctx.shape = (B, T, H, D)
        return y

    @staticmethod
    def backward(ctx, gout):
        x3, c2, s2 = ctx.saved_tensors
        B, T, H, D = ctx.shape
        g3 = gout.reshape(B * T, H, D).contiguous()
        return _rope_norm_bwd(x3, g3, c2, s2).reshape(B, T, H, D), None, None


# ---------------------------------------------------------------------------- validation
def stats(name, a, b):
    a32, b32 = a.float(), b.float()
    md = (a32 - b32).abs().max().item()
    den = b32.abs().max().item()
    nrm = (a32 - b32).norm().item() / max(b32.norm().item(), 1e-30)
    cos_sim = F.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item()
    print(f"  {name:28s} max_abs={md:.4e}  max|ref|={den:.4e}  rel_l2={nrm:.4e}  cos={cos_sim:.10f}")
    return md, nrm


def main():
    B, T, H, D = 2, 1024, 3, 128
    dt = torch.bfloat16
    torch.manual_seed(1234)

    x_cpu = (torch.randn(B, T, H, D, dtype=torch.float32) * 0.7).to(dt)
    base = 1024.0
    freqs = torch.arange(0, D, 2, dtype=torch.float32)
    inv = 1.0 / (base ** (freqs / D))
    tt = torch.arange(T, dtype=torch.float32)
    tab = torch.outer(tt, inv)
    cos_cpu = tab.cos().to(dt)[None, :, None, :]
    sin_cpu = tab.sin().to(dt)[None, :, None, :]
    g_cpu = (torch.randn(B, T, H, D, dtype=torch.float32) * 0.3).to(dt)

    # ---- CPU reference (fp32 math on bf16 inputs, matching what the kernel does)
    xr = x_cpu.clone().requires_grad_(True)
    yr = eager_ref(xr, cos_cpu, sin_cpu)
    yr.backward(g_cpu)
    ref_y, ref_dx = yr.detach(), xr.grad.detach()

    dev = "neuron"
    xd = x_cpu.to(dev)
    cd, sd = cos_cpu.to(dev), sin_cpu.to(dev)
    gd = g_cpu.to(dev)

    # ---- eager on device (the thing we must not change)
    xe = xd.clone().requires_grad_(True)
    ye = eager_ref(xe, cd, sd)
    ye.backward(gd)
    eag_y, eag_dx = ye.detach().cpu(), xe.grad.detach().cpu()

    # ---- NKI kernel on device
    xk = xd.clone().requires_grad_(True)
    yk = RopeNormNKI.apply(xk, cd, sd)
    yk.backward(gd)
    nki_y, nki_dx = yk.detach().cpu(), xk.grad.detach().cpu()

    print("\n--- forward ---")
    stats("eager_dev vs cpu_ref", eag_y, ref_y)
    stats("nki vs cpu_ref", nki_y, ref_y)
    fmd, _ = stats("nki vs eager_dev", nki_y, eag_y)
    print("\n--- backward (dx) ---")
    stats("eager_dev vs cpu_ref", eag_dx, ref_dx)
    stats("nki vs cpu_ref", nki_dx, ref_dx)
    bmd, _ = stats("nki vs eager_dev", nki_dx, eag_dx)

    ulp = 0.0078125  # bf16 relative resolution
    print(f"\nbf16 unit-roundoff = {ulp:.3e}")
    print(f"forward  max_abs / (ulp*max|ref|) = {fmd / (ulp * eag_y.float().abs().max().item()):.3f}")
    print(f"backward max_abs / (ulp*max|ref|) = {bmd / (ulp * eag_dx.float().abs().max().item()):.3f}")

    # ---- exact-match rate against the CPU fp32-rounded reference
    print(f"\nfwd exact-bit match vs cpu_ref: {(nki_y == ref_y).float().mean().item()*100:.2f}%"
          f"   (eager_dev: {(eag_y == ref_y).float().mean().item()*100:.2f}%)")
    print(f"bwd exact-bit match vs cpu_ref: {(nki_dx == ref_dx).float().mean().item()*100:.2f}%"
          f"   (eager_dev: {(eag_dx == ref_dx).float().mean().item()*100:.2f}%)")

    # ---- micro benchmark: DEPTH=6 does 12 fwd+bwd calls per step (q and k per layer)
    def bench(fn, n=20, warm=5):
        for _ in range(warm):
            r = fn()
        r.cpu()  # drain the queue before timing
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
        r.cpu()  # drain the queue inside the timed region
        return (time.perf_counter() - t0) / n

    def eager_call():
        xx = xd.clone().requires_grad_(True)
        y = eager_ref(xx, cd, sd)
        y.backward(gd)
        return xx.grad

    def nki_call():
        xx = xd.clone().requires_grad_(True)
        y = RopeNormNKI.apply(xx, cd, sd)
        y.backward(gd)
        return xx.grad

    def clone_only():
        return xd.clone()

    tc = bench(clone_only)
    te = bench(eager_call)
    tn = bench(nki_call)
    print(f"\nmicro-bench per call (clone+fwd+bwd): eager={te*1e3:.3f} ms  nki={tn*1e3:.3f} ms  "
          f"clone-only overhead={tc*1e3:.3f} ms")
    net_e, net_n = te - tc, tn - tc
    print(f"net of clone: eager={net_e*1e3:.3f} ms  nki={net_n*1e3:.3f} ms  "
          f"speedup={net_e/max(net_n,1e-12):.2f}x")
    print(f"projected per-step saving (12 calls/step): {(net_e - net_n)*12*1e3:.3f} ms")


if __name__ == "__main__":
    main()
