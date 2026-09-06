"""Characterise nisa rsqrt / reciprocal / sqrt accuracy vs argument magnitude.

The equivalence sweep showed 6% error for inputs with mean(z^2) ~ 1e-6, which
points at the activation-engine rsqrt table rather than at the kernel algebra.
"""
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
import nki
import nki.language as nl
import nki.isa as nisa

P, N = 128, 128


@nki.jit
def variants(m):
    """m: [P, N] fp32 positive. Returns 4 candidate rsqrt implementations."""
    a = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)  # activation rsqrt
    b = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)  # 1/sqrt via reciprocal(sqrt)
    c = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)  # sqrt(reciprocal)
    d = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)  # rsqrt of scaled arg, rescaled

    mb = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mb, src=m[0:P, 0:N])

    t = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=t, op=nl.rsqrt, data=mb)
    nisa.dma_copy(dst=a, src=t)

    s = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=s, op=nl.sqrt, data=mb)
    nisa.reciprocal(dst=t, data=s)
    nisa.dma_copy(dst=b, src=t)

    nisa.reciprocal(dst=s, data=mb)
    nisa.activation(dst=t, op=nl.sqrt, data=s)
    nisa.dma_copy(dst=c, src=t)

    # rsqrt(m) computed as rsqrt(m*K)*sqrt(K) with K a power of two (exact scaling)
    K = 2.0 ** 20
    nisa.tensor_scalar(dst=s, data=mb, op0=nl.multiply, operand0=K)
    nisa.activation(dst=t, op=nl.rsqrt, data=s)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=float(K ** 0.5))
    nisa.dma_copy(dst=d, src=t)

    return a, b, c, d


exps = list(range(-30, 21, 2))
rows = []
base = torch.rand(P, N, dtype=torch.float32) + 1.0  # mantissa in [1,2)
for e in exps:
    m_cpu = base * (2.0 ** e)
    ref = torch.rsqrt(m_cpu.double())
    outs = variants(m_cpu.to("neuron"))
    rel = []
    for o in outs:
        o = o.cpu().double()
        rel.append(((o - ref).abs() / ref).max().item())
    rows.append((e, rel))

print(f"{'2^e':>6} {'act_rsqrt':>12} {'recip(sqrt)':>12} {'sqrt(recip)':>12} {'scaled_rsqrt':>13}")
for e, rel in rows:
    print(f"{e:>6} {rel[0]:12.3e} {rel[1]:12.3e} {rel[2]:12.3e} {rel[3]:13.3e}")
print("\n(max relative error vs float64 torch.rsqrt; fp32 eps = 1.19e-07)")
