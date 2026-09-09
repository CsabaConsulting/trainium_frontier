#!/usr/bin/env python3
"""De-risk the Muon fusion before writing it: does an NKI kernel launch actually
cost the same as a small aten launch in eager on this backend, and what does it
cost to compile?

The whole sizing argument (research/scratch/muon_fusion_sizing.py: 184 launches
removed, -30.4ms) assumes "one fused NKI kernel == one launch == one aten op's
launch gap". If an @nki.jit call carries extra per-call host overhead the argument
collapses, so measure it rather than assume it.

Probe kernel is deliberately the real K3 shape from the fusion plan:
    out = a*x + p     (x, p bf16 [M, N]; a a runtime [P,1] fp32 operand)
against the eager form `a * x + p`, which is 2 aten launches.

Usage:  NEURON_RT_VISIBLE_CORES=0 python research/scratch/nki_launch_cost_probe.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def axpy_kernel(x, p, coef):
    """out = a*x + p, bf16, with `a` a runtime operand (coef[:, 0:1]).

    M is a compile-time constant (nki.jit specialises per shape), so the
    128-partition tiling and its remainder are both resolved at trace time.
    """
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    n_full = M // P
    rem = M - n_full * P
    for t in range(n_full + (1 if rem else 0)):
        start = t * P
        size = P if t < n_full else rem
        xb = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x[start:start + size, 0:N])
        pb = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pb, src=p[start:start + size, 0:N])
        ab = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ab, src=coef[0:size, 0:1])
        ax = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ax, data=xb, op0=nl.multiply, operand0=ab)
        yb = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=yb, data1=ax, data2=pb, op=nl.add)
        nisa.dma_copy(dst=out[start:start + size, 0:N], src=yb)
    return out


def bench(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.neuron.synchronize() if hasattr(torch, "neuron") else None
    import train as T
    T.synchronize(torch.device("neuron"))
    ts = []
    for _ in range(reps):
        T.synchronize(torch.device("neuron"))
        t = time.monotonic()
        fn()
        T.synchronize(torch.device("neuron"))
        ts.append((time.monotonic() - t) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def main():
    import train as T
    dev = torch.device("neuron")

    # the three real K3 shapes from the locked config's Muon groups
    shapes = [(15, 32), (15360, 640), (3840, 2560)]
    a_val = 8.156554524902461  # POLAR_EXPRESS_COEFFS[0][0]

    for M, N in shapes:
        x_c = torch.randn(M, N).to(torch.bfloat16)
        p_c = torch.randn(M, N).to(torch.bfloat16)
        coef_c = torch.full((P, 1), a_val, dtype=torch.float32)
        x, p, coef = x_c.to(dev), p_c.to(dev), coef_c.to(dev)

        t0 = time.monotonic()
        y = axpy_kernel(x, p, coef)
        T.synchronize(dev)
        compile_s = time.monotonic() - t0

        # correctness: CPU reference, exact-match check
        ref = (a_val * x_c.float()).to(torch.bfloat16) + p_c
        ref = ref.to(torch.bfloat16)
        got = y.cpu()
        eager = (a_val * x + p).cpu()
        n_exact_ref = int((got == ref).sum())
        n_exact_eager = int((got == eager).sum())
        tot = got.numel()
        maxdiff = (got.float() - eager.float()).abs().max().item()

        n_nki, lo_n, hi_n = bench(lambda: axpy_kernel(x, p, coef), 30)
        n_eag, lo_e, hi_e = bench(lambda: a_val * x + p, 30)

        print(f"\nshape {M}x{N}  ({tot/1e6:.2f}M elems)")
        print(f"  first-call (compile+run)   {compile_s:8.2f} s")
        print(f"  NKI  1 launch              {n_nki:8.3f} ms  [{lo_n:.3f}-{hi_n:.3f}]")
        print(f"  eager 2 launches           {n_eag:8.3f} ms  [{lo_e:.3f}-{hi_e:.3f}]")
        print(f"  per-launch implied (eager) {n_eag/2:8.3f} ms")
        print(f"  NKI vs eager               {n_eag - n_nki:+8.3f} ms")
        print(f"  exact vs CPU ref           {n_exact_ref}/{tot} "
              f"({100*n_exact_ref/tot:.4f}%)")
        print(f"  exact vs EAGER-on-device   {n_exact_eager}/{tot} "
              f"({100*n_exact_eager/tot:.4f}%)  max|diff| {maxdiff:.3e}")


if __name__ == "__main__":
    main()
