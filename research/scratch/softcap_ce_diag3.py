"""Where is the fused softcap+CE forward's REMAINING error, after the chunked reduce?

The equivalence harness shows two distinct plateaus:
  * ~4.2e-5 absolute on every case whose exponents are all near 0 (all-zero,
    all-equal, tiny, realistic lm_head logits) -- including all-zero logits, where
    d = sum of 8192 exact 1.0s is EXACT in fp32, so the accumulator and exp cannot
    be responsible and only the final log can be.
  * up to ~4.7e-4 on cases whose exponent range |y_j - y_c| approaches 2S = 30.

This isolates both: measure the Activation engine's log absolute error directly, and
its exp relative error as a function of argument magnitude.
"""
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def hw_log(x):
    """Activation-engine log of a [P, V] tile, elementwise."""
    N, V = x.shape
    out = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        xb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x[rows])
        ob = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ob, op=nl.log, data=xb)
        nisa.dma_copy(dst=out[rows], src=ob)
    return out


@nki.jit
def hw_exp(x):
    """Activation-engine exp of a [P, V] tile, elementwise."""
    N, V = x.shape
    out = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        xb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=x[rows])
        ob = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ob, op=nl.exp, data=xb)
        nisa.dma_copy(dst=out[rows], src=ob)
    return out


# ---- log: the floor on every case. d spans ~1 to ~4e16 in the real kernel.
print("Activation-engine log, absolute error of log(d) vs fp64 (this lands 1:1 on the loss)")
d = torch.logspace(0, 16, P * 64).reshape(P, 64).float()
got = hw_log(d.to("neuron")).cpu()
ref = torch.log(d.double())
err = (got.double() - ref).abs()
t_ref = torch.log(d).double()  # torch's own fp32 log, for comparison
print(f"   over d in [1, 1e16]:  hw max_abs={err.max():.3e}  "
      f"torch_fp32 max_abs={(t_ref - ref).abs().max():.3e}")
for lo, hi in ((1.0, 1.01), (8000.0, 8400.0), (1e12, 1e14)):
    m = (d >= lo) & (d <= hi)
    if m.any():
        print(f"   d in [{lo:.4g}, {hi:.4g}]: hw max_abs={err[m].max():.3e}  "
              f"(log(d) ~ {ref[m].mean():.3f})")

# specifically d = 8192 exactly, the all-zero-logits case
d8 = torch.full((P, 64), 8192.0)
g8 = hw_log(d8.to("neuron")).cpu()
print(f"   d == 8192 exactly: hw log = {g8[0, 0].item():.9f}  fp64 = "
      f"{torch.log(torch.tensor(8192.0, dtype=torch.float64)).item():.9f}  "
      f"abs_err = {(g8[0,0].double() - torch.log(torch.tensor(8192.0, dtype=torch.float64))).abs():.3e}")

# ---- exp: does the relative error grow with |argument|? that would explain the
# wide-exponent-range cases (the kernel's arguments live in (-2S, 2S) = (-30, 30)).
print("\nActivation-engine exp, relative error vs argument magnitude")
print("   (the kernel's exp arguments are y_j - y_c, bounded to (-30, +30))")
for lo, hi in ((-1.0, 1.0), (-5.0, 5.0), (-10.0, 10.0), (-20.0, 20.0), (-30.0, 30.0)):
    a = torch.linspace(lo, hi, P * 64).reshape(P, 64).float()
    ge = hw_exp(a.to("neuron")).cpu()
    re = torch.exp(a.double())
    rel = ((ge.double() - re).abs() / re)
    rel_t = ((torch.exp(a).double() - re).abs() / re)
    print(f"   arg in [{lo:+6.1f}, {hi:+6.1f}]: hw max_rel={rel.max():.3e}  "
          f"mean_rel={rel.mean():.3e}  torch_fp32 max_rel={rel_t.max():.3e}")

print("\nDONE")
