"""Attribute the fused softcap+CE kernel's residual error to its real source.

Three candidate error sources, separated here so the harness reports a cause and not
just a number:
  (a) the Activation engine's tanh approximation vs torch's tanh
  (b) the constant (-S) logsumexp shift vs torch's row-max shift
  (c) the fp32 reduction order inside sum_j exp(...)
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
S = 15.0


@nki.jit
def probe(z):
    """Return (t = hw tanh(z/S), d = sum_j exp(S*t - S)) so each stage is inspectable."""
    N, V = z.shape
    t_out = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)
    d_out = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    negS = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=negS, value=-S)
    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=zb, src=z[rows])
        tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)
        d = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=zb, op=nl.exp, data=tb, scale=S, bias=negS,
                        reduce_op=nl.add, reduce_res=d,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.dma_copy(dst=t_out[rows], src=tb)
        nisa.dma_copy(dst=d_out[rows], src=d)
    return t_out, d_out


def diag(N, V, scale):
    torch.manual_seed(0)
    z = (torch.randn(N, V) * scale).float()
    t_hw, d_hw = probe(z.to("neuron"))
    t_hw, d_hw = t_hw.cpu(), d_hw.cpu().reshape(-1)

    # (a) hardware tanh vs torch tanh (both fp32), and vs an exact fp64 tanh
    t_ref = torch.tanh(z / S)
    t_exact = torch.tanh(z.double() / S)
    print(f"  N={N} V={V} scale={scale}")
    print(f"    (a) tanh:  hw-vs-torch max_abs={(t_hw - t_ref).abs().max():.3e}  "
          f"hw-vs-fp64 max_abs={(t_hw.double() - t_exact).abs().max():.3e}  "
          f"torch-vs-fp64 max_abs={(t_ref.double() - t_exact).abs().max():.3e}")

    # (b)+(c): lse from the kernel's d, vs an fp64 lse of the *same* hw t values.
    # This isolates shift+reduction error from tanh error.
    lse_hw = S + torch.log(d_hw)
    y_hw64 = (S * t_hw.double())
    lse_exact_same_t = torch.logsumexp(y_hw64, dim=1)
    print(f"    (b+c) lse from d vs fp64 lse of the same t: "
          f"max_abs={(lse_hw.double() - lse_exact_same_t).abs().max():.3e}")

    # torch's own fp32 logsumexp on its own t, vs fp64: the eager reference's own error
    lse_torch = torch.logsumexp(S * t_ref, dim=1)
    lse_t_exact = torch.logsumexp(S * t_ref.double(), dim=1)
    print(f"    torch fp32 lse own error vs fp64: "
          f"max_abs={(lse_torch.double() - lse_t_exact).abs().max():.3e}")

    # total lse error attributable to tanh alone (propagate hw t through fp64 lse)
    lse_ref64 = torch.logsumexp(S * t_exact, dim=1)
    print(f"    lse(hw t, fp64) - lse(exact t, fp64) = tanh-only contribution: "
          f"max_abs={(lse_exact_same_t - lse_ref64).abs().max():.3e}")


for args in ((256, 512, 3.0), (256, 8192, 3.0), (256, 8192, 30.0), (256, 8192, 0.1)):
    diag(*args)
print("DONE")
