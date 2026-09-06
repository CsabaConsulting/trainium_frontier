"""Extended numerical-equivalence sweep for the fused rope+RMSNorm NKI kernel.

Compares the kernel against the eager `norm(apply_rotary_emb(x, cos, sin))` chain
(forward AND backward) across seeds, magnitudes and degenerate inputs.
Reference is computed on CPU (bit-identical to the device eager path, verified in
rope_norm_dev.py), so only the NKI kernel generates a NEFF.

The kernel under test is imported straight out of train.py, so this harness can
never drift from what actually ships.
"""
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

import train as T_

RopeNormNKI = T_._RopeNormNKI


def eager_ref(x, cos, sin):
    return T_.norm(T_.apply_rotary_emb(x, cos, sin))

B, T, H, D = 2, 1024, 3, 128
DT = torch.bfloat16


def rope_tables(base=1024.0):
    freqs = torch.arange(0, D, 2, dtype=torch.float32)
    inv = 1.0 / (base ** (freqs / D))
    tab = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    return tab.cos().to(DT)[None, :, None, :], tab.sin().to(DT)[None, :, None, :]


COS, SIN = rope_tables()
ULP = 2.0 ** -8  # bf16 relative resolution


def ulps(a, b):
    """diff in bf16 ULPs, floored at 1e-3 of the tensor max so that near-zero
    reference elements cannot manufacture astronomical ratios (an earlier version
    of this metric divided by max(|ref|,1e-30) and reported 1e37 "ULPs" for
    elements whose absolute error was 1e-38)."""
    a32, b32 = a.float(), b.float()
    floor = max(b32.abs().max().item(), 1e-30) * 1e-3
    scale = torch.clamp(b32.abs(), min=floor) * ULP
    return ((a32 - b32).abs() / scale)


def check(name, x_cpu, g_cpu):
    xr = x_cpu.clone().requires_grad_(True)
    yr = eager_ref(xr, COS, SIN)
    yr.backward(g_cpu)
    ref_y, ref_dx = yr.detach(), xr.grad.detach()

    xk = x_cpu.to("neuron").clone().requires_grad_(True)
    yk = RopeNormNKI.apply(xk, COS.to("neuron"), SIN.to("neuron"))
    yk.backward(g_cpu.to("neuron"))
    got_y, got_dx = yk.detach().cpu(), xk.grad.detach().cpu()

    rows = []
    for tag, got, ref in (("fwd", got_y, ref_y), ("bwd", got_dx, ref_dx)):
        d = (got.float() - ref.float()).abs()
        u = ulps(got, ref)
        exact = (got == ref).float().mean().item() * 100
        rel_l2 = d.norm().item() / max(ref.float().norm().item(), 1e-30)
        cs = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
        rows.append((tag, d.max().item(), u.max().item(), exact, rel_l2, cs))
    print(f"{name}")
    for tag, mx, mu, ex, rl, cs in rows:
        print(f"   {tag}: max_abs={mx:.3e}  max_ulp={mu:6.2f}  bit_exact={ex:6.2f}%  "
              f"rel_l2={rl:.3e}  cos={cs:.9f}")
    return rows


print(f"shape=({B},{T},{H},{D}) dtype={DT}\n")

for seed in (0, 1, 7):
    torch.manual_seed(seed)
    x = (torch.randn(B, T, H, D) * 0.7).to(DT)
    g = (torch.randn(B, T, H, D) * 0.3).to(DT)
    check(f"seed={seed} scale=0.7", x, g)

torch.manual_seed(3)
check("large magnitude (x*40)", (torch.randn(B, T, H, D) * 40).to(DT),
      (torch.randn(B, T, H, D) * 5).to(DT))

torch.manual_seed(4)
check("tiny magnitude (x*1e-3)", (torch.randn(B, T, H, D) * 1e-3).to(DT),
      (torch.randn(B, T, H, D) * 1e-3).to(DT))

torch.manual_seed(5)
check("mixed scale per head", (torch.randn(B, T, H, D)
                              * torch.tensor([0.01, 1.0, 100.0])[None, None, :, None]).to(DT),
      (torch.randn(B, T, H, D) * 0.3).to(DT))

torch.manual_seed(6)
check("x all zeros (degenerate rms)", torch.zeros(B, T, H, D, dtype=DT),
      (torch.randn(B, T, H, D) * 0.3).to(DT))

torch.manual_seed(8)
check("grad all zeros", (torch.randn(B, T, H, D) * 0.7).to(DT),
      torch.zeros(B, T, H, D, dtype=DT))

print("\nDONE")
