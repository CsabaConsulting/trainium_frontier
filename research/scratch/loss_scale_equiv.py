"""CPU equivalence checks for static loss scaling (--loss-scale).

Convention: research/scratch/*_equiv.py numeric checks, CPU torch only.
Run:  python3 research/scratch/loss_scale_equiv.py
Covers:
  1. unscale_grads_ restores scaled grads EXACTLY (buffer path and p.grad path).
  2. Full recipe (scale loss -> backward -> unscale -> step) is bit-identical
     to plain training for power-of-2 scales on CPU.
  3. scale == 1.0 is a strict no-op (helper returns before touching grads).
Hardware must still confirm the actual point: scaled grads stay representable
under fp8 auto-cast on Trainium (sub1-p2b-autocast-scale smoke).
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(f"FAILED: {name}")


torch.manual_seed(7)

# --- 1: unscale helper exactness, both paths ---
g = torch.randn(32, 16)
buf = g.clone().mul_(1024.0)
T.unscale_grads_(None, buf, 1024.0)
check("buffer path exact", torch.equal(buf, g))

lin = torch.nn.Linear(16, 8)
ref = [p.detach().clone() for p in lin.parameters()]
x = torch.randn(5, 16)
loss = lin(x).square().mean()
S = 1024.0
(loss * S).backward()
T.unscale_grads_(lin, None, S)
plain = torch.nn.Linear(16, 8)
with torch.no_grad():
    for p, r in zip(plain.parameters(), ref):
        p.copy_(r)
loss0 = plain(x).square().mean()
loss0.backward()
check("p.grad path exact",
      all(torch.equal(a.grad, b.grad) for a, b in zip(lin.parameters(), plain.parameters())))

# --- 2: end-to-end scaled recipe == plain, bit-exact ---
def run_scaled(scale):
    torch.manual_seed(11)
    m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    torch.manual_seed(13)
    xs = [torch.randn(6, 16) for _ in range(3)]
    ys = [torch.randn(6, 4) for _ in range(3)]
    for xb, yb in zip(xs, ys):
        opt.zero_grad()
        l = (m(xb) - yb).square().mean()
        (l * scale).backward()
        T.unscale_grads_(m, None, scale)
        opt.step()
    return [p.detach().clone() for p in m.parameters()]

a, b = run_scaled(1.0), run_scaled(1024.0)
check("scaled recipe bit-identical to plain",
      all(torch.equal(x, y) for x, y in zip(a, b)))

# --- 3: scale 1.0 is a strict no-op ---
m = torch.nn.Linear(8, 8)
m.weight.grad = torch.randn_like(m.weight)
before = m.weight.grad.clone()
T.unscale_grads_(m, None, 1.0)
check("scale 1.0 untouched", torch.equal(m.weight.grad, before))

print("ALL LOSS-SCALE CHECKS PASSED")
