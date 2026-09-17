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

# --- 4: dynamic transition sequences ---
s, c, skip = T.loss_scale_transition(1024.0, 0, False, 500, 8192.0)
check("overflow halves+skips", (s, c, skip) == (512.0, 0, True))
s, c, skip = T.loss_scale_transition(1.0, 0, False, 500, 8192.0)
check("overflow floors at 1.0", (s, c, skip) == (1.0, 0, True))
s, c, skip = T.loss_scale_transition(1024.0, 499, True, 500, 8192.0)
check("500th clean doubles", (s, c, skip) == (2048.0, 0, False))
s, c, skip = T.loss_scale_transition(8192.0, 999, True, 500, 8192.0)
check("cap respected", (s, c, skip) == (8192.0, 1000, False))
s, c, skip = T.loss_scale_transition(256.0, 10, True, 500, 8192.0)
check("clean steps through", (s, c, skip) == (256.0, 11, False))

# --- 5: grads_finite_ cases ---
m = torch.nn.Linear(4, 4)
m.weight.grad = torch.randn(4, 4)
check("finite grads true", T.grads_finite_(m, None) is True)
m.weight.grad[0, 0] = float("inf")
check("inf detected", T.grads_finite_(m, None) is False)
m.weight.grad[0, 0] = float("nan")
check("nan detected", T.grads_finite_(m, None) is False)
m.weight.grad = None
check("all-None is finite", T.grads_finite_(m, None) is True)
buf = torch.randn(16)
check("buffer finite", T.grads_finite_(None, buf) is True)
buf[3] = float("inf")
check("buffer inf detected", T.grads_finite_(None, buf) is False)

print("ALL LOSS-SCALE CHECKS PASSED")
