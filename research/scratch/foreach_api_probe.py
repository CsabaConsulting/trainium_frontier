"""Probe the real torch._foreach_* API surface on THIS installed torch, on CPU.

Written because the AdamW rewrite depends on three things that differ across torch
versions: whether the in-place ops accept a 0-dim Tensor scalar (the tensor-lr case),
whether they accept a per-tensor ScalarList (the per-parameter bias-correction case),
and whether mixed shapes in one call are allowed (our groups are ragged).
"""
import torch

print("torch:", torch.__version__)


def probe(name, fn):
    try:
        out = fn()
        print(f"  OK    {name}" + (f" -> {out}" if out is not None else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:150]}")
        return False


def mk(shapes, dtype=torch.float32, val=2.0):
    return [torch.full(s, val, dtype=dtype) for s in shapes]


RAGGED = [(4,), (3, 5), (), (2, 2, 2)]  # deliberately mixed shapes AND a 0-dim
print("\n[1] ragged shapes in one call (same dtype/device, different shapes)")
probe("_foreach_mul_(ragged, 0.5)", lambda: torch._foreach_mul_(mk(RAGGED), 0.5))
probe("_foreach_lerp_(ragged, ragged, 0.1)", lambda: torch._foreach_lerp_(mk(RAGGED), mk(RAGGED), 0.1))
probe("_foreach_add_(ragged, ragged, alpha=-0.3)", lambda: torch._foreach_add_(mk(RAGGED), mk(RAGGED), alpha=-0.3))
probe("_foreach_sub_(ragged, ragged)", lambda: torch._foreach_sub_(mk(RAGGED), mk(RAGGED)))
probe("_foreach_sqrt_(ragged)", lambda: torch._foreach_sqrt_(mk(RAGGED)))
probe("_foreach_div(ragged, ragged)", lambda: len(torch._foreach_div(mk(RAGGED), mk(RAGGED))))
probe("_foreach_copy_(ragged, ragged)", lambda: torch._foreach_copy_(mk(RAGGED), mk(RAGGED)))

print("\n[2] 0-dim Tensor as the scalar operand (the tensor-lr case)")
t = torch.tensor(0.5)
probe("_foreach_mul_(list, 0dim Tensor)", lambda: torch._foreach_mul_(mk(RAGGED), t))
probe("_foreach_div_(list, 0dim Tensor)", lambda: torch._foreach_div_(mk(RAGGED), t))
probe("_foreach_add_(list, 0dim Tensor)", lambda: torch._foreach_add_(mk(RAGGED), t))
probe("_foreach_lerp_(list, list, 0dim Tensor weight)", lambda: torch._foreach_lerp_(mk(RAGGED), mk(RAGGED), t))

print("\n[3] ScalarList (per-tensor scalars -- per-param bias correction)")
sl = [1.0, 2.0, 4.0, 8.0]
probe("_foreach_div(list, ScalarList)", lambda: [x.flatten()[0].item() for x in torch._foreach_div(mk(RAGGED), sl)])
probe("_foreach_div_(list, ScalarList)", lambda: torch._foreach_div_(mk(RAGGED), sl))
probe("_foreach_mul_(list, ScalarList)", lambda: torch._foreach_mul_(mk(RAGGED), sl))
probe("_foreach_add_(list, ScalarList)", lambda: torch._foreach_add_(mk(RAGGED), sl))

print("\n[4] mixed dtype in one call (should FAIL or silently split -- must know which)")
mixed = [torch.ones(4, dtype=torch.float32), torch.ones(4, dtype=torch.bfloat16)]
probe("_foreach_mul_(mixed dtypes, 0.5)", lambda: torch._foreach_mul_(mixed, 0.5))

print("\n[5] bitwise: _foreach_mul(g,g) vs g.square()")
g = torch.randn(1000, dtype=torch.float32)
a = torch._foreach_mul([g], [g])[0]
b = g.square()
print("   identical:", torch.equal(a, b))
gb = torch.randn(1000).bfloat16()
print("   bf16 identical:", torch.equal(torch._foreach_mul([gb], [gb])[0], gb.square()))

print("\n[6] writing INTO unbind() views (the muon stacked-buffer read-in case)")
buf = torch.zeros(3, 4)
views = list(buf.unbind(0))
srcs = [torch.full((4,), float(i + 1)) for i in range(3)]
probe("_foreach_copy_(unbind views, srcs)", lambda: torch._foreach_copy_(views, srcs))
print("   buf after:", buf.tolist())

print("\n[7] does foreach actually reduce op count? (CPU dispatch counter)")
from torch.utils._python_dispatch import TorchDispatchMode


class Count(TorchDispatchMode):
    def __init__(self):
        self.n = 0
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        self.names.append(str(func))
        return func(*args, **(kwargs or {}))


ps = mk([(8,)] * 20)
with Count() as c:
    for p in ps:
        p.mul_(0.9)
print("   loop  20x p.mul_:", c.n, "dispatches")
ps = mk([(8,)] * 20)
with Count() as c:
    torch._foreach_mul_(ps, 0.9)
print("   foreach _foreach_mul_:", c.n, "dispatches ->", c.names)
