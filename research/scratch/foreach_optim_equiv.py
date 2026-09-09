"""Exactness harness for USE_FOREACH_OPTIM (MuonAdamW._adamw_step_foreach + the batched
Muon read-in).

The claim being tested is strong: the foreach path is the SAME update math, only
restructured for fewer dispatches, so the resulting parameters and BOTH AdamW moment
buffers must match the loop path bit-for-bit -- not "within tolerance". Anything above
0.0 max abs diff is a real behavioural change and disqualifies the rewrite.

Runs entirely on CPU against the real model at the locked config, on the real optimizer
groups built by GPT.setup_optimizer(), with identical synthetic grads fed to both sides.
Also reports the per-group dispatch counts, which is the thing the change is actually
buying.
"""
import copy
import os
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
sys.argv = ["equiv"]  # keep train.py's argparse out of this

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train as T

STEPS = 5
SEED = 1234


class Count(TorchDispatchMode):
    def __init__(self):
        self.n = 0
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        k = str(func)
        self.ops[k] = self.ops.get(k, 0) + 1
        return func(*args, **(kwargs or {}))


def build():
    torch.manual_seed(SEED)
    cfg = T.build_config()
    m = T.GPT(cfg)
    return m


def make_grads(model, step):
    """Deterministic synthetic grads, identical for both sides."""
    g = torch.Generator().manual_seed(9000 + step)
    out = {}
    for name, p in model.named_parameters():
        out[name] = torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * 0.01
    return out


def run(model, grads_per_step, foreach, count_dispatch=False):
    T.USE_FOREACH_OPTIM = foreach
    opt = model.setup_optimizer()
    named = dict(model.named_parameters())
    counts = []
    for s in range(STEPS):
        gs = grads_per_step[s]
        for name, p in named.items():
            p.grad = gs[name].clone().to(p.dtype)
        # exercise the real per-step lr write, the thing _device_scalar exists for
        for grp in opt.param_groups:
            grp["lr"] = grp["initial_lr"] * (1.0 - 0.3 * s / STEPS)
        if count_dispatch and s == STEPS - 1:
            with Count() as c:
                opt.step()
            counts.append(c)
        else:
            opt.step()
    return opt, counts


def compare(label, m_ref, m_new, opt_ref, opt_new):
    print(f"\n--- {label} ---")
    worst = 0.0
    worst_where = ""
    nparam = 0
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_new.named_parameters()):
        assert n1 == n2
        nparam += 1
        d = (p1.float() - p2.float()).abs().max().item()
        if d > worst:
            worst, worst_where = d, f"param {n1}"
        if not torch.equal(p1, p2):
            print(f"   BITWISE MISMATCH param {n1}: max abs {d:.3e}")
    # optimizer state: both AdamW moments and Muon's buffers
    keys = ("exp_avg", "exp_avg_sq", "momentum_buffer", "second_momentum_buffer")
    nstate = 0
    for (pr, sr), (pn, sn) in zip(opt_ref.state.items(), opt_new.state.items()):
        for k in keys:
            if k in sr and k in sn:
                nstate += 1
                d = (sr[k].float() - sn[k].float()).abs().max().item()
                if d > worst:
                    worst, worst_where = d, f"state[{k}]"
                if not torch.equal(sr[k], sn[k]):
                    print(f"   BITWISE MISMATCH state[{k}] shape {tuple(sr[k].shape)}: max abs {d:.3e}")
        if "step" in sr and sr.get("step") != sn.get("step"):
            print(f"   STEP COUNTER MISMATCH: {sr['step']} vs {sn['step']}")
    print(f"   compared {nparam} params + {nstate} state tensors over {STEPS} steps")
    print(f"   max abs diff = {worst:.3e}  ({worst_where or 'exact'})")
    print(f"   VERDICT: {'BIT-IDENTICAL' if worst == 0.0 else 'DIFFERS -- NOT EQUIVALENT'}")
    return worst


print(f"torch {torch.__version__}  |  locked config DEPTH={T.DEPTH} ASPECT_RATIO={T.ASPECT_RATIO}")

for tensor_lr in (True, False):
    T.USE_TENSOR_LR_SCALARS = tensor_lr
    m_ref = build()
    m_new = copy.deepcopy(m_ref)
    assert all(torch.equal(a, b) for a, b in zip(m_ref.parameters(), m_new.parameters()))
    grads = [make_grads(m_ref, s) for s in range(STEPS)]

    opt_ref, c_ref = run(m_ref, grads, foreach=False, count_dispatch=True)
    opt_new, c_new = run(m_new, grads, foreach=True, count_dispatch=True)
    compare(f"USE_TENSOR_LR_SCALARS={tensor_lr}", m_ref, m_new, opt_ref, opt_new)
    if tensor_lr:
        print(f"   dispatches in ONE optimizer step: loop={c_ref[0].n}  foreach={c_new[0].n} "
              f"({c_ref[0].n - c_new[0].n} fewer, {c_ref[0].n / max(1,c_new[0].n):.2f}x)")

# group composition, so the launch-count arithmetic is on the record
T.USE_TENSOR_LR_SCALARS = True
T.USE_FOREACH_OPTIM = False
m = build()
opt = m.setup_optimizer()
print("\n--- real optimizer group composition at the locked config ---")
tot = 0
for i, g in enumerate(opt.param_groups):
    shapes = sorted({tuple(p.shape) for p in g["params"]})
    dts = sorted({str(p.dtype) for p in g["params"]})
    tot += len(g["params"])
    print(f"  group {i:2d} kind={g['kind']:6s} n={len(g['params']):3d} wd={g['weight_decay']:<6} "
          f"dtypes={dts} shapes={shapes if len(shapes) <= 3 else str(shapes[:3]) + '...'}")
print(f"  total params: {tot}  |  adamw groups: {sum(1 for g in opt.param_groups if g['kind']=='adamw')}"
      f"  muon groups: {sum(1 for g in opt.param_groups if g['kind']=='muon')}")

# per-kind dispatch breakdown
for kind in ("adamw", "muon"):
    n_loop = n_fe = 0
    for g in opt.param_groups:
        if g["kind"] != kind:
            continue
        for p in g["params"]:
            p.grad = torch.randn_like(p) * 0.01
        T.USE_FOREACH_OPTIM = False
        with Count() as c1:
            opt._adamw_step(g) if kind == "adamw" else opt._muon_step(g)
        T.USE_FOREACH_OPTIM = True
        with Count() as c2:
            opt._adamw_step_foreach(g) if kind == "adamw" else opt._muon_step(g)
        n_loop += c1.n
        n_fe += c2.n
    print(f"  {kind:6s}: loop={n_loop:4d} dispatches -> foreach={n_fe:4d} "
          f"({n_loop - n_fe:+d}, {n_loop/max(1,n_fe):.2f}x)")
