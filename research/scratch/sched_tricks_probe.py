"""Localiser for the (4a) neutral-per-depth mismatch in sched_tricks_equiv.py.

(4a) asserts that turning per-depth Muon scaling on with multipliers of exactly 1.0 reproduces the
0-dim operand path bit for bit. It does not, by ~3e-5 on fp32 weights -- a bf16-scale difference, so
something in the [K,1,1] path is rounding through a different dtype rather than merely reassociating.
This intercepts _muon_math and compares, per group and per step: first the five scalar operands it
receives, then every tensor it returns. Whichever diverges first names the line to fix.

Run: python research/scratch/sched_tricks_probe.py
"""
import os
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
sys.argv = ["probe"]

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import train as T

STEPS = 2
SEED = 1234
_ORIG_MATH = T._muon_math


def instrument(sink):
    def wrapped(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
                tall, ns_steps, red_dim, red_dim_size, one_beta2, to_bf16):
        key = tuple(g.shape)
        rec = {"mom": mom, "one_mom": one_mom, "lr": lr, "lrwd": lrwd,
               "one_beta2": one_beta2, "g_in": g.clone(), "w_in": w.clone(),
               "mbuf_in": momentum_buffer.clone(), "to_bf16": to_bf16}
        out = _ORIG_MATH(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
                         tall, ns_steps, red_dim, red_dim_size, one_beta2, to_bf16)
        rec["w_out"], rec["mbuf_out"], rec["second_out"] = (t.clone() for t in out)
        sink.setdefault(key, []).append(rec)
        return out
    return wrapped


def run(depth_on):
    torch.manual_seed(SEED)
    model = T.GPT(T.build_config())
    sink = {}
    # _MUON_COMPILED caches the function object per (shape, tall, ns_steps, red_dim), so without
    # clearing it the second run would keep calling the FIRST run's instrumented wrapper and log
    # into the wrong sink.
    T._MUON_COMPILED.clear()
    T._muon_math = instrument(sink)
    try:
        opt = model.setup_optimizer()
        named = dict(model.named_parameters())
        for s in range(STEPS):
            gg = torch.Generator().manual_seed(9000 + s)
            for n, p in named.items():
                p.grad = (torch.randn(p.shape, generator=gg, dtype=torch.float32) * 0.01).to(p.dtype)
            prog = s / max(1, STEPS - 1)
            for grp in opt.param_groups:
                grp["lr"] = grp["initial_lr"] * (1.0 - 0.3 * prog)
                if grp["kind"] == "muon":
                    grp["momentum"] = T.muon_momentum(s, prog)
            opt.step()
    finally:
        T._muon_math = _ORIG_MATH
        T._MUON_COMPILED.clear()
    return sink


def scalarize(x):
    if isinstance(x, torch.Tensor):
        return x.flatten().tolist(), tuple(x.shape), str(x.dtype)
    return [float(x)], (), "python-float"


T.MUON_DEPTH_LR_BOTTOM = T.MUON_DEPTH_LR_TOP = 1.0
T.MUON_DEPTH_MOM_BOTTOM = T.MUON_DEPTH_MOM_TOP = 0.0
off = run(False)
T.MUON_DEPTH_MOM_BOTTOM = T.MUON_DEPTH_MOM_TOP = T.MUON_DEPTH_MOM_REF
on = run(True)

print(f"groups OFF={sorted(off)}  ON={sorted(on)}")
bad = 0
for key in sorted(off):
    for i, (a, b) in enumerate(zip(off[key], on[key])):
        for name in ("mom", "one_mom", "lr", "lrwd", "one_beta2"):
            va, sa, da = scalarize(a[name])
            vb, sb, db = scalarize(b[name])
            uniq_b = sorted(set(vb))
            same = len(set(va)) == 1 and len(uniq_b) == 1 and va[0] == uniq_b[0]
            flag = "" if same else "   <-- DIFFERS"
            if not same:
                bad += 1
            print(f"{key} step{i} {name:>9}: OFF {da}{sa} {va[:2]}  ON {db}{sb} {uniq_b[:3]}{flag}")
        for name in ("g_in", "w_in", "mbuf_in", "w_out", "mbuf_out", "second_out"):
            d = (a[name].float() - b[name].float()).abs().max().item()
            if d != 0.0:
                bad += 1
                print(f"{key} step{i} {name:>9}: max abs diff {d:.3e}  "
                      f"dtypes {a[name].dtype}/{b[name].dtype}   <-- DIFFERS")
        print(f"{key} step{i} to_bf16={a['to_bf16']}")
print(f"\n{'FOUND DIVERGENCE' if bad else 'no divergence inside _muon_math'} ({bad} items)")
