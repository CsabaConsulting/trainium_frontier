#!/usr/bin/env python3
"""T1.8 empirical gate: are the two hand-written NKI kernels ALREADY traceable under
torch.compile, on NEURON tensors, on the current binary?

WHY THIS SCRIPT EXISTS RATHER THAN A SOURCE-LEVEL ARGUMENT. T1.8's premise is that the
kernels are opaque to Dynamo, so each one splits the graph and forms a compile island,
and that registering them via `torch_neuronx.nki_op` + `torch.library.register_autograd`
would fold them in. Reading nki/framework/kernel.py says the premise is already false --
`Kernel.__call__` does `if has_torch_input(...): return wrap_nki_to_torch_hop(self, ...)(...)`,
and `wrap_nki_to_torch_hop` IS `torch_neuronx.wrap_nki` (same `register_kernel_to_torch` +
`NKIHOPCaller`, differing only in grid `[kernel.lnc]` vs `[]`). But the project's standing
rule is "comment says dead, probe says dead, THEN delete", and an earlier probe of exactly
this question had a methodology bug: it traced CPU tensors, and `has_torch_input` routing
plus the `device.type == "neuron"` guards in train.py mean a CPU trace never reaches the
kernel at all -- it measured the wrong path and would have reported 0 breaks either way.

So this probe:
  * builds inputs on the NEURON device, at the real locked shapes;
  * asserts the kernel actually RAN (null test -- an all-fallback trace would otherwise
    look like a clean pass, which is the same failure mode the old probe had);
  * counts Dynamo graph breaks via torch._dynamo.explain, and separately tries
    fullgraph=True, which cannot succeed if the kernel is a genuine break;
  * prints the captured graph so the kernel's presence as a HOP node is visible rather
    than inferred.

Run under ~/workspace/native_venv on the box. Needs a free device.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402
import torch_neuronx  # noqa: E402

import train as T  # noqa: E402


def report_api():
    """What the plan asks for by name -- does it exist at all on THIS stack?"""
    print("=== API availability (T1.8 asks for nki_op + register_autograd) ===")
    for path in ("torch_neuronx.nki_op", "torch_neuronx.wrap_nki",
                 "torch.library.custom_op", "torch.library.register_autograd",
                 "torch.library.register_fake"):
        mod, _, attr = path.rpartition(".")
        try:
            obj = getattr(eval(mod), attr)  # noqa: S307 - fixed literal list above
            print(f"  {path}: PRESENT ({type(obj).__name__})")
        except Exception as exc:  # noqa: BLE001
            print(f"  {path}: MISSING ({exc})")
    try:
        import nki.framework.kernel as nfk
        import inspect
        src = inspect.getsource(nfk.Kernel.__call__)
        auto = "wrap_nki_to_torch_hop" in src and "has_torch_input" in src
        print(f"  nki.framework.kernel.Kernel.__call__ auto-routes torch inputs to a HOP: {auto}")
        same = getattr(nfk, "wrap_nki_to_torch_hop", None) is getattr(torch_neuronx, "wrap_nki", None)
        print(f"  wrap_nki_to_torch_hop IS torch_neuronx.wrap_nki: {same}")
    except Exception as exc:  # noqa: BLE001
        print(f"  nki.framework.kernel introspection failed: {exc}")
    print()


def explain(name, fn, args):
    """Graph-break count + graph text for one callable, on whatever device args live on."""
    torch._dynamo.reset()
    try:
        exp = torch._dynamo.explain(fn)(*args)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: explain() RAISED {type(exc).__name__}: {exc}")
        return None
    print(f"  {name}: graph_count={exp.graph_count} graph_break_count={exp.graph_break_count} "
          f"op_count={exp.op_count}")
    for i, r in enumerate(exp.break_reasons or []):
        print(f"    break[{i}]: {getattr(r, 'reason', r)}")
    for i, g in enumerate(exp.graphs or []):
        code = g.code if isinstance(g.code, str) else str(g.code)
        hop = [ln.strip() for ln in code.splitlines()
               if "nki" in ln.lower() or "hop" in ln.lower() or "custom" in ln.lower()]
        print(f"    graph[{i}] nki/hop-looking lines: {hop if hop else 'NONE'}")
    return exp


def fullgraph(name, fn, args):
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend="neuron", fullgraph=True)(*args)
        torch.neuron.synchronize() if hasattr(torch, "neuron") else None
        print(f"  {name}: fullgraph=True SUCCEEDED (no break) -> out "
              f"{tuple(out.shape) if torch.is_tensor(out) else type(out)}")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0][:300]
        print(f"  {name}: fullgraph=True FAILED {type(exc).__name__}: {msg}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="neuron")
    args = ap.parse_args()

    report_api()

    print(f"_HAS_NKI={T._HAS_NKI}")
    if not T._HAS_NKI:
        print("FAIL: NKI unavailable, nothing to measure")
        return 1

    dev = torch.device(args.device)
    print(f"=== device={dev} ===")

    # ---- locked shapes ----
    # rope+norm: x [B, T, H, D] with D = head_dim = 128, called from apply_rotary_emb.
    B, S, H, D = 1, 1024, 5, 128
    x = torch.randn(B, S, H, D, dtype=torch.float32, device=dev)
    cos = torch.randn(S, D // 2, dtype=torch.float32, device=dev)
    sin = torch.randn(S, D // 2, dtype=torch.float32, device=dev)
    x.requires_grad_(True)

    # softcap+CE: logits [N, V] fp32, targets int64.
    N, V = 1024, 8192
    z = torch.randn(N, V, dtype=torch.float32, device=dev)
    z.requires_grad_(True)
    tgt = torch.randint(0, V, (N,), dtype=torch.int64, device=dev)

    ran = {"rope": 0, "ce": 0}
    _orig_rope_fwd = T._rope_norm_fwd
    _orig_ce_fwd = T._softcap_ce_fwd

    # Null test. Without this an all-fallback trace (device guard not taken, or Dynamo
    # constant-folding the branch away) reports a clean 0 breaks for the WRONG reason --
    # which is exactly how the earlier CPU-tensor probe went wrong.
    def _rope_spy(*a, **k):
        ran["rope"] += 1
        return _orig_rope_fwd(*a, **k)

    def _ce_spy(*a, **k):
        ran["ce"] += 1
        return _orig_ce_fwd(*a, **k)

    T._rope_norm_fwd = _rope_spy
    T._softcap_ce_fwd = _ce_spy

    def rope_fn(x, cos, sin):
        return T._RopeNormNKI.apply(x, cos, sin)

    def ce_fn(z, tgt):
        return T._SoftcapCrossEntropyNKI.apply(z, tgt)

    print("\n=== eager sanity + null test (did the kernel actually run?) ===")
    y = rope_fn(x, cos, sin)
    l = ce_fn(z, tgt)
    print(f"  rope out {tuple(y.shape)} finite={bool(torch.isfinite(y).all())} calls={ran['rope']}")
    print(f"  ce   out {tuple(l.shape) if torch.is_tensor(l) else l} calls={ran['ce']}")
    if ran["rope"] == 0 or ran["ce"] == 0:
        print("FAIL: a kernel never ran in eager -- every number below would be vacuous")
        return 1

    print("\n=== torch._dynamo.explain on NEURON tensors (the measurement) ===")
    before = dict(ran)
    explain("rope_norm", rope_fn, (x, cos, sin))
    explain("softcap_ce", ce_fn, (z, tgt))
    print(f"  null test during tracing: rope calls +{ran['rope'] - before['rope']}, "
          f"ce calls +{ran['ce'] - before['ce']} (must be > 0)")

    print("\n=== torch.compile(fullgraph=True) -- cannot pass if the kernel breaks ===")
    ok_rope = fullgraph("rope_norm", rope_fn, (x, cos, sin))
    ok_ce = fullgraph("softcap_ce", ce_fn, (z, tgt))

    print("\n=== backward through the compiled region ===")
    for name, fn, a, seed in (("rope_norm", rope_fn, (x, cos, sin), None),
                              ("softcap_ce", ce_fn, (z, tgt), None)):
        torch._dynamo.reset()
        try:
            out = torch.compile(fn, backend="neuron")(*a)
            loss = out.float().sum()
            loss.backward()
            g = a[0].grad
            print(f"  {name}: backward OK, grad {tuple(g.shape)} "
                  f"finite={bool(torch.isfinite(g).all())}")
            a[0].grad = None
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: backward FAILED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:200]}")

    print("\nVERDICT INPUTS: fullgraph rope={} ce={}".format(ok_rope, ok_ce))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
