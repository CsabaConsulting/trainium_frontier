"""T1.8 graph-break measurement, minimal shapes, one print per device op.

This is the probe that produced the T1.8 verdict (research/run-logs/t18-min.log):

    rope_norm: graph_count=1 graph_break_count=0 op_count=2 kernel_calls_during_trace=1
    rope_norm: fullgraph=True SUCCEEDED

i.e. the existing torch.autograd.Function-wrapped NKI kernel is ALREADY captured in a single
unbroken graph, so there is no compile island for `torch_neuronx.nki_op` to remove.

Design notes, all of them scars from the fuller version (t18_nki_traceability_probe.py):
  * Inputs must be on the **neuron** device. An earlier probe of this same question traced CPU
    tensors; `nki.framework.kernel.Kernel.__call__` routes on `has_torch_input` and train.py guards
    on `device.type == "neuron"`, so a CPU trace never reaches the kernel and reports 0 breaks for
    the wrong reason.
  * `kernel_calls_during_trace` is the null test: the @nki.jit symbol is monkeypatched with a
    counter, and a 0 there would mean every number here is vacuous.
  * SMALL shapes on purpose. At [1024, 8192] the *`torch.randn` on device* compiles its own NEFF and
    stalls for >13 minutes -- nothing to do with the kernels. Build on CPU and `.to(dev)` (8.9 s).
  * The softcap-CE half is split into t18_nki_graphbreak_ce.py because running it in the SAME
    process after a fullgraph compile stalls on the next device allocation; in a fresh process the
    allocation is instant. See that file for its own (unresolved) hang.

Run under ~/workspace/native_venv, needs a free device.
"""
import os, sys, time
REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO); os.chdir(REPO)
T0 = time.time()
def p(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
p("importing torch"); import torch
p("importing torch_neuronx"); import torch_neuronx
p("importing train"); import train as T
p(f"_HAS_NKI={T._HAS_NKI}")
dev = torch.device("neuron")
p("alloc x cpu->neuron")
x = torch.randn(1, 8, 5, 128).to(dev); p("x done")
cos = torch.randn(8, 64).to(dev); sin = torch.randn(8, 64).to(dev); p("cos/sin done")
x.requires_grad_(True)
ran = {"rope": 0, "ce": 0}
_or, _oc = T._rope_norm_fwd, T._softcap_ce_fwd
T._rope_norm_fwd = lambda *a, **k: (ran.__setitem__("rope", ran["rope"]+1), _or(*a, **k))[1]
T._softcap_ce_fwd = lambda *a, **k: (ran.__setitem__("ce", ran["ce"]+1), _oc(*a, **k))[1]
rope = lambda x, c, s: T._RopeNormNKI.apply(x, c, s)
p("eager rope call (first = kernel compile)")
y = rope(x, cos, sin); p(f"eager rope OK shape={tuple(y.shape)} calls={ran['rope']}")
for name, fn, a in (("rope_norm", rope, (x, cos, sin)),):
    p(f"explain {name}"); torch._dynamo.reset()
    n0 = ran["rope"]
    e = torch._dynamo.explain(fn)(*a)
    p(f"{name}: graph_count={e.graph_count} graph_break_count={e.graph_break_count} "
      f"op_count={e.op_count} kernel_calls_during_trace={ran['rope']-n0}")
    for i, r in enumerate(e.break_reasons or []):
        p(f"  break[{i}]: {getattr(r,'reason',r)}")
    for i, g in enumerate(e.graphs or []):
        code = g.code if isinstance(g.code, str) else str(g.code)
        hits = [l.strip() for l in code.splitlines() if "nki" in l.lower()]
        p(f"  graph[{i}] nki lines: {hits or 'NONE'}")
    p(f"fullgraph {name}"); torch._dynamo.reset()
    try:
        o = torch.compile(fn, backend="neuron", fullgraph=True)(*a)
        p(f"  {name}: fullgraph=True SUCCEEDED shape={tuple(o.shape)}")
    except Exception as ex:
        p(f"  {name}: fullgraph=True FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:200]}")
p("alloc z for softcap_ce")
z = torch.randn(8, 8192).to(dev); z.requires_grad_(True)
tgt = torch.randint(0, 8192, (8,), dtype=torch.int64).to(dev); p("z/tgt done")
ce = lambda z, t: T._SoftcapCrossEntropyNKI.apply(z, t)
p("eager ce call (first = kernel compile)")
try:
    l = ce(z, tgt); p(f"eager ce OK {l.shape if torch.is_tensor(l) else l} calls={ran['ce']}")
    p("explain ce"); torch._dynamo.reset(); n0 = ran["ce"]
    e = torch._dynamo.explain(ce)(z, tgt)
    p(f"softcap_ce: graph_count={e.graph_count} graph_break_count={e.graph_break_count} "
      f"op_count={e.op_count} kernel_calls_during_trace={ran['ce']-n0}")
    for i, r in enumerate(e.break_reasons or []):
        p(f"  break[{i}]: {getattr(r,'reason',r)}")
    p("fullgraph ce"); torch._dynamo.reset()
    try:
        o = torch.compile(ce, backend="neuron", fullgraph=True)(z, tgt)
        p("  softcap_ce: fullgraph=True SUCCEEDED")
    except Exception as ex:
        p(f"  softcap_ce: fullgraph=True FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:200]}")
except Exception as ex:
    p(f"ce path FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:300]}")
p("ALLDONE")
