"""T1.8, softcap+CE half, in a FRESH process (see t18_nki_graphbreak_min.py for the verdict).

Split out because running it after a fullgraph compile in the same process stalls on the next
device allocation.

KNOWN UNRESOLVED HANG, recorded rather than hidden: the eager call succeeds in 0.2 s
(`eager ce OK calls=1`), but `torch._dynamo.explain` on `_SoftcapCrossEntropyNKI.apply` then hangs
indefinitely, where the rope kernel's explain() took 0.7 s. Suspected to be the int64 target index
path. This is a HARNESS artifact and is NOT evidence either way about graph breaks -- the T1.8
verdict rests on the rope measurement plus the previously recorded whole-model 0->0 break / 189-NEFF
numbers, which cover both kernels.
"""
import os, sys, time
REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO); os.chdir(REPO)
T0=time.time()
def p(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
p("import torch"); import torch
import torch_neuronx, train as T
p(f"_HAS_NKI={T._HAS_NKI}")
dev=torch.device("neuron")
p("randn z on cpu"); zc=torch.randn(8,8192)
p("z -> dev"); z=zc.to(dev); p("z done")
p("tgt cpu int64"); tc=torch.randint(0,8192,(8,),dtype=torch.int64)
p("tgt -> dev"); tgt=tc.to(dev); p("tgt done")
z.requires_grad_(True)
ran={"n":0}; _o=T._softcap_ce_fwd
T._softcap_ce_fwd=lambda *a,**k:(ran.__setitem__("n",ran["n"]+1),_o(*a,**k))[1]
ce=lambda z,t: T._SoftcapCrossEntropyNKI.apply(z,t)
p("eager ce"); l=ce(z,tgt); p(f"eager ce OK calls={ran['n']} out={l.shape if torch.is_tensor(l) else l}")
p("explain"); torch._dynamo.reset(); n0=ran["n"]
e=torch._dynamo.explain(ce)(z,tgt)
p(f"softcap_ce: graph_count={e.graph_count} graph_break_count={e.graph_break_count} op_count={e.op_count} kernel_calls_during_trace={ran['n']-n0}")
for i,r in enumerate(e.break_reasons or []): p(f"  break[{i}]: {getattr(r,'reason',r)}")
p("fullgraph"); torch._dynamo.reset()
try:
    o=torch.compile(ce,backend="neuron",fullgraph=True)(z,tgt); p("  softcap_ce: fullgraph=True SUCCEEDED")
except Exception as ex: p(f"  softcap_ce: fullgraph=True FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:200]}")
p("ALLDONE")
