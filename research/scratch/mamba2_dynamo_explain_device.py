"""Dynamo graph structure of the Mamba2 path AS IT ACTUALLY RUNS -- on device.

WHY THIS EXISTS (a real methodological correction to mamba2_dynamo_explain.py).
That script runs with NEURON_COMPETITION_R1_DEVICE_TYPE=cpu, so its tensors are
CPU tensors, so `ssd_scan`'s device guard

    if USE_NKI_SSD and _HAS_NKI and x.device.type == "neuron" and x.ndim == 4:

is FALSE and the traced body is `ssd_chunked` -- the pure-PyTorch eager scan. Its
headline result ("one graph, zero breaks, 534 ops") therefore describes a graph
that is NOT the one the real `--nki-ssd` run compiles. On device the same call
site dispatches to `_SSDNKI.apply`, a `torch.autograd.Function` whose forward
invokes an `@nki.jit` kernel -- exactly the construct the post-compile re-profile
already found Dynamo cannot trace through for rope_norm/softcap_ce ("small
serialized islands", trn2-experiment-results.md).

So: same measurement, on neuron tensors, with --nki-ssd both ON and OFF, so the
island count is attributable rather than inferred.

Run on the box (single core, no torchrun needed):
    python research/scratch/mamba2_dynamo_explain_device.py
"""

import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
import torch._dynamo as dynamo

import train as T_


def report(name, model, idx):
    dynamo.reset()
    exp = dynamo.explain(model)(idx[:, :-1], idx[:, 1:])
    reasons = {}
    for gb in exp.break_reasons:
        key = str(getattr(gb, "reason", gb)).strip().split("\n")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"\n===== {name} =====")
    print(f"  graphs produced : {exp.graph_count}")
    print(f"  graph breaks    : {exp.graph_break_count}")
    print(f"  ops captured    : {exp.op_count}")
    print(f"  ops per graph   : {[len(g.graph.nodes) for g in exp.graphs]}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}x  {k[:180]}")
    return name, exp.graph_count, exp.graph_break_count, exp.op_count


def build(mamba, nki_ssd, pattern="MMAMMM"):
    T_.USE_MAMBA2 = mamba
    T_.MAMBA2_PATTERN = pattern
    T_.USE_NKI_SSD = nki_ssd
    torch.manual_seed(0)
    cfg = T_.build_config()
    m = T_.GPT(cfg)
    m.init_weights()
    return m.to(torch.device("neuron")), cfg


def main():
    dev = torch.device("neuron")
    print(f"has_nki={T_._HAS_NKI}")
    rows = []
    for label, mamba, nki in (("transformer (compiled path)", False, True),
                              ("mamba2 MMAMMM, --nki-ssd", True, True),
                              ("mamba2 MMAMMM, --no-nki-ssd", True, False)):
        model, cfg = build(mamba, nki)
        # Short sequence keeps the trace cheap; graph STRUCTURE is shape-independent
        # for this model (no data-dependent control flow), and the NKI guard only
        # needs T % chunk == 0, which SEQ_LEN itself satisfies. Use the real
        # SEQ_LEN so the guard resolves exactly as it does in training.
        idx = torch.randint(0, cfg.vocab_size, (2, T_.SEQ_LEN + 1),
                            generator=torch.Generator().manual_seed(3)).to(dev)
        rows.append(report(label, model, idx))
        del model
    print("\n" + "=" * 88)
    print(f"{'config':32s} {'graphs':>8s} {'breaks':>8s} {'ops':>8s}")
    for name, g, b, o in rows:
        print(f"{name:32s} {g:8d} {b:8d} {o:8d}")


if __name__ == "__main__":
    sys.exit(main())
