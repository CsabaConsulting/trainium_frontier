"""Why does torch.compile help the transformer and NOT the Mamba2 path?

MEASURED ON DEVICE (4 ranks, lnc=2, DEPTH=6, SEQ_LEN=2048):

    transformer   eager 2.18 s/step  ->  compiled 0.92 s/step   (2.37x)
    Mamba2 MMAMMM eager 1.96 s/step  ->  compiled 2.17 s/step   (0.90x -- WORSE)

Eager-to-eager the Mamba2 hybrid is the FASTER of the two (1.96 vs 2.18) at 19%
fewer parameters, and a fixed-300-step A/B put their quality within noise
(1.270639 vs 1.269458 bpb). So the whole wall-clock deficit is torch.compile,
not the architecture -- which makes "where does Dynamo break the Mamba2 graph"
the single question worth answering.

Graph breaks are a Dynamo-level property of the Python code, independent of the
backend, so this runs on CPU and costs no device time. It reports, per
architecture, the number of graphs Dynamo produced and every break reason.

Run: python research/scratch/mamba2_dynamo_explain.py
"""

import os
import sys

os.environ.setdefault("NEURON_COMPETITION_R1_DEVICE_TYPE", "cpu")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
import torch._dynamo as dynamo

import train as T_

torch.set_num_threads(1)

SEQ, VOCAB, NEMBD, NHEAD = 256, 512, 128, 2
T_.MAMBA2_HEAD_DIM = 64
T_.MAMBA2_D_STATE = 16
T_.MAMBA2_CHUNK = 64
T_.COMPUTE_DTYPE = torch.float32
CFG = dict(sequence_len=SEQ, vocab_size=VOCAB, n_layer=T_.DEPTH, n_head=NHEAD,
           n_embd=NEMBD)


def build(mamba: bool, pattern="MMAMMM"):
    T_.USE_MAMBA2 = mamba
    T_.MAMBA2_PATTERN = pattern
    torch.manual_seed(0)
    m = T_.GPT(T_.GPTConfig(**CFG))
    m.init_weights()
    return m


def report(name: str, model):
    dynamo.reset()
    idx = torch.randint(0, VOCAB, (2, SEQ + 1),
                        generator=torch.Generator().manual_seed(3))
    exp = dynamo.explain(model)(idx[:, :-1], idx[:, 1:])
    print(f"\n===== {name} =====")
    print(f"  graphs produced : {exp.graph_count}")
    print(f"  graph breaks    : {exp.graph_break_count}")
    print(f"  ops captured    : {exp.op_count}")
    reasons = {}
    for gb in exp.break_reasons:
        key = str(getattr(gb, "reason", gb)).strip().split("\n")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}x  {k[:150]}")
    return exp.graph_count, exp.graph_break_count, exp.op_count


if __name__ == "__main__":
    rows = []
    rows.append(("transformer", *report("transformer (USE_MAMBA2=False)", build(False))))
    rows.append(("mamba2 MMAMMM", *report("Mamba2 hybrid MMAMMM", build(True, "MMAMMM"))))
    rows.append(("mamba2 all-M", *report("Mamba2 every layer", build(True, "MMMMMM"))))
    print("\n" + "=" * 72)
    print(f"{'arch':16s} {'graphs':>8s} {'breaks':>8s} {'ops':>8s}  breaks/layer")
    for name, g, b, o in rows:
        print(f"{name:16s} {g:8d} {b:8d} {o:8d}  {b / T_.DEPTH:12.2f}")
    print("\nA break per Mamba2 layer means the neuron backend never sees the mixer as")
    print("one graph, so it cannot fuse the ~25 elementwise/reshape ops the mixer")
    print("spends most of its time on -- which is exactly the 2.37x the transformer")
    print("gets and the Mamba2 path does not.")
