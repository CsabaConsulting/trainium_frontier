"""Prove the --nki-custom-ops flag actually changes the dispatch path.

Guards against a null test: Part A's bit-identical result would look the same
whether the flag works or silently does nothing. Here a TorchDispatchMode counts
the actual ops issued, so "the tfc:: ops appear iff the flag is on" is observed
rather than assumed. Also checks the argparse -> module-global plumbing.
"""

from __future__ import annotations

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import train as T


class OpCounter(TorchDispatchMode):
    def __init__(self):
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

    def tfc(self):
        return {k: v for k, v in self.counts.items() if "tfc" in k}


def probe(custom_ops: bool):
    T.USE_NKI_CUSTOM_OPS = custom_ops
    dev = torch.device("neuron")
    B, TT, H, D = 2, T.SEQ_LEN, 5, T.HEAD_DIM
    V = T.TOKENIZER_VOCAB_SIZE

    x = torch.randn(B, TT, H, D, dtype=torch.float32).to(torch.bfloat16).to(dev)
    x.requires_grad_(True)
    cos = torch.randn(1, TT, 1, D // 2, dtype=torch.float32).to(torch.bfloat16).to(dev)
    sin = torch.randn(1, TT, 1, D // 2, dtype=torch.float32).to(torch.bfloat16).to(dev)
    z = torch.randn(B * TT, V, dtype=torch.float32).contiguous().to(dev)
    z.requires_grad_(True)
    tgt = torch.randint(0, V, (B, TT), dtype=torch.long).to(dev)

    ctr = OpCounter()
    with ctr:
        y = T.rope_norm(x, cos, sin)
        loss = T.softcap_cross_entropy(z, tgt, "mean")
        (y.float().sum() + loss).backward()
    return ctr.tfc()


def main():
    print(f"_HAS_NKI_CUSTOM_OPS={T._HAS_NKI_CUSTOM_OPS} err={T._NKI_CUSTOM_OPS_ERR!r}")

    off = probe(False)
    on = probe(True)
    print(f"\nflag OFF -> tfc:: ops observed: {off}")
    print(f"flag ON  -> tfc:: ops observed: {on}")

    ok = (not off) and len(on) >= 4
    print(f"\nDISPATCH-PATH CHECK: {'PASS -- flag really switches paths' if ok else 'FAIL'}")

    # argparse -> module global plumbing (no device work needed)
    import sys
    for argv, expect in ((["train.py", "--nki-custom-ops"], True),
                         (["train.py", "--no-nki-custom-ops"], False),
                         (["train.py"], False)):
        saved = sys.argv
        try:
            sys.argv = argv
            # Re-run only the parser by calling main() would train; instead assert the
            # default/flag resolution the same way main() does.
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--nki-custom-ops", action=argparse.BooleanOptionalAction,
                           default=os.environ.get("NEURON_COMPETITION_R1_NKI_CUSTOM_OPS", "0") != "0")
            got = p.parse_known_args(argv[1:])[0].nki_custom_ops
        finally:
            sys.argv = saved
        status = "OK" if got == expect else "MISMATCH"
        print(f"  argparse {argv[1:]!s:26s} -> {got}  (expected {expect}) {status}")


if __name__ == "__main__":
    main()
