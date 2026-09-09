#!/usr/bin/env python3
"""Census of the aten ops MuonAdamW._muon_step actually dispatches, per group,
split into MATMUL (Tensor-Engine work we are NOT touching) vs ELEMENTWISE/REDUCE
(the fusion target).

Runs on CPU -- no device needed. TorchDispatchMode observes the real op stream, so
the split is measured rather than counted by eye off the source.

Also prints the real Muon group shapes at the locked config, which decides whether
an NKI fusion is even tileable.

Usage:  python research/scratch/muon_dispatch_census.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import train as T

# ops that run on the PE array (Tensor Engine) -- explicitly out of scope
MATMUL_OPS = {"aten.matmul", "aten.mm", "aten.bmm", "aten.baddbmm", "aten.linear"}
# pure view/metadata ops: no launch on any backend, must not be counted either way
VIEW_OPS = {
    "aten.view", "aten.reshape", "aten._unsafe_view", "aten.transpose",
    "aten.permute", "aten.t", "aten.expand", "aten.unbind", "aten.select",
    "aten.slice", "aten.detach", "aten.alias", "aten.squeeze", "aten.unsqueeze",
    "aten.as_strided", "aten.is_same_size", "aten.contiguous", "aten.numpy_T",
}


def short(func) -> str:
    s = str(func)                     # e.g. "aten.mul.Tensor"
    parts = s.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else s


class Census(TorchDispatchMode):
    def __init__(self):
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(short(func))
        return func(*args, **(kwargs or {}))

    def classify(self):
        mm = ew = vw = 0
        for o in self.ops:
            if o in MATMUL_OPS:
                mm += 1
            elif o in VIEW_OPS:
                vw += 1
            else:
                ew += 1
        return mm, ew, vw


def main():
    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = True
    T.COMPUTE_DTYPE = torch.bfloat16

    config = T.build_config()
    torch.manual_seed(0)
    model = T.GPT(config)
    model.init_weights()
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

    # give every param a grad so nothing is skipped
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.01

    muon_groups = [g for g in opt.param_groups if g["kind"] == "muon"]
    adamw_groups = [g for g in opt.param_groups if g["kind"] == "adamw"]
    print(f"config: n_layer={config.n_layer} n_embd={config.n_embd} n_head={config.n_head} "
          f"seq_len={config.sequence_len}")
    print(f"groups: {len(adamw_groups)} adamw, {len(muon_groups)} muon\n")

    # one untimed warm step so lazy state allocation is not counted
    opt.step()

    print(f"{'group':>5} {'shape':>16} {'N':>3} {'elems/M':>8} "
          f"{'matmul':>7} {'elem+red':>9} {'views':>6} {'launches':>9}")
    tot_mm = tot_ew = 0
    per_group = []
    for i, g in enumerate(muon_groups):
        c = Census()
        with c:
            opt._muon_step(g)
        mm, ew, vw = c.classify()
        n = len(g["params"])
        shape = tuple(g["params"][0].shape)
        elems = n * g["params"][0].numel() / 1e6
        print(f"{i:>5} {str(shape):>16} {n:>3} {elems:>8.2f} "
              f"{mm:>7} {ew:>9} {vw:>6} {mm+ew:>9}")
        tot_mm += mm
        tot_ew += ew
        per_group.append((shape, n, mm, ew, Counter(c.ops)))

    print(f"{'TOTAL':>5} {'':>16} {'':>3} {'':>8} {tot_mm:>7} {tot_ew:>9} {'':>6} "
          f"{tot_mm+tot_ew:>9}")
    print(f"\nMuon total launches = {tot_mm + tot_ew}  "
          f"(matmul {tot_mm} = {100*tot_mm/(tot_mm+tot_ew):.1f}%, "
          f"elementwise/reduce {tot_ew} = {100*tot_ew/(tot_mm+tot_ew):.1f}%)")

    # AdamW side, for the whole-optimizer picture
    c = Census()
    with c:
        for g in adamw_groups:
            opt._adamw_step_foreach(g)
    amm, aew, _ = c.classify()
    print(f"AdamW total launches = {amm + aew}")
    print(f"OPTIMIZER TOTAL = {tot_mm + tot_ew + amm + aew}")

    # per-op histogram of the elementwise stream on one representative group
    print("\nop histogram, largest Muon group:")
    shape, n, mm, ew, hist = max(per_group, key=lambda r: r[1] * 1)
    biggest = max(per_group, key=lambda r: r[3])
    for op, cnt in sorted(biggest[4].items(), key=lambda kv: -kv[1]):
        tag = "MATMUL" if op in MATMUL_OPS else ("view" if op in VIEW_OPS else "elem")
        print(f"  {op:<34} {cnt:>4}  {tag}")


if __name__ == "__main__":
    main()
