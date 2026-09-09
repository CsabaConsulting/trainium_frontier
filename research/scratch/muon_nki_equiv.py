#!/usr/bin/env python3
"""Offline equivalence gate for the NKI Muon elementwise fusion (train.py
USE_NKI_MUON): is the fused path BIT-IDENTICAL to the in-tree eager _muon_step?

The fusion is a mathematically lossless re-expression of existing elementwise ops --
not an approximation -- so the standard here is exact match, the same standard the
rope+QK-norm and softcap+CE kernels were held to, not the FP8 kernel's
inherent-precision-loss standard. Anything short of 0 differing elements is a bug.

Method: two independent optimizers over identically-initialised parameter sets, fed
the same synthetic grads and the same per-step lr/momentum/weight_decay schedule, one
with USE_NKI_MUON on and one off. After every step, compare the params AND every
optimizer state tensor (momentum_buffer, second_momentum_buffer, stacked_*) element by
element. Divergence anywhere shows up immediately rather than being averaged away.

The lr/wd schedule is deliberately made to change every step: the fused path carries
lr and lr*wd as runtime device operands, so a static schedule would not exercise the
one thing most likely to be wrong about them.

Usage (needs a Neuron device for the kernels):
    NEURON_RT_VISIBLE_CORES=0 python research/scratch/muon_nki_equiv.py --steps 6
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch

import train as T

STATE_KEYS = ("momentum_buffer", "second_momentum_buffer", "stacked_grads",
              "stacked_params")


def build_arm(device, seed=42):
    """A fresh GPT + MuonAdamW on `device`. Two calls with the same seed give two
    parameter sets that are equal element for element."""
    config = T.build_config()
    torch.manual_seed(seed)
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return model, opt


def cmp_exact(a: torch.Tensor, b: torch.Tensor):
    """(n_differing, max_abs_diff, numel) between two tensors, computed on the host."""
    x, y = a.detach().float().cpu(), b.detach().float().cpu()
    ne = int((x != y).sum())
    md = float((x - y).abs().max()) if x.numel() else 0.0
    return ne, md, x.numel()


def set_schedule(opt, step):
    """A schedule that moves every step, so lr / lr*wd are never accidentally constant."""
    lrm = 1.0 - 0.017 * step
    wd_progress = min(1.0, step / 7.0)
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = 0.95 - 0.01 * step
            group["weight_decay"] = T.WEIGHT_DECAY * (1.0 - wd_progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = True
    device = torch.device("neuron")
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)
    print(f"device={device} compute_dtype={T.COMPUTE_DTYPE} has_nki={T._HAS_NKI}")
    assert T._HAS_NKI, "no NKI on this host -- nothing to verify"
    assert T.COMPUTE_DTYPE == torch.bfloat16, "fused path is gated on bf16 compute"

    m_ref, o_ref = build_arm(device, args.seed)
    m_nki, o_nki = build_arm(device, args.seed)

    muon_ref = [g for g in o_ref.param_groups if g["kind"] == "muon"]
    muon_nki = [g for g in o_nki.param_groups if g["kind"] == "muon"]
    print(f"muon groups: {[(tuple(g['params'][0].shape), len(g['params'])) for g in muon_ref]}")

    # sanity: the two arms really do start identical
    for pr, pn in zip(m_ref.parameters(), m_nki.parameters()):
        ne, md, _ = cmp_exact(pr, pn)
        assert ne == 0, f"arms differ at init by {ne} elements (max {md})"

    worst = 0.0
    total_bad = 0
    for step in range(args.steps):
        torch.manual_seed(1000 + step)
        # identical synthetic grads for both arms, generated on the host so the two
        # arms cannot diverge through any device-side RNG difference
        grads = {}
        for name, p in m_ref.named_parameters():
            grads[name] = (torch.randn(p.shape, dtype=torch.float32,
                                       generator=torch.Generator().manual_seed(
                                           hash((step, name)) % (2 ** 31))) * 0.01)
        for model in (m_ref, m_nki):
            for name, p in model.named_parameters():
                p.grad = grads[name].clone().to(device)

        set_schedule(o_ref, step)
        set_schedule(o_nki, step)

        T.USE_NKI_MUON = False
        for g in muon_ref:
            o_ref._muon_step(g)
        T.USE_NKI_MUON = True
        for g in muon_nki:
            o_nki._muon_step(g)
        T.USE_NKI_MUON = False
        T.synchronize(device)

        # every Muon param, then every Muon state tensor
        bad = 0
        for gi, (gr, gn) in enumerate(zip(muon_ref, muon_nki)):
            shape = tuple(gr["params"][0].shape)
            for pi, (pr, pn) in enumerate(zip(gr["params"], gn["params"])):
                ne, md, n = cmp_exact(pr, pn)
                worst = max(worst, md)
                bad += ne
                if ne:
                    print(f"  [step {step}] group{gi}{shape} param{pi}: "
                          f"{ne}/{n} differ, max|diff| {md:.3e}")
            sr = o_ref.state[gr["params"][0]]
            sn = o_nki.state[gn["params"][0]]
            for key in STATE_KEYS:
                if key not in sr:
                    continue
                ne, md, n = cmp_exact(sr[key], sn[key])
                worst = max(worst, md)
                bad += ne
                if ne:
                    print(f"  [step {step}] group{gi}{shape} {key}: "
                          f"{ne}/{n} differ, max|diff| {md:.3e}")
        total_bad += bad
        print(f"step {step}: {'EXACT' if bad == 0 else f'{bad} DIFFERING ELEMENTS'}"
              f"  (running worst max|diff| {worst:.3e})", flush=True)

    print("\n=== MUON NKI FUSION EQUIVALENCE ===")
    print(f"steps compared      {args.steps}")
    print(f"differing elements  {total_bad}")
    print(f"worst max|diff|     {worst:.3e}")
    print("VERDICT: " + ("BIT-EXACT" if total_bad == 0 else "NOT EXACT -- do not ship"))
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
