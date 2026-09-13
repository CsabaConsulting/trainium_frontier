#!/usr/bin/env python3
"""CPU correctness smoke for the FROZEN Q/K-shift probe and Recursive's input-dependent x0 gate.

Arms: --qk-shift --qk-shift-beta 0 --qk-shift-freeze (the throughput probe), the trainable beta=0
contrast, a frozen beta=0.3, and --x0-gate.

The claim this arm makes is much stronger than "no-op at step 0": with beta frozen at exactly 0.0
the shifted branch contributes `0.0 * causal_shift_1(x)`, which is exactly 0 in IEEE fp for any
finite x, and its backward contribution to x is scaled by the same 0.0. So the arm must be
bit-identical to the baseline FOREVER, not just at initialization -- that is what makes it a clean
probe of the compiler-scheduling throughput anomaly (the elementwise ops stay in the graph; the
mathematics does not change at all).

Checks, at the locked config in fp32:

  1. every TRAINABLE parameter is in exactly one optimizer group, and the frozen beta is in NONE;
  2. beta.requires_grad is False and beta.grad stays None after backward;
  3. the loss is bit-identical to the baseline's at step 0 AND after 5 full optimizer steps;
  4. beta has not moved off 0.0 after those steps;
  5. causality: perturbing token T-1 moves no logit at positions < T-1.

For contrast it also runs the TRAINABLE beta=0.0 arm (the 1800s `t39-qkshift-b0` config), which
must be a step-0 no-op but must DIVERGE by step 5 -- that divergence is the mechanism the 1800s
run measured as +0.0016 bpb.

Usage (CPU only):
    python research/scratch/qk_freeze_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train  # noqa: E402


DEPTH = 6
B, T = 2, 128
NSTEPS = 5


class toggled:
    def __init__(self, **toggles):
        self.toggles = toggles

    def __enter__(self):
        self.saved = {k: getattr(train, k) for k in self.toggles}
        for k, v in self.toggles.items():
            setattr(train, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(train, k, v)
        return False


def build():
    cfg = train.build_config(DEPTH)
    cfg.sequence_len = T
    torch.manual_seed(42)
    model = train.GPT(cfg)
    model.init_weights()
    opt = model.setup_optimizer()
    return model, opt, cfg


def check_groups(model, opt, label):
    """Every TRAINABLE parameter in exactly one group; no frozen parameter in any group."""
    seen = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            assert id(p) not in seen, f"{label}: param in groups {seen[id(p)]} and {gi}"
            seen[id(p)] = gi
            assert p.requires_grad, f"{label}: group {gi} holds a requires_grad=False tensor"
            if g["kind"] == "muon":
                assert p.dim() == 2 and min(p.shape) > 1, \
                    f"{label}: muon group {gi} holds a {tuple(p.shape)} tensor"
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    assert not missing, f"{label}: trainable parameters in no optimizer group: {missing}"
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    return len(seen), frozen


def batches(cfg, n):
    torch.manual_seed(1234)
    return [(torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64),
             torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64)) for _ in range(n)]


def causality(model, cfg):
    torch.manual_seed(7)
    idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64)
    with torch.no_grad():
        a = model(idx)
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
        b = model(idx2)
    return (a[:, :-1] - b[:, :-1]).abs().max().item()


ARMS = [
    ("baseline", {}),
    ("qkshift-b0-frozen", dict(USE_QK_SHIFT=True, QK_SHIFT_BETA=0.0, QK_SHIFT_FREEZE=True)),
    ("qkshift-b0-trained", dict(USE_QK_SHIFT=True, QK_SHIFT_BETA=0.0, QK_SHIFT_FREEZE=False)),
    ("qkshift-b03-frozen", dict(USE_QK_SHIFT=True, QK_SHIFT_BETA=0.3, QK_SHIFT_FREEZE=True)),
    # Recursive's input-dependent x0 gate rides along here rather than in its own script: it is the
    # same class of claim (bit-exact step-0 no-op, per-token-only reduction so causality is
    # structural) and it needs the same three checks.
    ("x0-gate", dict(USE_X0_GATE=True)),
]


def main() -> int:
    train.COMPUTE_DTYPE = torch.float32
    ref = None
    failures = []
    for label, toggles in ARMS:
        with toggled(**toggles):
            model, opt, cfg = build()
            ntensors, frozen = check_groups(model, opt, label)
            betas = [b.qk_shift_beta for b in model.transformer.h if b.qk_shift_beta is not None]
            losses = []
            for idx, tgt in batches(cfg, NSTEPS):
                opt.zero_grad(set_to_none=False)
                loss = model(idx, tgt)
                assert torch.isfinite(loss), f"{label}: non-finite loss"
                loss.backward()
                if toggles.get("QK_SHIFT_FREEZE"):
                    bad = [i for i, b in enumerate(betas) if b.grad is not None]
                    assert not bad, f"{label}: frozen beta got a .grad on layers {bad}"
                opt.step()
                losses.append(loss.item())
            cz = causality(model, cfg)
            if label == "baseline":
                ref = losses
            same_step0 = losses[0] == ref[0]
            same_all = losses == ref
            beta_moved = max((b.detach() - float(toggles.get("QK_SHIFT_BETA", 0.0))).abs().max().item()
                             for b in betas) if betas else 0.0
            ss = [b.x0_gate_s for b in model.transformer.h if b.x0_gate_s is not None]
            s_moved = max(s.detach().abs().max().item() for s in ss) if ss else 0.0
            status = "ok"
            if label.endswith("-frozen") and toggles.get("QK_SHIFT_BETA") == 0.0 and not same_all:
                status = "FAIL not bit-identical over 5 steps"
            if label.endswith("-frozen") and beta_moved != 0.0:
                status = f"FAIL frozen beta moved by {beta_moved:.3e}"
            if label in ("qkshift-b0-trained", "x0-gate") and (not same_step0 or same_all):
                status = "FAIL expected step-0 no-op then divergence"
            if cz != 0.0:
                status = f"FAIL causality {cz:.3e}"
            print(f"{label:20s} tensors={ntensors:3d} frozen={len(frozen):2d} "
                  f"loss0={losses[0]:.8f} loss{NSTEPS - 1}={losses[-1]:.8f} "
                  f"same_step0={same_step0} same_all={same_all} beta_moved={beta_moved:.2e} "
                  f"s_moved={s_moved:.2e} causality={cz:.1e}  {status}")
            if status != "ok":
                failures.append(label)
    print("---")
    print(f"failures: {failures if failures else 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
