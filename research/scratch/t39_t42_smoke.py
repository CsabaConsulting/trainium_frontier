#!/usr/bin/env python3
"""CPU correctness smoke for T4.2 (Aster output bigram logit table) and T3.9 (Recursive's
cheap architectural one-liners).

Pure CPU, no device, no training. For each arm it checks, at the locked config:

  1. the model builds, the optimizer groups cover every trainable parameter exactly once, and
     no 1-D or single-row tensor has been routed into a Muon group;
  2. the forward runs and produces finite loss/logits;
  3. STEP-0 NO-OP, where the arm claims one: the loss is bit-identical to the same-seed
     baseline's (this is the property that makes the A/B honest -- the two arms start from
     the same point and differ only in what they can learn);
  4. CAUSALITY, exactly: perturb token T-1 of the input and assert not a single logit at
     positions < T-1 moves. Same property `prepare.causality_check` verifies on device, done
     here in fp32 so the tolerance is 0.0 rather than 1e-2;
  5. one optimizer step runs and leaves every parameter finite.

Usage (on the box, CPU only):
    python research/scratch/t39_t42_smoke.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train  # noqa: E402


DEPTH = 6
B, T = 2, 128


class toggled:
    """Apply `toggles` as train.py module globals for the whole `with` body.

    Deliberately NOT scoped to construction only: RELU2_TAU, USE_MLP_SANDWICH_NORM,
    HEAD_GATE_NORM and HEAD_GATE_MUL are read inside forward(), so a context that ended after
    GPT(config) would silently evaluate every arm with the DEFAULT forward behavior -- which is
    exactly the false pass this class exists to prevent (it happened on the first run of this
    script: two arms reported an impossible bit-exact step-0 no-op).
    """

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
    seen = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            assert id(p) not in seen, f"{label}: param in groups {seen[id(p)]} and {gi}"
            seen[id(p)] = gi
            if g["kind"] == "muon":
                assert p.dim() == 2 and min(p.shape) > 1, \
                    f"{label}: muon group {gi} holds a {tuple(p.shape)} tensor"
    named = {id(p): n for n, p in model.named_parameters()}
    missing = [named[i] for i in named if i not in seen]
    assert not missing, f"{label}: parameters in no optimizer group: {missing}"
    return len(seen)


def fwd(model, cfg, seed=0):
    torch.manual_seed(1234 + seed)
    idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64)
    return idx, tgt, model(idx, tgt)


def causality(model, cfg):
    """Max abs logit change at positions < T-1 when the LAST input token is replaced."""
    torch.manual_seed(7)
    idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.int64)
    with torch.no_grad():
        a = model(idx)
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
        b = model(idx2)
    return (a[:, :-1] - b[:, :-1]).abs().max().item()


ARMS = [
    ("baseline", {}, True),
    ("t42-out-bigram", dict(USE_OUT_BIGRAM=True), True),
    # relu2-tau and mlp-sandwich ARE exact step-0 no-ops, for a reason worth writing down: the
    # MLP's c_proj is zero-initialized, so the whole MLP branch contributes exactly 0 at step 0
    # whatever its activation or norm placement is.
    ("t39-relu2-tau", dict(RELU2_TAU=0.5), True),
    ("t39-mlp-sandwich", dict(USE_MLP_SANDWICH_NORM=True), True),
    ("t39-head-gate-norm", dict(USE_HEAD_GATE=True, HEAD_GATE_NORM=True), False),
    ("t39-head-gate-only", dict(USE_HEAD_GATE=True, HEAD_GATE_NORM=False), True),
    ("t39-out-pool", dict(USE_OUT_POOL=True), True),
    ("t39-qk-shift", dict(USE_QK_SHIFT=True), False),
]


def main() -> int:
    train.COMPUTE_DTYPE = torch.float32   # fp32 everywhere so "no-op" can mean bit-identical
    base_loss = None
    failures = []
    for label, toggles, expect_noop in ARMS + [
        # Every arm that is individually a step-0 no-op, together: catches an interaction that
        # breaks the exactness even though each piece holds on its own.
        ("stack-noop", dict(USE_OUT_BIGRAM=True, USE_MLP_SANDWICH_NORM=True, RELU2_TAU=0.5,
                            USE_OUT_POOL=True, USE_HEAD_GATE=True, HEAD_GATE_NORM=False), True),
    ]:
        try:
            with toggled(**toggles):
                model, opt, cfg = build()
                nparams = sum(p.numel() for p in model.parameters())
                ngroups = check_groups(model, opt, label)
                idx, tgt, loss = fwd(model, cfg)
                assert torch.isfinite(loss), f"{label}: non-finite loss {loss}"
                if label == "baseline":
                    base_loss = loss.item()
                noop = (loss.item() == base_loss)
                cz = causality(model, cfg)
                loss.backward()
                opt.step()
                bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
                assert not bad, f"{label}: non-finite params after one step: {bad}"
                status = "ok"
                if expect_noop and not noop:
                    status = "FAIL step-0-no-op"
                if cz != 0.0:
                    status = f"FAIL causality {cz:.3e}"
                print(f"{label:22s} params={nparams/1e6:8.2f}M tensors={ngroups:3d} "
                      f"loss={loss.item():.6f} noop={noop} causality_max_abs={cz:.1e}  {status}")
                if status != "ok":
                    failures.append(label)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:22s} EXCEPTION {type(exc).__name__}: {exc}")
            failures.append(label)

    print("---")
    print(f"failures: {failures if failures else 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
