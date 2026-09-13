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
    # The MINIMAL form of the same graph perturbation: `x = x + nudge*x` with nudge frozen at 0.
    # Two multiplies/adds per block instead of the shift's gather+multiply+add over [B,T,C], and
    # no [C]-wide parameter. Same bit-identity claim as qkshift-b0-frozen.
    ("nudge-qk", dict(USE_BLOCK_NUDGE=True, BLOCK_NUDGE_SITE="qk")),
    ("nudge-resid", dict(USE_BLOCK_NUDGE=True, BLOCK_NUDGE_SITE="resid")),
    # The op-REMOVING identity found while scoping the parallel block: three norm() calls on one
    # tensor collapse to one. Same bit-identity claim, opposite direction (fewer ops, not more).
    ("qkv-norm-cse", dict(USE_QKV_NORM_CSE=True)),
    # Item 3's parallel attn+MLP block. NOT an identity past step 0 (the MLP reads the pre-attention
    # normed residual from step 1 on), but it IS bit-identical at step 0 because mlp.c_proj is
    # zero-init and par_lambda_a starts at 1.0 -- same class of claim as x0-gate.
    ("parallel-block", dict(USE_PARALLEL_BLOCK=True)),
    # Both coalescing modes must be bit-identical to parallel-block (NOT to the baseline) at step 0
    # -- that is what the slice-wise init in init_weights buys, and it is checked against `par_ref`.
    ("par-coalesce-vfc", dict(USE_PARALLEL_BLOCK=True, PARALLEL_BLOCK_COALESCE="vfc")),
    ("par-coalesce-all", dict(USE_PARALLEL_BLOCK=True, PARALLEL_BLOCK_COALESCE="all")),
    # T3.2 Engram-lite. Same class of claim as x0-gate/parallel-block: bit-identical at step 0 (zero
    # tables => e_t = 0 => v = 0 => vt = 0 => SiLU(Conv(0)) = 0 => Y = 0, and the gate's
    # sigmoid(0) = 0.5 is deliberately NOT relied on), then divergent by step 5 once W_V's gradient
    # has moved the tables off the origin. ENGRAM_TABLE_MULT is dropped to 1 for all arms so the
    # harness allocates 8192-row tables instead of 262144-row ones -- the geometry is irrelevant to
    # every claim being checked here and 32x smaller keeps the CPU run seconds long.
    ("engram-own", dict(USE_ENGRAM=True, ENGRAM_TABLE_MULT=1)),
    # Multi-head: 2 heads x 2 orders = 4 tables, so e_dim = 4*d_mem. Checks the per-order/per-head
    # prime slicing and the concat width actually line up with w_k/w_v.
    ("engram-2head", dict(USE_ENGRAM=True, ENGRAM_TABLE_MULT=1, ENGRAM_HEADS=2)),
    # Conv off: Y = alpha*v. Still a step-0 no-op, and it must have strictly fewer parameters.
    ("engram-noconv", dict(USE_ENGRAM=True, ENGRAM_TABLE_MULT=1, ENGRAM_CONV_KERNEL=0)),
    # A narrow gate space, which is the only configuration that builds w_q.
    ("engram-keydim", dict(USE_ENGRAM=True, ENGRAM_TABLE_MULT=1, ENGRAM_KEY_DIM=64)),
    # Last-layer site: what the default was before the paired 900s site test showed mid-stack is
    # both free and 0.0016 better. Kept as an arm because it is the contrast the doc block cites.
    ("engram-last", dict(USE_ENGRAM=True, ENGRAM_TABLE_MULT=1, ENGRAM_SITE="-1")),
    # Share mode: no tables of its own, e_t is T3.1's already-gathered rows. The n-gram tables are
    # ALSO zero at init, so the step-0 no-op holds for the same reason, and the arm must add no
    # table parameters at all (checked by `tensors` versus the baseline's). ENGRAM_SITE is forced to
    # -1 here because NGRAM_VE_LAYERS is locked at "-1": share mode can only read rows that have
    # already been gathered by the time the loop reaches the injection site.
    ("engram-share", dict(USE_ENGRAM=True, ENGRAM_SITE="-1", ENGRAM_SHARE_NGRAM_TABLES=True)),
    ("engram-share-bigram-only", dict(USE_ENGRAM=True, ENGRAM_SITE="-1",
                                      ENGRAM_SHARE_NGRAM_TABLES=True, NGRAM_VE_TRIGRAM=False)),
]


def main() -> int:
    train.COMPUTE_DTYPE = torch.float32
    ref = par_ref = None
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
                if toggles.get("USE_BLOCK_NUDGE"):
                    bad = [i for i, b in enumerate(model.transformer.h)
                           if b.block_nudge is not None and b.block_nudge.grad is not None]
                    assert not bad, f"{label}: frozen nudge got a .grad on layers {bad}"
                opt.step()
                losses.append(loss.item())
            cz = causality(model, cfg)
            if label == "baseline":
                ref = losses
            if label == "parallel-block":
                par_ref = losses
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
            if label == "qkv-norm-cse" and not same_all:
                status = "FAIL not bit-identical over 5 steps"
            if label.startswith("nudge-"):
                nudges = [b.block_nudge for b in model.transformer.h if b.block_nudge is not None]
                assert len(nudges) == DEPTH, f"{label}: {len(nudges)} nudges for {DEPTH} blocks"
                n_moved = max(n.detach().abs().max().item() for n in nudges)
                if not same_all:
                    status = "FAIL not bit-identical over 5 steps"
                elif n_moved != 0.0:
                    status = f"FAIL frozen nudge moved by {n_moved:.3e}"
            if label in ("qkshift-b0-trained", "x0-gate", "parallel-block") and (not same_step0 or same_all):
                status = "FAIL expected step-0 no-op then divergence"
            if label.startswith("engram"):
                # The Engram claim, in three parts. (1) Step 0 must be bit-identical to the
                # baseline's -- zero tables + zero conv, NOT a closed gate. (2) It must DIVERGE by
                # step 5: if it does not, the tables are receiving no gradient (the W_V-must-not-be-
                # zero-init trap) and the whole mechanism is dead weight. (3) The site must be where
                # it was asked for, so a typo'd ENGRAM_SITE cannot silently pass as the default.
                eng = model.engram
                want_site = int(toggles.get("ENGRAM_SITE", train.ENGRAM_SITE)) % DEPTH
                ntab = len(eng.tables)
                want_tab = 0 if toggles.get("ENGRAM_SHARE_NGRAM_TABLES") else \
                    toggles.get("ENGRAM_HEADS", train.ENGRAM_HEADS) * len(train.engram_orders())
                if not same_step0:
                    status = f"FAIL step-0 differs from baseline by {losses[0] - ref[0]:.3e}"
                elif same_all:
                    status = "FAIL never diverged -- tables are getting no gradient"
                elif model.engram_site != want_site:
                    status = f"FAIL site {model.engram_site} != {want_site}"
                elif ntab != want_tab:
                    status = f"FAIL {ntab} tables != {want_tab}"
                elif (eng.conv is not None) != (toggles.get("ENGRAM_CONV_KERNEL", 1) != 0):
                    status = "FAIL conv presence does not match ENGRAM_CONV_KERNEL"
                elif (eng.w_q is not None) != (eng.key_dim != cfg.n_embd):
                    status = "FAIL w_q presence does not match ENGRAM_KEY_DIM"
                elif eng.w_k.weight.shape[1] != eng.e_dim or eng.w_v.weight.shape[1] != eng.e_dim:
                    status = f"FAIL w_k/w_v input width != e_dim {eng.e_dim}"
                elif toggles.get("ENGRAM_SHARE_NGRAM_TABLES") and \
                        any(int(k[1:]) > model.engram_site for k in model.engram_share_keys):
                    status = "FAIL share mode reads a table from AFTER the injection site"
            if label.startswith("par-coalesce-"):
                # The reference for these is the UNCOALESCED parallel block, not the baseline: the
                # step-0 claim is "merging four matmuls into one changed no arithmetic", and it is
                # only interesting relative to the architecture the merge is applied to.
                if losses[0] != par_ref[0]:
                    status = f"FAIL step-0 differs from parallel-block by {losses[0] - par_ref[0]:.3e}"
                elif losses[0] != ref[0]:
                    status = f"FAIL step-0 differs from baseline by {losses[0] - ref[0]:.3e}"
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
