# Independent Round13 C2 / C3 source and CPU verification

Recorded 2026-09-27 08:36:04 UTC by the separate standing verifier. **Both PASS_SOURCE_CPU_ONLY** against immutable c239 parent `c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0`. No new source/CPU blocker found. No source, host, queue, state, ledger, launcher, evaluator, or dataset was changed by this review.

| Arm | Exact source SHA256 | Verdict |
|---|---|---|
| C2: remove layer-0 MLP | `a84f8716b2f3fb54bc6793d9011f37f5d05ddb0ad06c64d0d49988a8a7c68cee` | PASS_SOURCE_CPU_ONLY |
| C3: AdamW-only final-40% cooldown | `9d3fdee28eda63e91b12559a4f86e3b9df9fa097ae19c2dccdaa119504ddfff8` | PASS_SOURCE_CPU_ONLY |

This is an isolated implementation review, not native qualification, a combined-stack review, launch authorization, adoption, or a BPB result. Both hashes intentionally retain the c239 value-embedding gate's known Muon assignment; neither includes the separately reviewed Round13 gate fix. Any rebase onto an adopted gate-fixed parent needs a new hash-bound diff/verification. The Supervisor retains responsibility for that known issue and the normal completed proposal/machine/artifact gates. These reviews add no extra native experiment requirement and do not block already-ready host jobs.

## Independent execution

Script `research/v2/research_cycles/20260927-r13-c2-c3-reference-verification.py`, SHA256 `136e0c94d6e1c8fa91e79c4c9fd0279b9ccb518d99089f19e9fd16b00e7253ae`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r13-c2-c3-reference-verification.py
```

First invocation exited 0, reported check-body time 0.8357 seconds; local Torch `2.13.0+cpu`. Author tests were not run or counted as independent evidence. Hash-bound actual definitions and actual scheduling statements were extracted by AST without importing the training runtime or `prepare.py`. No training data or validation data was read. No full-model optimizer update, distributed collective, Neuron compilation, or host execution occurred.

## C2 findings

- Independently reversed the four intended edits and recovered the parent byte-for-byte. Changes are only layer-0 MLP construction, conditional forward, initialization, and optimizer enumeration. Layer 0 still has attention, local convolution, residual/input scalars, and its original index. Attention/local-conv indices remain `[0,2,4,6,7]`; MLPs exist only at indices `[1,2,3,4,5,6,7]`, with no identity substitute or orphan module.
- Production-parameter meta check (`L=8,D=1024,H=8,V=8192,T=1024`): **63 unique tensors, 104,868,116 elements**, exactly 8,388,608 fewer elements than c239. Only `transformer.h.0.mlp.c_fc.weight` and `.c_proj.weight` disappear. Every retained parameter has its original optimizer policy and exactly one optimizer entry. The actual four-owner planner covers every parameter once with nonoverlapping ranges.
- Tiny CPU model (`B=1,T=7,D=128,H=8,L=8,V=128`, seed 42): deliberately nonzero remaining MLP projections, local convolutions and skip gates exercise connectivity. All 63 gradients are finite and nonzero. An explicit layer-0 no-MLP equation and independently index-addressed full-forward oracle match bitwise. Hooks verify encoder-to-decoder pairs `3→4,2→5,1→6,0→7`, including skip 0. Prefix causality and length-one finite output pass.
- Actual unchanged `load_for_eval` executes with an in-memory checkpoint, `weights_only=True` and strict loading, extends the rotary context as before, returns eval/no-grad mode, preserves seven MLPs and yields identical CPU predictions. The parent model state correctly fails strict loading into C2 because its two removed matrix keys are unexpected. No train/eval-only architecture switch was introduced.
- **RNG consequence, not a blocker:** same-seed constructor/initialization runs change 30 retained tensor initializations in the tiny comparison, including the token embedding and layer-0 attention matrices. Removing random draws changes the stream even though the retained initialization formulas are unchanged. Do not claim identical common-weight initialization or attribute a measured delta solely to compute removal without this qualification. This is the documented consequence of the requested architecture arm, not an additional edit.

The synthetic loss `4.849189281463623` is only a logic-check output, not a BPB/quality measurement.

## C3 findings

- Independently reversed the four intended edit blocks and recovered c239 byte-for-byte. Only `lr_multiplier`'s optional cooldown-length parameter and actual training-group LR selection change. Architecture, initializer, optimizer grouping/math, charged-time computation, budget/stop behavior, data, MTP, and eval paths remain byte-identical.
- The meta model still has 65 tensors and all original optimizer policies. The parent model state strictly loads, predictions match, and the actual `load_for_eval` in-memory round trip matches.
- Executed the **actual schedule statements** after their unchanged charged-time input over **14,784 group cases**: 9 actual groups plus synthetic AdamW/Muon `no_lr_decay` groups; warmup values 0 (actual), 3 and 300; steps 0,1,2,3,299,300,1200; budgets 0,1e-9,1,1800; 16 progress points including immediately around 0.25,0.60 and 1. Independent piecewise-linear oracle maximum absolute LR error `5.551115123125783e-17`.
- Every Muon group field is exactly parent-equal in every case: LR still uses final-75% cooldown, and momentum, beta2 and weight decay are unchanged. For AdamW, every field other than LR is exactly parent-equal; all actual AdamW groups use final-40%. `no_lr_decay` retains the original warmup/flat override. Actual `WARMUP_STEPS=0` and final fraction `0.05` are unchanged; the nonzero-warmup synthetic cases check preservation of that branch.

| Charged-budget progress | Muon LR multiplier (unchanged) | AdamW LR multiplier |
|---|---:|---:|
| 0.25 | 1.0 | 1.0 |
| 0.60 | 0.5566666667 | 1.0 |
| 0.80 | 0.3033333333 | 0.525 |
| 1.00 or later | 0.05 | 0.05 |

For an 1800-second schedule, this means Muon cooldown starts at charged time 450 seconds and AdamW at 1080 seconds; it does not change the official clock or stop rule.

The initial C3 spec accidentally copied C2's architecture-checkpoint incompatibility sentence. The verifier flagged it; the Supervisor corrected the **spec only**, now SHA `6cac6bcb3cbda3d9a9a7b26bdc0ed086cd3d80c9a8349523dba70534a658dc57`. The corrected text was reread and source SHA remained unchanged. This documentation issue is resolved, not a source blocker.

## Limits

No quality or speed gain is established. CPU Torch differs from the previously retrieved host Torch `2.11.0+cpu`; native behavior remains empirical. Proposals are author implementation stubs, not completed native run packets. The normal Supervisor/Operator workflow still binds exact source, unique label, command, environment, comparison and artifact retrieval. No new legality decision is needed for these exact source diffs; this statement does not authorize any later evaluator/data/command change.
