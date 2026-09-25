# Independent Recursive batch262144 review

2026-09-25 — **PASS_SOURCE_AND_BOUNDED_CPU_ONLY** for commit
`4e361d15baef94f041501f558620422c55ae8c88`, source
`workflow/proposals/research-v2/train.py`, SHA-256
`a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9`.

No concrete source/proposal defect found. This is not full-size G2, native
correctness, performance, BPB, cold or submission approval. No candidate, host,
queue, ledger, protected evaluator/data code or clean4083 port was changed.

## Lineage and narrow production change

Independently read the source from the actual parent of4e361d15 and compared it
with adopted commit `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`. Both are exact
**6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3**,
not an incidental canonical c5c snapshot. Extracting and removing the five
inserted lines from the actual Git diff reconstructs those base bytes exactly.

The additions declare the existing global in `main`, expose an integer
`--total-batch-size` default131072, reject nonpositive values and assign the
parsed value. No new environment override is introduced. The existing modulus
guard rejects positive values not divisible by the configured world microbatch;
the new positivity guard runs before runtime/model allocation. Default omission
and explicit131072 preserve behavior. Recursive itself remains explicit ON in
both approved comparison argv; this is not the separate baked submission port.

All model/optimizer kernels, data/fetch pipeline, scheduling, timing and
checkpoint/evaluation bodies are byte-unchanged. No architecture, FP8, tables,
LR scaling, ramp or batch-dependent inference path was added.

## Production accumulation and normalization

The selected profile has device batch1, pack32, context1024, world4, hence
`world_tokens_per_micro = 1 * 32 * 1024 * 4 = 131072`. The unchanged division
gives accumulation1 for131072 and accumulation2 for262144. Per-rank forward
shape stays32x1024; model width stays1024. The arm intentionally consumes two
successive microbatches per optimizer update, not a larger forward tensor.

With required batch-ramp OFF, the existing ramp helper returns the selected
accumulation count and `batch_lr_scale=1.0`. The actual loop clears gradients
once per update, divides each mean micro-loss by `grad_accum_steps`, accumulates
both backward passes, unscales once, then performs rank AVG and clips once
before the optimizer update (`adam_every_n=1`, loss scale1). There is no second
division by accumulation or automatic LR compensation. This produces the
global mean gradient for equal microshapes before clipping. Token accounting
uses `grad_accum_now * world_tokens_per_micro`, so production steps report the
selected batch size.

LR/WD/Recursive formulas and their clocks are unchanged; the step-based momentum
warmup still counts300 optimizer steps. Their relation to processed tokens
intentionally changes with the batch arm. Logged training loss remains the
last micro-loss, not a new average of both; the packet discloses this. Selection
uses the unchanged fresh public20M evaluator. Fixed microshape does not guarantee
identical native graph/cache behavior: /2 and accumulation can specialize.

## Independently executed checks

Both commands completed exit0, with **six tests passing**, including the tiny
tensor test (not skipped): normal2.610s, optimized5.194s.

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RUN_BATCH_TENSOR_CHECKS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/recursive-batch262144/test_batch.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RUN_BATCH_TENSOR_CHECKS=1 /home/poojith-devan/venv/bin/python -O -B research/v2/lanes/recursive-batch262144/test_batch.py
```

Tests execute extracted real parser/divisibility/micro-loop/token-accounting
AST. They cover default/explicit default/arm, zero/negative/noninteger rejection,
positive nondivisible rejection, ramp-OFF LR invariance and the global-mean
gradient oracle. That oracle uses tiny float64 linear squared-error tensors,
four simulated ranks, accumulation1/2 and loss scales1/128, tolerance1e-12.
It verifies normalization algebra, not BF16/native/model equivalence. No full
model or native device was allocated by this review.

All12 manifest hashes passed. The independent tensor-free
`audit_evidence.py` also completed exit0; results are in `evidence-audit.json`.
It independently checks actual parent lineage, exact five-line reconstruction,
all raw rank reports, approved/reduced argv, proposal control scope, and reruns
the unchanged default receipt comparator.

## Three raw reduced G2 receipts — audited, not rerun here

- Base6ef7 and new-source default each have four ranks/three updates, all12
  losses exactly equal, all62 final parameter digests equal, and matching
  ownership plan `806be765aee5bf7597ada264f58f60162a46dc5b9b7d098dc13a9e40c2bbfa57`.
- Explicit262144 has **six finite forward losses per rank** for three updates
  (24 total), consistent with accumulation2. All62 parameters update, ownership
  is unchanged, cross-rank mismatches are empty, and checkpoint reload is
  deterministic with `[1,16,8192]` logits. No arm-to-control weight identity is
  claimed or expected after changing the batch.
- All three use depth6/width128 and actual synthetic1x16 micro-inputs on four
  CPU ranks, recorded fake-clock/BF16/no-compile/stub-prepare support. Their
  physical inputs are **64 tokens per control update and128 per arm update**.
  Logged production counts131072/262144 are configuration arithmetic, not
  physical-token throughput evidence. These are not exact-size/native runs.

The reduced baseline argv equals the durable adopted argv with width128 only;
default uses the same argv, and the arm adds only `--total-batch-size 262144`.
The production proposal likewise appends only those two tokens to adopted argv.
Packet PyArrow/stub-prepare shims remain test-only and must not be deployed.

## Controls, pending work and authority

The packet correctly distinguishes historical schedule-arm measurements from
eligible selector controls on the new6ef7 base. Fresh same-host6ef7 control
labels/scores remain pending; no historical ledger row is relabelled, no fake
control is introduced, and the selector/thresholds are unchanged. No quality
verdict can be derived from the present synthetic receipts or pending controls.

Full-parameter CPU default identity/arm checks, native accumulation and trained
checkpoint/causality/finite/budget validation, source-bound Operator checks,
genuine fresh controls and full1800/fresh20M results retain their existing
scope. Architecture/FP8 priorities and the clean-port CPU route are not
preempted. Step counts remain diagnostics, not an invented quality veto.
There is no speedup, BPB-win, cold-cache or submission-readiness claim here.
