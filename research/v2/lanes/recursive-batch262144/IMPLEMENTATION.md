# Recursive base: global batch 262144, one-factor arm

Status: **IMPLEMENTER SELF-CHECK PASS; INDEPENDENT REVIEW / FULL-SIZE G2 /
NATIVE GATES PENDING. NOT HARDWARE-READY OR SUBMISSION-ELIGIBLE.**

Branch `codex/20260925-recursive-batch262144`, based on canonical
`a5b3064fdaf617ff3088d2886d5c6bd40275dcc6`. This does not modify the separate
clean4083 submission port, any host, queue, ledger or existing submission.

Source: `workflow/proposals/research-v2/train.py`, SHA-256
`a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9`.
Base: exact adopted6ef7 from `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`, SHA
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.

## Production change: five added lines

`train.patch` adds the `--total-batch-size` integer option, default131072,
rejects nonpositive values, and binds the existing global. No new environment
override is introduced. The original positive divisibility guard is retained.
Omitting the flag is OFF/default; explicitly selecting131072 is equivalent.
The actual forward/backward/optimizer loop, LR/schedules, data, dtype, model,
checkpoint/eval code, timing charges and predictor reset are byte-unchanged.
Removing those five added lines reconstructs exact6ef7 byte for byte.

The production profile remains dense6x1024, sequence1024, pack32, rank-local
device batch1, world4, matrixLR.015, Recursive ON, batch-ramp OFF:

| Quantity | Control | Arm |
| --- | ---: | ---: |
| Rows per rank per microbatch | 32 | 32 |
| Tokens per row | 1024 | 1024 |
| World tokens per microbatch | 131072 | 131072 |
| Accumulation passes per optimizer step | 1 | 2 |
| Global tokens per optimizer step | 131072 | 262144 |
| Automatic LR multiplier | 1.0 | 1.0 |

The existing loop divides each micro-loss by `grad_accum_steps`, accumulates
gradients, then averages across ranks and clips once before the optimizer
update. Data order/pipeline stay the same; the arm intentionally consumes two
consecutive microbatches per update. Its step-based momentum warmup still spans
300 optimizer steps, and wall-clock LR/WD/Recursive formulas are unchanged.
The existing logged step loss is the last micro-loss, not the mean of both;
selection remains the unchanged fresh20M evaluator, not this training log.

No batch ramp or LR compensation is used. Native input microshapes stay fixed,
but /2 scaling and accumulated-gradient execution may introduce compile
variants/cache misses; CPU evidence cannot promise identical native graphs or
cache reuse. This is a batch-size hypothesis, not a speedup claim. Fewer
optimizer/communication operations per processed token may help throughput,
but fewer updates and changed gradient noise may hurt final quality.

## Bounded self-checks

Six tests pass normally and with `-O` (2.775s /4.206s), including actual source
AST extraction of the parser, divisibility, micro-loop and token accounting.
The micro-loop test uses tiny float64 tensors and four simulated ranks, checks
accumulation1/2 and loss scales1/128 against one global mean gradient with
rtol/atol1e-12. It proves normalization algebra, not BF16/native equivalence.
The source-reconstruction test proves model/optimizer/LR/data/evaluator/timer
code outside the option binding is unchanged. Zero, negative, noninteger and
nondivisible batches are rejected; production131072/262144 yield1/2 passes.

The first normal and optimized unit attempts found a **test fixture** error:
the isolated parser namespace omitted imported TRAIN_TIME_BUDGET_SECONDS and
its MAX_TRAIN_SECONDS alias. The fixture now extracts that constant from
unchanged prepare's AST. Production source did not change during this fix.
All three model G2 attempts passed on their first execution.

Exact commands for unit checks, adding `-O` for the second invocation:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RUN_BATCH_TENSOR_CHECKS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/recursive-batch262144/test_batch.py
```

Tiny G2 uses existing `workflow/tools/preflight_dryrun.py`, Torch2.13.0+cpu,
four ranks, 3 optimizer steps, width128/depth6 and **1x16 actual synthetic
micro-inputs**, one OMP/MKL/OpenBLAS thread. Its fake clock/BF16 CPU/no-compile
and checkpoint shims are recorded in the receipts. Local runtime dependencies
require existing `--stub-prepare` plus this packet's import-only PyArrow stub,
which rejects every real Parquet read. No packages were installed. No stub or
test_support path may be deployed for independent production/native gates.

- `tiny-g2-base.json`: exact6ef7, 4.9s, PASS.
- `tiny-g2-default.json`: new source, flag omitted, 5.9s, PASS.
- `default-lockstep.json`: three-step losses, all62 state digests and optimizer
  ownership plan are bit-identical to6ef7 on all four ranks.
- `tiny-g2-262144.json`: explicit arm, 5.7s, PASS; six finite forward losses per
  rank for three optimizer steps, strict checkpoint/deterministic reload,
  cross-rank consistency and unchanged ownership plan. Its log records
  `micro_tokens=131072 grad_accum_steps=2` from the production configuration.

**The G2 physical inputs are only1x16.** Its logged token accounting is not a
claim that262144 real tokens were processed. The independent actual-source
arithmetic test separately establishes full production token divisibility.
G2 checks flag binding and real loop execution only at reduced width/input;
exact-size G2 and native accumulation/correctness remain pending.

## Controls, priorities and handoff

`proposal.json` uses only same-host measured adopted6ef7 baselines: trn21
.9900560897366009/3669steps; trn22 .9902787975734985/3657steps. Those rows are
historical `schedule_arm` records keyed to c5c, not `kind=control` records keyed
to6ef7. Unchanged decide.py cannot silently treat them as new-base controls.
Supervisor has requested genuine fresh exact6ef7 controls on both hosts;
their labels/scores remain pending in this packet. Bind those new-base controls
before a mechanical verdict; no history is rewritten, old row reclassified or
fake control row invented. All standard BPB thresholds/two-host replication
remain.

Hardware priority remains FP8 diagnosis and architecture first. This packet
does not allocate or launch a host job. Independent source/full-size G2 and
native gates, exact source/argv/proposal/host preflight and current approved
runner remain required. Warm cache is allowed for research and must be
disclosed, not called cold. Budget is always1800, seed42, full chip4xLNC2,
fresh public20971520/context1024. Step counts are reported; below60% of matched
control or spikes/stalls prompt diagnosis, not automatic rejection. New-source
cold validation and human approval remain separate submission gates.
