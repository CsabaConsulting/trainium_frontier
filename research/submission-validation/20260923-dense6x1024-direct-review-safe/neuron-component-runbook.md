# Direct-wiring Neuron component gate

This is the hardware-only component gate requested after the fail-closed operator preflight. It
does not authorize training or submission. Run it only through the approved experiment protocol
on a clean, idle full-chip Trn2 host, from an immutable worktree at the supervisor-frozen commit.
The governing approved proposal is `proposal.json` in this directory; this runbook adds evidence
requirements without changing that proposal or broadening its authority.

## Frozen inputs and environment

- Candidate: `submissions/dense6x1024_direct_review/train.py`
- Candidate SHA-256: `c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537`
- Python: the host's competition `native_venv`, with `torch-neuronx`, NKI, and Neuron distributed
  support already installed.
- Hardware: one clean full-chip Trn2 host; no concurrent run.
- Topology: `NEURON_LOGICAL_NC_CONFIG=2`, four `torchrun` ranks, one LNC2 device per rank.
- Seed: 42. No dataset, tokenizer, evaluator, checkpoint, or remote service is used.
- Compiler flags: the candidate launch flags, `--optlevel=1 --auto-cast matmult
  --auto-cast-type bf16`.

The operator must create a unique run label and replace `<RUN_LABEL>` below. The report path and
outer protocol receipts must not overwrite an earlier run.

## Exact command

```bash
timeout --foreground --signal=TERM --kill-after=60s 1200s \
  env PYTHONUNBUFFERED=1 \
      NEURON_LOGICAL_NC_CONFIG=2 \
      NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache \
      NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16' \
  /home/ubuntu/workspace/native_venv/bin/torchrun \
    --standalone --nproc_per_node=4 \
    research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_component_direct.py \
    --train-py submissions/dense6x1024_direct_review/train.py \
    --report research/run-logs/<RUN_LABEL>.json \
    --seed 42 --host-threads 4
```

The external hard timeout is 1,200 seconds, followed by a 60-second termination grace period.
The expected envelope is at most 420 seconds for the production-shape MLP compilation/check, 300
seconds for both CE grids, and 420 seconds for optimizer compilation/collectives/state checks.
Those are diagnostic expectations, not independent stage watchdogs; exceeding the outer timeout
is a failed gate and requires clean postflight, not a retry inside the same run.

## Predeclared numerical gates

| Quantity | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Fused MLP output | 0.0625 | 0.02 |
| Fused MLP input gradient | 0.002 | 0.03 |
| Fused MLP up-weight gradient | 0.02 | 0.03 |
| Fused MLP down-weight gradient | 0.02 | 0.03 |
| Adaptive CE loss | 0.0002 | 0.0002 |
| Adaptive CE logit gradient | 0.00005 | 0.0005 |
| Owned-optimizer parameter update | 0.000001 | 0.000001 |

Every compared tensor must also be finite. The MLP fixture is the launch trigger shape
`[32, 1024, 1024]` with a 4096-wide hidden layer and a differentiable BF16 weight cache. The
reference is an independent eager `linear -> relu -> square -> linear` autograd expression. The
gate requires an `_FusedMLP` node in the actual autograd graph.

CE uses real vocabulary width 8192 at 128 rows (one lane) and 512 rows (two lanes). Its reference
is independent PyTorch `cross_entropy(15*tanh(z/15), target)`. Temporary call-through probes wrap,
record, and immediately restore the real NKI grid objects; each forward and backward grid must be
called exactly once with the expected lane count, and the direct NKI autograd node must be present.

The optimizer gate uses all four LNC2 ranks. It compares one explicit `OwnedOptimizer` update with
the same candidate optimizer updating every parameter independently on every rank. It requires an
exact, omission-free ownership plan, all four owners, finite collectives, the intended base/owner
API split, complete AdamW and stacked-Muon local state, exact state round-trip, and rejection of a
bad schema. All three component stages must pass on all four ranks.

## Fail-closed output contract

Rank 0 atomically writes one JSON report and prints it after the marker
`DIRECT_NEURON_COMPONENT`. Every rank prints a compact `DIRECT_NEURON_COMPONENT_RANK` record.
Success requires report `passed: true`, `global.all_ranks_passed: true`, pass counts of four for
all three stages, and `distributed_cleanup_succeeded: true`. Any exception, missing metric,
non-finite value, tolerance violation, wrong hash/topology/lane, incomplete state, timeout,
nonzero exit, or dirty postflight is failure. A component pass does not establish full-model
compilation, timing, checkpoint replay, BPB, reviewer acceptance, or submission readiness.

Local static-only command (never a hardware substitute):

```bash
python research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_component_direct.py \
  --static-only \
  --train-py submissions/dense6x1024_direct_review/train.py \
  --report /tmp/dense6x1024-direct-neuron-component-static.json
```
