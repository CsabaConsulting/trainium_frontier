# PASS_FOR_NATIVE_DIAGNOSTIC

Date: 2026-09-26. One bounded independent source review; no native execution.

Candidate SHA256: `3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99`

Base SHA256: `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`

This verdict applies only to the explicit default-OFF component diagnostic in
the supplied bundle. It is not launch authorization, scoring eligibility,
quality adoption, a native correctness result, or a performance improvement.
No blocking source defect was established within this scope.

## Independence and evidence

Inputs reviewed: bundle `train.py`, `base.py`, `DIFF.patch`, `SPEC.md`,
`cpu-result.json`, and `test_p3.py`. I did not access the implementer's branch,
notes, or shared P3 analysis/status. The user-level repository contract required
initial reading of README, CURRENT_STATE, experiment-protocol and DECISION_LOG;
that mandatory contract read was disclosed to the supervisor and is the sole
departure from bundle-only context. Findings here rely on bundle source and
the independently authored reference, not historical experiment claims.

`reference.py` and `result.json` in this directory are my independent evidence.
Command executed successfully (exit 0):

```
/home/poojith-devan/venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/verification/20260926-p3-sprint-independent/reference.py
```

The reference extracts exact AST definitions from the candidate. It does not
rerun the author's tests or import repository test stubs. Small BF16 integer
inputs permit an independently written scalar-loop matrix calculation to
match full MLP output, all three original custom gradients, and every split
activation/matrix result bit for bit. Five VJPs on the same retained graph
also match. This establishes a concrete exactness claim on the tested domain,
not blanket BF16 equivalence across all inputs or devices.

A separate dense masked-softmax attention reference checks SDPA forward and
Q/K/V gradients on small FP32 CPU tensors; max forward difference was
`2.384185791015625e-07`, with checks at at most `3e-6` absolute/relative
tolerance. Transposed BTHD layout is preserved. No full-size CPU test ran.

## Source findings

- OFF equivalence: deleting precisely the added diagnostic functions, four
  parser options and early dispatch yields an AST identical to base. Existing
  model/evaluation/data/optimizer/training-clock statements are unchanged.
  The store-true flag defaults false; early return occurs before data loading,
  model construction, training, checkpointing or scoring.
- Geometry: the fixed SDPA tensors are B32/T1024/H8/D128 in transposed BTHD
  storage, with matching upstream layout. MLP tensors use 32768x1024 inputs,
  4096x1024 up weights, 1024x4096 down weights, BF16 throughout. These match
  the SPEC's chosen configuration and eligible production MLP boundary;
  they do not claim to represent the baseline's unoverridden CLI defaults.
- Callable boundaries: SDPA uses the original causal call and native autograd.
  Full MLP directly invokes unchanged `_FusedMLP`; decomposition products and
  saved activation orientation agree with its actual forward/backward. Weight
  cache refresh, FP32 parameter cast-back and surrounding model operations
  are explicitly excluded. Split operations are materialized, nonadditive
  diagnostics and are correctly described as such.
- Timing control: barriers and synchronization precede blocks, and device
  synchronization ends them. First forward/backward calls are separately
  recorded and excluded from steady samples. The backward closure refers to
  the newly assigned current output, so each backward block uses its latest
  saved graph rather than a stale first-call graph. Exact nested controller
  execution with small CPU tensors and disclosed identity-compile/block
  substitutes verified two warmup blocks, four timed blocks, three calls per
  block, and separate backward records. This is a control-flow test, not a
  native compiler/timing emulation.
- Outputs are checked for finiteness outside timing and component records are
  saved incrementally. Bounds reject invalid runtime/block/repetition settings.
  Runtime exceptions fail rather than substituting an alternate primitive.
  The watchdog remains active through teardown and exits 124 on expiry.
  Startup failure before the inner try may lack a FAILED JSON snapshot, but
  raises and exits; it cannot produce a successful diagnostic record.
- No evaluator, tokenizer, data pipeline, scoring-function, official clock,
  AWS, process-lock, or external-process mutation was introduced. Random
  synthetic tensors do not draw on training or validation data.

## Limits for interpretation

Native backend support, exact native VJP behavior, synchronization behavior,
memory footprint and compile completion within the deadline are unproven.
The existing synchronization helper assumes the installed native stack exposes
`torch.neuron.synchronize`; the native environment must support that contract
for these times to be meaningful. Repeated saved-graph backward failure is a
failed diagnostic, not permission to change the measured operation.

The watchdog is not whole-job supervision: a Python thread cannot guarantee
interrupting a native call that indefinitely holds the GIL, and process exit
does not reap compiler children or establish host idleness. The SPEC already
requires Operator hard process supervision, a lifetime lock and child cleanup;
that obligation remains. Compiler cache provenance must be recorded before
describing first-call measurements as cold. Use the prescribed seed 42.

Rank files are separate; assess contention with the slowest-rank component
result. First-call fields include execution and any compilation, steady fields
include host dispatch, and none of these values alone establishes production
step savings or BPB gain. Partial snapshots or timeout/failure logs must not be
reported as complete. No additional source revision or review iteration is
requested by this verdict.
