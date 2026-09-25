# Independent review: bounded FP8 VJP diagnostic

2026-09-25 — **PASS_DIAGNOSTIC_PACKET_ONLY** for commit
`fe781c226a7c700a71324783b70018613c1dbcaf`. No blocking checker/packet defect
found. This is independent source/CPU review of one proposed diagnostic, not a
native result or launch authorization. The original `c1ac162c` component failure
and `357ba6e2` unresolved-cause audit remain in force. All 1% gates are retained;
timing and quality remain blocked on the existing applicable correctness gates.

## Identity and unchanged acceptance path

Production `train.py` is byte-identical to `50a901374ae6f76b35b8556d32fd72b1ece28fc6`:
`921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
`prepare.py` is also unchanged:
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
Only checker additions, its CPU test, and four packet files changed.

Reviewed support hashes:

| File | SHA-256 |
|---|---|
| `check_native.py` | `d8187ad2f3e68d204203a6252e16062e52521e8b93f119c7f0a9fbe82e4367fd` |
| `test_native_diagnostic_cpu.py` | `62313338a415f71baa74bf290b204160a5c248efb9b31038eaba4976879683b1` |
| `EXPERIMENT_PROPOSAL.json` | `321fe230688a61bfe9c921048d7a9e8ace9bf523b8023f96bebdbc530df058f2` |
| `RUNBOOK.md` | `133f761170955935e8a76a07553e12bbc329f16382365fd616f9f07af71468d4` |
| `SOURCE_MANIFEST.json` | `fb423227616f572bad8464bf813c0dfb8bb412b5198687aab14e2dd66e65717b` |
| `CPU_SELF_CHECK.json` | `713f16b16cc84de7d174ef41f21968070726340668a720d65aaf30ce8e65178d` |

The independent `audit_static.py` verifies all manifest identities and the
original checker hash `7ba967b06834071f33cf9158c1f2a76555e9850d0abc988421b4b870ce1e3bc0`.
Removing only the two opt-in argument declarations and two diagnostic branches
from the new `main` produces an AST **identical to the complete original main**.
The metric function is AST-identical separately. Thus the original four shapes,
backward comparisons, strict `>0.01` failure condition, prefix/padding check,
receipt, cleanup and acceptance flow are unchanged, not merely similar text.

Diagnostic mode requires a label and exact9212 hash before importing the source.
It returns before the original gate, writes only `diagnostic-rank*.json` with
exclusive creation, and always labels the result
`DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS`, `full_component_gate_passed=false` and
`whole_step_measured=false`. No observed comparison can promote it to acceptance.

## What the observations can establish

The fixed seed42 first256-cube preserves zero/tiny/outlier rows and BF16/FP32
types. Original compiled bound-apply is first and performs no pre-backward saved
tensor inspection. Fresh detached leaf clones isolate later gradient buffers.
The enclosing function puts `.apply` inside the compiled frame. Eager and
already-compiled replay inspections are explicitly marked as materializing
saved tensors before backward, so a changed inspected result cannot replace
the original uninspected result.

Finite state, actual gradient maximum/norm/nonzero count, expected statistics
and separately named `error_max_abs` distinguish zero gradients from the old
ambiguous error maxima. Missing/nonfinite gradients remain visible failures.
Upstream round-trip checks address transfer; native BF16 control using transferred
reference operands distinguishes ordinary device GEMM from a direct NKI operand
handoff. Direct-operand VJP and actual autograd-context records are kept separate.
Same-shaped saved tensors are compared against named references without assuming
ordering. Traversal is bounded to16 nodes/tensors; unavailable inspection is
reported, not inferred as success. Four case logs are flushed before final JSON.

Limitations are correctly diagnostic, not blocking defects: the enclosing
fullgraph wrapper is an inner capture-boundary contrast, not the full compiled
production Linear/model; the upstream round trip is not direct observation of
the G arriving inside compiled backward. Saved-context materialization may
perturb execution, and SDKs may expose different/no saved tensor records. These
contrasts can localize a failure but do not automatically prove its root cause.
The runbook retains non-reproduction as non-reproduction, not a retroactive PASS.

## Independently executed checks

From the isolated review worktree, with one OMP/MKL/OpenBLAS thread and no native
imports, both commands completed exit0:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/fp8-projections/test_native_diagnostic_cpu.py -v
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -O -B research/v2/lanes/fp8-projections/test_native_diagnostic_cpu.py -v
```

Normal: **3 tests PASS**, unittest1.445s. Optimized: **3 tests PASS**, unittest4.266s.
These cover exact source, original threshold, normal/missing/nonfinite statistics,
all four CPU diagnostic paths with AOT-eager, actual saved-context inspection,
manual controls and deliberate zero-VJP fault injection. Injected zero VJPs
retain RMS1.0 and actual maxima0 versus error maxima4.875/326, while manual
controls agree; the receipt remains diagnostic-only. The explicit CPU kernel
adapters test plumbing, not NKI behavior. Largest tensors are256-square; no
model, optimizer, data, checkpoint or host was used.

The tensor-free independent audit also completed exit0:

```sh
/home/poojith-devan/venv/bin/python -B research/v2/verification/20260925-fp8-vjp-diagnostic-review/audit_static.py
```

Its source/hash/AST/flags results are in `static-audit.json`.

## Proposed native bounds and binding handoff

The packet specifies one unique `fs-20260925-l1-921271-vjpdiag-r1` label, one next
free host, world4/LNC2, exact source/support hashes, unchanged SDK/cache/compiler
settings, no overlap/interruption and **300 seconds wall plus60 seconds kill
grace**. It uses only the failed first shape, two compiler entry points, one eager
call, an inspected replay and four manual BF16 GEMMs. A timeout is incomplete,
not a pass or automatic retry. There is no new launcher, optimizer workload,
full-model run or training-budget claim.

The existing manual driver still needs its explicitly described mechanical
diagnostic-only binding: fresh immutable paths/label, pinned hashes/review, exact
absolute argv including `--diagnose-first-case --label`, bounded timeout and
diagnostic receipt predicate. Its old component-PASS predicate must not be used.
This review covers the packet, not an unprovided driver/binding modification.

**Binding detail:** caught per-path exceptions are recorded as
`cases[*].status="ERROR"` and can coexist with exit0 and four diagnostic files.
Honor the runbook's exception/incomplete rule by inspecting these statuses;
exit0/files alone do not establish all contrasts were observed. Numeric
comparison failures with `status="OBSERVED"` are valid completed diagnostic
evidence, never component acceptance. Root and Implementer were notified of
this distinction; no checker/source edit is required for this review.

No native action was taken. The original failure, component gates and matched
whole-step >=3% requirement remain unchanged.
