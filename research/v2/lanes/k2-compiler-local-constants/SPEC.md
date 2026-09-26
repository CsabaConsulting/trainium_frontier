# K2 compiler-local geometry constants

Implementer delivery, 2026-09-26. Off-host source/CPU/parser evidence only.
This is a distinct evidence-backed scope fix. The consumed-retry a318 native
variant remains parked; this delivery grants no review or launch approval.

Candidate SHA-256:
`662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.
Frozen parent remains
`a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`.

## Exact change and evidence

The full `SOURCE.diff` moves the two existing immutable tuple assignments
and their comment from the outer factory into the beginning of the actual
`@nki.jit resident_consumer`:

```
vocab, tokens, channels = 8192, 32768, 1024
partition, features, groups = 128, 512, 64
```

Their values and every subsequent NKI operation are unchanged. There are no
metadata, hash, int32/int64 boundary, FP32 accumulation, gather, gradient,
precision, flag-off, eval, optimizer, timer or other source changes.

The exact-version real-parser diagnosis is committed at 5d060b09:
`research/v2/analysis/20260926-k2-parser-failure/REPORT.md`.
It reproduces the native `tokens` unbound error, then demonstrates failures
for the other five closure cells as earlier names are locally bound.
The selected repair removes all six captured cells. Plain Python resolves
these closures, explaining why the previous CPU instruction model passed.

## Executed CPU tests

The prior `test_k2.py` and `cpu_instruction_model.py` are copied byte for
byte from a318's lane. No interpreter edit was necessary: its existing AST
walk locates the tuple assignments within the kernel body and substitutes
only reduced test geometry as before.

Command, exit 0:

```
/home/poojith-devan/venv/bin/python -B test_k2.py --base /home/poojith-devan/Desktop/Trainium/workflow/candidates/confirmed-clock36-116188/train.py
```

`cpu-result.json` contains exact source/test/base identities and raw gradient
bits. Both int32 and int64 inputs separately pass 12 finite output/FP32-gradient
operation fixtures, production metadata ordering/count checks, duplicate
collision/order/subnormal fixtures, reused input storage and two tables/two
microsteps. Four NaN/Inf/overflow fixtures per dtype remain explicitly
`FAIL_NONFINITE_DIAGNOSTIC`, never counted as valid training passes.
All 21 input-contract rejection checks pass.

For each dtype, three OFF and three ON reduced-eight-layer steps match the
audited e1e2 base in loss, every gradient and updated parameter bytes; normal
evaluation matches. Geometry: eight layers, width64, context8, batch2, test
SGD0.01. ON uses the CPU-interpreted actual consumer. Existing test-only
prepare and unused parquet imports remain disclosed shims. This is not a
native/full-size optimizer/full-training-loop proof.

## Exact candidate real-parser test

Command, exit 0:

```
/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B test_parser.py
```

`test_parser.py` extracts the exact unmodified factory AST from this
candidate, retains the actual file/line source, executes that factory with
the real NKI package, and uses `source_to_mlir(..., FrontendKind.PARSER,
target='trn2', lnc=2)` followed by `CompilationResult.verify()`.
There is no source/body/constant rewriting, virtual diagnostic variant,
fake NKI import, data tensor payload or device execution in this test.

Production metadata arguments are BF16 gradient[32768,1024] and int32
counts/starts[8192,1], positions[32768,1], group_maximums[64,1].
The kernel has no Python closure cells. The exact host-version NKI package
`0.5.0+28631259367.ga768afa6` under Python3.12.13 produces
`PARSER_MLIR_PASS`, verified MLIR1,062,963 bytes, empty stderr.
MLIR SHA-256:
`404acb968b16f1bb922b57b8ee2422eab3aedb7de57a54a59512597889dbc76b`.
The parser script hash and full stdout/stderr/exit receipt are retained in
`parser-result.json`.

This reuses the already authorized isolated diagnostic environment; no
packages were installed or changed. NKI version is matched to archived host
metadata, not a separately authenticated host wheel binary. The prior
diagnosis documents local dependency versions and package acquisition.

## Remaining boundaries

Verified MLIR ends before BIR/backend lowering, neuronx-cc, allocation/liveness,
executable generation, framework capture and native execution. Native DMA and
floating-point behavior, cold startup, whole-step timing and fixed-budget BPB
remain unknown. No numerical tolerance, resource floor, budget, idle/lock
gate or failed-run verdict is changed.

Root assigns one isolated fresh review of these exact bytes, then only the
sole Operator may execute a separately completed/qualified new proposal.
No SSH, host action, new review, launch, state or ledger write occurred here.
