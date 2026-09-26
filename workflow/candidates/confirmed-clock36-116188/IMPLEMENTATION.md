# Implementer handoff — not approval or launch authorization

Base repository commit a5da356a; isolated branch codex/20260926-confirmed-clock36-clean.
Implementation follows completed proposal20260926-confirmed-clock36-clean-116188.
Source is a disclosed reuse of the preserved old3f5035 architecture draft, with
only the exact confirmed36 helper and schedule consumers added. The source test
reverses precisely those additions and requires byte equality to the old draft.
Full diffs versus original116188 and exact36 are supplied, so review is not
restricted to the small incremental change.

Architecture attribution/provenance is retained in the old source and proposal;
the architecture/cache work came from architecture3e79-116188. Clock helper and
operation order come verbatim from research commit7419927c. No new scientific
factor, parser option, launch flag, eval route or real-time budget change.
No protected source/artifact was replaced; no host contact, training launch,
package installation, data read, AWS action, upload or Downloads change occurred.

Test command (one process/thread, tiny models only):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python workflow/candidates/confirmed-clock36-116188/test_port.py

Attempt1 raw streams are retained. It reached all three-step tensor comparisons
and detached-cache control but failed its reused negative skip probe with
OSError: could not get source code. Cause: in-memory import advertised a nonexistent
module file. Test-only spec metadata was corrected to the actual candidate file.
Before attempt2, the reused probe's successful sys.exit was replaced by an assertion
so additional retained optimizer/cache comparisons actually execute. No model
source change or tolerance relaxation occurred. Attempt2 raw streams are retained.
Use their final status as evidence, not this narrative alone.

Attempt2 exited0:596/596 numerical records,420 scalar clock cases and all three
optimizer/cache trees equal. Attempt3 added context256 B1/B2/B3 ordinary trained
versus loaded candidate, independent rows, changed-suffix causality and reference
loader comparisons; all had max absolute/relative error0 and the process exited0.
The real row-route discrepancy is retained in hook traces. Core test observed
43,590,044 live tensor bytes; that watcher covers the original bounded probe,
not the subsequently added context256 tests. Those also use only D128/V256 tiny
models, sequential sources and one thread. Native kernels are not executed.
Attempt3 artifacts: /tmp/confirmed36-tiny-4lt28iv2.

The harness writes checkpoint/tensor artifacts only to unique /tmp/confirmed36-tiny-*
directories. Their paths are in stdout; checkpoints are not committed. Raw results
contain per-tensor errors, memory bounds, environment and explicit native=false.
The inherited word COMPILED in optimizer diagnostics denotes the CPU Python
fallback in these tests, not a compilation measurement.

Remaining limitations and required fresh review are specified in REVIEW_SPEC.md.
No claim of independent PASS, full113M G2, cold startup, native identity, new BPB,
legality approval or upload readiness is made. Old REFUSE stays preserved.

Reversible decision: reuse the old clean architecture implementation as disclosed
source evidence, then add confirmed clock semantics in a new directory. Reason:
minimal exact proposal scope; undo by ignoring this new candidate, preserving all
existing artifacts and failures. No adoption/state/queue mutation made.
