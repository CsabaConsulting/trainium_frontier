# Fresh independent verifier: REFUSE

Reviewed frozen packet `/home/poojith-devan/k2-f0f2-fresh-review-PC6TPV`.
The supplied `SHA256SUMS` digest is
`7481b87b2d6f8b1ee8842734058444fda3790b5bf21d931286b78aac503e4a8b`;
all listed files verified. All 22 added diff files also reconstruct byte-identically
to the supplied snapshots. Review used only this packet and local tiny CPU probes,
not implementer branches/history, host access, network, native SDK or model execution.

**Verdict: REFUSE_IMPLEMENTATION_MECHANICS_REVIEW. No release or launch authority.**
The disabled `BINDING=None` remains appropriate. No scientific/kernel adoption,
full K2 PASS, BPB result, speed result or native correctness conclusion follows.

## Blocking findings

### 1. Failed terminal persistence can bypass the required original-lock hold

`host_driver.py:198-205` invokes `record('terminal.json', terminal)` before the
protection hold, without containing exceptions from that write. `record` itself
can raise on the driver's 120 MiB cap, exclusive file creation, disk-full, or
ordinary filesystem failure. `monitor_log.close()` at line 172 also precedes
containment without exception protection.

Independent reproduction executes the actual `host_driver.main()` with mocked
host-only preconditions, a real local reviewer-owned flock, a failed idle check,
and injected `OSError(ENOSPC)` specifically on opening terminal.json. Main escapes
with that error instead of entering the protection hold; after process exit the
same lock is acquirable. This is not a native flock/host test: it demonstrates
the exact failing control flow and actual process-lifetime flock release locally.
The production outer flock also releases when its synchronous driver exits.

Raw evidence: `raw-results.json:terminal_hold`; reproduction modes `terminal`
and `terminal_test` in `review_checks.py`. Failed containment must retain the lock
even when terminal evidence cannot be persisted. The present implementation does not.

### 2. Preflight descendants escape child-only timeout containment

`host_driver.bounded_run`, lines 20-46, kills and waits only for its immediate
Popen child. The driver runs git and idle helpers before enabling subreaper and
constructing its tracker at lines 121-122. The initial pidfd self-probe does not
provide descendant containment.

Independent reproduction executes the actual bounded_run with a tiny fork helper
and 0.2-second deadline, in a process that has not enabled subreaper, matching the
preflight order. The helper's direct parent is killed, but its TERM-ignoring child
remains alive. A separate reviewer subreaper adopts that child and demonstrates
its live PID/start/RSS; the reviewer's pidfd tracker then sends TERM and KILL and
reaps it. No test process was left running.

Raw evidence: `raw-results.json:preflight_orphan`. A successful standalone
Processes.stop double-fork test cannot waive this actual driver sequencing defect.
The historical child-only timeout/orphan problem is therefore not resolved for
the whole process lifetime by the added monitor.

### 3. Postflight helpers can create survivors after the last termination census

`host_driver.py:177` completes tracker.stop before postflight invokes bounded_run
again at line 187. There is no descendant recensus/reap after the postflight
helpers. A helper can exit zero and close its child's stdout/stderr while the
child survives, so bounded_run returns success.

Independent reproduction enables the actual subreaper, completes actual
Processes.stop, then runs a tiny zero-exit fork helper through actual bounded_run.
The helper returns 0 with a live adopted descendant, verified by PID/start/RSS.
The reviewer subsequently cleans it with actual pidfd TERM/reaping. That last
cleanup exists in the probe, not in the candidate postflight sequence.

Raw evidence: `raw-results.json:postflight_orphan`. Parsed ps/device-idle checks
are not a replacement for census of all owned descendants; generic compiler/helper
descendants need not match the idle parser's selected command names.

### 4. Separate provenance issue: one PID is accepted as all four ranks

`coordinator.py:129-145` requires distinct claimed rank integers, but does not
require distinct authenticated `(PID,starttime)` peers or exclude the torchrun
parent itself. The ancestry loop accepts the exact child PID without traversing
to a worker. The hello's rank/world/source pins are self-reported.

Independent reproduction runs actual coordinator.main with only generation,
torchrun launch, and route bodies mocked. A single real local child opens four
Unix connections, reports ranks 0-3 with identical PID/starttime and correct public
pins, and passes the actual SO_PEERCRED, source-claim and ancestry checks. All four
environment receipts are written and all four route mocks are scheduled.

Raw evidence: `raw-results.json:rank_association` plus `rank-probe/artifacts/`.
The route mocks intentionally raise; this test does NOT claim an accepted entire
native diagnostic, an unmodified-worker exploit, or arbitrary same-UID attack
resistance. It establishes that the coordinator's purported four-rank process
association does not itself enforce four distinct worker processes. This needs
separate correction/verification; dictionary claims are insufficient provenance.

## Independently verified positive evidence

The reviewer wrote and executed an exact rational reference using Python Fraction
and a binary search of the ordered finite binary32 lattice, selecting nearest
with ties to an even low bit. This differs from the candidate's integer
bit-length/shift rounding implementation and uses no NumPy addition reference.
It matched 1,386 finite input pairs including cancellation, signed zero,
normal/subnormal boundaries and overflow. It also matched all 1,024 F0 channel
four-addition recurrences and all eight F2 classes on rows 0,127,128,4095,4096,8191.
The full scalar F2 token-ID recipe was independently checked. F2 expected positive
and negative reachable subnormals were `00010000` and `80010000`; zero outcomes
retain positive initialization semantics. No full geometry tensor was allocated.

Tiny full-byte comparisons reject a changed last word, a missing mock second
owner, signed-zero change and nonfinite values. Identical NaN bytes also fail
the finite diagnostic. Actual socket-stream input-before, input-after and retained
blob corruption tests reject with FULL_BYTE_MISMATCH or OUTPUT_NOT_EXACT_FINITE;
full corrupt blobs and associations remain in byte-probes. These tests exercise
the real evidence layer with eight words, not actual F0 device storage.

All seven resource threshold/gap negative probes fail closed. UNKNOWN precision
fields remain explicit nulls with a reason, and the implementation does not claim
sampled memory/disk guards are exact peaks. The monitor's actual pidfd/subreaper
mechanics successfully cleaned the reviewer's controlled escaped descendants.
That positive helper behavior does not repair findings 1-3 in driver sequencing.

## Source review and limits

Worker source calls the pinned `build_kernel(enabled=True)` callable directly;
ordinary source uses embedding -> immediate BF16 -> actual backward. It retains
F0 output through repeat/F2, recaptures it after both, reuses F0 inputs for repeat,
and replaces nonweight tensors once when switching to F2. The three F2 routes
then reuse those tensors, and a single compiled callable spans F0/F2. These are
source observations, not independently executed SDK behavior. The reviewer did
not execute a full seven-route worker mock or four-rank native test after the
blocking failures were established and the supervisor requested a prompt freeze.

Complete input/output streaming, finite comparisons and staged challenges are
useful checks, but hardware execution counters remain unknown. This review did
not demonstrate that all source-preserving consumer bypasses are impossible.
The supplied CPU-object/fabricated-dictionary negatives are not proof of native
invocation provenance. The concrete same-PID admission issue above remains open.

Actual frozen-SDK device compatibility, compilation, full-size generation/runtime/
disk feasibility, full 32 MiB comparisons on native returns, real four-rank
distinctness, native ordinary-control exactness and host containment remain
unobserved. No selected F0/F2 evidence can qualify the six-fixture suite, waive
ordinary-path failures, or establish BPB/speed. Supplied normal/optimized receipts
were read as implementer evidence; they were not relabeled as independent tests.

No packet source was edited. No fixes, launch binding, state change, cleanup of
preexisting artifacts, host command, AWS action or native allocation was performed.
Only local reviewer-created child processes were signaled/reaped. Review-generated
files are retained. Commands: `python -B review_checks.py`, then
`python -B freeze_evidence.py`. `MANIFEST.sha256` freezes every regular review file
except itself, including nested raw byte/association evidence; control.sock is a
closed, inert Unix socket pathname, not a running service.
