# trn22 incremental final-continuity timing r4 — DISABLED / HOLD

This is a distinct read-only diagnostic proposal for the held trn22 e65→r9
source stage. It is **not** source-stage, native-run, evaluation, AWS, or
submission authority. No trn22/SSH contact was made in preparing it. The
checked-in `GATE_CONTRACT.HOLD.json` has `dispatch_authorized: false` and the
three separate authority files named below do not exist. Do not run the
collector on a host from this package. A separate Verifier must review these
exact bytes and the trn22 r2 source-stage status first.

## Pinned subject

The proposed writer is commit
`a7643f05c5468da6537c2d7b403a05ac048679f0`, file
`../trn22-c5c-a0-r9-incremental-stage-r2-hold/remote_incremental.py`, SHA-256
`2b84ef670811b3f0015b30e619cb3ec5b23e914326cdd3e74523fdc5b68ee7a7`.
The local collector checks that exact file hash before it can read any
authority file. It **does not execute or transmit that writer source**. The
diagnostic is a standalone read-only equivalent of r2 lines 395–417:

1. check STOP/process activity, `/dev/neuron0` holders, `neuron-ls`, then
   STOP/process activity again;
2. scan the attempt label across `/home/ubuntu` twice with `find -xdev`;
3. directly check every pinned STOP path plus the bundle and target names;
4. sample a monotonic `would_open` clock without opening either name, and
   compare activity and second-scan ages with the one-second bound.

The host, source linked-worktree identity, attempt label, bundle, target,
STOP-path set, process markers and existing cooperative lock are copied from
the exact r2 source. The diagnostic has **shorter** read-only child-command
timeouts (2 s for `fuser`/`neuron-ls`, 3 s for each `find`) than the r2 writer's
20/45 s limits. A timeout is an abort, not a speed result. The diagnostic
therefore measures the same *ordering* and commands with conservative
timeouts, but does not prove exact writer latency, the final return-to-O_EXCL
scheduling gap, absence of a same-UID/root race, or that a future transfer is
safe. The current independent source-stage HOLD remains controlling.

The remote identity is exact trn22 (`ip-172-31-32-253`, UID 1000, persistent
device 66305) and the accepted e65 linked source, including its `.git`
pointer, clean HEAD, unchanged `train.py` and `prepare.py` hashes. A full
stage source/protected-asset/ACL/mount review is **not** replaced by this
diagnostic. The old source is never patched. All remote operations take the
existing nonblocking cooperative flock by a read-only, no-follow descriptor;
they never create the lock, archive, target, marker, journal or any host file.

## Raw evidence and bounded execution

If later separately authorized, one fixed local attempt runs three pinned
SSH processes: `probe`, `timing`, `probe`. The second `probe` is attempted even
after timing timeout/failure, including if a local STOP appears during timing.
The initial and final probes check host/source/lock identity, absent STOP and
target, process/Neuron idle state, a broad label scan, and the 13,177,973,911
byte/10,000 inode preflight floor. They are read-only observations, not source
stage preflight or independent approval.

The timing process emits canonical bounded JSONL immediately: a `start` line,
one flushed line after **each completed trial**, then an `end` line if it
finishes. It attempts at most 20 trials. A STOP, target, holder, changed
identity or malformed helper output aborts; an age over one second is a
`TIMING_MISS` and may be sampled again. The shell command has a 30 s hard
probe limit and 120 s hard timing limit; local SSH has 45/135 s timeouts.
Those timeouts may terminate only diagnostic-owned children, never signal an
existing workload. No arbitrary host process is signaled. SSH has `/dev/null`
stdin, a sanitized environment, Python `-I -B`, exact pinned ED25519 known
hosts, no agent forwarding, no host-key updates, no multiplexing, and no
transfer operation. No AWS action is present.

The collector exclusively creates only the local
`OPERATOR_RAW_TIMING_RESULT.json` before first contact, consuming the one
attempt even if it fails. It stores raw base64 stdout/stderr and hashes for
each reached phase, the longest complete strict JSONL prefix, any trailing
partial bytes, timeout/return code, errors, and before/after comparison.
Thus completed trials remain visible when the 120/135 s hard limit ends a
later trial. No retry, overwrite or cleanup is implemented. This fixes the
prior final-r3 diagnostic's end-only emission, which lost completed in-process
timings on timeout; its separate SyntaxError is not reused here.

## Authority and interpretation

The absent `TIMING_REQUEST.ENABLED.json` must match every fixed field and the
disabled contract hash. An independent
`INDEPENDENT_TIMING_REVIEW.PASS.json` must bind the actual request and source
hashes. Only a distinct
`SUPERVISOR_TIMING_ASSIGNMENT.json` may assign one attempt. The collector
checks all three, local STOP/breakers, pinned known-host bytes and host
fingerprint, and fixed result absence *before* SSH or result creation.
Without them it fails locally. No agent may infer those permissions from this
runbook or a passing local unit test.

Even a complete, all-under-one-second 20-trial receipt is **only timing
evidence pending independent review**. Any incomplete/partial/timeout result
is failed closed and retained. The r2 source-stage repair still needs its own
independent static and exact live gates, plus the separate G2/native/monitor
chain. In particular, the r1 source-stage HOLD and any independent r2 HOLD
for late activity/label/STOP changes or the last check-to-O_EXCL gap are not
lifted by this study.

Implementer-local synthetic tests, not independent verification:

`python3 -B -m unittest -v test_timing_readonly.py`

`python3 -O -B -m unittest -v test_timing_readonly.py`
