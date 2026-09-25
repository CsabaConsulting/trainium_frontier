# trn22 e65→r10 source stage: enabled request, final review pending

**DISPATCH HOLD.** The exact request
`REQUEST.ENABLED_PENDING_FINAL_REVIEW.json` has
`dispatch_authorized=true` and `source_stage_authorized=true`, but those
bits are **not sufficient** for execution. The package deliberately contains
neither `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json` nor
`SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json`. The repaired
`operator_once.py` reads and authenticates both before local STOP/breaker,
SSH identity, the one-attempt marker or any host contact. Its direct
entrypoint currently exits `HOLD` under normal Python and `-O`.

This is a source-stage request only. It cannot launch native training,
evaluate, clear a breaker, remove a provisional file, or submit. The
one-shot label
`fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1`, target
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1`,
remote bundle
`/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn22-stage-r1.bundle`,
and local marker
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r10-incremental-stage-r1.one-attempt.jsonl`
are unchanged from the disabled r3 subject. This is the **first** possible
one-shot stage attempt; no pre-open retry was added. A failed first attempt
consumes the marker and leaves any empty or nonempty provisional bundle
intact, with no automatic cleanup/retry.

## Exact source and evidence pins

This package descends from the original clean science-base stage subject
`b3ec7406f0e7a3df6848698876e2106305dd283f`. Its root `train.py`
is SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and unchanged `prepare.py` is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
Do **not** run this operator from an archival Desktop controller checkout
with different root science bytes. The active Desktop controller is used
only for the explicitly pinned `research/v2/state.json` breaker/STOP
check; former Documents and package STOP roots also remain checked.

| Item | SHA-256 or exact evidence |
| --- | --- |
| Enabled request | `d19086a3f0c8aff9835f14f641d528980b12b21f45310070f6613ae261a33958` |
| Unchanged `remote_incremental.py` | `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f` |
| Repaired `operator_once.py` | `af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f` |
| Unchanged `e65-to-r10.bundle` | 109,800 B; `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Independent disabled r3 static PASS | subject `b3ec7406`, review `460f3a9f4694eaab46ffe9818ab620e1d33bd7d6`; `research/v2/verification/20260925-trn22-r10-incremental-stage-r3-controller-root-independent-static-PASS.md` SHA-256 `c7fb259490a1c5f3313439457966024b653e6104f74273d2ea03dba5151e25dc` |
| Independent read-only timing PASS | raw `23d704cfc919ab6b476898438f92396b4866f10a`, bracketing snapshots `c129bf60`, independent review `02f02d7bb1cc0a8d4b404c7373ba656bc6725562`; `research/v2/verification/20260925-trn22-r10-readonly-timing-raw-independent-PASS.md` SHA-256 `7f7ee3a298561a43305143b865cc0bdfab3436f904ef6cee4a14e261afaf62a0` |
| Independent enabled r3 final HOLD | subject `16e1c982d8e7ce5dcd53ca966cce65d5a57c7301`, review `22b5744115394ce349dcf208e3ca7fc1d6c75837`; `research/v2/verification/20260925-trn22-r10-stage-r3-enabled-request-independent-final-HOLD.md` SHA-256 `e79a0515c8007bb0927497cd3bd69fe9ea84a6727ca274a8fe1d6ac0f5c33805` |

The enabled r3 final review found that a desktop STOP or breaker arising
during `_ssh_identity()` could consume the one-attempt marker before the
next local STOP check. This r4 subject adds exactly one
`_local_stop_and_breakers()` recheck immediately after `_ssh_identity()` and
before `_marker()`. Offline tests inject each late desktop change at that
seam and assert the marker remains absent and no remote call occurs, under
normal and optimized test bytecode. The actual `-O` entrypoint still holds
before authority.

`REQUEST.HOLD.json` and the original `RUNBOOK.md` remain historical disabled
r3 artifacts with the former operator hash. Only this enabled request pins
the repaired operator. Neither the disabled static PASS nor the enabled r3
final HOLD is a final PASS for this r4 operator/request.

The timing result has 20/20 read-only modeled trials within its one-second
freshness bound, but its worst double-label-scan duration was 980,357,752 ns:
only **19.642 ms** of margin. It never executed a real `O_EXCL`, payload
write, fsync, checkout or readback. The independent reviewer recommended
investigating a bounded pre-open-only retry; this **first** request does not
implement one. That old timing result measured the exact **unchanged**
`remote_incremental.py` final pre-open sequence; it did not bind or test the
new local `operator_once.py` hash or this r4 commit. Carry-forward of that
bounded timing evidence is a decision for the independent final reviewer,
not this Implementer. The final reviewer must explicitly weigh the narrow margin
against the consequence of consuming the one-shot marker on a safe
pre-open timing refusal. No timing result waives fresh identity, activity,
STOP, label, disk or inode checks.

## Before any Operator command

1. A different Verifier independently reviews this exact canonical request
   byte string, the one-line local operator repair and its tests, and the
   unchanged r3 transfer source/bundle; it must check normal and optimized-
   Python refusals, the source-root identity and the 19.642-ms
   timing margin. Only then may it write the exact
   `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json` required by
   `operator_once.py::_authority`; this Implementer has not created it.
2. The Supervisor independently verifies that review, active Desktop
   STOP/breaker state, exact request and source hashes, and absence of the
   one-attempt marker, then may issue the exact
   `SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json` for this label
   only. No assignment exists now.
3. The Operator performs a fresh, complete read-only local/host preflight:
   pinned SSH host identity; source e65 clean HEAD, expected science bytes,
   target/bundle absence, protected checkpoints/caches, host lock and
   Neuron/process quiescence, STOP/breakers, ≥13,177,973,911 root free
   bytes and ≥10,000 free inodes before material write. A failed gate is
   HOLD, not authority to clean or retry.

After those gates, the proposed **single-attempt** command from this
verified clean science-base worktree is:

```text
/usr/bin/python3 -B /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r3-enabled-request-r1/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
```

That command must **not** be run yet. The Operator must retrieve its raw
marker journal, cross-session readback and attempted unconditional
postflight. A separate Verifier must accept the stage result, followed by
a separate G2 source-identity disposition. Only after identical clean r10
HEAD is established locally and on both trn21/trn22 may separately reviewed
native A0 controls be considered. There is no new BPB result here.

Offline-only Implementer checks, from the science checkout root:

```text
sha256sum -c research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/SHA256SUMS
git bundle verify research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/e65-to-r10.bundle
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold -p 'test_*.py' -q
PYTHONDONTWRITEBYTECODE=1 python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold -p 'test_*.py' -q
```

The new tests also run the unchanged entrypoint and assert HOLD without SSH
or a marker. These checks are not independent final review or source-stage
authority.
