# trn22 e65→r10 source stage: r5 remote-timeout repair pending review

**DISPATCH HOLD.** The exact request
`REQUEST.ENABLED_PENDING_FINAL_REVIEW.json` has
`dispatch_authorized=true` and `source_stage_authorized=true`, but those
bits are **not sufficient** for execution. The repaired operator requires
distinct, currently absent r5 authorities:
`INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW_R5.PASS.json` and
`SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT_R5.json`, each with its
new exact schema. The earlier r4 PASS and assignment JSON remain here as
audit evidence, but cannot satisfy either r5 path/schema or the new
operator/request hash. The direct entrypoint currently exits `HOLD` under
normal Python and `-O`, before SSH or the one-attempt marker.

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
| Enabled request for this disabled r5 repair | `25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2` |
| Unchanged `remote_incremental.py` | `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f` |
| Remote-timeout repaired `operator_once.py` | `6224763c0acdbba8bfadf4281498fe9801a8b8d4aaee3d37f8b58f2f97b412be` |
| Unchanged `e65-to-r10.bundle` | 109,800 B; `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Retained historical r4 independent PASS | `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json` SHA-256 `c9be026c2609fbbdcfdc85005630675959a2b6661befb91268dd4cf73d59d615` |
| Retained historical r4 assignment | `SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json` SHA-256 `ed07f024e45b5a606c4c8ed3f0ad6bc335a6bcaead548bf89c1e0cbcd2ec5bb4` |
| Independent disabled r3 static PASS | subject `b3ec7406`, review `460f3a9f4694eaab46ffe9818ab620e1d33bd7d6`; `research/v2/verification/20260925-trn22-r10-incremental-stage-r3-controller-root-independent-static-PASS.md` SHA-256 `c7fb259490a1c5f3313439457966024b653e6104f74273d2ea03dba5151e25dc` |
| Independent read-only timing PASS | raw `23d704cfc919ab6b476898438f92396b4866f10a`, bracketing snapshots `c129bf60`, independent review `02f02d7bb1cc0a8d4b404c7373ba656bc6725562`; `research/v2/verification/20260925-trn22-r10-readonly-timing-raw-independent-PASS.md` SHA-256 `7f7ee3a298561a43305143b865cc0bdfab3436f904ef6cee4a14e261afaf62a0` |
| Independent enabled r3 final HOLD | subject `16e1c982d8e7ce5dcd53ca966cce65d5a57c7301`, review `22b5744115394ce349dcf208e3ca7fc1d6c75837`; `research/v2/verification/20260925-trn22-r10-stage-r3-enabled-request-independent-final-HOLD.md` SHA-256 `e79a0515c8007bb0927497cd3bd69fe9ea84a6727ca274a8fe1d6ac0f5c33805` |
| Prior r4 Operator HOLD | reviewed/assigned source `e827fd2abecf813474f2d6c95518b4ac331e04b4`; observation `532ded624ffe04caa9af22f94e79ea792052b4f2`, `research/v2/observations/20260925-trn22-r10-stage-operator-timeout-hold-r1/RECEIPT.md` SHA-256 `b12cb963b188bae60edc9ef9ba74e94b6f6121afc72bd209d441884b62fad2bd` |

The enabled r3 final review found that a desktop STOP or breaker arising
during `_ssh_identity()` could consume the one-attempt marker before the
next local STOP check. The prior r4 subject added exactly one
`_local_stop_and_breakers()` recheck immediately after `_ssh_identity()` and
before `_marker()`. Offline tests inject each late desktop change at that
seam and assert the marker remains absent and no remote call occurs, under
normal and optimized test bytecode. This r5 repair preserves that recheck
and adds a second local STOP/breaker read after the new bounded remote
timeout probe, before the irreversible marker. The actual `-O` entrypoint
still holds before authority.

The Operator receipt found local `/usr/bin/timeout` is uutils coreutils
0.8.0 and the old `_ssh_phase()` incorrectly demanded local GNU even though
the fixed deadline executes on trn22. The repaired source performs a
read-only, 25-second-bounded probe of the **remote absolute**
`/usr/bin/timeout` through the same pinned strict SSH identity/options
before the marker and again before each host phase. It accepts only a
well-formed GNU coreutils version line, empty stderr, bounded stdout and
zero exit; a remote uutils, wrong version, warning, nonzero exit or timeout
is a HOLD. The fixed remote command remains
`/usr/bin/timeout --foreground --signal=TERM --kill-after=15s 320s ...`
and the local 350-second phase wait remains intact. A fresh local
STOP/breaker check runs after each probe and immediately before the marker
or phase SSH. No local fake timeout can satisfy this gate. No probe or
stage was run by this Implementer.

`REQUEST.HOLD.json` and the original `RUNBOOK.md` remain historical disabled
r3 artifacts with the former operator hash. Only this enabled request pins
the r5 operator. The retained r4 review and assignment authenticate only
their previous exact operator/request and cannot authorize this disabled
repair. A new independent final review and new Supervisor assignment are
required before any operational attempt.

The timing result has 20/20 read-only modeled trials within its one-second
freshness bound, but its worst double-label-scan duration was 980,357,752 ns:
only **19.642 ms** of margin. It never executed a real `O_EXCL`, payload
write, fsync, checkout or readback. The independent reviewer recommended
investigating a bounded pre-open-only retry; this **first** request does not
implement one. That old timing result measured the exact **unchanged**
`remote_incremental.py` final pre-open sequence; it did not bind or test the
new local `operator_once.py` hash or this r5 commit. Carry-forward of that
bounded timing evidence is a decision for the independent final reviewer,
not this Implementer. The final reviewer must explicitly weigh the narrow margin
against the consequence of consuming the one-shot marker on a safe
pre-open timing refusal. No timing result waives fresh identity, activity,
STOP, label, disk or inode checks.

## Before any Operator command

1. A different Verifier independently reviews this exact canonical request,
   the remote timeout probe and pinned SSH/deadline commands, STOP checks,
   adversarial normal/optimized offline tests, and the unchanged transfer
   source/bundle. It must weigh the source-root identity and 19.642-ms
   remote timing margin. Only then may it write the exact **r5**
   `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW_R5.PASS.json` required by
   `operator_once.py::_authority`; this Implementer has not created it.
2. The Supervisor independently verifies that review, active Desktop
   STOP/breaker state, exact request and source hashes, and absence of the
   one-attempt marker, then may issue the exact
   `SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT_R5.json` for this label
   only. No r5 assignment exists now; the old r4 assignment is not reusable.
3. The Operator performs a fresh, complete read-only local/host preflight:
   pinned SSH host identity; source e65 clean HEAD, expected science bytes,
   target/bundle absence, protected checkpoints/caches, host lock and
   Neuron/process quiescence, STOP/breakers, ≥13,177,973,911 root free
   bytes and ≥10,000 free inodes before material write. A failed gate is
   HOLD, not authority to clean or retry.

After those gates, the proposed **single-attempt** command from this
verified clean science-base worktree is:

```text
/usr/bin/python3 -B /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-remote-timeout-repair-r1-disabled/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
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

The offline tests run an isolated entrypoint copy with r5 authorities absent
and assert HOLD without SSH or a marker. This remains safe after a future
r5 PASS/assignment is added; the tests are not independent final review or
source-stage authority.
