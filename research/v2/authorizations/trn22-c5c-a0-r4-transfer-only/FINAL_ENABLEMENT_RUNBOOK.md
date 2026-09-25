# Historical b80c final request (SUPERSEDED; current package HOLD)

**Do not use the procedure or hashes below for dispatch.** The independent
final Verifier put this candidate on HOLD at canonical `de7e29bb` because
preflight-to-write identity and live host STOP/process gates were missing.
The repaired draft is documented in `CONTINUITY_REPAIR_HOLD.md`; its frozen
request now says `dispatch_authorized: false`, and new code hashes invalidate
the prior static PASS. Everything below describes the superseded b80c
candidate as historical evidence only.

**Current dispatch verdict: HOLD.** This is an Implementer-prepared request,
not an Operator assignment. No SSH, host/AWS contact, source transfer, checkout,
G2, native training, evaluation, monitor operation, cleanup, breaker reset or
submission occurred while preparing it. No final independent review JSON or
Supervisor assignment JSON exists at the fixed paths. The request's
`dispatch_authorized: true` records **transfer-only request intent**, not
effective permission: `operator_transfer_final_once.py` cannot reach its
one-shot marker, journal or SSH until it authenticates both later files.

## Exact frozen subject

| Item | Exact value |
| --- | --- |
| Frozen source commit | `446b657695f086f9b484fa601c9a3b61c2495912` (not package HEAD) |
| Capsule | outside Git, `/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`; 104,154,875 bytes, SHA-256 `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| Sole proposed host write | `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz` on `ip-172-31-32-253`, UID 1000, persistent ext4 |
| Fixed local one-shot marker | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker` |
| Fixed local raw journal | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl` |
| Enabled request SHA-256 | `3cd3c091d782dbaa535f67c399041900f9ceaf29b74ee046c6472bd3f44919f3` |
| Final wrapper SHA-256 | `ccdc7175e68f0e27eeff1dab71f17d8c49f4a9089ee88d75e73b9fcc29ffce7d` |
| Reused reviewed core wrapper SHA-256 | `b81744bba97c1fd0c16ce90b9571a964f347b19908614fae16e617396624439d` |
| Reused reviewed transport/remote SHA-256 | `58a2fc899470137fed940b0b96850d5ce304a432d24093bd6e49e38b1104e38b` / `2920dfc78b1bf3941e9eaf500e41a44f7f5e29f9799273c04eb7ddc7de3d756e` |
| Prior independent static receipt SHA-256 | `7dcfc65cde4dd29a409e8541ace24ada72465b15964d89af5852a5fd89f896db` |

The prior receipt is **static PASS for the disabled core only** and dispatch
HOLD. It cannot approve this new request/wrapper. The final wrapper rehashes
the *actual files* for that receipt, the reviewed core code and the pinned
source-stage contract/review before reading either future authority file.
Its request is pinned by exact SHA in the wrapper; no request-provided PASS or
assignment SHA is accepted.

## Separate authority chain, still missing

1. A different Verifier reviews this exact enabled request, wrapper, tests,
   reused code and fixed source/target/marker. The Verifier authors
   `INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json` at the fixed path with the
   exact structured fields returned by `_expected_final_review(...)`,
   including `verdict: PASS` and the **actual final wrapper/request hashes**.
   A narrative independent receipt should record the reviewer identity,
   commit, mutation tests, limits, and whether the file was written by that
   reviewer. This prior Implementer has no authority to author it.
2. A fresh read-only trn22 feasibility/preflight review must verify host,
   fixed ED25519 key, boot/mount/parent, `/dev/root` and
   `/dev/nvme0n1p1` kernel-device equivalence, all old r1/r2/r3 paths,
   absent new paths, STOP/breakers, protected checkpoint/six caches,
   process/Neuron/lock quiescence and disk/inodes. The 02:42:28 UTC snapshot
   at `83047a14` is historical only and does not grant this gate. The live
   wrapper performs its own full fresh preflight again before the write.
3. Only then may a distinct Supervisor verify the independent authorship and
   issue the fixed-path `SUPERVISOR_FINAL_TRANSFER_ASSIGNMENT.json`, with
   the exact fields from `_expected_assignment(...)`. It must bind the
   **actual review file SHA-256**, exact request and both wrapper/code hashes,
   source, host, sole target, marker, journal and one attempt. JSON role labels
   alone cannot prove human/agent independence; this is a Supervisor duty.

Until both fixed files exist and pass exact structured comparison, the
wrapper raises HOLD **before** touching the marker or starting SSH. Tests
that create synthetic review/assignment bytes are local fixtures, not actual
authority. There is no final review or assignment in this package.

## One attempt and failure behavior

If all future authority gates pass, the wrapper first rehashes the capsule,
checks local global/trn22 breakers and controller/worktree STOP files, then
durably creates the **fixed** marker with `O_EXCL|O_NOFOLLOW`, mode `0600`,
and fsync. An existing marker, including one left by a failed attempt, is
a terminal HOLD for this assignment. It never creates a new marker name,
unlinks, overwrites or resumes the old one. A second fixed `O_EXCL` journal
must be created and fsynced before any SSH process. A journal failure after
marker creation still consumes the assignment; neither file is cleaned up.

The unchanged reviewed core makes a fresh read-only probe, checks exact
absent target and all safety gates, reauthenticates both actual authority
files and the marker plus local STOP/breakers, and only then streams the one
capsule to the fixed target. The remote writer exclusively creates a
regular/single-link mode-0600 file on ext4, hashes exact bytes, fsyncs file
and parent, and rereads the same descriptor. A distinct SSH process rehashes
the completed file. A final distinct read-only full probe is attempted even
if preflight or writing fails. Raw SSH argv, PID, exit, complete stdout and
stderr, hashes and receipts remain in the fixed private journal. A short or
partial remote file is preserved; the same path is never retried, replaced,
renamed or deleted. Even a clean transfer ends only at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`; a separate result
review and distinct checkout authorization must precede extraction. There is
no G2/native/evaluation/submission path here.

## Local-only tests

The supplied `test_operator_transfer_final_once.py` exercises exact pins,
missing-review no-dispatch, request drift, byte-bound review/assignment,
exclusive private marker, and prewrite-STOP failure/postflight using fakes.
The prior `test_stage_r4_transfer.py` and independent
`independent_r2_static_tests.py` cover the unchanged core. All tests are
offline; they do not contact a host or read private-key contents. An
independent final Verifier must repeat and extend these tests before any
assignment.
