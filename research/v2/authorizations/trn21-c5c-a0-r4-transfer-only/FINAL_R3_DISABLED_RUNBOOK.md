# trn21 r4/446b final-r3 transfer-only repair — DISABLED / HOLD

**No dispatch authority.** This is a separate Implementer-prepared, offline-only
successor to the independently held final-r2 candidate. Its frozen request has
`dispatch_authorized: false` and `source_transfer_authorized: false`; the r3
entry point rejects before its one-attempt marker or SSH. Neither an old static
PASS nor a future review/assignment file can enable this request. A different
Verifier must review the exact r3 bytes and a Supervisor must decide whether
to prepare a separately named enabled request and one-shot assignment. No host,
AWS, source transfer, marker, checkout, CPU G2, native run, evaluation,
cleanup, breaker reset or submission action was performed in preparing r3.

## Exact disabled subject

| Artifact | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R3_HOLD.json` | `3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e` |
| `operator_transfer_final_r3_once.py` | `6bf2aea104e9dbdefc69ee48a5bfe095e5b8f46c3d99bdd9ba9ba9aecf913a1a` |
| `stage_r4_transport_final_r3.py` | `0b676d2105010a6d5e6770dc507133ca958dc677ae21079cf673713a9894a083` |
| `stage_r4_remote_final_r3.py` | `3a738cff02b26824ea0be1fb50ee16b6cacce96941a9e2162506c7cf608ead32` |
| `test_operator_transfer_final_r3_once.py` | `d404c81111c38a298f897af1026b08b22dd96163400df60fde0723175969225d` |

The exact starting canonical commit is
`8c005717913bd65f76e0834c0f53a7f904c63e7f`. The prior r2 request,
wrapper, transport, remote, tests and runbook remain untouched; their hashes
are pinned in the r3 request/wrapper. The independent r2 HOLD is
`SUPERVISOR_FINAL_R2_HOLD.md`, SHA-256
`1f82c8d13c4abd8099a4a13a778628398088ba19dd6f8e7718ff0837b50e5639`.
The original static-reviewed disabled core/transport/remote and final-r1
HOLD/negative evidence likewise remain unchanged. The original disabled
static PASS is not an r3 review or transfer authority.

## Repair and evidence chain

R2's last clean science-label scan came before another potentially slow
activity gate. A label created during that gate could still reach target
`O_EXCL`. R3 first completes a fresh full writer probe and two unowned
STOP/process/checkpoint/shared-lock/Neuron/target gates. It then opens the
**existing** 446b launcher lock by descriptor-relative, no-follow,
read-only path. It never creates, writes, truncates or deletes the lock file.
An absent, replaced, unsafe or already-held lock raises HOLD before the
target is opened. The nonblocking exclusive `flock` is held across the final
post-acquisition activity gate, the final science-label scan, target
`O_EXCL|O_NOFOLLOW`, streaming, fsync, same-descriptor readback and write
receipt.

The frozen 446b launcher uses the same
`/home/ubuntu/.cache/trainium_freshstart/device0.lock` with nonblocking
exclusive `flock` and holds its descriptor through its guard. R3's acquired
lock gate expects `fuser` to report only the writer PID. On the local offline
psmisc check, an active `fuser` writes `path:` plus newline to stderr; r3
requires the exact SHA-256 of that expected output and rejects additional
diagnostics. The final clean `find` scan, with start/end UTC and empty-output
digest, occurs **after** every other potentially slow safety check and
immediately before target creation. No subsequent remote gate can reopen the
r2 label race. The write receipt duplicates and binds the final label and
lock evidence; the distinct readback and unconditional full postflight each
record new clean label scans and stable lock identity/idle-holder evidence.
The transport rejects missing, dirty, mismatched or time-reordered records.

The lock check deliberately fails closed if `.cache` or
`trainium_freshstart` is group/world-writable, or if the existing regular
lock is group/world-writable or lacks owner read/write permissions. This is
an offline trust rule, **not** evidence about trn21's live directory modes.
A separate host's `trainium_freshstart` directory was reported as mode 0775;
trn21's mode and group trust are unknown. If trn21 has the same mode, this
candidate will HOLD without a separately reviewed trust decision. No mode or
group membership was changed or checked on trn21.

There is no atomic exclusion of a manual, uncooperative same-UID label
creator: such an actor can create a label after or during the last scan, and
other manually created activity can race after its last observation. The
existing 446b launcher is cooperative through the shared lock; this does
not claim protection against arbitrary manual filesystem actors. A reviewer
must assess that residual limit before any later enablement.

The accepted e65 checkout's pinned `.git/HEAD` remains required: a
file-backed `.git` there is protected-source drift and HOLD. A file-backed
`.git` in the other old r1/r2 checkouts alone is safely inventoried at top
level without descent and does not force HOLD.

## Offline result and future gates

`python3 -B -m unittest -v` on all four transfer-package test modules
passed **41/41**; the r2 and r3 modules alone passed **15/15**.
The r3 negatives include a label appearing after the last slow activity
gate, STOP/process/Neuron activity at that gate, absent/unsafe/contended
lock, real local self-held `fuser` output, missing or reordered label/lock
receipt evidence, partial target retention/no same-path retry, and no
marker/SSH while disabled. These are Implementer-local tests, not an
independent PASS or live-host feasibility result.

A future separate request would still need an independent exact-file review,
Supervisor assignment, fresh strict trn21 host/boot/mount/source/old-asset
preflight, trusted existing-lock metadata, adequate root space/inodes, and
all original one-attempt source-transfer gates. It would authorize only the
single capsule transfer, never checkout, CPU G2, native training or cleanup.
