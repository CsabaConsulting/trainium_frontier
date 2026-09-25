# trn21 r4/446b transfer-only final request — pending separate authority

**Current dispatch verdict: HOLD.** This is an Implementer-prepared, exact
transfer-only request. `dispatch_authorized: true` in the new request expresses
intent for one possible source transfer; it is not effective permission. The
future fixed-path final independent PASS and Supervisor assignment files do
not exist. `operator_transfer_final_once.py` checks both actual files before
creating its marker or starting SSH. No trn21 or AWS contact, transfer,
checkout, CPU G2, native run, evaluation, cleanup, breaker reset or
submission occurred while preparing this package.

## Exact subject and reviewed predecessor

| Artifact | SHA-256 |
| --- | --- |
| New `TRANSFER_REQUEST.FINAL_PENDING_REVIEW.json` | `04a6b215ac796927677a29aebdf414ffe0f361495cca8a45d2dec070bc60f136` |
| New `operator_transfer_final_once.py` | `4c02dd1891009effe00d22b01966943ae85041a8ccbc0006792f3b76859feec1` |
| New `stage_r4_transport_final.py` | `8c1e362fd586a00137016d7e0fad6f6ed758124cf7f66716a143895c91b512ba` |
| New `stage_r4_remote_final.py` | `9b73cd84a46af46e9ce7e8729e328878a7b29c16ac1556e9adeb93785fd9b4b8` |
| Reviewed disabled request | `5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b` |
| Reviewed disabled core | `81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621` |
| Reviewed disabled transport / remote | `fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647` / `be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68` |
| Actual independent disabled-package static PASS, not final authority | `3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3` |
| trn21 source-stage HOLD contract / runbook | `2f7dca0e1e07543e8b2ad7ea2c20241d6841b8ea3524e01c1c3f3028730b0219` / `e31b9df3b4b1b2ac0f0fef355d773ef980875c4a3841bc0355387af6af9627f5` |
| Existing local capsule review, source-only | `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |

The disabled implementation is canonical commit
`96f02153fc65e6236c84b3a76983ab91301d1da7`; its static review was
integrated at `3d5e66806c98e3a635c51b595c1cdcc0579eb866` from review
commit `9cdc545e92cf65bea08aeb912b9b21a8f30e07f5`. The actual static
receipt is hash-pinned and explicitly covers only the disabled files. The
new final transport/remote and this request need their **own** independent
final review; the old PASS does not approve them.

The source is the existing private, outside-Git one-commit capsule at
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`:
104,154,875 bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`,
detached source HEAD `446b657695f086f9b484fa601c9a3b61c2495912`.
The only possible host write is the new persistent-ext4 target
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz`
on pinned trn21 `ip-172-31-34-85`, UID 1000. The accepted e65 checkout,
two checkpoints, six warm caches and earlier stage files are protected.
The future clean r4 checkout, monitor, observer, output and science label
must remain absent. No `/dev/shm` target or second source is allowed.

## Final authority still missing

1. A different Verifier must inspect these exact new request/wrapper/final
   payload bytes, the disabled static PASS and the local offline evidence.
   Only a genuine PASS may be authored at the fixed path
   `INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json`, with the exact fields required
   by `_expected_final_review(...)`. The Implementer cannot create it.
2. Before assignment, a distinct Operator and Supervisor must obtain and
   assess a fresh full read-only trn21 feasibility/preflight: strict pinned
   ED25519 host identity, boot/mount and `/dev/root` ↔ `/dev/nvme0n1p1`
   `259:1` equivalence, safe ext4 ancestry, absent proposed targets and
   science label, no STOP/breaker/process/Neuron/lock/checkpoint holder,
   unchanged e65/checkpoints/caches/prior stage files, at least
   13,177,973,911 free root bytes and 10,000 free inodes. Historical host
   snapshots do not meet this live gate. The wrapper repeats the full
   read-only preflight in its own SSH session after assignment.
3. Only a separate Supervisor may issue the fixed-path
   `SUPERVISOR_FINAL_TRANSFER_ONLY_ASSIGNMENT.json` for the exact request,
   **actual** final review file SHA-256, wrapper, both final payload files,
   disabled static chain, source/host/target, fixed marker/journal and one
   attempt. `_expected_assignment(...)` gives the exact required fields.
   A role string or hash copied into the request is not this assignment.

The final wrapper reads both actual files through no-follow, owner-checked
opens and requires exact structured equality. If either is absent, malformed,
stale, or broader than transfer-only, it raises HOLD before marker or SSH.
The local capsule is then rehashed; the literal trn21 known-hosts SHA and
ED25519 fingerprint are checked without reading private-key bytes.

## One attempt and host-side continuity

After those future gates, the wrapper consumes the separate fixed
`trn21-c5c-a0-r4-transfer-final-r1-one-attempt.marker` with
`O_EXCL|O_NOFOLLOW`, mode `0600`, file and parent fsync. The fixed private
raw journal is also created exclusively and fsynced before SSH. Neither
is removed after failure. A second invocation or partially created host
target is HOLD; there is no retry, rename, replacement, deletion or cleanup.

Each SSH process uses `-F /dev/null`, strict pinned ED25519 host-key
checking, one literal endpoint and no caller-supplied remote command. The
first session performs the full live read-only probe. The writer's distinct
session receives that probe's stable boot/mount/parent identity digest and
runs its own full protected-asset, STOP, process, Neuron, lock, label,
disk/inode and target-absence probe before opening the target. It checks
STOP/process/target after the holder commands, then the **new final remote
payload repeats the complete target/STOP/process/checkpoint-holder/lock/
Neuron gate immediately before its descriptor-relative `O_EXCL` create**.
The final receipt carries both gates and their order; the new transport
requires both complete records. The accepted e65 checkout's `.git/HEAD` is
SHA/inode-pinned, so a file-backed `.git` **there** is protected-source drift
and causes an explicit non-directory-ancestor HOLD before a write. Other
old r1/r2 checkouts are inventoried at top level without descending into
their `.git`; a file-backed `.git` inside either is not itself a blocker.

The writer streams the exact capsule into a mode-0600 regular file on the
pinned root ext4, fsyncs file and parent, and hashes a same-descriptor
readback. A later distinct SSH process independently verifies bytes, SHA,
inode, mode, boot and mount. A further separate **full read-only postflight
is attempted even if the preflight or writer fails**; it checks the protected
assets and old paths against preflight and retains any partial target.
Every SSH argv, local PID/birth, return code, full stdout/stderr and digest
is saved in the raw journal. A clean result ends only at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`.

## Offline evidence and remaining gates

The supplied final tests exercise missing authority, exact hash chains,
exclusive marker behavior, literal SSH argv without execution, a second
late STOP/process/lock/Neuron/target gate, file-backed `.git` HOLD,
cross-session receipt validation and unconditional failure postflight.
They are Implementer-local tests, not independent verification or host
observation. The separate checkout-only request and result review, CPU G2
or independent source-identity carryforward, native safety monitors,
training, fresh public20M evaluation and submission are later gates.
