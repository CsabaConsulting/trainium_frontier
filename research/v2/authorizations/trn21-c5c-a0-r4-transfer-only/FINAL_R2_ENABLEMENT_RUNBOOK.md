# trn21 r4/446b transfer-only final-r2 successor — HOLD pending new authority

**Current dispatch verdict: HOLD.** This is a separate Implementer-prepared
successor to the final-r1 candidate held for a late science-label race.
`dispatch_authorized: true` in the r2 request expresses one-attempt
transfer-only intent, not effective permission. The fixed r2 independent
final PASS and r2 Supervisor assignment files do not exist. The r2 wrapper
rejects before a marker or SSH without both exact actual files. Preparing
and testing r2 contacted no host or AWS and made no transfer, checkout,
CPU G2, native run, evaluation, cleanup, breaker reset or submission.

## Exact r2 subject and held predecessor

| Artifact | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R2_PENDING_REVIEW.json` | `097b4e904c3d4cb0584f8973843110928388649f35f31efcd481e1863780e31e` |
| `operator_transfer_final_r2_once.py` | `494180db56ab0bbdf34e68a78b8d25fc6a240c8a02b8ca76fcb80a06bb906729` |
| `stage_r4_transport_final_r2.py` | `d7d3b279929ae9cb764033178b5b48fd5aa8ded77e8e9111a6e1c148ddb03b58` |
| `stage_r4_remote_final_r2.py` | `6b1cfe6b0b0d178a48bc881c389a2b2c3949736d7e9457c2aa4fa840581803c7` |
| `test_operator_transfer_final_r2_once.py` | `7f88d74d08d28669b28ba5e21b4b0d7dc3d0ee8ec5238ad70aef53e75b84986c` |
| Held final-r1 request / wrapper | `04a6b215ac796927677a29aebdf414ffe0f361495cca8a45d2dec070bc60f136` / `4c02dd1891009effe00d22b01966943ae85041a8ccbc0006792f3b76859feec1` |
| Held final-r1 transport / remote | `8c1e362fd586a00137016d7e0fad6f6ed758124cf7f66716a143895c91b512ba` / `9b73cd84a46af46e9ce7e8729e328878a7b29c16ac1556e9adeb93785fd9b4b8` |
| Independent final-r1 HOLD / immutable negative tests | `368c39aabd5437a0b2937773de3ad7aee4548e93a7255aa1a5012f792bbc49d0` / `ec73e4a510d5b4010d92804b98285a128f967a01623063395e3c00f26cff59d4` |
| Reviewed disabled request / core | `5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b` / `81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621` |
| Reviewed disabled transport / remote | `fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647` / `be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68` |
| Actual disabled-package static PASS, not r2 authority | `3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3` |
| Source-stage HOLD contract / runbook | `2f7dca0e1e07543e8b2ad7ea2c20241d6841b8ea3524e01c1c3f3028730b0219` / `e31b9df3b4b1b2ac0f0fef355d773ef980875c4a3841bc0355387af6af9627f5` |
| Existing local capsule review, source-only | `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |

The disabled implementation is canonical commit
`96f02153fc65e6236c84b3a76983ab91301d1da7`; the independent disabled
static PASS is canonical `3d5e66806c98e3a635c51b595c1cdcc0579eb866`.
Final-r1 candidate source `2cda547669059f49b173a08e273bff7f81bd734c`
was integrated at `0a5fc72cb77161f902feeee2620d69cb93740c05`;
its independent HOLD is canonical
`25fb21d9002e063bf3750e2d2e0bb34c93959f28`. The old PASS never
reviewed r1 or r2, and the r1 HOLD remains in force for r1.

The source remains the existing private outside-Git one-commit capsule
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`,
104,154,875 bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`,
detached source HEAD `446b657695f086f9b484fa601c9a3b61c2495912`.
The sole possible host write is the new persistent-ext4 target
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz`
on pinned trn21 `ip-172-31-34-85`, UID 1000. No old file, accepted e65
checkout, checkpoint or warm cache may be overwritten, renamed, deleted or
cleaned. The clean r4 checkout and science output remain future work.

## The r2 late-label repair

The r1 full writer probe scanned the science label, but a label appearing
after it could pass both subsequent activity gates and reach `O_EXCL`.
R2 retains the initial complete gate and runs a second complete activity
gate, then a fresh `find` science-label scan. Any nonzero exit, match,
truncation, stderr or nonempty stdout digest raises HOLD before target
creation. Because the scan can take time, r2 runs a third complete
STOP/process/checkpoint-holder/shared-lock/Neuron/target gate after it.
Only a descriptor-relative `O_EXCL|O_NOFOLLOW` target open follows.

The writer receipt embeds both the pre-label and final post-label activity
records, plus the label scan result and scan start/completion UTC. The r2
transport requires a complete clean label record with the SHA-256 of empty
stdout, validates UTC order from first gate through final gate to write,
and reapplies the disabled full activity validator to the pre-label and
post-label records. A missing or dirty label record now fails the
cross-session continuity gate. There is still no theoretical atomic
snapshot of an independently changing host; all checks fail closed on
observed activity and the final gate is immediately before `O_EXCL`.

The accepted e65 checkout's `.git/HEAD` is SHA/inode-pinned: a file-backed
`.git` **there** is protected-source drift and explicitly HOLDs. Other
old r1/r2 checkouts are inventoried at top level without descending into
`.git`, so a file-backed `.git` in either alone does not block.

## Still-missing authority and one attempt

1. A different Verifier must independently review the exact r2 bytes,
   offline tests and held-r1 evidence. Only an actual PASS with exact fields
   from `_expected_final_review(...)` at
   `INDEPENDENT_FINAL_R2_ENABLEMENT_REVIEW.json` can satisfy this gate.
   The Implementer has not authored it.
2. A distinct Operator and Supervisor must assess a fresh read-only trn21
   feasibility snapshot: strict pinned ED25519 identity; boot, root ext4
   mount/source/alias and parent continuity; absent targets/STOP/label and
   local breaker; idle processes, Neuron, lock and checkpoint holders;
   unchanged protected e65/checkpoints/caches/old stage files; and at least
   13,177,973,911 root free bytes and 10,000 free inodes. Historical
   snapshots cannot substitute for this live gate.
3. Only a separate Supervisor may issue
   `SUPERVISOR_FINAL_R2_TRANSFER_ONLY_ASSIGNMENT.json` with exact fields
   from `_expected_assignment(...)`, binding the actual r2 review file
   digest, request, wrapper, payload, source, host, target, separate marker
   and fixed journal. A role string or prior assignment cannot satisfy it.

The wrapper reads both future files with no-follow, owner-checked opens and
requires exact structured equality before marker or SSH. It rehashes the
local capsule and checks the literal trn21 known-hosts SHA and fingerprint
without reading private-key bytes. Once authorized, the separate
`trn21-c5c-a0-r4-transfer-final-r2-one-attempt.marker` is consumed with
`O_EXCL|O_NOFOLLOW`, mode 0600, file and parent fsync. The fixed raw
journal is separately exclusive and fsynced. Failed attempts retain both;
there is no same-path retry, replacement or cleanup. The preflight, writer,
independent readback and unconditional separate full read-only postflight
remain as in r1. Each SSH argv, PID/birth, return code, full stdout/stderr
and digest is saved. The success state ends at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`, not training.

Offline `python3 -B -m unittest -v test_operator_transfer_final_r2_once.py`
passed 7/7, including the preserved independent r1 negative suite 3/3
against r2 module aliases. Those formerly failing tests now reject both
a post-probe label before `O_EXCL` and a final receipt lacking label
evidence. Additional r2 tests reject a label introduced after the second
gate, every STOP/process/checkpoint/lock/Neuron/target activity class
arising during the label scan, dirty/missing/reordered
receipt evidence, and missing future authority before marker/SSH.
These are Implementer-local tests, not independent verification.
