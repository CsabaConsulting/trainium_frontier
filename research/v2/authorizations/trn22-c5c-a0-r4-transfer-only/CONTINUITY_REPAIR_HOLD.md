# trn22 r4 transfer: preflight-to-write continuity repair (HOLD)

This Implementer revision is **disabled**. It changes the remote writer, SSH
transport, core wrapper and final wrapper after the earlier independent static
PASS. That PASS applies only to the old bytes and is **not** an authorization
for these bytes. No SSH, host contact, transfer, checkout, G2, native run,
evaluation, cleanup, AWS action or submission was performed for this repair.
The independent HOLD receipt for the prior final candidate is
`INDEPENDENT_FINAL_ENABLEMENT_HOLD.md` at canonical `de7e29bb`, SHA-256
`5bc37139209b7a87ff9103902be0f7162f0e3afcef5ebe83f44f78ce5e576e96`.
It reports precisely the two repaired blockers below; its verdict remains
HOLD and cannot be treated as a review of this revised package.

## Frozen scope

- Source commit: `446b657695f086f9b484fa601c9a3b61c2495912` (not package HEAD).
- Capsule outside Git: `trainium-c5c-r4-446b-one-commit-20260925.tar.gz`,
  104,154,875 bytes, SHA-256
  `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
- Sole proposed target: `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`
  on trn22 persistent ext4. Exclusive mode-0600 creation, no overwrite,
  deletion, rename, cleanup or retry remains the only write path.
- Both `TRANSFER_REQUEST.HOLD.json` and
  `TRANSFER_REQUEST.ENABLEMENT_CANDIDATE.json` now have
  `dispatch_authorized: false`. The latter filename is historical; it is not
  an enabled request. Both CLI wrappers fail before marker, journal or SSH.

| Repaired local artifact | SHA-256 |
| --- | --- |
| Remote writer | `023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d` |
| Transport | `ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0` |
| Disabled core wrapper | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` |
| Disabled final wrapper | `b3ad1290f5e9cc6655b57380b269b9c2e4082391ea27d4f81545388b892beb1d` |
| Frozen HOLD request | `2b7249002ec301962153efdfc3b003454dff9c2ba6fe80a09b780e10adb8ab0c` |
| Disabled former enablement candidate | `833737bbe76a7b7acc287c247616527b199b5f515a46fcf8de0190595600ee57` |

## What was repaired

The old local comparator checked write-to-verify and preflight-to-postflight,
but did not compare preflight identity with the write process. A synthetic
pre/post boot A and write/verify boot B therefore passed. The old remote
writer also relied on the earlier probe's STOP/process results and did not
recheck them before creating the archive.

The repaired transport now requires a fresh validated preflight receipt for
the sole `write` command. It passes a canonical hash-bound token containing
the exact preflight receipt digest, time, process identity, frozen source and
target, and host/boot/mount/block-alias/parent-inode identity. The remote
writer rejects an absent, stale or mismatched token before creating anything.
It reruns live STOP, process, checkpoint-fuser, shared-lock, Neuron and label
checks, then checks identity plus STOP/process again immediately before its
`O_EXCL` create. The local wrapper compares preflight-to-write identity and
the writer's live-safety evidence before accepting a result; the distinct-SSH
readback and final read-only postflight still follow. A failure leaves any
partial target in place and consumes the one attempt; it never cleans/retries.

Local synthetic negative tests prove the specific boot-A/write-B/post-A and
post-probe host-STOP cases now HOLD. A post-probe matched-process case also
HOLDS before target creation. These tests do not establish host readiness or
grant dispatch.

## Required next gates (not completed by this Implementer)

1. A different Verifier must review the new exact code/request bytes and
   adversarial tests, including the preflight token, live safety, one-shot
   behavior and the mandatory postflight on failure. The earlier static
   receipt cannot be reused.
2. A fresh independent final request/review must bind the reviewed hashes and
   actual review file. The disabled request cannot be enabled by flipping one
   field or by placing a stale review/assignment file at a fixed path.
3. Only a distinct Supervisor may consider a narrowly scoped, one-attempt
   transfer-only assignment after fresh read-only host feasibility checks.
   Historical snapshots are not preflight evidence. Extraction, checkout, G2,
   native run and evaluation each remain outside this package.

The maintained local synthetic suite is:

`python -B -m unittest -v research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_stage_r4_transfer.py research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_operator_transfer_final_once.py`
