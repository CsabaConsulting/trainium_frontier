# Independent static review: trn22 r4 transfer-only draft

**Verdict: HOLD.** The frozen request is correctly non-dispatchable, and the
local transfer mechanics passed the checks below. The package must **not** be
enabled or used on trn22. Three authority/preflight defects are reproduced;
the old root-device alias also remains unresolved. This review is not a
transfer, checkout, G2, native-run, evaluation or submission approval.

Reviewed at 2026-09-25T03:10Z by a separate Verifier, against Implementer
commit `fcaf3780f0b54db6266231144d0ff3667b4620de` on
`origin/codex/20260925-trn22-r4-transfer-only` (parent `1c4aad53873381853b1f92f3b6cdfe5c090a9b69`). I did not author that
implementation and did not SSH, inspect key contents, contact either host,
transfer, extract, run G2, train, evaluate, alter AWS, clean up, or reset a
breaker. This receipt and independent offline tests are on an isolated review
branch; no scored source, evaluator or host file was changed.

## Exact subject and positive checks

| Frozen input | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.HOLD.json` | `3c583226a6dd1d87b7626f62c19771b96d217d680c3d2cf93bcaf9c3f311de39` |
| `operator_transfer_once.py` | `1889e87f765a9ed7bd8de6c59b6c52b152032a9e19942baf1f96236382cfa663` |
| `stage_r4_remote.py` | `d02dbf6d9efee6062a20fe620743460ab247240964e9266875a1599f95bcfd50` |
| `stage_r4_transport.py` | `515b27d6afefe35a1b85fd9bc3bb1418d0b487d32f421a75722b4931d49d75e3` |
| Implementer `test_stage_r4_transfer.py` | `50592d093075d025aa8eae91ac1a3a3b44193caebc2f6ee174505ef52839c566` |
| `RUNBOOK.md` | `361078d79f29810d0525940516545de64a1e9f25b88aca23fe56fc0342bd44d9` |
| Independent `independent_negative_tests.py` | `7c67a8c3e916eaccbf228990b91c18a006edcae754108c0cbe6e17d49edcd8a1` |

The outside-Git capsule is regular, mode `0600`, single-link, exactly
`104154875` bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
Its archived `.git/HEAD` and `.git/shallow` both identify frozen source
`446b657695f086f9b484fa601c9a3b61c2495912`, **not** package HEAD
`fcaf3780...`; archived `train.py` and `prepare.py` hashes are respectively
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The prior independent capsule/static receipt and point-in-time trn22
feasibility receipt were read, but neither substitutes for a fresh live
pre-transfer preflight.

The frozen request's `dispatch_authorized` is `false`; direct wrapper use
stops before SSH. Its fixed journal uses private `O_EXCL|O_NOFOLLOW` creation
and fsync before SSH, so a second attempt cannot reuse the path. The
remote writer uses descriptor-relative `O_EXCL|O_NOFOLLOW`, mode `0600`,
regular/single-link checks, file and parent fsync, exact stream SHA/byte
checks and same-FD readback. Local fixtures proved success and preservation
of a short partial file with no same-path retry. The intended wrapper records
literal SSH argv/hash, local PID/times/exit/timeout and complete stdout/stderr
bytes, hashes and Base64; remote receipts include PID/start ticks. It opens
separate SSH processes for write, read-only persistence verification and the
final read-only probe; a `finally` path attempts that final probe even after
a failed preflight. The transport validator inventories fixed old r1/r2/r3
source/cache/output roots, protected checkpoint hash/inode and six cache
identities, STOP paths, processes/Neuron/lock, disk/inodes and exact new-path
absence. These are **static/local** findings, not fresh host observations.

The Implementer suite passed 8/8. Nine independent, no-SSH tests passed 9/9,
including the exact capsule/HEAD/source hashes, HOLD, exclusive writer and
partial-file behavior, fixed journal, failed-preflight postflight attempt,
and the three bypasses below. `git diff --check` passed.

## Blocking findings

1. **SSH host identity is not fixed.** In `stage_r4_transport.py:94-104`,
   `build_ssh_argv()` sets `StrictHostKeyChecking=yes` but omits a fixed
   `UserKnownHostsFile`, `GlobalKnownHostsFile=/dev/null`, ED25519-only host-key
   algorithm, and `-F /dev/null` (or equivalent isolation from mutable user
   SSH config). The prior read-only Operator observation named an ED25519
   known-hosts fingerprint, but this argv does not bind to that entry. The
   independent test reproduced the missing options without opening SSH.
   Before any host contact, freeze and verify the exact known-hosts entry and
   fingerprint without printing key material; then review the literal argv.

2. **A local global STOP does not stop the wrapper.** In
   `operator_transfer_once.py:88-98`, `_check_controller_breakers()` reads only
   two breaker booleans. It never checks the controller's
   `research/agentic/STOP` or `research/v2/STOP`. An independent fixture with
   a present STOP and clear breakers returned success. Host STOPs are probed,
   but cannot replace the local stop gate. Add fail-closed local STOP checks
   before the journal/first SSH and again immediately before the write.

3. **The supposed independent review and Supervisor assignment are
   self-attested by the request.** In `operator_transfer_once.py:54-85`, the
   mutable request supplies both receipt hashes; any text containing `PASS`
   and not `HOLD` in its first 200 bytes is accepted as an independent review.
   The assignment has no exact enabled-request SHA or review SHA binding.
   An independent offline fixture with a fabricated `PASS` file and a
   self-authored assignment (neither containing an exact reviewed-request
   hash) passed `authorize()` after the request's HOLD bit was changed.
   Require a structured independent receipt bound to the exact final request
   and reviewed code, and a separately issued one-shot assignment binding
   those exact hashes; do not solve this by merely flipping the HOLD JSON.

The historical `/dev/nvme0n1p1` versus `/dev/root` root-source discrepancy
is **not yet reconciled**. The draft pins `/dev/root` and checks ext4 plus
parent-device equality against the reported major:minor, which is useful,
but it does not establish that the old name is the same underlying block
device. The exact new pre-transfer probe must record both aliases' read-only
device identities or otherwise obtain a separately reviewed reconciliation;
any mismatch remains HOLD. The 02:42:28 UTC feasibility snapshot is planning
evidence only and omitted raw SSH transport and a complete old-path catalog;
the draft does not inherit authority from it.

After a separate Implementer repairs these issues, freeze a **new** package,
repeat independent static/mutation review, then require fresh full trn22
preflight and a one-attempt transfer-only Supervisor assignment. A successful
transfer would still require raw-result review before any separate checkout.
There is no trn22 r4 host transfer, checkout, G2, native control, public BPB
result or authority from this receipt.
