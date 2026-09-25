# Independent trn22 r4 continuity-r3 review — STATIC PASS, DISPATCH HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. The
subject is Implementer commit `76485b9623d84351433cc3d2cd350f830c624d1e`
on `codex/20260925-trn22-r4-final-r3-hold`. The seven r3 repair files are
byte-identical at canonical cherry-pick
`e21135cc84fa941784e6a52fa142dadd0973de3f`. I read the repository
contract in its required order, then the r3 runbook, prior independent
continuity static PASS and final-r2 HOLD, the changed code/tests, the exact
request/wrapper, and the archived read-only host inventory. Old chats and
receipts were evidence, not authorization.

**Verdict: STATIC PASS for this disabled repair only; DISPATCH HOLD.** The r3
request has `dispatch_authorized: false`; its wrapper checks exact bindings
and unconditionally raises DISPATCH HOLD before a one-attempt marker,
journal, source/SSH identity check, or SSH. The older final-r2 request and
wrapper retain their old pins and fail closed against the changed remote and
transport. The unchanged disabled core still refuses dispatch. This receipt
is **not** an independent final enablement PASS, Supervisor assignment,
current-host readiness finding, or permission to transfer.

## Exact reviewed bytes

| Artifact | SHA-256 |
| --- | --- |
| `CONTINUITY_R3_HOLD_RUNBOOK.md` | `0bb8d7c25282a4845de05bd924d0103a10f6ec54060dac6d252f0f19d6d8993b` |
| `TRANSFER_REQUEST.CONTINUITY_R3_HOLD.json` | `247c4eb25e9667276e83ecaba5cb570ef2d400464185b54be073e640dd773fa1` |
| `operator_transfer_continuity_r3_hold.py` | `6160246baba29e525545299bb28db9a4310873b4033926e72655dc382f7358a5` |
| `stage_r4_remote.py` | `e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086` |
| `stage_r4_transport.py` | `1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51` |
| Supplied `test_continuity_r3_hold.py` | `bcc91fd6367a2c48c479146a7a15f9e28cc3a199195a7ab6a866f4ea919f140d` |
| Updated `test_stage_r4_transfer.py` | `8ff48e102dddcb29e75b746d2f99572d3e747146cd7842b8667b914608477cbe` |
| New independent r3 adversarial tests | `4a950d2fb02f7bac0923542665894f2dff361f5c8d76d5eec5bf8caa0b07fe47` |

The prior `INDEPENDENT_CONTINUITY_REPAIR_STATIC_REVIEW.md` and
`INDEPENDENT_CONTINUITY_FINAL_R2_HOLD.md` remain unchanged at SHA-256
`669d99837e25ee634e4db418125b1b8bf13862a1907baad10ac213b5e89c7c57`
and `873bf131898e014adbb79a9cb6dd5cefd5f653897f8c80ae63b13e296f1e38f1`
respectively. The old final-r2 request/wrapper and disabled core remain
unchanged at SHA-256 `99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5`,
`14dac3c538d3d08e2876c1327e177f2a4c4696cfadb9884004725305383ccf23`,
and `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736`.
Neither historical PASS nor HOLD authorizes the r3 bytes or erases the final-r2
HOLD.

The frozen outside-Git capsule independently rehashed to
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`
(104,154,875 bytes, regular mode `0600`, one link). Pinned `known_hosts`
rehashes to `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`.
The private key was inspected by metadata only; no key contents were read.

## Independent findings and limits

The writer now runs a complete **second** live gate after its second
host/boot/mount identity and target-absence checks, immediately before the
sole descriptor-relative `O_CREAT|O_EXCL|O_NOFOLLOW` target creation. It
checks STOP paths, unreadable or matched processes, source-science label,
checkpoint and shared-lock holders, `/dev/neuron0` holder, and `neuron-ls`;
a final STOP/process scan follows the slower checks. The receipt records this
last full snapshot. The transport comparator rejects absent or busy final
evidence, not merely a clean initial prewrite snapshot. Deterministic local
tests introducing each busy signal during the intervening identity check
blocked creation. These are point-in-time checks, not a claim that activity
cannot begin after the last check.

The remote inventory now explicitly covers the old r1/r2/r3 `.git` and
`.git/HEAD` paths. A no-follow descriptor and inode check classifies only an
actual regular file-backed `.git` as making its nested `HEAD` absent; a
symlink, special file, mismatched metadata, or an arbitrary nondirectory
ancestor still fails. The transport validates that the nested classification
matches the inventoried `.git` metadata. Local fixtures reproduced the
archived host's r1 directory-backed `.git`, r2 148-byte file-backed `.git`,
and absent r3 form over the full 41-path old inventory, then passed the
actual probe validator. The archived Operator read-only snapshot is a
2026-09-25 04:18–04:21 UTC point-in-time observation, not a current/fresh
preflight and not evidence of an r3 host run.

The supplied r3 suite passed **7/7**, the two preserved independent tests
that exposed the final-r2 defects passed **2/2**, and the new independent r3
suite passed **4/4**: **13/13 targeted tests PASS**. The latter also tested
the full old-git inventory shape, strict exception scope, final-receipt
mutations for every safety signal, and disabled new/old/core wrappers with
marker, journal and SSH mocked as forbidden. `git diff --check` of the
Implementer commit and exact seven-file canonical byte-parity checks passed.
The older hash-pinned `test_stage_r4_transfer.py` suite passed **21/22**:
its sole failing assertion expected the old words `structured PASS`, while
the deliberately invalidated final-r2 binding now fails earlier with
`transfer request/code/binding drift`. This is the intended fail-closed
behavior. The full historical final-r2 independent suite passed **5/6**;
its only error likewise calls the old final-r2 request reader expecting
obsolete code hashes, which fails immediately on transport drift. Neither
old hash-pinned suite is an r3 enablement test or permission to relax a gate.

No SSH, host/AWS action, marker or journal creation, transfer, target write,
checkout, extraction, G2/native training, evaluation, cleanup, or submission
occurred in this review. No independent final-r3 PASS JSON or Supervisor
assignment was created. A separate exact-byte enabled request, independent
final review, distinct Supervisor one-shot transfer-only assignment, and
fresh host preflight remain required. Until then, **DISPATCH HOLD**.
