# Independent r2 static review: trn22 r4 transfer-only disabled draft

**Static verdict: PASS for the exact disabled package at `7aa72f9c`.**
**Dispatch verdict: HOLD.** This is a code/evidence-gate review, not an
`INDEPENDENT_ENABLEMENT_REVIEW.json`, host preflight, Supervisor assignment,
or permission to contact trn22. The frozen request has
`dispatch_authorized: false`, its SHA is hard-coded in the wrapper, and the
separate enablement review and one-shot assignment files do not exist. The
one-attempt journal does not exist either. Flipping the JSON flag fails its
pinned hash before SSH. A future enabled request necessarily changes the
wrapper, and the **new exact bytes** require another independent enablement
review, fresh read-only feasibility, and a separate Supervisor assignment.
This PASS cannot be inherited by that later package.

Reviewed at 2026-09-25T03:29Z in isolated branch
`codex/20260925-trn22-r4-transfer-r2-static-review` against canonical
Implementer package commit `7aa72f9c70f0dcbd1b83565b1b251f2f85a9c486`
(repair `e07781f2`). I did not author the repair. I read `AGENTS.md`,
`README.md`, `research/agentic/CURRENT_STATE.md`,
`research/experiment-protocol.md`, and `research/agentic/DECISION_LOG.md` in
the required order. No SSH connection, host/AWS action, transfer, checkout,
G2, native training, evaluation, key-content read, breaker reset, or
submission occurred in this review.

## Frozen subject and local results

| Reviewed file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.HOLD.json` | `9b36cead881506cca3c2bb95fe5bb0f03c1f0a773482e1a8bb1ce1f7ba2af85d` |
| `operator_transfer_once.py` | `b81744bba97c1fd0c16ce90b9571a964f347b19908614fae16e617396624439d` |
| `stage_r4_transport.py` | `58a2fc899470137fed940b0b96850d5ce304a432d24093bd6e49e38b1104e38b` |
| `stage_r4_remote.py` | `2920dfc78b1bf3941e9eaf500e41a44f7f5e29f9799273c04eb7ddc7de3d756e` |
| `RUNBOOK.md` | `bc9735f0d7eba44bac1b3cac432b4d4ec55e7d7ef7fa683f47805b0e751ab1ac` |
| Implementer `test_stage_r4_transfer.py` | `1243192d6b790afa978e671564199fdf017b411d1296c51128a6e01cdf394f8d` |
| This review's `independent_r2_static_tests.py` | `cb1ca393d5c9da0b972a51e11afc0ad68ddc59fcb6a03667e123825bda48e28a` |

The fixed outside-Git capsule remains a regular, single-link, mode-`0600`
file of 104,154,875 bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
An independent archive read found `.git/HEAD` and `.git/shallow` both at
frozen **source** commit `446b657695f086f9b484fa601c9a3b61c2495912`,
not package HEAD. The capsule's `train.py` SHA is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
its `prepare.py` SHA is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The pinned source-stage contract and independent static receipt rehashed to
`79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`
and `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e`.

The Implementer no-SSH suite passed **15/15**. My separate local adversarial
suite passed **9/9**. It tests the frozen HOLD and flag-flip, actual pinned
local ED25519 identity and literal config-isolated SSH argv, all four
controller/worktree STOP paths and both breakers, a valid one-way
request→review→assignment byte-hash chain and malformed mutations, root
alias `st_rdev` mismatch, exclusive one-attempt journal and preserved partial
writer target, successful fsynced writer/readback, raw SSH
argv/PID/exit/stdout/stderr archival with a fake
process, and the final read-only postflight attempt after failed preflight.
Both suites run with `PYTHONDONTWRITEBYTECODE=1 python3 -B`; the fake process
test does not spawn SSH. The obsolete pre-repair negative suite has fixtures
that intentionally expect the former defects; a legacy writer fixture also
lacks the new `block_aliases` field, so it is not a current gate.

## Prior HOLD findings: repaired in the disabled draft

1. **SSH identity/config.** Before building each SSH argv, the code checks
   private metadata of the existing key without reading it, the whole
   `known_hosts` file SHA, exactly one ED25519 entry for the fixed endpoint,
   and the expected fingerprint. The argv starts `ssh -F /dev/null -T` and
   pins `UserKnownHostsFile`, `GlobalKnownHostsFile=/dev/null`,
   `HostKeyAlgorithms=ssh-ed25519`, strict checking, no DNS/host-key updates,
   batch mode, and identities-only. Deliberate SHA/fingerprint drift fails
   locally. This is local identity proof only, not an SSH handshake.
2. **Local stop/breakers.** The wrapper checks the controller and isolated
   worktree `research/agentic/STOP` and `research/v2/STOP`, plus global and
   trn22 breakers, before the journal and again after the fresh preflight
   immediately before writing. All four STOP mutations failed closed in my
   fixture. A journal is fixed-path, private, `O_EXCL|O_NOFOLLOW`, and fsynced
   before the first SSH process; an existing path consumes the attempt.
3. **Authority chain.** The current request's exact SHA is frozen as HOLD.
   The validator rejects request self-attestation keys; it reads actual
   fixed-path structured review and assignment bytes. A future independent
   PASS would have to bind request/code/capsule/source/host/scope hashes, and
   the Supervisor assignment would have to bind that **actual review SHA**,
   request SHA, target, journal and one attempt. Forged, stale and unbound
   mutations fail locally. Role labels alone are not proof that the people
   were independent; the future Supervisor must verify authorship separately.
4. **Root-device alias.** The remote read-only probe requires ext4 mountinfo
   source `/dev/root` at `259:1`, `/dev/nvme0n1p1` resolving to a block node
   whose kernel `st_rdev` is `259:1`, `/dev/root` if present matching that
   same device, and `/sys/dev/block/259:1` resolving under `/sys/devices` to
   `nvme0n1p1`. A mismatched `st_rdev` fails. This is a **future live gate**;
   the historical `/dev/root` versus `/dev/nvme0n1p1` discrepancy has not
   been observed/reconciled on trn22 by this review.

The unchanged core transfer mechanics remain scoped to one new ext4 target:
remote descriptor-relative `O_EXCL|O_NOFOLLOW`, mode `0600`, SHA/byte and
same-FD readback, file+directory fsync, no extraction/deletion/overwrite or
same-path retry. The wrapper archives raw transport evidence, checks a fresh
full preflight including all old r1/r2/r3 roots and protected assets, verifies
the target from another SSH process, and attempts a distinct final full
read-only probe even on failure. Synthetic success/failure checks passed;
they cannot establish trn22's present disk, process, Neuron, STOP, old-path,
protected-checkpoint, mount or alias state.

**Next authority boundary:** a separate human-approved, exact enabled
transfer-only request/revised wrapper and independent enablement review,
fresh trn22 read-only preflight, and Supervisor one-shot assignment are all
still required. Even a successful transfer would end at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`; checkout, G2,
training and evaluation each remain unauthorized here.
