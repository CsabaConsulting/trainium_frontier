# trn22 r4 transfer continuity-r3: DISPATCH HOLD

This is an Implementer repair of the two reproducible findings in
`INDEPENDENT_CONTINUITY_FINAL_R2_HOLD.md` (SHA-256
`873bf131898e014adbb79a9cb6dd5cefd5f653897f8c80ae63b13e296f1e38f1`).
The prior disabled continuity static PASS is **historical evidence only**; it
does not approve the changed bytes below. The previous final-r2 request and
wrapper remain unchanged, retain their old pins, and fail closed against the
new transport/remote bytes. No r3 review or Supervisor assignment exists.

## Frozen scope and exact repaired bytes

| Item | SHA-256 |
| --- | --- |
| Frozen 446b capsule, outside Git, 104,154,875 bytes | `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| `stage_r4_remote.py` | `e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086` |
| `stage_r4_transport.py` | `1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51` |
| Unchanged disabled `operator_transfer_once.py` | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` |
| New `TRANSFER_REQUEST.CONTINUITY_R3_HOLD.json` | `247c4eb25e9667276e83ecaba5cb570ef2d400464185b54be073e640dd773fa1` |
| New `operator_transfer_continuity_r3_hold.py` | `6160246baba29e525545299bb28db9a4310873b4033926e72655dc382f7358a5` |

The fixed target remains
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`.
The fixed controller marker and journal remain the final-r1 one-attempt paths
already pinned in the r3 request. No new target, cleanup, overwrite, rename,
retry, checkout, G2, native run, evaluation, or submission is authorized.

## Repair behavior

- The writer now repeats **all** live gates after the second host/boot/mount
  identity check and immediately before the sole `O_EXCL` creation: old
  checkpoint holder, shared lock, label, `/dev/neuron0` holder, `neuron-ls`,
  STOP and process scan. The slower label scan precedes the quick holder and
  Neuron checks; a final STOP/process check is last. The write receipt carries
  the complete second live snapshot, and the controller rejects an absent or
  busy final snapshot. These are best-effort point-in-time checks, not an
  assertion that hardware activity cannot start after the check.
- Old r1/r2/r3 `.git` paths are explicitly inventoried. A regular file-backed
  `.git` is classified with no-follow descriptor and inode checks; its nested
  `.git/HEAD` is recorded absent *because of that exact file*, not because an
  arbitrary ancestor failed. A symlink, special file, unexpected regular
  ancestor, or mismatched `.git` metadata fails the preflight.

## Current gate and next ownership

`TRANSFER_REQUEST.CONTINUITY_R3_HOLD.json` has `dispatch_authorized: false`.
`operator_transfer_continuity_r3_hold.py` unconditionally raises DISPATCH HOLD
and has no host action path. This package is for an **independent Verifier** to
review the new remote/transport/request/wrapper bytes and local adversarial
tests. Only after a distinct static review should a separate, hash-bound
enablement request be prepared for independent final review. A Supervisor
one-shot transfer-only assignment and fresh host preflight would still be
required after that. No old PASS or self-attestation substitutes for these
gates.

Local-only verification (no SSH or key contents):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_continuity_r3_hold.py
```

The two original independent failure tests in
`independent_continuity_final_r2_negative_tests.py` are preserved unchanged
and now pass individually. The remaining historical r2 tests must not be
treated as enablement tests because their old exact source hashes are
intentionally invalidated by this repair.
