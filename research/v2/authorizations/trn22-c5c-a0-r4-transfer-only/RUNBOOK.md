# trn22 c5c A0 r4: exact-source transfer-only implementation (HOLD)

This is an **Implementer draft**, not dispatch authority. No SSH/AWS/host
command, source transfer, checkout, G2, native training, evaluation, monitor
operation, cleanup, breaker reset, or submission was performed while preparing
it. `TRANSFER_REQUEST.HOLD.json` has `dispatch_authorized: false`, no
independent implementation-review hash, and no Supervisor-assignment hash.
Running either local Python module directly exits with `DISPATCH HOLD`.

## Immutable subject and one new target

- Frozen training source commit: `446b657695f086f9b484fa601c9a3b61c2495912`;
  this is **not** the source-stage package's own HEAD.
- Independently static-PASS source capsule: outside Git at
  `/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`,
  104,154,875 bytes, SHA-256
  `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`,
  mode `0600`, one link. Its source-stage contract SHA-256 is
  `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`
  and independent static receipt SHA-256 is
  `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e`.
- Sole proposed host write: new
  `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`
  on trn22 (`ip-172-31-32-253`, UID 1000) under persistent ext4, created
  `O_EXCL|O_NOFOLLOW`, mode `0600`, regular file, one link. Never use
  `/dev/shm`, overwrite, unlink, rename, resume, retry, or sweep it. A failed
  write consumes this exact path and one-attempt assignment; leave any partial
  target in place for read-only diagnosis and request a *new* reviewed path.

The read-only feasibility snapshot at commit `83047a14` (02:42:28 UTC) is
historical planning evidence only. It observed ext4 mount source `/dev/root`,
major:minor `259:1`, device `66305`. The earlier source-stage contract
describes `/dev/nvme0n1p1` as historical root source; this alias discrepancy
is surfaced, not silently reconciled. The proposed remote writer pins the
fresh-observed `/dev/root` and major:minor is checked against the parent
device. Any fresh mismatch is HOLD pending a new review. That feasibility
snapshot omitted raw SSH transport metadata and a full old r1/r2/r3 path
catalog; neither omission is inherited by this draft.

## Disabled future Operator sequence

Only after a separate independent review of **these exact frozen bytes**, a
fresh pre-transfer host feasibility review, and a narrow Supervisor
`TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT` assignment may a new enablement
request be authored and independently checked. Do not flip the HOLD flag in
place or treat this Implementer draft as the Operator assignment. The actual
future Operator must be a different role/person than this Implementer.

The guarded wrapper is designed to:

1. Rehash the fixed local capsule, source-stage contract/receipt, remote code,
   transport code, its own code, the fixed-path independent PASS receipt and
   the fixed-path one-shot Supervisor assignment. Check current controller
   global/trn22 breakers. The independent review and assignment must be
   **read and rehashed**, never accepted merely because a SHA-looking string
   appears in a request.
2. Create a fixed private journal **with `O_EXCL` and fsync before the first
   SSH process**. It cannot be pointed at a fresh filename per retry. Archive
   each exact SSH argv and argv SHA, local SSH PID, UTC start/end, exit code,
   timeout flag, full stdout/stderr as Base64 with byte counts and hashes,
   plus each parsed remote receipt. The journal stays outside Git and must
   never be committed or printed if it could contain credentials.
3. Run a fresh read-only `probe` in one SSH process. It records host/UID/boot,
   mount namespace, ext4 mount ID/source/major:minor, no-follow `/`, `/home`,
   `/home/ubuntu` descriptor identities, root free bytes/inodes, all exact new
   target paths, global/old/new STOPs, protected checkpoint SHA/inode and six
   cache identities, checkpoint/lock/Neuron holders, `neuron-ls -j -a`,
   process and science-label scans, and **all fixed old r1/r2/r3 source,
   cache and output roots** plus dynamic matching old top-level roots.
   Pre-transfer floor: 13,177,973,911 free bytes and 10,000 free inodes.
   The source-stage historical snapshot cannot substitute for this probe.
4. If and only if the fresh probe passes, stream the fixed capsule to the
   fixed remote writer exactly once. It hashes the incoming 104,154,875 bytes,
   fsyncs file and parent, then rereads through the same descriptor and
   rechecks target inode/owner/mode/link count. There is no extraction.
5. Re-open the target **in a different SSH process** for a full read-only
   104,154,875-byte SHA-256 verification. Require exact host/boot/mount/parent
   continuity, target device/inode/mode and time order across receipts.
6. In a `finally` path, run another distinct read-only full `probe` even if
   preflight, transfer, verification or parsing failed. Archive its raw SSH
   transport and compare old inventory, protected assets, STOPs, activity,
   mount identity, and the target. Post-stage floor: 10,737,418,240 bytes.
   If postflight itself fails, retain the journal and mark HOLD; do not retry.

The wrapper's **successful transfer terminal state is still only**
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`. The source archive
must not be extracted/checked out until a separate result Verifier authenticates
the raw transports and a distinct checkout request, review, and assignment
exist. No G2/native/eval is authorized by this package.

## Local checks performed while drafting

`PYTHONDONTWRITEBYTECODE=1 python3 -B
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/test_stage_r4_transfer.py
-v` runs synthetic no-SSH checks for the frozen HOLD, literal SSH argv,
capsule SHA/bytes, fresh-gate rejection, distinct readback process, fixed
journal O_EXCL, and failure-path read-only postflight. These are **not** host
preflight, transfer evidence, a code review, or source-checkout authority.
