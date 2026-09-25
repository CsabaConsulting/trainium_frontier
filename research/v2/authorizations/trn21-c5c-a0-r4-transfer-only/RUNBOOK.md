# trn21 r4/446b capsule transfer-only implementation — DISABLED

This package implements the **transfer gate only** from the integrated
[`trn21 r4 source-stage HOLD plan`](../trn21-c5c-a0-r4-source-stage-hold/RUNBOOK.md).
It is an Implementer artifact, not a Verifier PASS, fresh Operator preflight,
Supervisor assignment, or authorization to contact trn21. The frozen
`TRANSFER_REQUEST.HOLD.json` has `dispatch_authorized: false`; the operator
module's command-line entry exits `DISPATCH HOLD` without opening SSH.
Changing that flag alone fails its pinned request hash. The independent
enablement-review and Supervisor-assignment files are intentionally absent.

## Frozen source and sole possible write

The only source is the existing private local one-commit capsule for detached
HEAD `446b657695f086f9b484fa601c9a3b61c2495912`, 104,154,875 bytes,
SHA-256 `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The only remote write named anywhere in this package is the *new, currently
unverified-absent* persistent-ext4 path
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz`.
It is never extracted, checked out, renamed, deleted, overwritten, or retried
by this package. The accepted e65 checkout at
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
and both checkpoints, six warm caches, and three earlier stage files are
read-only protected assets. The future clean r4 checkout, monitor, observer,
science outputs and label must remain absent. `/dev/shm` is not a target.

The prior independent capsule review has scope **local capsule and trn22
non-dispatchable plan only**. Its hash pin is evidence of source identity,
not a trn21 host-transfer approval. The earlier trn21 checkout postflight
is historical, not the required fresh transfer preflight.

## Reviewed mechanics awaiting separate authority

The proposed, still-disabled one-attempt wrapper has these fixed checks:

1. Read actual hash-pinned request/plan/capsule-review bytes and require a
   distinct structured independent trn21 PASS plus a separately issued,
   exact one-attempt Supervisor transfer-only assignment. Neither exists.
   Reject any local global-correctness or trn21 breaker and any canonical or
   worktree STOP. Rehash the owner-private local capsule and verify the
   literal trn21 ED25519 known-hosts entry and key *metadata* locally; never
   print or read private-key bytes.
2. Consume a **fixed**, owner-private, `O_EXCL|O_NOFOLLOW`, fsynced marker
   and a separate fixed exclusive fsynced raw-evidence journal before any
   possible SSH. An existing marker or journal is HOLD; neither is removed
   after failure. The marker records the actual request, review and assignment
   hashes. Each SSH invocation is separately configured with `-F /dev/null`,
   strict pinned host-key checking, literal trn21 endpoint, and no shell
   interpolation of user input. Raw argv, local SSH PID/birth, exit code,
   stdout/stderr bytes and digests are journaled.
3. In its own SSH process, a **fresh full read-only** preflight proves host
   `ip-172-31-34-85`, UID 1000, boot and mount namespace, ext4 root source
   `/dev/root` on `259:1`/device 66305 with the historical
   `/dev/nvme0n1p1` block alias reconciled to that same kernel device, safe
   no-symlink `/home/ubuntu` ancestry, all target absences, old-path catalog,
   exact protected SHA/inode/size identities, no STOP, process, Neuron, lock
   or checkpoint holder, and no r4 label. It requires at least
   **13,177,973,911** free root bytes and **10,000** free inodes.
4. The one writer SSH process receives the SHA-256 of the *stable host
   identity* from that separate preflight. Before target creation, it runs
   its **own full live probe**, rejects any boot/mount/parent or protected
   asset drift, then rechecks all future target absences, STOP, process,
   checkpoint holders, Neuron and shared lock; the fast target/STOP/process
   checks repeat after the holder commands, immediately before a
   descriptor-relative, no-follow `O_EXCL` mode-0600 create. It never writes
   through a symlink.
   File and parent are fsynced, and the same open descriptor is read back
   and checked against exact capsule bytes/SHA. A partial file is preserved
   and consumes the attempt. The postwrite root floor is **10 GiB**.
5. A **distinct** SSH process reopens the sole target read-only and proves
   identical bytes, SHA, inode, mode and host identity. Regardless of any
   earlier failure, the wrapper attempts a further distinct **full read-only
   postflight** and compares protected assets and old paths to preflight.
   Any gap, nonzero exit, uncertain evidence, or drift is HOLD. The only
   possible success status is transfer complete **pending independent raw
   result review**; it grants no checkout or training authority.

The wrapper itself still cannot run from this package. Any future enabled
package must be newly pinned and independently reviewed, then assigned for
exactly one attempt by the Supervisor. No same-path retry or clean-up is
part of that assignment. The separate checkout, CPU G2/source-identity
decision, native A0 run, evaluation, and submission remain later gates.

## Offline evidence and limits

`test_stage_r4_transfer.py` is synthetic and local only. It covers the
frozen HOLD, flag-flip/forged-review rejection, capsule and trn21 host-key
pins, exact SSH argv construction without execution, local STOP/breaker
failures, mount-alias and protected-asset drift, positive receipt comparison,
preflight-to-writer boot/mount drift, target/STOP/process appearing both in the full
writer probe, **after** that probe, and during holder checks before target
creation, partial-target
preservation/no retry, marker/journal exclusivity, and unconditional
postflight on a failed write. Its passing result is not independent review
or a live host observation. No host was contacted to prepare this package.

Pinned source-code chain in this draft:

| Artifact | SHA-256 |
| --- | --- |
| `stage_r4_remote.py` | `be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68` |
| `stage_r4_transport.py` | `fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647` |
| `TRANSFER_REQUEST.HOLD.json` | `5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b` |
| `operator_transfer_once.py` | `81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621` |

These hashes allow a separate Verifier to reproduce the exact draft; they
do not confer execution authority.
