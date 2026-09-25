# trn21 r4/446b final-r4 private-lock successor — DISABLED / HOLD

**No dispatch authority.** This is an additive, Implementer-prepared successor
to the independently static-PASSed *disabled* final-r3 package. The exact r4
request has `dispatch_authorized: false` and
`source_transfer_authorized: false`. Its entry point rejects before creating
the separate r4 one-attempt marker, opening the journal, or starting SSH.
The r1/r2 enablement HOLDs and the r3 disabled-only scope remain in force.
No host or AWS action, source transfer, checkout, training, evaluation,
cleanup, breaker reset, or submission occurred while preparing r4.

## Exact disabled subject

| New file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R4_HOLD.json` | `58390b53ae09c666b705d153355baa12dd6392d8e3ea550bd44451fa41ab09e6` |
| `operator_transfer_final_r4_once.py` | `e36742d05bc517e4da0357b47144c329ec666cf197f8673f1b5f9570928bbf04` |
| `stage_r4_transport_final_r4.py` | `8907b6fe1e9f775cb1b14bbfe0be867be54fbe071098bee507762c54146e405f` |
| `stage_r4_remote_final_r4.py` | `ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b` |
| `test_operator_transfer_final_r4_once.py` | `802481a6c303a4c1ce3aadd9ef92656a80b584e066d8cca5bbae5074ae4e0563` |

Starting canonical commit: `bdbf7050fc6a2d26e27eccbaf3578446fd1f9620`.
The r3 request, wrapper, transport, remote, tests, runbook and independent
disabled static PASS are byte-identical to their frozen SHA-256 values in the
r4 request. The r4 wrapper verifies them before considering any authority
file. The r3 independent review SHA-256 is
`57e50f32bb13b29a4a9b7e41f88c09c09d26dc0eb29efe38a3e81538d53d6000`;
that PASS is not r4 review or transfer authority.

The two trn21 read-only host observations are pinned by the r4 request and
wrapper: the [mode/flock receipt](../trn21-c5c-a0-r4-source-feasibility/OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T053029Z.md)
is SHA-256 `0557d5b940808fa3549123fe881c92f6c611d08f1a66a1401b22eeb344781f3a`
with raw SHA-256 `687b4d7b57c4de041adad37352a385e15ead7cfba0b83c4d3a3df5bd12058d14`.
The later [ACL/group receipt](../trn21-c5c-a0-r4-source-feasibility/OPERATOR_GROUP_ACL_READONLY_20260925T053728Z.md)
is SHA-256 `065636a908c2b8b5aa8feb99e952a14a2dc63c2803140228b5d85cd496e515c4`
with raw SHA-256 `484ea4e76b6a59d5e5ec6757db6936b0bc3e7fc3ad89f69aa304a84da483443c`.
Both are point-in-time, read-only evidence and neither authorizes transfer.

## Narrow lock policy

The first trn21 receipt found the existing lock path, all on ext4 device
`66305`, owned by UID:GID `1000:1000`: `.cache` mode `0700`, inode `295736`,
link count 6; `trainium_freshstart` mode `0775`, inode `596822`, link count 3;
and the zero-byte regular `device0.lock` mode `0664`, inode `596823`, link
count 1. The later receipt found no POSIX access/default ACL xattr on the
directories and no access ACL on the lock file. Its bounded NSS checks found
no other listed GID-1000 member; its 30-entry mountinfo scan in the same
namespace found no explicit relevant bind path.

R4 retains r3's non-group-writable acceptance rule. Its only added
group-writable case requires **all** observed owner, group, device, inode,
link-count, size and mode values above. It opens `.cache`, its child and the
existing lock through descriptor-relative `O_NOFOLLOW` paths, opens the lock
`O_RDONLY|O_NONBLOCK`, and rechecks named/opened identity. It checks
applicable POSIX ACL xattrs on those opened inodes; only `ENODATA` is accepted.
Any ACL entry, unsupported/denied ACL query, changed metadata, symlink,
hardlink, FIFO or other nonregular lock causes HOLD. It parses its complete
current mount namespace's mountinfo, requires the pinned root ext4 binding,
and rejects a mount into `.cache` or a same-device bind root at/below the
lock directory. A malformed or unavailable mount view also causes HOLD.
The lock identity, ACL result and mount proof are bound into the preflight,
writer, distinct readback and full postflight receipts; a changed identity
fails continuity.

The frozen 446b launcher opens this exact `device0.lock` and takes
nonblocking exclusive `flock`. R4 retains that compatible lock mode and
holds its descriptor over the post-acquisition STOP/process/checkpoint/
Neuron/target checks, the **last broad science-label scan immediately before
`O_EXCL`**, target streaming, fsync, same-descriptor readback and write
receipt construction. It then closes the descriptor. The distinct readback
and unconditional full read-only postflight still require clean label and
idle-lock evidence. A failed or partial target remains in place; same-path
overwrite, rename, delete or retry is not authorized.

This exception is scoped to ordinary DAC traversal through the observed
private `.cache` path and cooperation by the frozen launcher. The read-only
receipt does not exclude a privileged actor, another process with UID 1000,
an already open directory descriptor, or an alias in another mount namespace.
It also cannot make the final STOP/process/label scan atomic against an
uncooperative writer. The independent Verifier must judge these stated
limits before any later, separately named enabled request; a future fresh
preflight must repeat the live checks. Unknown ACL or observed relevant
alias remains a hard HOLD, not an inferred permission to dispatch.

## Offline checks and remaining gates

`python3 -B -m unittest -v` in this transfer package passed **51/51** local
tests. The ten new r4 tests cover disabled entry-point order, the exact
private case and launcher flock compatibility, unsafe mode/device/inode/GID/link/
symlink/FIFO substitutions, a raced FIFO requiring `O_NONBLOCK`, present or
unknown ACL, relevant mount aliases/malformed mountinfo, literal pinned SSH
argv construction without host contact, receipt drift, a
label appearing after the last slow gate, and lock ownership through write
receipt construction. These are Implementer-local tests, not an independent
PASS or a live r4 probe.

A different Verifier must review these exact bytes, the bounded private-parent
inference, receipt validation and the r3-to-r4 diff. A Supervisor may then
consider a separately named exact enabled transfer-only request with new
independent final review and one-shot assignment. The Operator would still
need fresh strict trn21 identity, protected-asset, quiescence, disk/inode,
STOP/breaker, lock/ACL/mount and label preflight. No r4 transfer, checkout,
CPU G2, native run, evaluation, monitor operation, cleanup or submission is
authorized by this disabled package.
