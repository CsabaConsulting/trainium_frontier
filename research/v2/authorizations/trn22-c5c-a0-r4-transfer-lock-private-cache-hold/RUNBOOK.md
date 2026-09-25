# trn22 r4 private-cache cooperative-lock successor — DISPATCH HOLD

This is a **distinct, disabled Implementer successor** to the earlier lock/final-label
package `37ef4f73`. Its independent static review
`INDEPENDENT_LOCK_FINAL_SCAN_STATIC_HOLD.md` (SHA
`e34f289bb253b7c9d7bf0ab86a07fbd5962401519ff7a75a6f7c62a4b7b0ca9a`)
found that package cannot pass the observed lock-directory mode `0775`, and
that its runbook misstated the transport SHA. Both findings remain HOLD until
this successor receives its own independent review. It does not alter the
archived r3 writer, request, PASS, later HOLD, or earlier repair. Its `main()` and
transport command builder both fail before SSH or host file access. No marker,
journal, transfer, checkout, G2, native run or evaluation is authorized here.
The earlier r3 PASS SHA `1069bba48756a5977498fb214094413add9a67c8d06633c9171d45f10fd5210f`
covers different code. The independent
`INDEPENDENT_CONTINUITY_FINAL_R3_LATE_LABEL_HOLD_ADDENDUM.md` (SHA
`94de8bc3d12a4c36cdad6d2160c1db081d4ea0c0035b6024bf7471e389ffaa0b`)
supersedes that PASS and keeps r3 dispatch on HOLD. Its exact timing finding
is the reason for this successor.

| Exact local file | SHA-256 |
| --- | --- |
| `stage_r4_remote_lock_hold.py` | `1ad01ed1e141d283c620bf39080d69c6d1adc184b1160d9c4edfea5507dffab1` |
| `stage_r4_transport_lock_hold.py` | `cf8ac477e7001f7c992fd063cf72585948d1a63e85b5e5bcde9c929f7c696837` |
| `test_lock_final_scan_hold.py` | `2eae8ac26fb30258cb05989a9bdeedd5920520a628f2a49e4d57831437061411` |

The frozen source remains commit `446b657695f086f9b484fa601c9a3b61c2495912`.
The sole capsule is 104,154,875 bytes, SHA
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`;
the sole proposed host archive target is
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`.
It must still be absent before an independently authorized unique attempt.
The existing lock is
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`. Frozen 446b
`fresh_start_launch.py` opens that file and takes a nonblocking exclusive
`flock`, then passes the descriptor into the guard and training child. The
repair opens only that **existing** lock read-only with no-follow directory
descriptors and takes the same nonblocking exclusive lock; it never creates,
writes, truncates, renames, deletes or changes permissions on the lock. A
missing file, unsafe ancestry, changed inode or held lock is HOLD. The archived
Operator read-only ACL receipt
`OPERATOR_GROUP_ACL_READONLY_20260925T053454Z.md` (raw receipt SHA
`753cd90daee7ab595383cff8b1f02b79766986f452bb6a3521dde5065b88e112`)
observed `.cache` UID:GID `1000:1000` mode `0700`, the inner directory
UID:GID `1000:1000` mode `0775`, and the lock UID:GID `1000:1000` mode `0664`.
The access/default POSIX ACL xattrs on both directories and the access ACL
xattr on the lock were absent (`ENODATA`). Bounded NSS checks found no other
local GID-1000 account/member, and mountinfo named no relevant bind path;
neither check excludes privileged actors, alternate namespaces or all indirect
aliases. The prior receipt is **planning evidence only**. Every new preflight
and write reopens the exact no-follow inodes, requires the private parent to
be **exactly** `0700` and owner/group `1000:1000`, and requires absent access
and default ACLs on both directories and absent access ACL on the lock. Only
then may the exact observed `0775` inner directory and `0664` lock be accepted;
other group-writable, any world-writable or special-bit descendant mode is
rejected. A non-`ENODATA` ACL error is HOLD. This rule relies on the private
parent's ordinary DAC path barrier, not on completeness of the NSS listing.
All inode/mode/ACL observations are bound to the fresh preflight and rechecked
before creation. This exception is **unreviewed and disabled**; do not change
host permissions just to pass it.

The bounded threat model is ordinary DAC traversal through the pinned private
`.cache` pathname plus cooperative exclusion with the frozen 446b launcher.
Known root-device aliases, STOP markers, process/Neuron/label activity and
checkpoint holders remain full fail-closed gates. This does not claim a proof
about every possible alternate mount namespace or malicious same-UID/root
activity. The write binding includes the fresh preflight's
host/boot/mount/ancestry and exact lock path/device/inodes/modes/ACL absence.
Before creation the writer repeats the
old full STOP/process/label/checkpoint-holder/Neuron checks, then acquires the
cooperative lock. Under that lock it repeats the full live checks; fuser on
the lock is replaced by proof of our own retained descriptor so the writer
does not reject itself. It checks STOP/process and the held lock again, then
does a final empty label scan as the **last potentially slow pre-creation
operation**. The scan's start/end UTC and monotonic times, lock-acquisition
time, and exclusive-open attempt/completion times are recorded. The scan to
open-attempt interval must be at most one second; a slow open fails with its
partial target preserved. Lock identity is rechecked after exclusive create
and after the write, and the lock remains held through fsync/readback and
receipt construction. The offline transport comparator requires these exact
lock and final-scan observations in the write receipt and binds their hashes
into the separate-session comparison result. None of this changes source
contents or permits a same-path retry.

This closes the reported late-label gap for **cooperating 446b launchers**:
one cannot obtain the same exclusive lock while this writer holds it. It is
not an atomic guarantee against a non-cooperating manual process that ignores
the lock, a privileged actor using another path or namespace, or a same-UID
actor deliberately replacing the lock path. The
remaining STOP/process/Neuron/label probes are fail-closed snapshots, not a
host-wide transaction. A reviewer must evaluate that residual risk; this
Implementer package does not assert absolute exclusion.

An independent Verifier must now review these **actual bytes** and issue a
new static PASS or HOLD. A separate Implementer would then need an exact
enabled transfer-only request/wrapper, new independent final review, and a
Supervisor one-shot assignment. The future wrapper must retain the existing
one-attempt marker and raw SSH journal semantics and must run a fresh full
read-only postflight in a distinct SSH process even if writing fails. Any
partial archive, marker and journal remain in place; no overwrite, delete,
rename, cleanup, retry, alternate target, checkout or score claim follows.
The disabled package has no such wrapper or authority.

Local-only tests: `test_lock_final_scan_hold.py` passes **13/13**, including
actual Linux read-only flock contention on a disposable synthetic file,
private-cache mode and ACL adversarial cases, a late label that appears after
the slow held checks but before exclusive creation, stale-scan rejection,
held lock, malformed receipt,
partial target preservation and postflight comparison. Archived r3 independent
final tests pass **7/7**, and r3 static negative tests pass **4/4**. The old
r2 negative suite passes five cases but its historical actual-file pin test
errors because canonical r3 changed `stage_r4_transport.py`; another older
b80 negative suite calls pre-r3 function signatures and errors. These are
historical test-fixture incompatibilities, not a claim that the new repair
passed them. No host, SSH, AWS or experiment action occurred in these tests.
