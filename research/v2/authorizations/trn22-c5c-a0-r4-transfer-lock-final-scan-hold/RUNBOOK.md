# trn22 r4 cooperative-lock and final-label transfer repair — DISPATCH HOLD

This is a **new, disabled Implementer package** based on canonical control-plane
`8c005717` (a fast-forward from `762a744ca618c9c79b8f1574ef9a6acf9758653e`). It does not alter the
archived r3 writer, request, PASS or later HOLD evidence. Its `main()` and
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
| `stage_r4_remote_lock_hold.py` | `26ca03e94a005513a648ce968326c6d2dfbd160cc099872f5ebed3706dfeeafd` |
| `stage_r4_transport_lock_hold.py` | `50865a842b5568b21bdbd08b18a1fdf7cc02bffbbb43b0f6f7e2beab57806494` |
| `test_lock_final_scan_hold.py` | `db03bec8f73bbccbb0edaeaec1108b0cf7d769183ae9a1c5f135ed8ab59590a5` |

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
missing file, unsafe/writable ancestor, changed inode or held lock is HOLD.
The historical read-only host snapshot did not pin both lock-ancestor modes:
a **fresh** preflight must supply them, and an unsafe value may make this
repair infeasible without a separately approved design. Do not change host
permissions just to pass this gate.

The write binding includes the fresh preflight's host/boot/mount/ancestry and
exact lock path/device/inodes/modes. Before creation the writer repeats the
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
the lock, or a same-UID actor deliberately replacing the lock path. The
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

Local-only tests: `test_lock_final_scan_hold.py` passes **9/9**, including
actual Linux read-only flock contention on a disposable synthetic file,
late-label and stale-scan pre-create rejection, held lock, malformed receipt,
partial target preservation and postflight comparison. Archived r3 independent
final tests pass **7/7**, and r3 static negative tests pass **4/4**. The old
r2 negative suite passes five cases but its historical actual-file pin test
errors because canonical r3 changed `stage_r4_transport.py`; another older
b80 negative suite calls pre-r3 function signatures and errors. These are
historical test-fixture incompatibilities, not a claim that the new repair
passed them. No host, SSH, AWS or experiment action occurred in these tests.
