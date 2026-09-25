# Independent trn22 FIFO-lock successor review — PASS disabled static only

Reviewed 2026-09-25 06:00 UTC by a Verifier separate from the Implementer.
The exact pushed subject is `5740a0e05b23e70acb9fb353c335d09d8fb4b257` on
`origin/codex/20260925-trn22-r4-transfer-lock-nonblock-hold`; the origin ref
was independently checked. I read the repository contract in its required
order, then the subject package and cited prior HOLDs. All inspection and
tests were local. The Implementer checkout remained clean and unchanged.

**Verdict: PASS_DISABLED_STATIC_ONLY for these bytes; DISPATCH HOLD.** The
prior private-cache package's FIFO blocking defect is repaired. This is not
an enabled request, fresh host preflight, final transfer review, Supervisor
one-shot assignment, transfer result, or native-launch approval. The earlier
r3 PASS remains superseded by its late-label HOLD addendum.

| Exact subject file | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.HOLD.json` | `df3a8a776b10fdf31bf307ff4b036131395b8a744c88699dc80d7e0f64ea0b03` |
| `RUNBOOK.md` | `18342e895e1ecb38032dd7d965a3820abf29ded152161ba381d8824bc0cbe431` |
| `stage_r4_remote_lock_hold.py` | `cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27` |
| `stage_r4_transport_lock_hold.py` | `4a8bfcc7ca091288e69b395cc28fe86607d257eef5f6fdbf2ba6d54a55781dfb` |
| Supplied `test_lock_final_scan_hold.py` | `c73a1f8b813d40e0ca4b2a2b110f931522376ccd99221a157d98e449cedf2f6b` |
| Verifier-owned `test_trn22_fifo_nonblock_5740.py` | `6027f472c8455fe8b62907c618767436d1d23043920070258d123a95bfab1093` |

The contract and runbook pin the actual subject hashes. The local capsule
rehashes to 104,154,875 bytes and
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The frozen `446b657695f086f9b484fa601c9a3b61c2495912` launcher, host, and
guard blobs match the contract's three hashes. Its launcher obtains
`LOCK_EX|LOCK_NB` on the exact existing lock and passes the descriptor to
its guard and child. Only the five disabled successor files were added by
the subject commit; no protected source or earlier HOLD/PASS file changed.

The remote helper classifies the named lock inode with no-follow `stat`
before opening it, then uses descriptor-relative
`O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC`. It compares named and opened
regular-file identity and closes all descriptors on failure. Independent
bounded child tests observed prompt HOLD for a pre-existing FIFO and actual
regular-to-FIFO and regular-to-symlink swaps between classification and open.
The read-only descriptor acquired an exclusive nonblocking flock, and a
second process could not acquire it. The writer retains that lock from before
its held live checks through the final label scan, exclusive target creation,
fsync/readback, and receipt construction. The final empty scan is the last
potentially slow check before `O_EXCL`; monotonic evidence limits scan-to-open
and open duration to one second. A late label or expired scan prevents target
creation; a slow open or later failure leaves any partial target intact.

The private-cache exception is bounded and consistent with the archived
read-only observation: `.cache` must be exactly owner/group `1000:1000`, mode
`0700`, with absent access/default ACLs; only under it can the exact
group-writable `0775` inner directory and `0664` regular lock pass. The code
rejects other group-write modes, world write, special bits, ACL uncertainty,
wrong device/owner, link count, or named/opened inode drift. Fresh preflight
and write recheck and bind this identity. Transport validation binds the
preflight receipt hash, host/boot/mount/ancestry, lock record, held-check
evidence, final scan and target-open ordering; mutated lock, STOP, process,
Neuron and timing fields fail the independent comparison. The file-backed
old `.git` inventory and protected checkpoint/cache gates remain present.

Local results: supplied successor suite **14/14**, verifier negatives
**4/4** (three bounded race cases inside one test), archived r3 final suite
**7/7**, archived r3 static suite **4/4**. `git diff --check` passed. Both
disabled entrypoints still refuse under `python -O`; there is no enabled
wrapper in this package. The frozen launcher's native dispatch behavior is
outside this transfer-only static verdict and needs its own gate.

The accepted threat model covers ordinary traversal through this private
path and launchers that cooperate on the same flock. It does not atomically
exclude same-UID/root lock replacement, alternate namespaces, or manual
processes that ignore the lock. No host or SSH contact, marker/journal,
transfer, checkout, G2, training, evaluation, cleanup, breaker reset, or
submission occurred. A distinct enabled transfer-only request, independent
final review, fresh host preflight and Supervisor assignment remain required.
