# trn22 FIFO-lock r4 transfer-only final-r2 request — REVIEW HOLD

This is a **new exact request candidate**, not an Operator assignment or host
authorization. It repairs the two persistent local STOP timing gaps found in
the independent final-r1 HOLD (SHA
`8d1bf8caa6fa0643b067c9242cbeed1e8adb756592bbedc274fefcc328ca04ad`).
The final-r1 package, its HOLD, and the disabled FIFO successor remain
unchanged. The disabled successor `5740a0e0` (canonical `53a02d7d`) has a
**disabled-static-only** independent PASS `248730a7` (canonical `3c3d9c15`),
receipt SHA
`67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d`.
Neither receipt authorizes this r2 wrapper, SSH, transfer, checkout, or a run.

| Exact final-r2 file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FIFO_FINAL_R2.json` | `b8bcfbb5d898bfb6c3b6308600b9c2e91d474fdccee66620bd6ff055f30f2ec2` |
| `operator_transfer_lock_core_r2.py` | `42d21f959ca2c7bab156bb5bdc320203c1278971092fd40e95da27e24e0846b8` |
| `stage_r4_transport_lock_final_r2.py` | `fde41bbe3ab5367840d1b225316b2d961a562a2dee4e6ca9509120e83b060d84` |
| `operator_transfer_lock_final_r2.py` | `f761ce825691a2ac911adb65f3e6dbdc9c90ec613e1de11b8f9e46f1690e66ae` |
| `test_final_fifo_request_r2.py` | `375d3257752d842b8f6519b17c9657ccb1f2b28848254707944ab8507d2c1ad1` |
| Unchanged disabled remote `stage_r4_remote_lock_hold.py` | `cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27` |

The only source is frozen commit
`446b657695f086f9b484fa601c9a3b61c2495912`, capsule 104,154,875 bytes,
SHA `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The only remote archive target is
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`.
The new assignment label is `trn22-c5c-a0-r4-fifo-lock-transfer-final-r2`.
Its unique local marker and journal are respectively
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-fifo-final-r2-one-attempt.marker`
and the same directory with basename
`trn22-c5c-a0-r4-fifo-final-r2-one-attempt.jsonl`. Both must be absent before
the one attempt; a partial target, marker, or journal must be retained, never
overwritten, deleted, renamed, cleaned, or retried.

## Exact authority boundary

`dispatch_authorized: true` in the request denotes **intent only**. The r2
wrapper hashes the actual request, historical HOLD and static PASS, disabled
subject files, source-stage evidence, and new core/transport bytes. It then
requires two currently absent fixed-path files: an independently written
`INDEPENDENT_FIFO_FINAL_R2_REVIEW.json` with an exact structured PASS binding
the actual wrapper SHA, and a separate
`SUPERVISOR_FIFO_FINAL_R2_ASSIGNMENT.json` binding that review SHA and this
one-shot scope. The package creates neither file. Missing or changed evidence
stops before marker, journal, or SSH. A separate Verifier must review **these
exact r2 bytes**; the old static PASS is not substituted for that review.

## Repair and bounded transfer shape

After authority, local source SHA and STOP/breakers are checked. The wrapper
rechecks STOP/breakers after slow second authorization and again immediately
before the `O_EXCL|O_NOFOLLOW`, mode-0600 one-attempt marker. It creates an
exclusive, fsynced raw SSH journal. A fresh read-only probe binds host, boot,
ext4 mount, lock, labels, and old-path inventory. Before the sole write, the
wrapper rechecks authority, marker, and local breakers. The core opens the
exact capsule `O_NOFOLLOW`, compares its file identity with the pre-marker
hash binding, rehashes that **same FD** and rewinds it for SSH stdin. After
host-key setup, FD rehash and journal fsync, it rechecks local STOP/breakers
immediately before write `Popen`. This closes the two *slow* intervals in the
r1 HOLD, not the unavoidable small race after a final check.

The transport pins ED25519 known-hosts/fingerprint, the exact private-key
**path only** (never contents), `-F /dev/null`, and strict host checking. It
embeds only the unchanged reviewed disabled remote under a non-main namespace
and invokes `probe`, `write`, or `verify`; no checkout operation exists. The
remote writer retains the bounded private-cache 0700/no-ACL, existing
`device0.lock` nonblocking flock, fresh STOP/process/Neuron/label, ext4,
space/inode, and old-path gates. It holds the cooperating launcher lock across
final label scan, `O_EXCL` target creation, write, fsync and readback. This
does **not** absolutely exclude same-UID/root, alternate-path or uncooperative
manual activity. A distinct SSH verifies persistence, and a fresh read-only
postflight runs even if the write fails. Raw SSH argv, PID/exit, stdout/stderr
bytes/hashes/base64, receipts and failures remain in the one-attempt journal.

The frozen 446b native launcher may have assertion-based gates; transfer
staging does not approve that launcher. Checkout, a clean source update, CPU
G2, native run, evaluation, score, submission, or AWS lifecycle action each
remain outside this request and need their own authority.

Local-only evidence: the r2 suite passes **11/11** under both normal and
optimized (`python -O`) Python. Its negative tests inject persistent STOP
during slow reauthorization, marker-parent work, and the interval after FD
rehash but before write SSH spawn; the latter still attempts read-only
postflight. Tests use synthetic files and intercepted spawns, not the host.
Next: a separate independent **final** review of these exact bytes, followed
only if PASS by a narrow Supervisor one-shot transfer-only assignment and an
Operator fresh preflight. Until then: **REVIEW HOLD; no dispatch**.
