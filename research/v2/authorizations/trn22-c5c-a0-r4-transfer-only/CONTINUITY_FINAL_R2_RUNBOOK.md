# trn22 r4 exact transfer-only successor — final review pending

**Current effective dispatch verdict: HOLD.** This is an Implementer-prepared
request, not an Operator assignment. The new request records transfer-only
intent with `dispatch_authorized: true`, but its new wrapper cannot reach its
one-shot marker, journal or SSH until it reads an **actual fixed-path**
independent final PASS JSON and a **separate** Supervisor one-attempt
assignment JSON that binds that PASS file's SHA-256. Both authority files are
absent in this package. No SSH, host/AWS contact, transfer, checkout, CPU G2,
native run, evaluation, cleanup, breaker reset or submission occurred while
preparing it.

This successor adds four files and leaves the earlier HOLD archive untouched:
`TRANSFER_REQUEST.CONTINUITY_FINAL_R2.json`,
`operator_transfer_continuity_final_r2.py`, and
`test_operator_transfer_continuity_final_r2.py`, plus this runbook. The source code and disabled
repair reviewed at `d682a90b1e75bdac128938feda884484ea87a8d5` are reused
byte-for-byte. Their independent static PASS/dispatch HOLD receipt is
`INDEPENDENT_CONTINUITY_REPAIR_STATIC_REVIEW.md`, SHA-256
`669d99837e25ee634e4db418125b1b8bf13862a1907baad10ac213b5e89c7c57`.
That receipt is not this successor's final PASS.

## Frozen exact subject

| Item | Exact value |
| --- | --- |
| Frozen training-source commit | `446b657695f086f9b484fa601c9a3b61c2495912`, not package HEAD |
| Outside-Git source capsule | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`; 104,154,875 bytes, SHA-256 `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| Sole proposed trn22 target | `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz` on the pinned host's persistent ext4 root |
| Fixed one-shot local marker | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker` |
| Fixed raw SSH journal | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl` |
| New exact request SHA-256 | `99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5` |
| New final wrapper SHA-256 | `14dac3c538d3d08e2876c1327e177f2a4c4696cfadb9884004725305383ccf23` |
| Reused reviewed core/transport/remote SHA-256 | `7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736` / `ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0` / `023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d` |

The wrapper pins the actual static review file, unchanged disabled requests,
reviewed marker-helper wrapper, core, transport, remote, source-stage contract
and source-stage independent receipt by SHA-256 before considering any future
authority. The request cannot provide its own final PASS or assignment hash.
The proposed SSH command remains config-isolated (`-F /dev/null`) and pinned
to one ED25519 known-hosts fingerprint; private-key bytes are not read by
these local checks.

## Separate authority still required

1. A different Verifier must review this **new exact request, wrapper, tests
   and their hashes**. Only that Verifier may write
   `INDEPENDENT_CONTINUITY_FINAL_R2_REVIEW.json` at the fixed path with the
   exact structured fields from `_expected_final_review(...)`, including
   `verdict: PASS`, the actual new-wrapper/request SHA-256 and the reviewed
   static receipt SHA-256. A narrative receipt should record authorship,
   tested mutations and limitations. A role string in JSON is not proof of
   independence; the Supervisor must verify provenance.
2. A fresh read-only trn22 feasibility check must confirm exact host,
   `/dev/root` versus `/dev/nvme0n1p1` block identity, boot/mount/parent,
   all old r1/r2/r3 paths, absent new target/checkout/label, protected
   checkpoint/six caches, STOP/breaker and process/Neuron/lock quiescence,
   free space and inodes. Historical snapshots cannot grant this gate.
3. Only then may a distinct Supervisor issue
   `SUPERVISOR_CONTINUITY_TRANSFER_R2_ASSIGNMENT.json` at its fixed path,
   exactly binding the **actual final review file hash**, this request and
   wrapper, source capsule, static review, host, sole target, fixed marker,
   journal and one attempt. It authorizes transfer only. No checkout,
   extraction, CPU G2, native run, evaluation or submission is included.

Absent either fixed file, `authorize_final()` raises HOLD before touching the
marker or opening SSH. Forged, stale, mismatched or extra review/assignment
fields fail exact comparison. Local synthetic review/assignment bytes in
tests do not create authority.

## One attempt, then independent result review

If all future gates pass, the wrapper checks local source SHA and STOP/global
and trn22 breakers, then durably creates the **same fixed** private marker
with `O_EXCL|O_NOFOLLOW` and fsync, followed by the fixed exclusive raw
journal. A failed or partial marker/journal consumes the assignment; neither
is removed or renamed. Its first SSH process performs a fresh full read-only
preflight. The sole writer receives an exact digest-bound preflight token and
rechecks host identity, STOP, processes, checkpoint/lock/Neuron and label
before `O_EXCL` creates the exact target. The wrapper joins preflight to
write, then a distinct SSH readback and mandatory read-only final postflight
even after a failure. Raw SSH argv, exit, stdout/stderr and receipts remain
in the fixed journal. A partial target is preserved and the same path is
never overwritten, deleted or retried. Even success ends at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`, not checkout
authority.

Local-only tests cover the new request and both previously reported safety
gaps. Run the maintained and independent repair suites plus the new request
suite with `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest` on the three
existing repair test files and `test_operator_transfer_continuity_final_r2.py`.
The new tests do not open SSH or read private-key contents.
