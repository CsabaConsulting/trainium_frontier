# trn22 e65 → r9 incremental source-stage r3 — DISABLED / HOLD

This is a distinct Implementer-only response to the [r2 independent
HOLD](../../verification/20260925-trn22-r9-incremental-stage-r2-independent-HOLD.md).
No host/SSH/AWS contact, source transfer, checkout, marker, assignment,
native run, evaluation, cleanup, breaker reset or submission occurred. This
package contains no enabled request, independent final PASS or Supervisor
assignment; `operator_once.py` refuses before a marker or SSH. A future stage
result would still not authorize native A0.

## Exact disabled subject

| Item | SHA-256 / pin |
| --- | --- |
| `remote_incremental.py` | `dcfde1310f143b561c24c97efc1cb0a193975cbadff80013fc8c76ef33197b6b` |
| `operator_once.py` | `d7a0b7467a3dca649b3056974b7d5e6e649fa900be0d47a2debf9a7f364d3794` |
| `REQUEST.HOLD.json` | `2daf85561b7683ec4ee01a350f4f03fd4887ac66d64c7bafcb5988cbfeca595b` |
| `test_incremental_stage.py` | `b873ccc43021a73a5b7bba8174113fdec024eeeefb42a0382360c9f3f9d7fa14` |
| retained r1/r2 race regression `test_late_scan_r2.py` | `0ffd411514fa3a7a175789a2f5b4ab26e7f1c5b460f52aa75994bf5cd7aa4546` |
| new r3 boundary regression `test_final_boundary_r3.py` | `7ebd131ca6e215150e8e11c92d7e8c7a90648cc53a1a65c28b9194c82c778469` |
| reused frozen r1 `e65-to-r9.bundle` | 101,237 B, `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |

R3 is a direct successor to r2 subject
`a7643f05c5468da6537c2d7b403a05ac048679f0`; r1/r2 packages and
their HOLD evidence are unmodified. The r2 independent receipt is canonical
at `c5c9a8bd`, SHA-256
`859e70b4737e2f1ac638e20b22889931d26425396389acfcee8ae0bb54a231aa`;
its nine-case witness is SHA-256
`3be2b006052f7a5913cedb7d2ecf6ee90d09c71d425cd8cb52e738c4a1e88a4e`.
R2's earlier source-only HOLD remains controlling for r2.

The exact bundle remains in the frozen r1 sibling package and is hashed and
length-checked before any marker. The independent r2 review confirmed its
r9 HEAD `e046de16826bfc3b954136e3c60804aea0554ecc`, sole prerequisite
e65 `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, and the eight pinned
r9 science/safety files; `train.py` and `prepare.py` are unchanged. The r1
inventory counted 131 objects, 14 commits, 57 trees, 60 blobs and 117 named
paths, with a bounded credential-path/blob scan finding zero hits. That scan
is not a proof against arbitrary prose secrets. No `.env`, private key,
token or credential bytes are copied or printed.

The source remains the observed group-writable `0775` linked e65 checkout,
not a private `0700` checkout. The exact `.git` pointer, Gitdir ancestry,
owner/inode/mode, private `.cache` lock path, ACL/NSS ordinary-DAC condition,
ext4 mount, protected checkpoint/caches, clean source commit and file hashes
are remeasured by every live full probe. The 07:33 Operator observation and
independent bounded review are planning evidence only. Unknown or changed
facts HOLD. The new target is the same absent private r9 stage-r1 path, never
an in-place edit of e65. R3 uses a distinct local one-attempt marker and
refuses if either r1 or r2 marker exists, because the remote label and target
remain the same. No previous request/review can authorize r3.

## Final boundary and narrow rollback

Under the existing nonblocking host flock, the write phase retains its full
fresh preflight and r2 early activity check. It then performs two broad
attempt-label scans, requiring their combined duration to fit the inherited
one-second limit. A **full process/fuser/Neuron activity check and root
byte/inode check follow both scans**, before the first exclusive open. STOP
and exact bundle/checkout absence are checked again. This catches persistent
process/Neuron/disk changes introduced during either scan before any write.

The code checks monotonic ages of the final label, activity and space
observations immediately before `O_EXCL`, then again **at the return of the
actual `os.open`**. Before writing bundle bytes it performs another broad
label scan, full process/fuser/Neuron check and root byte/inode check while
the provisional file remains empty. These post-open checks catch persistent
STOP/label/process/Neuron/disk changes entering between the final helper and
the open. The successful write receipt reports conservative ages at open,
post-open ages before payload, combined pre-open scan duration, open-call
duration, both root space observations and both post-label activity gates.
The local wrapper requires every age/duration to be finite and at most one
second and both space records to meet the unchanged 13,177,973,911-byte /
10,000-inode pre-write floor. The hard 6.5-GiB floor and post-write floor
are unchanged.

If a scheduling delay or persistent STOP/label/activity/disk change is
detected only after `O_EXCL`,
the code may remove **only its own same-inode, single-link, still-zero-byte
provisional bundle name**, fsyncing the parent before returning HOLD. It
never deletes a nonempty, replaced or ambiguous file, or a checkout. Such
partials, plus the one-attempt marker, remain for inspection without retry.
This narrow pre-payload rollback is part of the disabled proposal, **not**
general host cleanup authority; a future enabled request and independent
review must explicitly accept it. Cross-session readback and unconditional
full postflight remain required.

No userspace sequence makes activity, disk and a broad label traversal
atomic. In particular a noncooperating same-UID/root actor could add a
label after the last broad traversal, including during the final activity
check, or mutate state after the last observation. The cooperative flock,
private-parent ordinary-DAC evidence and one-second upper bound define the
limited threat model; they do not rule out privileged aliases or malicious
same-UID processes. An independent Verifier must decide whether this bounded
residual is acceptable. The actual r3 sequence (three full activity checks,
three broad scans, disk checks and post-open freshness check) has **not** been
timed on trn22. An r2 timing result is not a substitute; r3 may
systematically HOLD.
Do not enable or assign it until a separate read-only r3 feasibility receipt,
independent source/final reviews, fresh host preflight and one-shot authority
all pass.

The Implementer ran 35/35 local synthetic tests under normal Python and
35/35 under `python3 -O`. The suite retains r1/r2 late STOP/label and slow
scan negatives and adds the r2 Verifier's process/Neuron/disk/open-delay
witnesses plus persistent helper-to-open arrivals. A 1.1-second open delay
leaves neither bundle nor checkout;
unexpected nonzero mutation is retained and HOLDs without checkout. This is
Implementer regression evidence, not independent verification or live host
feasibility.

Offline commands:

`python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r3-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r3-hold -p 'test_*.py' -q`
