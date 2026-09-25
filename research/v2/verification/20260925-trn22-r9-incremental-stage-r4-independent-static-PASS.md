# Independent trn22 e65→r9 incremental source-stage r4 review — static PASS only

Reviewed 2026-09-25 09:01 UTC by a Verifier separate from the Implementer.
Exact disabled subject: `19fbfdc88763b6b5762e219339dcd156e7977a30`,
direct child of r3 `80c96851edbb6cbc707c82c68773a95ea1231f91` on
`origin/codex/20260925-trn22-r9-incremental-source-stage-r4-hold`.
The r3 independent HOLD is pinned to
`524f761fb470db86460d851b8f3af8caa50bce95`, with receipt SHA-256
`5a2b24a5b27661e61773004be35ea60308bb16385f199f5b2a24e719f43054ed`.

**Verdict: PASS for the exact disabled r4 source bytes under the documented
cooperative-flock/ordinary-DAC threat model, not PASS to dispatch or stage.**
The r3-blocking unauthorized `os.unlink` is removed. Failed gates after
exclusive creation retain even a zero-byte provisional bundle and the
one-attempt marker; failed transfer/durability does not create a checkout or
return success. No enabled request, independent *final* review, Supervisor
assignment, live timing measurement, fresh host preflight, or postflight
receipt exists in this package. This static receipt is **not** the
`INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json` accepted by the wrapper.
No SSH/EC2/AWS contact, host marker, source transfer, checkout, cleanup,
training, evaluation, breaker reset, or submission occurred in this review.

## Findings at the exact boundary

- Compared r3 and r4 implementations directly. R4 deletes the entire
  `_rollback_unpopulated_bundle()` and its call. `remote_incremental.py`
  contains no `unlink`, `remove`, `rmdir`, `rename`, `replace`, or retry path
  for the provisional host bundle. The only normal target creation is
  `os.mkdir(TARGET.name, 0o700)` after the full bundle hash/length check;
  reconstruction and checkout follow that gate. The same-UID replacement
  witness now preserves both names rather than deleting either. This fixes
  the exact r3 authority defect without weakening the inherited age bound.
- Two broad final label scans are followed by full STOP/process/Neuron and
  byte/inode checks before `O_EXCL`; actual exclusive-open return rechecks
  the finite ≤1-second ages and open-call elapsed time. A third broad label
  scan, full activity check, root-space check, direct STOP/target check, and
  finite post-open age checks occur before payload. Ordinary persistent
  late STOP, label, training process, Neuron holder, or disk loss in the
  tested intervals returns HOLD. If after `O_EXCL`, the empty provisional
  inode stays and no checkout is created. The pre-write floor remains
  13,177,973,911 bytes and 10,000 inodes, alongside the 6.5-GiB hard floor.
- The bundle descriptor and parent directory are fsynced in `finally` on
  success or failure. Independent interrupted-payload and file-fsync
  witnesses retain the nonempty partial, create no checkout, and return no
  success. A failed fsync makes durability uncertain, so a later human
  inspection/decision is required; the code does not clean or retry.
- The local wrapper's r4 marker is distinct and checks all r1/r2/r3 markers
  before `O_EXCL`. It demands exact canonical enabled request, independent
  final review and Supervisor assignment records bound to the r4 source
  hashes, request hash, label, scope, bundle and r3 HOLD. Its checked-in
  request disables dispatch, staging, training and evaluation; the three
  enabling artifacts are absent. The CLI refuses under normal and `-O`
  before a marker or remote contact. The one-attempt claim is conditional on
  the later authenticated wrapper and persistent marker/remote one-shot
  gates, not authority supplied by this static receipt.
- Preflight, write, readback and postflight are separate SSH processes in
  the wrapper. After a failed phase, it **attempts** postflight and records
  `POSTFLIGHT_HOLD` if that attempt fails. A newly appearing local STOP may
  block the postflight connection; no completed postflight may be inferred
  then. Success demands an actual completed postflight receipt and the
  source/protected identity comparison. Failed postflight cannot yield
  `STAGE_EVIDENCE_READY`.
- The source checks pin the observed group-writable `0775` linked e65
  checkout, `0664` `.git` pointer, Gitdir ancestry, clean HEAD, science
  hashes, ownership/ACL/group condition, ext4 mount, cooperative lock,
  protected checkpoint/caches and root floor. Unknown or changed live facts
  HOLD. These checks are code assertions, **not fresh trn22 observations**.

## Exact subject and offline evidence

| Item | SHA-256 |
| --- | --- |
| `REQUEST.HOLD.json` | `8cd3b2debb0c309ab72faef78b057e344fc0525b1269332d3d46b13337797f56` |
| `RUNBOOK.md` | `ffe6b55505759e1a71cbd24d810a3b0d71f9d6a58d597aa7096cf8d2e0babdd9` |
| `operator_once.py` | `64d0ed49261cd231b3193f315d114d43e39a94f3b3dea21833a37dd15c0c2d7e` |
| `remote_incremental.py` | `871fd00aa61d0db2df8597949c8cad43c4c0a9a94ffda218a44851d60091c9b7` |
| `test_incremental_stage.py` | `ffde1ed789f2c54aaac171036f533263fd9e551a633ebad3a4de44c1487bb02f` |
| `test_late_scan_r2.py` | `36104c13dd52c2d7c81a79f6016a6360082be919822dce209bf92badefe5fe30` |
| `test_final_boundary_r4.py` | `c68fba09ad56495419a0f5583fcd33e8122cfee98f6dfb6779d7e8c3ee94f7da` |
| independent disposable-directory witness | `1693f76582fa4d6979a980a96181f3264e1d1173af9ab3cccb7ccbf04009d7d4` |
| reused r1 bundle | `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |

The checked-in bundle is 101,237 bytes and `git bundle verify` succeeds:
r9 `e046de16826bfc3b954136e3c60804aea0554ecc` is HEAD and e65
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` its sole prerequisite.
Independent `git show` hashes confirm both commits have unchanged `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The r4 subject adds seven authorization-package files only; it does not edit
science, evaluator, data, the old r1–r3 packages, or the r3 HOLD.
`git diff --check 80c96851 19fbfdc8` passes.

Supplied suite: **36/36 PASS** under normal Python and **36/36 PASS** under
`python3 -O`. Independent `test_trn22_stage_r4_independent.py`: **3/3 PASS**
normal and **3/3 PASS** under `python3 -O`, with late post-open label, partial
payload-write, and file-fsync-failure cases. Each witness forbids `unlink`,
checks the retained empty/nonempty bundle and absence of checkout, and is
confined to a disposable local directory. No native Neuron work was run.

## Scope and next gates

This PASS does not establish that r4's three broad scans, three full activity
checks and disk checks fit the inherited one-second live boundary on trn22;
that timing remains **unknown**. These userspace checks are not atomic against
same-UID/root interference or alias paths. As in the r3 HOLD, a persistent
label introduced after the last broad scan can reach material write in a
noncooperating race, and a delayed payload write can occur after the sampled
`O_EXCL` freshness window. The documented numeric criterion is at actual
exclusive open, not payload completion. Those residuals were not silently
reclassified as fixed by the narrow r4 change. An independent final review
must consciously bound or reject them.

Before any source-stage attempt: obtain separate read-only live r4 timing
evidence and its review; create an exact enabled source-stage request and
independent final review; require Supervisor one-attempt assignment and
fresh host preflight. Any failed provisional bundle, even empty, remains
human-reserved for cleanup/retry. Successful source staging still needs
cross-session readback, a **completed** full postflight and independent
stage-result/G2 identity review. R9 on trn22 versus planned r10 on trn21
does not satisfy experiment-protocol's same-HEAD requirement for a scored
run without later exact rebinding or explicit human-approved reconciliation.
Native A0/training authority is separate. There is no BPB result here.
