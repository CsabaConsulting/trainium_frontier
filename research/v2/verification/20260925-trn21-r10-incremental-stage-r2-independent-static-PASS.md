# Independent trn21 e65→r10 incremental source-stage r2 review — static PASS only

Reviewed 2026-09-25 09:08 UTC by a Verifier separate from the Implementer.
Exact disabled subject: `3e84658933cf7f1c9758df894dd57fb265aba366`,
direct child of independently held r1 `c03e3ac385d9ff4384bba1f1ca71ffa2486cad7c`,
on `origin/codex/20260925-trn21-r11-incremental-stage-disabled`.
The r1 HOLD at
`research/v2/verification/20260925-trn21-r10-incremental-stage-independent-HOLD.md`
reproduced a pre-`O_EXCL` STOP/label/activity/disk/freshness gap and an
overstated postflight claim.

**Verdict: PASS for these exact disabled source-stage r2 bytes under the
documented cooperative-flock/ordinary-DAC model only.** This is not the
independent *final* stage PASS consumed by `operator_once.py`, and is not
dispatch, marker, source-stage, native training or evaluation authority.
The checked-in request has every authorization flag false. The enabled
request, independent final review and Supervisor assignment are absent.
The CLI exits HOLD before marker or contact under normal and optimized
Python. No SSH/EC2/AWS contact, stage, checkout, marker, cleanup, native run,
evaluation, breaker reset or submission occurred in this review.

## Resolved r1 findings

- The first full activity check and two broad label scans are now followed
  by another full STOP/process/Neuron check, a root byte/inode check, direct
  STOP and target-absence checks, and finite ≤1-second evidence ages. The
  actual `O_EXCL` **return** revalidates those ages and the open-call time.
  A third broad label scan, full activity/STOP check and root-space sample
  run after open before payload; a named/opened same-empty-inode check then
  precedes `os.write`. An ordinary persistent STOP, label, training/Neuron
  activity or disk-floor drop injected after the former final helper now
  returns HOLD with an empty provisional bundle, not a checkout. A 1.05 s
  helper-to-open delay also refuses. This closes the r1 witnesses in the
  tested cooperative model without relaxing the one-second criterion.
- The pre-write floor remains 13,177,973,911 bytes and 10,000 inodes, plus
  the 6.5-GiB hard floor. Finite pre/post-open ages and both root-space
  records are checked again by the local wrapper before readback. There is
  no `unlink`, `remove`, `rmdir`, `rename`, `replace`, or automatic retry in
  the host writer. A failed post-open gate retains even an empty file; an
  interrupted payload retains its nonempty partial. Neither creates a
  checkout or yields a write success receipt. The marker remains consumed
  if a later authorized stage entered it; cleanup/retry is human-reserved.
- The wrapper's postflight is now accurately described and recorded as an
  **attempt**. A new local STOP can refuse `_ssh_phase("postflight")` before
  SSH; this records `POSTFLIGHT_HOLD` rather than claiming a completed full
  probe. Only an actual full postflight result with the expected phase and
  scope and source/protected identity can support stage evidence. Failed
  postflight is terminal HOLD, including after a partial write.

## Exact pins and offline checks

| Item | SHA-256 |
| --- | --- |
| `REQUEST.HOLD.json` | `7d8d0bda4269f0863b828c1a284c2046adca1e4ba139a006cb786cbe954fa78e` |
| `RUNBOOK.md` | `9d3fe52171a6222303dc846afe56891852a4784c3240d461fa8b82b746685a13` |
| `SHA256SUMS` | `ff1f035d73ac4dff72d0d3545570ecf720b7e9d6ecff2588c26bb0c9f756e37a` |
| `remote_incremental.py` | `47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b` |
| `operator_once.py` | `cdfdd5c731a73dd190268108a6234579f7547379006d51f1bcac5ef160cdacf7` |
| `test_final_boundary_r2.py` | `e3ff8f1057dfd83ea88533a86662c5240025059deec12d20bbf6281a85876c91` |
| `test_incremental_stage.py` | `ccf51f57599f69ee49751185a578ae8f94dbd3ac758d0f452db1726d912a4506` |
| independent witness | `eeef562c08cd7cfd6cb772096ee50bab26480f3a08ad282004db14a0eeebdc36` |
| `e65-to-r10.bundle` | `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |

`sha256sum -c SHA256SUMS` passes for all seven listed package files.
The bundle is 109,800 bytes and `git bundle verify` passes with r10
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1` as its only HEAD and e65
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` as its sole prerequisite.
The e65→r10 range has 15 commits. Independent `git show` SHA-256 checks
confirm both commits contain identical `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
the r10 launcher is
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
The cited separate r10 launcher-source PASS receipt hashes to its bound
`5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`.
The subject adds eight r2 authorization-package files only, and
`git diff --check c03e3ac3 3e846589` passes.

Implementer suite: **26/26 PASS** normal and **26/26 PASS** under `-O`.
Independent `test_20260925_trn21_r10_incremental_stage_r2_independent.py`:
**5/5 PASS** normal and **5/5 PASS** under `-O`, with subcases covering the
five r1 gap events after the final helper and four event types after open,
nonempty interrupted payload retention, and local-STOP postflight refusal.
All use disposable local paths and synthetic sentinels. A fifth witness
deliberately reproduces a documented **noncooperating** residual: a label
created after the third/last broad scan can coexist with a write-phase
bundle and checkout before the later readback notices it. Its test PASS
means that limitation reproduced, **not** that staging is race-free.

## Remaining gates and limits

No finite userspace scan and final check are atomic against same-UID/root
interference or scheduler delay. The recorded one-second ages are sampled
at actual open return and before payload, **not** at payload completion.
The new full checks have not been timed on trn21; the strict live bound may
refuse. Source/private Git, ACL/group, host lock, STOP/process/Neuron,
checkpoint/cache and disk facts cited by the package are historical
point-in-time observations, not this review's live preflight.

Before any attempt, a separate final review must explicitly accept the
bounded residual, verify exact enabled request and Supervisor one-attempt
assignment, and obtain fresh host evidence. A successful stage requires
distinct-session readback, a **completed** full postflight, and independent
stage-result/G2 source-identity review. Even then it does not authorize
native A0. In addition, experiment-protocol §1 requires the same clean Git
HEAD on local and both hosts for a scored run. Trn21 r10 paired with trn22
r9 would not meet that condition; trn22 needs a separately reviewed r10
rebind/stage or an explicit human-approved protocol reconciliation. No new
BPB result follows from this static review.
