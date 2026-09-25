# Independent source-stage review — trn21 e65 → r10 — HOLD

**Verdict:** `HOLD_SOURCE_STAGE_ONLY`. Do not create an enabled request or
one-attempt assignment for this frozen package. This is a static/offline
review, not a host preflight, transfer, native-run approval, or result review.

## Exact reviewed subject

- Implementer commit: `c03e3ac385d9ff4384bba1f1ca71ffa2486cad7c`
  (`codex/20260925-trn21-r10-incremental-source-stage-hold`). Scope is the six
  files under `research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-hold/`.
- Remote writer SHA-256:
  `eb1b1d9150dc35d6c96c00ad49c321bc9944c0cbaeef0b1494272346eb78c9f0`.
  Operator SHA-256:
  `8ed2266c20774c5f4f1f750bbe9f94c8d68a8e8459670308ef689c8e714901c1`.
  Bundle SHA-256:
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
- `git bundle verify` passed: the only prerequisite is accepted private e65
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, and the only HEAD is
  reviewed r10 `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. The e65→r10
  range has 141 Git objects. `train.py` and `prepare.py` are unchanged from
  e65; their r10 SHA-256 values match the writer's pins
  (`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`).
  The r10 launcher SHA-256 is the pinned
  `1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
- The cited independent *r10 launcher-source* receipt at commit
  `ee9a4b14cc30ab372162fd434692e59b13ac44ab` hashes to its pinned
  `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`.
  It is not a review of this stage writer.
- The exact request is `REQUEST.HOLD.json` with `dispatch_authorized: false`;
  the fixed enabled request, independent stage PASS, and Supervisor assignment
  files are absent. Normal entry therefore exits before marker or SSH. The
  trn21 private `0700` e65 source, separate new r10 target, `0700` Git directory,
  checkpoint/cache pins, nonblocking existing host lock, and 6.5-GiB/10,000-inode
  minima are represented in the writer and existing offline fixtures. This
  review did not establish their live state.

## Blocking finding 1 — pre-write continuity and age can expire before O_EXCL

The package's own `RUNBOOK.md` lines 98–118 requires final activity, two label
scans, direct STOP/target checks, and activity/label evidence no older than one
second **at the first write**. The explicit one-second limit is a c03 package
contract (`remote_incremental.py` lines 41, 446–468); it is not inferred from
the top-level competition rules.

The writer runs `_final_no_write_continuity()` at line 561, then immediately
uses its returned receipt and calls the first bundle `O_EXCL` at lines 562–563.
The age is measured *inside* the helper, and there is no subsequent STOP,
label, process/Neuron, disk/inode, or monotonic-age recheck before the write.
The last disk check is earlier in `_probe()` at lines 485–496; the subsequent
final activity and label scans can be slow.

The independent disposable-directory witness
`test_20260925_trn21_r10_incremental_stage_boundary.py` injects each event
**after the original final helper returns but before `_write_bundle` resumes**.
Current code still creates both the bundle and target when: (a) a STOP appears,
(b) the attempt label appears, (c) a train/Neuron activity sentinel becomes
active, (d) simulated free bytes/inodes fall below the hard floor after a good
precheck, and (e) 1.10 seconds elapse before the first `O_EXCL`. These are
observed behavior tests, not claims of live-host occurrence. The 1.10-second
case proves the returned `activity_age_seconds` and `label_age_seconds` can
both be ≤1 while the evidence is already >1 second old at the first write.
This gap is **before** the syscall, not the irreducible race with a
noncooperating actor after an immediately adjacent final check.

Repair must close the interval between the final check and first exclusive
open to the declared bound, revalidate STOP/label/activity and the hard disk
floor at a defensible last no-write point, and carry enough timing evidence
to prove the claimed age at that point. Then obtain new independent review;
do not relax the numerical gate or call this source dispatch-ready.

## Blocking finding 2 — “unconditional full postflight” is not guaranteed

`operator_once.py` lines 314–322 attempts `remote_call("postflight")` in a
`finally` block, but the production `_ssh_phase()` first runs
`_local_stop_and_breakers()` at lines 202–204. A local STOP arising after a
partial write causes that call to raise before any SSH or remote full probe.
The sixth independent test demonstrates this without contacting a host. The
marker would record `POSTFLIGHT_HOLD`, not an unconditional full postflight,
contrary to `RUNBOOK.md` line 95 and `remote_incremental.py` lines 594–599.
Resolve the precedence explicitly without weakening STOP safety; if a STOP
necessarily prevents read-only postflight, state that limitation truthfully
and preserve the partial-stage HOLD evidence.

## Offline checks and limits

- Frozen subject suite: **17/17 PASS** under normal Python and **17/17 PASS**
  under `-O`. These positive tests do not cover the post-helper interval.
- Independent boundary witnesses: **6/6 PASS** under normal Python and
  **6/6 PASS** under `-O`. “PASS” here means each negative witness reproduced
  the blocking behavior, **not** that staging is safe.
- `git bundle verify`, checked-in SHA-256 pins, source-review receipt digest,
  science-file identity, and launcher digest checks passed. No remote machine
  was contacted; no marker, host file, checkout, training, evaluation, AWS
  action, or leaderboard submission was performed. No private SSH key or
  `.env` was accessed or printed.
- The synthetic filesystem and patched activity/disk sentinels establish a
  source-level control-flow defect, not real trn21 timing, host quiescence,
  disk state, or security against a privileged/same-UID adversary.

**Next gate:** a different Implementer may revise this disabled package; the
revised exact bytes need another independent stage review and the separately
required authority chain. This HOLD does not negate the separate r10
launcher-source PASS or authorize a native A0 job.
