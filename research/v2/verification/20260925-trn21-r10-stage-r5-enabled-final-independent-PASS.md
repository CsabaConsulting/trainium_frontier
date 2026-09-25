# Independent final PASS — trn21 r10 r5 source-stage request only

**Verdict: PASS_SOURCE_STAGE_ONLY** for the exact clean final-pending subject
`origin/codex/20260925-trn21-r10-stage-r5-enabled-final-pending@0ccfb3731f85728973b2d2dacd430804b1a59f16`.
The canonical machine-readable review is
`research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json`,
SHA-256 `d1170f0ac27285d4ca5670dda9828da2a7cc76a7527f2a6eb0b90239479b9ff3`.
This is an independent **request review**, not a Supervisor assignment or
proof of a completed stage. I did not contact trn21, create a one-attempt
marker, stage a bundle, train, evaluate, alter AWS or breakers, clean any
artifact, or submit.

## Bound subject and offline evidence

- The enabled request is exact canonical JSON, SHA-256
  `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a`.
  Its stage-only scope binds trn21 hostname, e65 source commit/path, r10
  target commit/path, unique r2 stage label and one-attempt marker, 109,800
  bundle bytes/hash, remote source hash, operator source hash, and the r10
  source-review receipt. `native_training_authorized` and
  `evaluation_authorized` are false; adding either authority bit fails the
  operator's exact-dictionary gate. The frozen review schema itself has no
  extra advisory fields. A hypothetical exact Supervisor assignment passed
  `_authority` **in memory only** when bound to this actual review hash;
  changed review request-hash or native-authority fields were rejected.
- The operator is unchanged from the independently static-PASSed disabled r5
  source: SHA-256
  `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b`.
  Remote source remains
  `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`;
  bundle remains
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `git bundle verify` confirms e65
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` as the only prerequisite
  and r10 `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` as HEAD.
  `train.py`/`prepare.py` are unchanged SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` /
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The pinned r10 source-review receipt at commit `ee9a4b14` was independently
  hash-checked as `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`.
- All 13 preexisting manifest entries verified; 41/41 package tests passed
  normally and 41/41 under optimized Python. `git diff --check` passed.
  Before this review file was added, final review, assignment, and marker
  were absent; the actual normal entrypoint exited HOLD for missing review.
  With this review file present but **no assignment**, the actual normal
  entrypoint exits HOLD for missing assignment and the optimized entrypoint
  rejects optimization. The marker remained absent at review time.

## Safety judgment and limits

The r4 final review **HOLD** at `06451f16d48026b9a167edf803ce57fdec787ce4`
(receipt SHA-256 `0b9200787ed6a1d79b9b056d7f3068329fd89cf15fbbc0bd4df96aa1f654a515`)
found a STOP/breaker-to-read-only-probe gap after local SSH identity. The
distinct r5 repair at `2db60693` received my disabled-source **static-only
PASS** at `739c5dac9e04dae56700e4d3c3d5a6493150cf89` (receipt SHA-256
`99e1fd9770171467c3679a396dff63d648f858513c0dcf1176eb0ad9257b7b2a`).
This enabled request binds exactly those r5 operator bytes. Both
`stage_once` and `_ssh_phase` recheck active Desktop STOP/breakers immediately
after bounded local identity, before the first possible remote timeout-probe
SSH, and again after that probe before marker or phase SSH. The probe checks
the **remote** absolute GNU `/usr/bin/timeout` through pinned SSH; the local
controller's uutils `timeout` is not used as a false gate. The independent
static witness saw zero r5 probe, marker or phase calls after STOP at this
seam; the held r4 source had made one probe call.

The unchanged r4 remote writer retains one broad final pre-open label scan,
full activity/Neuron/STOP and byte/inode checks, a finite ≤1-second freshness
test at the actual exclusive-open return, and a post-open broad scan plus
activity/space/same-empty-inode checks **before payload**. The local receipt
rejects absent, nonfinite or contradictory proof. This is bounded to the
documented cooperative-lock/ordinary-DAC model: a persistent duplicate
arriving behind the final pre-open scan can create a retained **empty**
provisional bundle and consume the one-shot marker; a noncooperating
same-UID/privileged actor after the last post-open scan remains a residual.
No cleanup, unlink or retry follows from this PASS. The r4 one-scan **live**
duration is unmeasured; the older two-scan timing failed at 1.161466714 s.
The later Operator attempt may therefore fail closed and consume its marker.

The separate Supervisor one-attempt assignment, fresh local/host preflight,
retrieved raw stage evidence, independent stage-result review, G2
source-identity carry-forward and distinct native A0 authority are still
required. This PASS does **not** authorize training, evaluation, or any BPB
claim.
