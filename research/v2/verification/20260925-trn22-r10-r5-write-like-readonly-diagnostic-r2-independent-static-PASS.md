# Independent static review: trn22 r5 write-like read-only diagnostic r2

**Verdict: STATIC PASS — DISABLED SOURCE ONLY.** This is not an enabled
request, final dispatch review, Supervisor assignment, host-contact approval,
source-stage retry approval, training/evaluation approval, or cleanup approval.
No SSH or other host contact was made in this review. The cause of the
historical r5 write-phase failure remains **unknown**.

## Exact reviewed identity

- Implementer commit: `e9ef3115c61aaac6740db4e84069645533f6bfa9` on
  `codex/20260925-trn22-r10-write-like-readonly-diagnostic-r2-disabled`.
- Package:
  `research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r2-hold/`.
- `operator_readonly.py`: SHA-256
  `c1c740c22d6127b032b10763c5e5c3202abb199d99be5c0cdc3c846d663e6e18`.
- `remote_readonly.py`: SHA-256
  `e9cd37be06d644d99b6be1cc4ee73541a66d028998fd172fe03813fb51ac4ada`,
  byte-identical to the r1 diagnostic remote.
- Canonical `REQUEST.DISABLED.json`: SHA-256
  `9a7bfa79a5d8ab18fcfe8e50b99023476c46670f768231abfebf24ad02090e92`.
  It has `dispatch_authorized=false` and false source-stage, native, eval,
  AWS, cleanup, and marker-reset authorities. Its r2 label/schema cannot
  reuse an r1 request or an r5 source-stage assignment.
- Exact r5 stage remote: SHA-256
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
  exact incremental bundle: 109,800 bytes, SHA-256
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  The pinned e65 source is `e65a3097a5b00f6178188a935bb2acb9a5042c6c`;
  the intended r10 target is `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`.
  Root science SHA-256s remain `train.py`
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py`
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Prior r1 independent HOLD: commit
  `d8e70c2b848c88573d09432953d73b3090bb6d0a`, receipt SHA-256
  `af64f1bd289ad757f455ce927251944637efecdaf3195a0343bfe219f61e9968`.

## Independent offline checks

- `sha256sum -c SHA256SUMS`: **13/13 passed**, including the prior HOLD,
  consumed-stage evidence, exact r5 remote/bundle, and science files. The
  reviewed source worktree was clean at the pinned commit.
- Normal and `-O` unit suites: **20/20 passed** in each mode. The generated
  remote Python compiled. Direct normal and `-O` entrypoints returned
  `HOLD_PRE_DISPATCH` (exit 1), with no enabled request and before any SSH.
- The r1 blocking defect is corrected. `operator_readonly.py` now makes a
  separate pinned, read-only SSH to
  `/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version` before **each** timed
  phase. The probe is bounded to 25 seconds and rejects nonzero return,
  nonempty stderr, oversized stdout, or a non-GNU/malformed first line.
  Local active-Desktop/former-Documents/package STOP and breaker checks run
  before the probe, after it, and immediately before the phase SSH. The
  existing in-child check remains defense-in-depth.
- Independently mocked `_ssh_phase` observed exactly the separate version
  probe followed by the outer-timeout phase. A STOP injected after the probe
  caused a HOLD after **one probe SSH and zero phase SSH**. This witness
  performed no network call. The package tests additionally exercise bad
  version, stderr, timeout, and intervening STOP refusal.
- The `write_like` branch invokes only `remote_readonly.perform_readonly`,
  never r5 `perform("write")`, `_write_bundle`, or `_reconstruct`. The remote
  diagnostic uses the existing read-only flock, exact full probe/stability,
  source/protected/label/activity/disk and bundle-stdin identity checks,
  `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` on the root directory, and unchanged
  one-second pre-open freshness checks. No new `O_CREAT`, `O_EXCL`, truncate,
  unlink, rename, checkout, or marker-write path is reachable. The consumed
  r5 marker is checked by hash, not reset.
- Published results contain bounded typed statuses, finite whitelisted
  timing/space fields and byte counts/SHA-256 of streams, not raw SSH
  stdout/stderr, exception messages, key bytes, or environment contents.

## Remaining boundary

This diagnostic can characterize a **current** no-write pre-open path. It
cannot prove an actual `O_EXCL` succeeds or identify why the historical r5
write SSH returned a generic failure. It is still disabled. Any future host
contact requires a distinct exact enabled diagnostic request, independent
final request-bound review, Supervisor assignment, and fresh STOP/breaker,
source/bundle/science, host identity, protected-artifact, disk, and
one-run-per-host checks. This static PASS is not a source-stage retry verdict.
