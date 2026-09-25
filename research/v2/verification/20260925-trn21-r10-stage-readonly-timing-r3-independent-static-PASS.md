# Independent trn21 r10 read-only timing r3 review — static PASS, dispatch HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
disabled subject: `3aa258d31d62b5f59f2dfcf34149b63f8eabd453` on
`origin/codex/20260925-trn21-r10-stage-readonly-timing-r3`. **PASS for the
disabled read-only source only**: it closes the contradictory-chronology
positive-result witness in independent r2 HOLD `ca3578e7`. This is not an
enabled timing request, host timing observation, source-stage authority,
native run or BPB result. No SSH/host/EC2 contact, operational receipt,
marker, transfer, checkout, cleanup, training, evaluation, AWS action or
submission occurred in this review. Synthetic mocked receipts existed only
in disposable local test directories.

The r3 parser retains r2's empty-stderr, exact start identity (host, UID,
source, target, bundle, label, lock and commits), boot/mount, disk/inode,
canonical ordered JSONL, finite numeric range, and terminal-count gates.
It now requires each positive trial's
`preopen_seconds + 1e-6 >= label_scan_duration_seconds + label_age_seconds`
and `label_age_seconds + 1e-6 >= activity_age_seconds`,
`activity_age_seconds + 1e-6 >= space_age_seconds`. The pinned remote timing
source samples these quantities in that order within one trial; one
microsecond covers float subtraction roundoff and is negligible against
the strict one-second freshness bound. These relations are checked before
any parsed result can become a positive timing observation.

I independently supplied canonical, full-identity, empty-stderr ordered
start/three-trial/end JSONL. The former r2 witness (`preopen=0.0`, scan
`0.4`, label/activity/space ages `0.3/0.2/0.1` seconds) now raises a
chronology error and fully mocked `capture()` archives HOLD. Forged
label-before-activity and activity-before-space ages, NaN, wrong hostname,
wrong terminal count and nonempty stderr also archived HOLD; a valid
record archived only `READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY`.
A 0.5-microsecond boundary discrepancy was accepted, while a 2-microsecond
discrepancy was rejected, as declared.

| Exact package item | SHA-256 |
| --- | --- |
| `collect_readonly.py` | `64cd69560d36aad7515437893792af8e567df82688a59c30509b29798b671cd5` |
| unchanged `remote_timing.py` | `5a1298febfb67f6a61eb1f6fa02f974c7cbe15c6eb0bc221338b85d892be0db7` |
| unchanged `REQUEST.HOLD.json` | `9f2ed81e1c04decc16e0747d6e71f52037da759241ea1c8774dbbf790604d897` |
| `SHA256SUMS` | `416d0e97a45063fe6e06ed8e608dcfa61aa549f40df15e4814b2cedadc9374ba` |
| `RUNBOOK.md` | `dd997c03139c531364cdceeaeb53b35706c3a30a58abfc7dadf5fda0ceb96a43` |
| `test_readonly.py` | `dbe73f1b68228d3d06f76477096430fc4b84f1c45920a25ba673cfd75a510868` |

`sha256sum -c SHA256SUMS` passed **5/5**. `--check` verified the unchanged
stage r3 remote/operator SHA-256 values
`47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b`
and `6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f`,
plus independent stage receipt SHA-256
`ab0a63e47ece6d9055167b62679b40bee949b1d8e1e948195967960a77bf859e`.
Offline tests passed **11/11** normal and **11/11** under `-O`;
`git diff --check` passed. Default and `--capture` entrypoints in both
modes returned nonzero HOLD at absent separate enablement before SSH or
operational output. The remote timing source and stage writer are unchanged;
this probe creates no host file or one-shot marker.

Remaining gates: a different role must prepare the exact read-only enabled
request, independent final timing review and Supervisor assignment, followed
by fresh local/host checks. The three-trial clock-only probe has not run on
trn21; it cannot measure actual `O_EXCL` syscall/scheduler latency, payload
or post-open behavior. Stage/source and native authorization remain
separate. Any eventual execution must use the exact clean reviewed c5c
package worktree (`train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`),
not the archival Desktop controller checkout with different science bytes.
