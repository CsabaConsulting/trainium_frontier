# trn22 e65 → r10 direct incremental source stage r1, package correction r2 — DISABLED / HOLD

The original disabled package commit `3d26e963df349f444cd560114400788bf572b062`
remains in history but had a mistyped SHA-256 pin for the inherited r4 static
PASS receipt. This r2 package correction binds the actual receipt bytes
(`5f1b18e45fda3e726ccde6063523ee2679f3a8dbc42480cc1e84936d1b78bb1d`)
without changing the remote stage source, unused host label, bundle, target,
one-attempt marker or any launch gate. Neither revision has contacted trn22.
The review hash is reproducible offline from
`git show efd5f2396b8bb2c37865d3cb751f45ed2c07bb81:research/v2/verification/20260925-trn22-r9-incremental-stage-r4-independent-static-PASS.md | sha256sum`.

This is an Implementer-only, distinct **disabled** rebinding of the reviewed trn22 r4 no-cleanup source-stage design. It exists because a trn22 r9 checkout and a planned trn21 r10 checkout would not meet `research/experiment-protocol.md` §1's same-Git-HEAD requirement for comparable scored runs. **No source stage or host contact has occurred through this package.** The request has `dispatch_authorized: false` and `source_stage_authorized: false`; no enabled request, independent final review or Supervisor assignment is present. Neither this file nor the r4 static PASS authorizes SSH, a host marker, cleanup, native training, evaluation or submission.

## Exact immutable subject

| Item | Pin |
| --- | --- |
| Accepted trn22 source | linked e65 checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2` at `e65a3097a5b00f6178188a935bb2acb9a5042c6c` |
| New, distinct proposed trn22 target | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1` at `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` |
| Remote bundle name | `/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn22-stage-r1.bundle` |
| Checked-in direct e65→r10 bundle | `e65-to-r10.bundle`, 109,800 bytes, SHA-256 `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Bundle graph | sole prerequisite e65, HEAD r10, 15 commits e65..r10; `git bundle verify` passes locally |
| One-shot label | `fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1` |
| One-shot local marker | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r10-incremental-stage-r1.one-attempt.jsonl` |
| Unchanged train / evaluator | `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| r10 launcher | `research/fresh_start_launch.py` SHA-256 `1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7` |

The target r10 commit descends from r9 `e046de16826bfc3b954136e3c60804aea0554ecc`. Its launcher source has an independent **disabled-source-only** PASS at `research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R10_TRN21.md` (SHA-256 `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`, review commit `ee9a4b14cc30ab372162fd434692e59b13ac44ab`). That review found the trn22 r9 profile retained, but it grants no native trn22 authority. The inherited r4 trn22 stage design is source-reviewed at `research/v2/verification/20260925-trn22-r9-incremental-stage-r4-independent-static-PASS.md` (SHA-256 `5f1b18e45fda3e726ccde6063523ee2679f3a8dbc42480cc1e84936d1b78bb1d`, review commit `efd5f2396b8bb2c37865d3cb751f45ed2c07bb81`, exact r4 subject `19fbfdc88763b6b5762e219339dcd156e7977a30`). That PASS is **static and r4-specific**; it does not review or authorize these new r10 bytes. The older r3 independent HOLD remains pinned and historical; it is not an authority source.

The local bundle and all source files are listed in `SHA256SUMS`. Bundle bytes are hash/length checked before a future marker or host action. No `.env`, private key, token or credential bytes were copied into this package. `train.py`, `prepare.py`, data and evaluator paths are unmodified.

## Preserved gates, exact failure policy

The proposed source remains the observed group-writable `0775` linked e65 checkout with a `0664` `.git` pointer. The inherited code refuses if its inode, mode, owner, Gitdir ancestry, clean e65 HEAD, science hashes, ordinary-DAC ACL/group condition, ext4 mount, cooperative lock, protected checkpoint/caches, STOP/process/Neuron/label state or root disk/inode floor changes. It checks all old trn22 r9 r1–r4 one-attempt markers before its own marker; this is deliberately conservative because a prior r9 stage may have left partials requiring human resolution. Unknown or changed live facts HOLD. These are source assertions, **not new live trn22 observations**.

The unchanged floor is 13,177,973,911 free root bytes and 10,000 free inodes before material write, 10,737,418,240 bytes after reconstruction, plus the hard 6.5-GiB root floor. The wrapper requires a distinct fresh preflight, hash-checked e65→r10 transfer, cross-session readback and a **completed** postflight receipt before success. It attempts postflight even after failure; a newly appearing local STOP may block the postflight connection, in which case the state remains HOLD. Source-stage success would only produce evidence for an independent stage-result and G2 carry-forward review, never native run authorization.

The r4 no-unlink boundary is carried forward: two broad final label scans followed by process/Neuron, STOP and root-space checks; finite ≤1-second evidence at actual `O_EXCL` return; then a third broad label scan, process/Neuron, STOP and root-space checks before payload. A failed post-open gate retains even an empty provisional bundle; any partial or unexpected replacement remains intact, the one-attempt marker stays consumed, no checkout is created and no successful write receipt is returned. There is **no automatic unlink, rename, retry or cleanup**. Durability uncertainty is HOLD. Cleanup or retry requires a separate human decision.

No finite userspace scan is atomic against a same-UID/root actor or alias path. The r4 static review explicitly left two residuals: a label introduced after the last broad scan may race into a write, and payload completion can occur after the numeric `O_EXCL` freshness window. Actual timing of this r10-bound sequence on trn22 is **unknown**; a static test does not prove its ≤1-second live feasibility.

## Independent gates before any host action

1. A different Verifier reviews this **exact r10 package** and its bundle, normal and `-O` tests, target file hashes, no-cleanup policy, stale-evidence behavior and residual risks. Record an exact disabled static verdict. Do not reuse the r4 verdict as if it covered r10.
2. Obtain separately authorized read-only live r10 timing evidence and its independent review. No timing attempt follows from this disabled file.
3. If proceeding, create the absent `REQUEST.ENABLED_PENDING_FINAL_REVIEW.json` with the exact schema and fields checked by `operator_once.py::_authority`; independently review that final request/source; then obtain `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json` and `SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json`. The checked-in HOLD request is **not** an enabling artifact.
4. The Operator must freshly check host identity, source/target absence, source and protected assets, STOP/breaker/lock/process/Neuron/label state, byte/inode floors, known-host identity and one-attempt marker absence before any stage. The marker is consumed once; failures are retained and reported, not cleaned or retried.
5. Independently review any resulting stage receipt and G2 source identity. Confirm the same exact r10 Git HEAD on local, trn21 and trn22 before treating subsequent controls as comparable. A separate native A0 request, monitors, fresh preflight and one-shot authority are still required.

This source-only rebinding does **not** resolve the separate A2 CPU-G2 tool-path problem: e65/r9/r10 trees do not contain `workflow/tools/preflight_dryrun.py` or `workflow/tools/contained.py`. No such tools were silently added to the target, and A2 remains separately gated.

## Offline-only checks

From the repository root, with no host access:

```text
sha256sum -c research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/SHA256SUMS
git bundle verify research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/e65-to-r10.bundle
python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold -p 'test_*.py' -q
python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold -p 'test_*.py' -q
python3 -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
python3 -O -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
```

Both entrypoint commands must refuse with `HOLD` before SSH or a marker.
The Implementer suite passed **36/36** under normal Python and **36/36** under `-O` in disposable local fixtures. These are not independent verification, live host feasibility, or a BPB result.
