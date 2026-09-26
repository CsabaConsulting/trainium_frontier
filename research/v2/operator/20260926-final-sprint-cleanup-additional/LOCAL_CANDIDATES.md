# Additional local cleanup leads — not a deletion receipt

No host inspection/deletion was performed. Current P0 and attached evaluations
remain protected. Historical logical sizes below are not freshly measured
allocation or guaranteed reclaim. Exact ownership/hash/non-symlink/protection,
idle-device and original-lock checks are mandatory before any future removal.

## Explicit-DISCARD checkpoint, trn21

Path:
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-lr012-trn21-r1-packet/train/out/final.pt`

SHA256 `3d2a0663fa285c4d2be9f3214aee2ace58e07fd4311b98370f87c45f3ac80fbc`;
recorded 453,048,594 bytes. Actual public-20M 0.9880655476948598, selector DISCARD
for ambiguous band/low-prior kind versus genuine control 0.9881207925.
Evidence: `research/v2/operator/20260926-trn21-clock36-lr012/completed/results-bundle.json`
and `research/v2/analysis/20260926-anchored-control-final/selector-clock36-matrix-lr012.json`.
This is the separate LR.012 arm, not adopted36 LR.015. Its caches are shared and
excluded from cleanup. No matching current state label/hash was found locally.

## Scored discarded checkpoint, trn22 — Supervisor classification received

Path:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-lr012-trn22-r1-full-packet/train/out/final.pt`

SHA256 `4fc42f95dd6f597bb1b1f1c3854464a3db2088f1a5d151063d0dd28dd3356f26`;
recorded 453,048,594 bytes. Public-20M 0.9883603019664198, about +0.0020199967
versus its prespecified parent. Ledger verdict is NO_IMPROVEMENT_VS_PRESPECIFIED_PARENT,
not literal DISCARD; independent review says not to promote and lacks formal selector
eligibility. The Supervisor has now explicitly classified this already-scored
NO_IMPROVEMENT / DO_NOT_ADOPT arm as discarded under the standing cleanup scope.
The ledger is not relabelled. Shared caches and the original 0.986340 checkpoint
stay excluded.
Evidence: `research/v2/operator/20260926-trn22-arch3e79-lr012/completed-eval/post-eval-checkpoint.json`,
`research/v2/verification/20260926-overnight-terminals/REVIEW.md`, ledgerline18.

No third sufficiently supported candidate found in the bounded local pass.

## Narrow assignment received, no deletion performed

`research/v2/assignments/20260926-scored-lr-checkpoint-cleanup.md` authorizes
only these two exact checkpoint files, after each host's protected P0 training
AND attached public-20M evaluation finish, artifacts are retrieved and explicit
quiescence is established. Fresh original-lock, idle/device, exact regular-file
identity/hash and protection checks remain mandatory. All caches, logs, sources
and verdicts remain protected. Actual reclaimed allocation and available bytes
must be recorded. This local note does not assert either file still exists.
