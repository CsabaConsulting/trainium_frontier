# Exact discarded-scored checkpoint cleanup

26 September 2026, 13:16 IST. Scope: Final Sprint standing authorization,
not new deletion authority. Sole Operator, after each affected host's current
P0 training AND attached fresh20M finish and its original lock is safely free.
Do not interrupt those jobs or perform concurrent cleanup under their lock.

Only these two regular checkpoint files may be removed, after fresh exact
path/hash/size/ownership and non-protection checks, under the original exclusive
host lock with fresh ps/neuron-ls idle checks. Preserve every log, JSON, verdict,
source and ALL caches. Any identity mismatch means no deletion of that target.

## trn21

`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-lr012-trn21-r1-packet/train/out/final.pt`

SHA256 `3d2a0663fa285c4d2be9f3214aee2ace58e07fd4311b98370f87c45f3ac80fbc`;
logical size453048594 bytes. Actual fresh20M0.9880655476948598, unchanged
selector DISCARD at
`research/v2/analysis/20260926-anchored-control-final/selector-clock36-matrix-lr012.json`.
This is LR.012, not the adopted LR.015/clock36 or audited e1e2.

## trn22

`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-lr012-trn22-r1-full-packet/train/out/final.pt`

SHA256 `4fc42f95dd6f597bb1b1f1c3854464a3db2088f1a5d151063d0dd28dd3356f26`;
logical size453048594 bytes. Archived ledger row is evaluated20M0.9883603019664198,
`NO_IMPROVEMENT_VS_PRESPECIFIED_PARENT`, `DO_NOT_ADOPT`, submission_eligible=false.
This already-scored unadopted LR.012 arm is discarded for artifact retention;
its historical verdict/score/comparator are NOT rewritten or reclassified into
a new selector result. It is not the earlier3e79 architecture parent checkpoint,
current clock36, audited e1e2, candidate or submission checkpoint.

Record current identity/protection evidence, actual allocated bytes and before/
after free space. Unlink only the two verified exact files, no recursive target
or unresolved glob. No verified backup is known; deletion is not assumed
recoverable. Preserve the receipt. This recovers up to906097188 logical bytes,
not enough to claim the25GB maintenance target achieved. It must not lower
the unchanged resource admission floor or authorize unscored-checkpoint/cache
cleanup. Reassess qualified next-job resources after the cleanup.
