# Exact human-approved checkpoint cleanup

Completed 2026-09-26 06:00:21.993257 UTC on trn21, under original exclusive
lock device66305/inode596823. Human authorization is the exact three-file
approval recorded at
`research/v2/verification/20260926-trn21-three-discarded-checkpoints-cleanup-authorization.md`,
SHA-256 `3cf0200e455ee58b57433e02af6784f4a98b0524c013a178b1cb93a55813b2c2`.

Only these three discarded `final.pt` files were unlinked:

1. `/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt`
2. `/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/fs-20260925-fp-c5c-matrixlr0175-trn21-r1/out/final.pt`
3. `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt`

All three exact SHA-256 values, archived inode/device/size/owner/timestamps,
single-link regular-file type and unchanged open-file identities were verified
before the first unlink. Every path component was opened without following
symlinks. Exact parent descriptors were retained for the three file operations.
Fresh ps, neuron-ls and device-holder checks passed before and after cleanup.
The same original lock remained held through the measured absence and disk checks.

- Before: 13,067,206,656 bytes free.
- After: 14,746,214,400 bytes free.
- Reclaimed: **1,679,007,744 bytes**, matching the three files' allocated space.
- All three exact paths were absent afterward; no error or partial failure.

No directory, cache, log, source, result, best/current/candidate/submission
checkpoint or AWS resource was changed. No training or diagnostic was launched.
These unlinks are permanent unless a separate existing backup is later verified;
no backup is claimed and no duplicate checkpoint was created.

This measured free space exceeds 13.5 GiB only narrowly; it does not meet the
16 GiB K2 gate. Future work still requires its own fresh resource checks and
launch assignment. The one-use cleanup script is consumed and must not be rerun.
