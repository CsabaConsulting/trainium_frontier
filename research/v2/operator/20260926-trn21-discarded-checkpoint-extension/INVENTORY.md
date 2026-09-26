# Two-file read-only extension — no cleanup authorization

At2026-09-26 05:10:17 UTC, both explicitly assigned archived DISCARD files were rehashed successfully on idle trn21 under original66305:596823 lock and fresh process/device checks. Total4.39seconds, exactly two files,64KiB streaming buffer, no checkpoint transferred, no host writes or shared-cache inspection.

| Exact target | Logical bytes | Allocated bytes | Stable identity and archive hash |
| --- | ---: | ---: | --- |
| `/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt` |739,322,371|739,328,000|dev66305/inode3176828,uid/gid1000,nlink1;SHA6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b|
| `/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/fs-20260925-fp-c5c-matrixlr0175-trn21-r1/out/final.pt` |469,834,849|469,839,872|dev66305/inode3178506,uid/gid1000,nlink1;SHAf722109658e944dc6fcb27fc0e838759f07733e4b116d1bf98e8ce1e61870187|

No symlink exists in either path. All identity fields, including full integer mtime/ctime nanoseconds, were unchanged before/open/after hash. Provenance is pinned in `inventory.json`: A1 negative-result report and LR0.0175 independent result review both explicitly say DISCARD. These are not current/best/candidate checkpoints. Logs and source remain preserved.

Current free bytes:13,067,255,808. The two files together allocate1,209,167,872 bytes; alone that is not enough for13.5GiB.

Combined with the previously inventoried explicit-DISCARD bcbd checkpoint at `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt` (allocated469,839,872 bytes; see prior `20260926-trn21-discarded-scratch-inventory` receipt), the three checkpoints total1,679,007,744 allocated bytes. If separately authorized and actually released, estimated free space would be14,746,263,552 bytes:250,748,928 bytes above13.5GiB, but still2,433,605,632 bytes below16GiB. Fresh post-cleanup space must decide, not this estimate.

**No files were deleted.** Human cleanup approval and exact-target revalidation remain separate. Nothing here authorizes a launch, threshold reduction, broader scan or deletion of caches, logs, source, P traces, or protected artifacts.
