# Human-approved three-checkpoint cleanup — 26 September 2026

Recorded 05:57 UTC / 11:27 IST. Poojith answered the exact pending question:
"Yes—delete those three discarded checkpoints" (question call_Hw6EfImkw8GwNGOswUh8t4BO).
This is approval only for the three files below. No cache, log, result, source,
best/current/candidate/submission checkpoint, directory, or AWS action is included.

Sole Operator: /root/trn22_queue_operator, trn21 only. Complete the already
assigned read-only trn22 observation first; preserve its protected job. Then
take original trn21 exclusive lock device66305/inode596823, require fresh
successful ps plus neuron-ls/device-holder idle checks, and verify all three
exact regular files before deleting any. Reject symlinks or changed ownership,
identity, bytes or archive hash; no recursive or wildcard deletion. Retain the
lock through cleanup and the measured post-cleanup disk/absence checks.

| Exact file | Logical bytes | Expected SHA-256 |
| --- | ---: | --- |
| `/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt` | 739322371 | `6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b` |
| `/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/fs-20260925-fp-c5c-matrixlr0175-trn21-r1/out/final.pt` | 469834849 | `f722109658e944dc6fcb27fc0e838759f07733e4b116d1bf98e8ce1e61870187` |
| `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt` | 469834913 | `edd9adca6f6f782f7ccb220f567ffed7177291a65f3e28ec269b38403e1a47ec` |

Evidence: operator/20260926-trn21-discarded-checkpoint-extension/INVENTORY.md
and inventory.json; operator/20260926-trn21-discarded-scratch-inventory/;
bcbd archived completed/results-bundle.json checkpoint.json record. All three
have archived DISCARD verdicts. Only the first two were recently content-hashed;
the third must also match its archived completed-run checkpoint hash now.

Expected allocation released is 1,679,007,744 bytes, not a guarantee. Archive
the before/after identities, independently streamed hashes, exact operations,
timestamps, errors and actual free space. If partial failure occurs, preserve
the exact outcome and stop; do not broaden targets or retry blindly. Deletion
is permanent unless another existing backup is independently established;
do not make a duplicate checkpoint that consumes the intended reclaimed space.

No training/native launch or threshold reduction is authorized by this file.
Post-cleanup free space decides resource readiness separately. Estimated
13.73 GiB may clear native-consistency/CPU checks but not K2's 16 GiB floor.
