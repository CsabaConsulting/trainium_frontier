# Read-only discarded-artifact inventory — no deletion authority

Observed 2026-09-26 04:57:37 UTC, trn21, under original lock66305:596823 with fresh empty process/device checks. Only three exact previously bound paths were inspected; 131 metadata entries, no content reads, symlinks or mutations. All belong to uid/gid1000. Best/current/candidate checkpoints, source/results, P traces, and shared research train/eval caches were excluded.

All paths below begin with `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/`.

| Exact relative target | Allocated bytes | Terminal/disposition provenance |
| --- | ---: | --- |
| `fs-20260926-k2-symmetric-consumer-trn21-r1-packet/caches` | 692,224 | Four-rank `EXACTNESS_FALSIFIED: unique/fp32`, child exit1 at00:09:18.565322; parked/no retry. `research/v2/operator/20260926-trn21-k2-symmetric-native/completed/OPERATOR_RESULT.md`, bundle SHA16dc0f7d84eed33eaaa821cb6afd21d00fc5b6d60b4ad1b9f93e55f3e07b9c7b. |
| `fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches` | 380,928 | `DIAGNOSTIC_CHILD_FAILED`, exit1 at02:21:42.218705; outer exit1 at02:21:42.224650. `research/v2/operator/20260926-trn21-ordinary-subnormal/completed/results-bundle.json`, SHAb91f230736f522d20174749b2792da1e3b34c9c842c3cf82889c41ec22152c0f. |
| `fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt` | 469,839,872 | Completed quality run public20M0.9906857009017018, explicit `DISCARD` in ledger and DECISIONS. Outer exit0 at01:04:12.730857. `research/v2/operator/20260926-trn21-basebcbd-full/completed/results-bundle.json`, SHAabb30ec0f23fd9d5c792113b6f8b3d1ac9627d8163cf6cf3dc3834e630c8c328. |

Each target's before/after device/inode/mode/uid/gid/mtime/ctime and full bounded size inventory are in `inventory.json`. This is selection evidence for a future scoped decision, not permission to delete. Checkpoint content was not rehashed in this metadata-only check.

Free disk: **13,067,276,288 bytes** (12.17GiB). The three targets total only **470,913,024 allocated bytes** (0.439GiB). Even if separately authorized and actually freed, they would **not** reach either pending gate:

- 13.5GiB gate presently lacks1,428,238,336 bytes; after these three it would still lack957,325,312 bytes.
- 16GiB gate presently lacks4,112,592,896 bytes.

No deletion is recommended as sufficient remediation from this limited inventory. Any broader inventory/cleanup needs exact additional targets and protected-artifact ownership review; no filesystem-wide scan was performed. trn22's running clean full-cold job was not inspected or changed by this inventory.
