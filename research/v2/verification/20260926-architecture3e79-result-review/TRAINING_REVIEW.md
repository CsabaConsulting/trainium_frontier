# Architecture 3e79 full-run training and eval-handoff audit

**PASS for training-result authentication and recorded through-exit budget compliance. Quality remains PENDING_FRESH20M.** No score or selector quality verdict is invented before the terminal evaluator archive. This audit makes no host calls, tensor allocations, source changes or ledger edits.

Exact raw commit `8d9ee4b03f0042fc633a96b90637887e0d864490`, canonical import `55da2532`, contains eight retrieved artifacts under `research/v2/operator/20260926-trn22-arch3e79-full/completed-training/`. All eight match the retrieval SHA256/size records and canonical import byte-for-byte. The complete log, all 3,629 observer records and append-only step lines agree on consecutive boundaries 0 through 3628.

Source SHA256 `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` matches committed `b4d83b5d`. Exact driver, unmodified prepare, pinned Sep24 harness, guard and dispatcher hashes agree with binding `83db5fe3`. Actual isolated source/prepare, training argv and environment match the binding. This is the old-schedule architecture with `--unet-8x1024 --attention-layers 0,2,4,6,7 --initial-cache-refresh`, width/context1024, pack32, seed42, four ranks/LNC2, matrixLR0.015 and MAX1800. It does **not** enable the adopted Recursive schedule. The live launch archive independently shows ranks0–3/world4; torchrun/harness exits0 and the fresh eval guard observes no remaining training/device holder.

| Independent observed timing | Seconds |
| --- | ---: |
| Uncapped observed fifth boundary | 814.6809981259285 |
| Allowed exclusion, min(fifth,900) | 814.6809981259285 |
| Measured torchrun wall through process exit | 2603.889482168015 |
| Charged through exit | 1789.2084840420866 |
| Headroom under1800 | 10.791515957913361 |
| Last observed boundary to process exit, included above | 4.482617728994228 |

This uses the unchanged harness's monotonic launch-to-boundary/process-exit observations, not the source's self-estimate. Nominal50ms polling plus scheduling delay is not an exact device-completion timestamp. Harness time begins immediately before torchrun, after earlier sandbox/prepare setup. The same-driver UTC interval from `20:02:37.478643` to `20:46:04.317992` encloses that measured process wall. Source summaries remain separately labeled: training1784.7, startup allowance estimate812.5, charged estimate1785.1, `over_budget=False`. They are not substituted for external observations.

The run intentionally reused the exact-source cold diagnostic's SDK cache directories. `fresh_empty_sdk_caches={}` and no `--fresh-caches` flag confirm it is **warm research**, not a cold full replay. A separate cold-five receipt is not relabeled as this full run's cache state. The original unprimed93ae result also remains a distinct historical result, not erased or replaced by this audit.

The retained checkpoint is453048594bytes, SHA256 `068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25`. Harness and post-training identity receipts agree; no checkpoint was downloaded or independently loaded here. The final archive must still establish post-evaluation identity/retention and terminal result.

Fresh eval handoff occurred after training exit and checkpoint recording. Both fresh process/device checks passed under the **same live lock** (device66305/inode832589, descriptor3, driver3112491, parent3112490). Eval guard began `20:46:04.324331`, ended `20:46:04.385762`; new eval process launch was recorded `20:46:04.386031` UTC. The bound unchanged harness requests20,971,520tokens, context1024, batch1 and the exact new checkpoint. Actual final evaluator metrics/exit remain pending at this phase; no native causality outcome is inferred.

The primary comparator remains `fs-20260925-v2-c5c-a0-control-trn22-r10`, exact c5c base on trn22, BPB0.991513958330512 and3,633steps. Its archived raw evaluator hash and `total_nats/(num_bytes*ln2)` arithmetic were independently checked again. No 6ef7 Recursive or other-host control is substituted. Selector SHA256 remains `fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`; the current ledger is only read. An unchanged-selector query against current history plus the new authenticated candidate will follow the actual score.

`audit_training.py` passes65explicit stdlib checks in normal and optimized Python. `TRAINING_RESULT.json` records hashes, exact values, source estimates, control identity and ledger snapshot hash. Reproduce with `python3` or `python3 -O` on this script; it reads immutable Git archives and local support/control files, never a host or training module.
