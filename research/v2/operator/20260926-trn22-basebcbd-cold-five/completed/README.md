# Exact bcbd cold-five startup diagnostic — completed

Label fs-20260926-ct-basebcbd-cold-trn22-r1; sourcebcbd148d,
release7c1423e7, frozen packet9881a3c8. Five steps only, unchanged MAX1800,
6x1024 Recursive, seed42, global131072, LNC2/four ranks.

- Actual uncapped fifth boundary: **493.3377903739456s**, target600s met.
- Original6ef7 cold reference855.071994281s, improvement361.7342039070544s.
- Process wall496.8015910499962s; charged3.463800676050596s, within1800.
- First observed boundary489.7261545449728s; source first-step dt467.90s.
- Five steps and all-process exit0; driver ended2026-09-25T21:24:30.121620Z.
- Checkpoint469834849bytes, SHA721d272268dec5d5472c0b0246e10626ad49da7a29686445cd754caa6efd7f63.

All six private SDK caches were absent before staging and asserted empty by
the unchanged rehearsal harness immediately before the torchrun launch.
All four workers' effective cache paths and changing compiler descendants
were captured in352b59e0 and terminal-quiescence.json. No copied/shared warm
cache, flags/runtime/source edits or prewarm was introduced.

Fresh21:26:18UTC postflight showed no descendants/compiler/device holders,
original common lockfree,31.63GiB available and old queuePIDs absent.
The full closed training/harness/guard/dispatch receipts have ten verified
host hashes. Four graph-stat CSV texts are stored losslessly with hashes in
graph-stats-raw.json (5042bytes total); no heavy trace decode or checkpoint
transfer. Source-capped allowance is separate from actual observed boundary.

This passes the assigned startup screen only; no BPB, model-quality,
steady-throughput, full1800s, adoption or submission claim follows. Printed
four postcompile rates are270755/266820/266732/266547tok/s but5steps do not
establish a fair steady-speed sample. Independent receipt audit is pending.
No automatic full run or evaluation has been started.
