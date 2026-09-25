# Startup-clock 36a862: accepted cold-five diagnostic

**PASS_COLD_FIVE_STARTUP_DIAGNOSTIC_ONLY.** Independently authenticated terminal `15261d14` (canonical `e08afdec`) and launch `4fa3355b` against exact binding `f35ac4947cfcb9a06a5833d00afe098ed79f07f9`. This accepts the bounded startup result, not a full training, quality, control, cache-hit, throughput, adoption or submission claim. It creates no automatic follow-on authority.

Own stdlib audit passes **130 explicit checks normal and -O**, with identical output. All 20 terminal raw text/size/SHA records (59,544 bytes) and all 11 launch records (43,805 bytes) authenticate. Source `7419927c`/SHA `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`, prepare/harness, exact package/driver/shared guard, actual argv/environment and copied sandbox source hashes match. Actual command is the frozen architecture command plus only `--startup-adjusted-schedule`, with five steps, MAX1800, seed 42, 113,256,724 parameters, world 4/LNC2 and unchanged microshape.

## Measured result

- Uncapped observed fifth boundary: **845.6273377500474 s**, below target 870 and hard ceiling 900.
- Torchrun-process wall through exit: **850.0966789470986 s**.
- Actual charged through exit: `850.0966789470986 - min(845.6273377500474, 900)` = **4.469341197051108 s**, within 1800.
- Five ordered rank-0 step boundaries 0–4; finite printed losses `[9.0101, 8.7779, 8.4047, 8.0711, 7.7681]`, each printed LR multiplier `1.000`.
- Harness, driver and outer flock exits 0. Natural driver terminal: **2026-09-25 23:24:10.327537 UTC**. No evaluation/full follow-on.
- Checkpoint retained remotely: **453,048,594 bytes**, SHA `053211cd99b1d6b0494a1027d157dfc85784b183d5db8f5442fd6fde8f2531bb`, hashed by the unchanged harness after the training process exited. No checkpoint was downloaded or independently loaded here.

The source's rounded `startup_allowance_est=843.4` and `charged_seconds_est=0.4` are separate estimates, not substituted for the observer values. The observer clock starts immediately before torchrun and includes its process exit, but excludes earlier sandbox/cache creation and prepare check. Fifth timing is first observation with nominal 50 ms polling plus scheduling delay; no exact UTC boundary is inferred from it.

## Cold paths, ranks and quiescence

Six job-private SDK caches were absent at staging, created and asserted empty by the pinned `--fresh-caches` harness, then recorded identically in all four actual worker environments, the isolation log and final rehearsal. Compiler URL equals DIR. All four actual worker argv/rank mappings, optimizer integrations and graph tables are present. The aggregate zero torchrun exit is the exit evidence; separate per-worker terminal files are not claimed.

The original lifetime lock is inode 832589/device 66305. Fresh in-lock process and Neuron checks were clean before launch; the live snapshot shows that lock held. Explicit postflight **23:24:50.616576 UTC** lists 16 observed driver/worker/compiler identities absent, including every one of the 10 independently matched launch PIDs. The PPID/PGID/SID/compiler query is empty, Neuron is idle and the original lock is free. The earlier 23:24:37 retrieval snapshot is also quiet. These are historical observations, not a claim of perpetual idleness or permission for another job. No manual kill or cleanup occurred in the recorded postflight.

Four graph tables contain 5/6/5/5 rows and retain forward/backward compilation evidence. Zero compile timers are **not** interpreted as cache hits. The short post-compilation intervals do not establish steady throughput. Five startup updates do not exercise the schedule after the fifth-boundary transition or establish final BPB. A full experiment still requires the separately reviewed packet, Supervisor selection and fresh sole-Operator preflight.

## Reproduction and evidence pins

Run from this repository:

```sh
python research/v2/verification/20260926-startup-clock-cold-result-independent/audit.py
python -O research/v2/verification/20260926-startup-clock-cold-result-independent/audit.py
```

The script reads immutable local Git objects only; no host calls, tensors, source changes or ledger writes.

- Terminal bundle SHA: `6c45190b3c1db2242073fdb04a1a5ca997ec2eb51c7abf655e70488b0b0f7e18`.
- Rehearsal SHA: `9811b186e3f5bf543a65b0b874a147d1a1df898691df2371358e6806946644ac`.
- Explicit postflight SHA: `c34998431372f2b7ec855dd5ec1d3203606d88276c752155b4878432a30ff4a1`.
