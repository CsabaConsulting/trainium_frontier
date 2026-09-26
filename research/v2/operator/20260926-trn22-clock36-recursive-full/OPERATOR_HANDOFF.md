# Recursive-on-clock36 full run — actual detached launch

Assignment `38d4a8635356b0c7310231c6573e68bbd281b015`, authoritative host-state release `f994951c`, exact frozen packet `0090fd37ed605461f9e15414cf8f3bfc4d43ad66`. This is a full 1800-second research run with attached fresh public20M/context1024/B1, not a cold full replay, submission, adoption or automatic successor.

The exact one-use transport `stage-dispatch.py` SHA `ab9ccbadff29ca5d724fddbd3af8ebc23cfa65c35ddc840a8ae9cc4e63e4a862` is **CONSUMED: DO NOT RERUN**. First local and first remote invocation both exited 0. No source, bound driver, shared guard, harness, evaluator, environment or budget edits were made.

Actual dispatch: **2026-09-26 03:01:15.439763 UTC / 08:31:15 IST**, PID **3252200**. Driver training phase started **03:01:15.555974 UTC**. Cold predecessor terminal was 02:55:11.715961, so turnover was **363.723802 seconds** (6m03.724s), below the 15-minute incident threshold.

At **03:01:47.879551 UTC**, source, actual working-directory train.py hash, argv, all-rank environment and process identities were confirmed:

- flock **3252201**, driver **3252202**, timeout **3252218**, harness **3252219**, torchrun **3252233**.
- Workers **3252248, 3252249, 3252250, 3252251**; RANK/LOCAL_RANK 0–3, WORLD_SIZE4, LNC2, seed42, budget1800. Each worker's PGID/SID equals its PID; each starttime is **100573985** ticks.
- Exact actual source SHA `4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea`. All six actual whitelisted cache paths match the accepted child-cold cache roots; compiler URL equals directory. This verifies configuration, not a cache hit.
- Original common lock **66305:832589** held. The actual driver under-lock receipt passed fresh ps and neuron-ls before spawn, with no busy jobs or device holders. Exact reviewed driver retains the same inherited lock through training and attached evaluation, checking again before eval.
- Old queue PIDs **3050509 and 3053905** absent; pause retained. No second launcher.
- Disk at staging **21,695,897,600 bytes** free; at worker observation **21,596,905,472 bytes**. No cleanup performed.

First actual worker evidence SHA: `6bb597aba9190d8bd265009b5cae7b6a2aef174e5560a524430e5fad2ca16898`. Under-lock receipt SHA: `9b16b04c33af12f1586269d3e05ddeb7794f424eb1f357a1eb3abbb2b66087eb`. Training-launch receipt SHA: `6e3079db293b07121566d4ee2ea4d505a3dc6edb49bf8d11be7ffd9a2494dbc8`.

Remote root:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clock36-recursive-trn22-r1-full-packet`.

No training steps were yet present at this early startup observation. Later retrieval must preserve actual first-five/process-exit charging, all observed worker/compiler identities, complete training and fresh20M logs, checkpoint pre/post-eval hashes, and explicit every-outcome descendant/device/original-lock quiescence. No restart on observation timeout. The existing trn21 LR0.012 full run and its attached evaluation remain protected.
