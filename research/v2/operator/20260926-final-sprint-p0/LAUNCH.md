# Final Sprint P0 — actual two-host launch

Both independently reviewed P0 arms are running, observed07:06:43UTC on26September2026. No score, first-step throughput, cold-start duration or quality PASS is claimed yet.

| Host | Exact arm | Actual training phase start UTC | Driver / torchrun / workers | Original lifetime lock |
|---|---|---|---|---|
| trn21 | `fs-20260926-sprint-clean-recursive-trn21-r1`; clean e1e2-derived Recursive ON, normal batch/LR | 07:05:49.019276 | 3203817 /3203849 /3203864–3203867 | device66305/inode596823, held |
| trn22 | `fs-20260926-sprint-clean-batch-lr1-trn22-r1`; batch262144, matrixLR0.015 | 07:05:49.285254 | 3315643 /3315676 /3315691–3315694 | device66305/inode832589, held |

Detached dispatch timestamps07:05:48.897609UTC and07:05:49.162334UTC respectively; dispatch PIDs3203815 and3315641. This is12:35:49IST first native-job start. Every actual worker has rank/local_rank0–3, world_size4 and LNC2. Actual copied `train/train.py` SHA matches the reviewed source50fa6d144f19758dcdf9980e03f2302453174b956f0fa1a3e0845c4c0b72e6fb ontrn21 and8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee ontrn22.

Both native logs show8layers×1024,113,256,724 parameters, source budget1800seconds and pack32/context1024. trn21 has grad_accum_steps1; trn22 has2 as requested. Logs contain usual Neuron operator-registration warnings, not an actual runtime error. Neither driver terminal nor completed rehearsal exists at this observation. Each reviewed driver will run its already-attached unchanged fresh-public20M eval after clean training, under the same lifetime guardian. No automatic successor was installed.

## Gate and binding evidence

Authority is the direct Final Sprint and `research/v2/assignments/20260926-final-sprint-p0.md`, conditional on the SINGLE fresh reviewer PASS. That PASS completed07:02UTC and covers both source/CPU changes, guardian5df6897d and all six frozen variants. Each deployed source/support file was authenticated against the actual reviewed68-entry manifest SHA c28e772438afd2d8c46be423700dc6d8cdeea085e3465c901558c762cce356a3.

Only each driver's TRAIN_ENV cache-path literals changed mechanically. The original source bytes, science argv, optimizer/architecture, evaluator, harness, official budget, idle guard, lifetime guardian and dispatch wrapper are unchanged. The cold-bound driver SHA is017b41c92bc58e028bb4e9170a4b9f85b58ade47408a9fcfe05711959e6e9c41 ontrn21 and0cd808457a8cbe2db5e397531ad0d5f0545c850bce8e4b18f3601ad95892407a ontrn22. The literal restoration test recovers the reviewed host-driver bytes exactly.

Each root was absent before staging. Six new private training cache directories were created empty under `<run-root>/cold-training-caches/{compiler,neff,local,hlo,inductor,debug}`. Both CACHE_URL and CACHE_DIR bind the compiler child; the six effective path values were confirmed in all four actual rank environments. These directories are outside the harness's previously absent `train` sandbox, avoiding the earlier containment/setup trap. This establishes an initially empty requested/effective cache setup, not a measured startup duration or future submission qualification. Existing evaluation research caches were preserved unchanged; eval remains a fresh process.

Actual stage checks ran under each original exclusive lock: correct hostname/uid, no source/root reuse or symlink parents, unchanged prepare5807c46e and harness7a55cb87, actual harness containment, retired queue PIDs absent, ps/neuron-ls/fuser idle, STOP/breakers clear, executable availability, at least64GiB available RAM and the existing13.5GiB initial cold reserve (10.5GiB runtime floor plus3GiB projected footprint). The separate final launch reacquires that original lock and repeats fresh in-lock idle checks before spawning; it keeps the lock across owned descendants and attached eval. The25GB cleanup target was not treated as a new gate.

The old top-level launch bits in the archived controller snapshot still describe obsolete one-use labels. The new direct assignment and Final Sprint explicitly supersede that ceremony; no STOP/breaker was bypassed. Supervisor was notified before dispatch and owns current state reconciliation. Local proposal provenance remains historical/disabled; the separate explicit RELEASE records this actual scoped authority.

One LOCAL-only preparation error occurred: the initial manifest assertion compared the repository's69-entry manifest to the reviewed68-entry bundle SHA and raised `AssertionError` before any SSH/staging. It was corrected to read the actual frozen reviewed manifest; both local checks then passed. No remote attempt or scientific retry occurred.

## Current resources and next observation

At07:06:43UTC disk free is15,166,464,000B ontrn21 and17,846,312,960B ontrn22, above each10.5GiB runtime floor. Both original locks are held by their expected live guardian/flock chain, with four actual workers each. No intervention is indicated. The next useful expected cold-start boundary is approximately07:19–07:21UTC; actual uncapped fifth boundary, source charging and initial steady throughput must be read from the natural run, not inferred from prior candidates or source timers. Do not restart these detached jobs because a local observer exits or times out.

Raw dispatch receipts: trn21 SHA75bbce3d14549ebbcc5e039369456f3e262c0a43fc3f70ea56ad7543ccc3fe10; trn22 SHA173d869f4a47f6950bd046bd8495990961e66aada4e766414f8d7d08c27ee4d8.
Raw actual-worker observations: trn21 SHA79eaadfabeb8ec34dda3c38a0306e4cd4e62dc6f08ecd6a1caad979be17d8bcb; trn22 SHA311f65e4f8b5a6354f587c59a4d975169c7b397858b84574b82825a06eb67cc7.
All captured text artifact hashes were replayed locally, along with source/rank/argv/cold-cache bindings and stable PID startticks. Ordinary task-state changes between two process reads are not misreported as PID replacement.
