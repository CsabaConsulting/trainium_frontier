# Experiment protocol — Trainium Frontier research

Version3,2026-09-18. Applies to every device training run, phase probe, standalone
evaluation and kernel diagnostic on both provisioned hosts. The user explicitly
requires setup and cleanup before and after every experiment, and preservation
of work locally. Kernel failures require diagnosis, not blind repetition.

## Before each run

1. Use `fresh_start_launch.py` with a unique label, bounded command, explicit
   environment, expected commit and source fingerprints. Commit locally before
   transferring a Git bundle to a new immutable remote worktree. Different arms
   may use different commits; each must identify its exact source. Never overwrite
   an active worktree or the user's original dirty checkout.
2. Acquire the host-wide `/home/ubuntu/.cache/trainium_freshstart/device0.lock`.
   It spans all campaign worktrees and stays held through postflight. The existing
   training-runner lock additionally protects its own checkout. One device job per
   host; never run a profiler or component probe alongside training on that chip.
3. Run the pre-cleanup routine for the previous completed job, if recorded. Only
   same-UID processes carrying that unique inherited run marker and an unchanged
   process birth identity may be terminated. Unknown processes are inspected,
   never killed by a broad pattern or `pkill` command.
4. Save `<label>.preflight.json`: no `/dev/neuron0` holders; `neuron-ls` reports no
   active Neuron processes; no remaining training/compiler workers; at least16GiB
   `MemAvailable`, 6.5GiB disk space and10,000 free inodes for the verified dense
   candidate (approximately 0.47GiB checkpoint). The controller and launcher must
   hard-abort below5GiB; a proposal with a larger checkpoint or measured scratch
   requirement must declare and enforce a higher operational floor. These thresholds
   are operational minimums, not sufficient capacity estimates for an arbitrary model.
5. Capture boot ID/uptime, SDK versions, complete explicit Neuron flags, memory,
   disk/inodes, cache sizes, a bounded `neuron-monitor` sample and available kernel
   messages. Record telemetry failures rather than inventing measurements. Review
   new runtime errors, thermal/throttling indicators and ECC changes before a
   comparison. The installed monitor may not expose every counter or temperature.
6. Retain the same cache policy: reuse immutable compilation caches in
   `/tmp/neuron_cache` and `/tmp/neff_cache`; do not clear Linux page cache, unload
   drivers, reboot or reset hardware as routine cleanup. File cache is reclaimable
   memory; use `MemAvailable`, not `MemFree`, to assess pressure. A deliberate cold
   compile is a separately labeled experiment. Do not delete compiler artifacts
   from a failed run until the error and needed reproducer have been preserved.
7. Record the hypothesis, primary control and advancement gate. Seed42 is the
   comparison default. A small predeclared independent repeat can assess variation;
   do not search seeds for a favorable public score. The working competitive target
   is around0.9749BPB, with margin for leaderboard movement.

## During the run

- `fresh_start_guard.py` supervises the manifested timeout command. It assigns a
  unique `TRAINIUM_FRESH_START_RUN_ID` to descendants, records process/session IDs,
  and keeps the host lock. Full training still uses `run_experiment.py --no-commit`;
  the runner owns its own `.launch.json` and raw log. The outer guard uses a driver
  log in that case, so it cannot pre-create or overwrite the runner's files.
- Keep a local live-log mirror, normally every60s. The dispatch intent is written
  locally before SSH. Record all failures, not just successful results.
- Preserve the installed native PyTorch/NKI0.5 environment and LNC2. Do not apply
  XLA guidance, unverified flags, new packages or driver changes to these hosts.
- Stop a hung experiment through its owned timeout/process tree. Never interrupt
  an unrelated process or another experiment to make a test appear clean.

## After every outcome, including assertions, NaNs and compiler crashes

1. The guard waits for the workload, then runs bounded owned-process cleanup:
   TERM, grace period, then KILL only for verified remaining descendants. This also
   covers marked children that changed session or were reparented. Normal exits
   ordinarily leave nothing to terminate.
2. Allow a short release interval, then record `<label>.postflight.json` with the
   command result, cleanup actions, device/process/memory/disk checks and monitor/
   kernel evidence. No unknown holder is killed. A failed health check is not a
   clean run; inspect and resolve it before continuing. An instance loss or kill of
   the guard can prevent a receipt: its absence must be recorded and the full
   preflight repeated, never assumed to have passed.
3. Retrieve raw logs, manifests, health/process receipts and complete research
   records locally. Merge experiment records by source identity without overwriting
   older differing records. Hash-verify checkpoints and raw profiles against their
   remote files; preserve failures and useful reproducers too. Temporary artifacts
   may be removed only after the needed evidence is local and verified, with the
   removed paths recorded. Valid retained caches/checkpoints are not debris.
4. Commit source, protocol changes, notes, logs and receipts locally, then push
   the research branch to `origin` (the user's CsabaConsulting fork). The user
   explicitly authorized regular GitHub backups after the local GUI crash on
   September18. Verify the remote ref after pushing. Keep a current
   `research/RESUME.md` with completed jobs, unresolved failures, next commands
   and artifact locations. Heavyweight checkpoints/profiles remain outside Git
   with tracked hash receipts and require a separate off-machine artifact backup;
   a receipt alone is not a backup of the data. Competition submission remains
   a separate action. Preserve original EC2 snapshots instead of blindly merging
   stale files over the canonical checkout.
5. Read the result and failure mechanism before choosing the next experiment. A
   completed process is not a correctness pass; a component speedup is not a
   training speedup; a small public prefix is not the private leaderboard score.

## Failure diagnosis and repeatability

Preserve the exact flags, SDK/source identity, inputs or deterministic input
recipe, shapes/dtypes, first incorrect stage and complete error text. Compare CPU
references, eager device execution and compiled composition. Inspect finite values
before the first backward/update; test allocation/lifetimes, index bounds, dtype
casts and synchronization when evidence points there. Instrumentation can change
allocation, so re-test the original composed path after a fix. A failure closes
that implementation only when its cause/cost is understood; it does not establish
that all implementations of the idea fail.

Use unprofiled, warmed full-chip measurements for speed, with all input/metadata,
layout, communication, optimizer and control costs included. Prefer nearby
same-host controls; report ranges and compiler/startup differences. If throughput
drifts beyond normal control variation, investigate host/runtime state before
calling it an algorithmic result. A repeat cannot be made reliable merely by
renaming its log or clearing a cache.

For paired compiled probes, verify compiled execution on later microbatches and
after weight-cache refreshes for both the reference and candidate. Give them
distinct root callables when only one permits graph breaks. Source177/178
reproduced a shared `Probe.forward` interaction: after the candidate call, the
reference returned `LinearBackward0` with no compiled derivative node. Keeping
the original reference assertion and isolating the candidate root restored all
checks. A graph-node wrapper is not itself a numerical error, but an eager
reference cannot support a claimed compiled timing comparison. Preserve the
failed original probe and verify the cause before changing its checks. See
`run-logs/fs-w178-nkilib-shared-root-diagnosis-r1.json`.

Full-budget BPB comparisons legitimately include different step counts: gaining
useful updates is the purpose of kernel optimization. They do not isolate
matched-exposure algorithmic quality. Use separate matched-step/exposure studies
for that question. Startup exclusion follows the trusted first-five-boundaries
protocol and900s cap; later compilation/preparation and checkpoint completion
still count. Cache state can change startup and potentially the number of charged
updates; verify timing rather than assuming it cannot influence the score.

## Transition from the earlier protocol

Existing completed/current jobs were launched with idle-device, disk, exact-source
and environment checks, but not the automatic version2 guardian. After completion,
`fresh_start_host.py --adopt-completed --remote <worktree> --label <label>` records
a read-only postflight. It explicitly does not claim a version2 preflight happened
retroactively, and cannot terminate old unmarked processes. Both new kernels and
training runs must use the version2 launcher once deployed.
