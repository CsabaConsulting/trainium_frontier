# Experiment Protocol — Trainium sub-1.0 campaign

Formal routine for every training run. Written 2026-09-17 after the
speed-collapse incident showed an undocumented environment variable
(`NEURON_CC_FLAGS`) could silently halve throughput with zero log trace.

## 1. Pre-run (both boxes, every time)

1. **Idle check.** `ps aux | grep -E "train.py|phase_timer|reconcile"` must be
   empty. At most 1 run per box, always.
2. **Disk.** `df -h /` — abort the launch under ~5G free. Compiling under
   ENOSPC poisons the build (incident 2026-09-17: first dynamic smoke).
3. **Device holders.** `fuser /dev/neuron0` must be silent. Orphans from
   traumatic kills hold cores/memory — kill strays first.
4. **Code sync.** Same `git rev-parse HEAD` on both boxes AND local, and
   `git status --porcelain` clean. A scored run with `git_dirty=True` is
   void on arrival. (trn22 once sat a dozen commits behind with no remote —
   sync by file copy, md5-verified, when pull is impossible.)
5. **Env capture.** The full launch environment goes INTO the record:
   `NEURON_CC_FLAGS` verbatim plus box id (`trn21`/`trn22`) appended to the
   `--description` string, e.g. `"... [trn21] CC='--optlevel=1 --auto-cast
   matmult --auto-cast-type bf16'"`. Rationale: speed-collapse post-mortem
   was blocked for hours because no record tied runs to flag strings.
   MANDATORY: every `NEURON_CC_FLAGS` MUST contain `--optlevel=1`
   (Day-2 winner, adopted campaign-wide). `train.py` sets it as the
   in-code default via `setdefault` — any external export WITHOUT it
   silently demotes the run to optlevel=2 (incident 2026-09-17: six
   consecutive 2x-slow runs from reconstructed flag strings).
6. **Seed.** 42 for all comparisons (no seed hunt — selection bias).

## 2. Launch

- Via `research/run_experiment.py` with `--no-commit`, seed 42.
- Verify the process started (`ps` shows workers) before ending the turn.
- Never relaunch a crashed label without diagnosing its log first.
- Never edit `TOTAL_BATCH_SIZE` or other `train.py` constants without
  restoring them after (verify with `diff`).
- Never stop a running instance.

## 3. Post-run (watchdog fire)

1. Check procs + new files in `research/run-logs/`, `out/*/final.pt`.
2. `rsync` new run-logs home; pull both `research/experiment-log.jsonl`
   to /tmp; merge by label keeping latest timestamp (strip NUL bytes
   before JSON parse — box1's file has a corrupt NUL region; a parse stop
   is never completion).
3. Read verdict (`public_val_bpb`, steps, `over_budget`); append to plan.
4. `git add` ONLY new run-logs + `research/experiment-log.jsonl` + plan
   doc (+ this protocol when it changes). Commit, push.
5. Launch next arm only on a satisfied gate (smoke win in a retrieved log,
   or an explicitly queued full run).

## 4. Between-run hygiene

Modern Neuron (1.16+) links the runtime (`libnrt`) into each process —
there is NO resettable daemon and no nvidia-smi-style reset
(`neuron-ls/top/monitor` are read-only). Device memory frees on process
exit. So hygiene = steps 1–3 of §1, every time, plus: stale
`/tmp/neuron_compile_service_*.sock*` regenerate alone — clear only after
an odd compile failure, retry once, then escalate.

## 5. Comparability rules (anti-contamination)

- **Rate comparisons** (steps/s, tok/s) only between back-to-back runs on
  the SAME box, same day, same code — the foreach ON/OFF rows are the
  template (paired, non-overlapping step counts).
- **Quality comparisons** (bpb) require comparable step counts: a full run
  >10% short on steps vs the record is CONFOUNDED, not negative
  (bf16-full 1449 vs 2135 — voided, not closed).
- **Warm vs cold compile** affects wall time, never step rate; the budget
  clock resets past startup (`startup_allowance_est`), so cold cache does
  not bias scored runs — but verify with `training_seconds`, not wall.
- **Crash cascades**: any kill (OOM, ENOSPC, SIGTERM) triggers the full
  §1 hygiene before the next launch on that box, no exceptions.
