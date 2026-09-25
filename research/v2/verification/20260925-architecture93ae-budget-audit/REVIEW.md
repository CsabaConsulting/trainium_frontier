# Independent 93ae startup/budget audit

Verdict: **PASS — source charges excess startup correctly; final process-wall
compliance PENDING.** No demonstrated timer arithmetic bug. This read-only audit
does not certify the still-running job, cold startup, quality, or submission.

Reviewer: independent Verifier, 2026-09-25. Evidence frozen at canonical import
`9be08a23`, particularly `steady-progress-1814` retrieved at
`2026-09-25T18:13:54.026370+00:00`. No host access, model/tensor allocation,
source/driver edit, queue change, or interruption. This receipt does not require
a new replay or observer to be attached to the active process.

## Exact bindings

- Source commit `1e21c3d797eed764f108c41546b270065ad5650c`,
  `workflow/proposals/research-v2/train.py` SHA256
  `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
- Unchanged `prepare.py` SHA256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Native packet prefix:
  `research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/`.
  `host_driver.py` SHA256
  `da58b422aeb136b1cc80cea336cc103c9a25694fe7d7bfc750c86858782d7a59`.
- `steady-progress-1814/training-launch.json` SHA256
  `2ce195961d62b03f4267563e62366bfcef87cbba00dc0b791a6ac15adeda2190`;
  `training.log` SHA256
  `f2d2460cea18157a9b417f57166ab0320539e74de98d82106f927808da8e04cf`.
  Both sizes/hashes independently match the retrieval manifest.
- Launch binds four ranks, LNC2, seed42, 8x1024, context1024/pack32,
  `--max-train-seconds 1800 --num-steps 1000000 --no-eval-public` and the
  reviewed architecture flags. No Recursive/FP8 combination. Cache directories
  are reused; **not cold-start proof**, even though substantial compilation ran.

Policy: `workflow/COMPILE_CONSTRAINT_CORRECTION_2026-09-25.md` and
`workflow/FIXED_BUDGET_SELECTION_2026-09-25.md`: cap900, warning about870,
charged limit1800, startup excess paid, source estimates separate from actual
process timing, step count diagnostic only.

## Source arithmetic and observed first five

Exact source lines4095–4111 synchronize, report the optimizer boundary, and at
the fifth completion execute, with `P = _PROCESS_STARTED`, `S = now - P`:

```
startup_allowance_est = min(S, 900)
excess = max(0, S + 30 - 900)
budget_started = now - excess
recent_step_times.clear()
```

Therefore at the fifth boundary source-budget elapsed is
`max(0, S - 870)`. For this run, where startup exceeds900, it equals
`(S - 900) + 30`: **all excess plus the30-second launch margin is charged**.
Afterward `budget_started = P + 870`, not the fifth-boundary clock. Clearing the
step-duration predictor does not clear elapsed budget. Stop prediction still
adds the next-step estimate and15-second checkpoint reserve. Before five steps
the minimum-step guard remains; an excessively long startup can still fail the
budget and is not excused by this audit.

The first five logged durations are847.90,695.40,0.49,0.49,0.49 seconds,
sum1544.77. They are disjoint in-step intervals, not total startup; import,
initialization and inter-step work are additional. Even allowing .005 seconds
rounding per duration gives a lower bound1544.745, already well above900.
Thus the capped allowance is900 without needing to invent an exact fifth-step
timestamp. Actual startup duration remains unmeasured by this driver.

At step00004, remaining1111 seconds means approximately689 seconds are already
consumed on the source budget clock. It is not a fresh1800-second window.
Synthetic `S=1559` gives689 and illustrates the equation; **1559 is not a
measured startup value**. Do not turn the coarse first-print absence/presence
polls into an exact optimizer timestamp. The early LR drop and later recovery
use the pre-/post-boundary budget origins already present in the source; they
are not evidence that elapsed startup was forgiven.

## What establishes final budget compliance

The source final `charged_seconds_est` / `over_budget` uses
`budget_elapsed_after_save` (lines4144–4150,4184–4185), after checkpoint/barrier
but before final output, `cleanup_runtime()` and runtime atexit work. It is not
an external elapsed-through-exit measurement. For startup>900, its clock also
includes the30-second launch margin. Keep this value as a named source estimate;
do not silently relabel it actual charged seconds.

The exact clean driver does not set `NEURON_COMPETITION_R1_STEP_LOG`; unchanged
`prepare.record_step_boundary` is therefore a no-op. No external fifth-boundary
timestamp can be recovered from that nonexistent stream. That does not require
a replay merely to determine the allowance here: the existing first-five lower
bound already proves saturation at900. The driver's5400-second watchdog is an
infrastructure timeout, not enforcement of the1800-second charged limit.
Likewise its eventual `terminal.json: COMPLETE` means successful training and
eval exits, not an independent budget verdict.

At exit, retrieve and hash the complete `training.log`, `run-exit-code.txt`,
`checkpoint.json`/checkpoint identity and existing process-timing evidence.
Require successful exit, checkpoint and source `over_budget: False`, with no
crash/nonfinite/truncation. Then classify official-rule timing using:

```
actual_charged_through_training_process_exit = training_process_wall - 900
accept budget only if actual_charged_through_training_process_exit <= 1800
equivalently: training_process_wall <= 2700
```

Use training process exit including worker teardown/export, **not** final step,
checkpoint time, source summary, or the later fresh-evaluation completion time.
An existing monotonic process measurement is strongest. The driver's
`TRAINING_START_UTC=2026-09-25T17:41:38.131774+00:00` is immediately before the
subprocess call and `TRAINING_END_UTC` is after its return. With a consistent
wall clock those enclose training process lifetime and can give a conservative
upper bound. An end by `18:26:38.131774+00:00` yields a logged2700-second bound;
this is a derived reference, not an inferred exact startup/exit timestamp.
Allow for any demonstrated clock discontinuity or material bracketing delay.

- An upper bound at/below2700 establishes timing compliance conservatively.
- A measured process wall or lower bound above2700 establishes budget failure,
  even when the source printsFalse or BPB is good.
- Missing end/exit evidence, contradictory clocks, or a bracket straddling2700
  leaves timing unproven; resolve from available end/process evidence, not a
  guessed timestamp or an automatic cold replay.

The frozen retrieval explicitly lacks `run-exit-code.txt` and `terminal.json`,
so this receipt cannot yet classify the completed run. The later fresh public
20M/context1024 evaluation is a separate quality/correctness gate, not charged
training wall. Warm-cache research still cannot establish cold submission cost.

## Independent checks

`python3 research/v2/verification/20260925-architecture93ae-budget-audit/audit.py`
and the same command with `python3 -O` both exited0: **24 checks PASS** each.
Stdlib only; no model imports. The checker hashes exact source/prepare and raw
archive, verifies launch budget/no-observer state, parses the raw five lines,
and executes the actual extracted fifth-boundary AST for eight startup cases
(0,800,870,899,900,1000,1559,3000). It verifies non-refunding,30-second margin,
the900 cap and predictor reset, including a case that is already over budget.
Checker SHA256:
`bb2921cf12c317b5d9f8dc71faa74b240d4f1612b395c92c61b86c3c6d09a279`.

Conclusion: no timer-source correction justified by this evidence. The concrete
reporting hazard is treating driver success or a source estimate as measured
official compliance; retain the exit-time calculation above in the final audit.
