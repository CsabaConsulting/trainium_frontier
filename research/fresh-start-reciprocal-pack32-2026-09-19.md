# Reciprocal optimizer with the current larger microbatch

Prepared2026-09-19 17:40UTC. This is a bounded composition experiment alongside
the FP8 output-layout comparison. No new approximation or quality result.

Source98's refined-reciprocal NKI AdamW plus optimizer ownership passed its
independent numerical gates and24 four-rank ownership/state comparisons. Its
pack16/GA2 whole-step pair was503.328809ms native versus497.650549ms candidate,
a1.128% gain below the3% advancement threshold. The current winner uses
pack32/GA1; reviewed dispatch records contain no corresponding reciprocal
timing. The NKI kernel and integration adapter are byte-identical to source98:

- `adamw_fused_fp32.py`:d01d69d662e8f9ae99861281e8bfd0d33c3a8b76dc4860f59bdb716f20cc3f96
- `adamw_fused_integration.py`:f99e3c15228b1a2ff11e4dfc667ac030aa3b326aac0216ef09f0e1fe9ff0572f

The same-host/source148 native control completed cleanly on trn21 at
17:36:34.847072:488.035319ms median,30 timed updates after5 warmups,
four passing finite/causal/batch/repeated-call checks. Optimizer time is
26.761235ms and the complete non-forward/backward tail54.784983ms. This is
the reference for the new candidate; it does not establish a cross-host FP8
comparison. Detailed raw log, summary and postflight hashes are preserved.

The fixed one-hour coordinator plan adopts that completed control, then runs:

1. `fs-w148-owned-reciprocal-current-gates-r1`: current actual model inventory,
   four-rank mode2/ownership equivalence,24 expected rank/stage pairs, cache
   updates, missing/zero gradients, skipped updates, schedules and state restore.
   The replicated reference uses identical NKI math; the earlier independent
   arithmetic gates remain its separate authority. Do not claim exact equality
   to the original optimizer's floating-point trajectory.
2. `fs-w148-owned-reciprocal-pack32-phase-r1`: otherwise unchanged native model,
   actual data, seed42, pack32/GA1, exact CE and ownership,5 warmups/30 timed
   updates, followed by four whole-model numerical checks. The remote wrapper
   independently verifies all24 unique rank/stage pairs, all four composition
   results and the native control before execution. It also requires positive
   kernel update counts after timing to reject a silently unused candidate.

Both jobs use the already deployed immutable60f5374 source,1500-second bounds,
host-wide device locks and full pre/post cleanup. Any failed gate, changed
request/helper/plan, ambiguous launch or deadline stops automatic advancement.
No quality run starts automatically. At least3% whole-step improvement plus
numerical review is still required before considering a separate30-minute
quality trial. Preserve negative results; do not sweep seeds.

The prepared request hashes and exact completion requirements are in
`run-logs/fs-w148-reciprocal-pack32-sequence-r1.plan.json`. Its launch/state
receipts establish whether it actually started. The local plan check validates
the gate requirements against source98's real24 equivalence/four composition
records and parses the remote wrapper without running a device command.
