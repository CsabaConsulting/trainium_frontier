# Compile time is a first-class constraint

Poojith's direct correction, 25 September 2026. Supplements FINAL PUSH and
RESUME_AND_TABLE_CORRECTION, superseding conflicting queue priorities.

1. A2-r1 is invalid as a quality test: only five steps. Rerun A2 on trn22 with
   populated persistent caches; mark its warm result RESEARCH_ONLY. Finish
   the current eval and respect the common device lock before starting.
2. Record measured cold startup for every candidate, binding source, argv,
   compiler environment and cache-empty evidence. Unknown stays unknown.
   A table layout is not submittable until its cold penalty is measured and
   paid in a valid cold fixed-budget replay. Never extrapolate warm BPB into
   an official score or subtract a guessed BPB penalty.
3. Investigate in order: gather table rows outside the compiled graph and
   pass rows as inputs; reduce distinct compiled graphs; compiler flags last.
   Removing cross-core resharding remains a hypothesis until measured.
4. Fix the research training stop predictor: discard excluded startup step
   durations at the exclusion boundary. CPU-test the actual source. Preserve
   charged elapsed time, checkpoint reserve and the official cap. Inspect the
   submission file and prepare a separate prospective patch only if needed;
   do not overwrite the already-scored116188 vehicle.
5. Report A1 final public-20M BPB together with training seconds, steps and
   startup. Distinguish measured startup from the capped startup allowance.

## Timing convention

Unmodified prepare.py has TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900. The research
file separately reserves STARTUP_LAUNCH_MARGIN_SECONDS=30: about870 seconds is
the conservative target, not a different official cap. Official charging uses
its own process timing and excludes at most900 seconds. Clearing predictor
history never clears already charged compilation time.

## Current evidence

- A1/trn21: training_seconds1784.9; charged_seconds_est1785.5;3051steps.
  First five logged durations total991.79s:541.61+448.56+0.54+0.54+0.54.
  Import/model/loader startup is additional and not separately logged by its
  manual driver. Startup allowance was capped at900. Fresh eval is pending.
- A2-r1/trn22:5steps, process wall1997.294311970938, harness allowance900,
  charged-through-exit1097.294311970938, exit0. Invalid quality test.
- A2-r2 is an exactc5c warm research-only rerun through the installed serial
  controller. It does not contain the separately developed predictor patch,
  so it also tests whether existing cache reuse actually helps.

No evaluator, SDK, compiler flags, live run or submitted file is modified by
this policy record. Submissions remain human-only.
