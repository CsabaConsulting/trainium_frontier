# Prospective K2 timing control reuse — no source or gate relaxation

26 September2026,13:44IST. Supervisor selects this under FINAL SPRINT.
This supersedes ONLY the unstarted extra OFF45 run and its prospective disk
allocation in the earlier K2 timing assignment. Preserve the already-running
full control and its20M. No job is killed, shortened or relabelled.

## Exact comparison

- Control: actual trn22 run fs-20260926-sprint-e1e2-control-trn22-r1,
  audited parent e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43,
  launched08:07:14UTC. It remains kind=control/full1800/quality in the ledger.
- Candidate: frozen63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5,
  exact existing K2 ON45 source/argv, on the same host/runtime/seed42,
  batch131072/context1024/pack32/four ranks/LNC2. No source edit or other flag.
- Primary timing statistic stays **35*131072/(t44-t9)** for BOTH runs.
  Select this window now, before the new control's steady-state window or
  any K2 timing result is observed. Do not choose a faster later interval.

The K2 patch's existing fresh independent review2ddb87ac verified full parent
and candidate source identities, three OFF model steps against e1e2 by bytes,
and unchanged parameter creation/ordinary branches/optimizer/budget/eval code.
Reference: verification/20260926-k2-sprint-independent/REVIEW.md. Thus reusing
the actual unchanged parent is a matched causal baseline, not an old6L, batch
or schedule-arm comparator. It does not prove nativeK2 correctness or speed.

Supervisor source inspection: args.num_steps controls loop termination and
printed remaining_steps, not shapes, data order, optimizer or LR schedule.
Budget and LR progress remain1800 seconds in both. record_step_boundary(step)
occurs before the stop-decision collective; the t44 endpoint therefore does
not include the candidate's different stop flag, final save or exit. Both
windows include identical preceding per-step control, data and logging work.
The source's existing K2-enabled path is the sole effective model-op change.

## Native release and resources

After the current control+eval exits, retrieve/authenticate its terminal
observations and verify45 sequential finite boundaries including t9/t44.
If the control failed, recompiled/stalled in this interval, runtime/science
differs, or timing is missing/untrustworthy, do NOT use this comparison.
Keep the current control's checkpoint/caches protected.

Mechanically rebind the existing trn22 ON45 prepared packet to this proposal
and actual completed-control predecessor. Use a new unique label
fs-20260926-k2-sprint-on-parentcontrol-trn22-r1. All scientific source, launch
argv, existing reviewed guardian, cold six-cache setup, original lifetime
lock and idle/hash checks remain unchanged. No extra source review.

Candidate admission is the UNCHANGED per-run13.5GiB=14,495,514,624B initial
floor (3GiB prospective footprint plus10.5GiB runtime floor). Parent-control
artifacts already exist and remain allocated in the measured free-space
check; do not reserve a second3GiB for an OFF job that will not be run. This
does not lower any per-job floor or delete protected/unscored artifacts.
If actual measured free space fails, skip launch and report exact deficit.

ON must complete45 finite steps with actual cold startup and budget recorded;
preserve checkpoint and timings without running a fake quality eval. K2 speed
PASS still requires rateON/rateCONTROL >=1.03, same precommitted statistic,
and all native/correctness/source gates. Record cold startup on BOTH and
charge beyond900 seconds normally. Any native failure remains a failed gate.
Only then may the prepared same-source full1800+fresh20M follow-up run.
This update changes no decide.py threshold, official accounting or upload rule.

Undo: cancel the unstarted ON binding if any comparison precondition fails;
restore the explicit same-candidate OFF/ON design when resources permit.
Do not reinterpret unrelated data as a substitute. Historical disk failure
and the original unexecuted design remain archived, not erased.
