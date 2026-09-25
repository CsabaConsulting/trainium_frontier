# General fixed-budget selection amendment — Poojith, 25 September 2026

This direct human instruction supersedes every older step-window gate and
per-arm exemption requirement. It does not change post-A1 scheduling priorities.

- Every arm (architecture, FP8/throughput, schedule, tables and other approved
  training changes) is judged on final fresh-process public-20M/context-1024 BPB
  at the fixed **1,800-second official training budget**.
- Step count is reported, not gated. Do not request another step-window exemption.
- `over_budget` must be false. Record the actual charged time and the declared
  1,800-second limit; do not promote a short compile/timing smoke to a quality
  run. The existing checkpoint/exit reserve remains part of staying under that
  limit, not a requirement to perform exactly 1,800 seconds of optimizer work.
- Record startup for every run, distinguishing actual measured startup, capped
  allowance and a source-native estimate. Unknown is not zero and warm-cache
  directory reuse is not a cold-start measurement.
- About 870 seconds is the planning warning; the official allowance cap remains
  900 seconds. Actual startup above 900 is charged to the 1,800-second budget.
  Never subtract more than the allowance or forgive the excess in a comparison.
- Measure a genuine cold start before every submission candidate is accepted.
  Check the 8x1024 candidate's startup early; a warm research result alone cannot
  certify its cold compilation cost or submission readiness.
- A step count below 60% of its same-host control, step-time spikes or a stall
  raises a **diagnostic alarm**, not an automatic BPB verdict. Investigate a hang,
  recompile or incomplete run before trusting the score. Actual failures retain
  their failed/invalid status; the alarm never turns a truncated run into a pass.
- Correctness, same-host controls, two-host replication, all numeric BPB
  thresholds in `workflow/tools/decide.py`, the big-win audit and the submission
  audit remain unchanged. Submissions remain human-only.

The selector rules version and regression tests must reflect this instruction.
Historical table-specific exemption metadata may remain as evidence but is no
longer a prerequisite for any full-budget arm. Startup and cold-validation
evidence belong in the durable run records, not an invented score adjustment.
