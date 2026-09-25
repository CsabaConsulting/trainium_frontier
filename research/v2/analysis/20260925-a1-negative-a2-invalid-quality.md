# A1 result and A2 invalid-quality exclusion

Independent Analyst: root; did not implement the candidate or operate these runs.
The raw Operator records, not earlier estimates, determine this report.

## A1: DISCARD for quality, not throughput

`fs-20260925-v2-c5c-a1-m64d128-trn21-r1-full` completed 3,051 steps.
Native `training_seconds=1784.9`, `charged_seconds_est=1785.5`,
`startup_allowance_est=900.0`, `over_budget=False`. The first five logged
steps total 991.79 seconds. Initial process/model/loader overhead is additional;
uncapped process-to-fifth-boundary startup was not recorded by this driver.
900 is the allowance cap, not the measured startup duration.

The fresh-process organizer evaluation completed at 16:09:41 UTC with exit0:
20,971,520 tokens, context1024, 83,462,186 bytes, 57,772,392 nats,
BPB **0.9986312057378084**. Recomputing nats/(ln2*bytes) matches the reported
BPB to <1e-12. Same-host c5c/LR0.015 control is0.9914739939934344:
delta **+0.007157211744374**. Current unchanged authorized decision rules
return **DISCARD**, explicitly exempting the0.8302 step ratio.

Exact source/evaluator pins, training and evaluation commands, checkpoint
hash and both exit codes are archived under
`research/v2/operator/20260925-trn21-a1-full-final/`.
All files enumerated in its training and final retrieval receipts were
independently rehashed and matched. The739,322,371-byte checkpoint was
rehashed after eval and remains on the host. No checkpoint was deleted.
This is a valid negative research result, not a proven cold-start replay.

## A2-r1: INSUFFICIENT, not a quality loss

`fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full` completed only5steps, exit0.
The old predictive stop retained compile-heavy startup durations. Its
20M/context1024 BPB2.850923606746586 is preserved as observed output but
is excluded from quality selection with status `startup_truncated`.
The deterministic decision is **INSUFFICIENT**.

The harness observed the fifth boundary1988.6475348719396seconds after
process start; whole-process wall1997.294311970938seconds, capped
allowance900 and charged-through-exit1097.294311970938seconds. Native
training1116.4/charged1117.2seconds use different clock origins; do not
combine them with the harness timing. Existing cache directories were
reused, so none of these are proven-empty cold measurements.

Raw logs, timing and eval are under
`research/v2/operator/20260925-trn22-conditional-queue-r1/`.
Source/preflight/provenance are in `research/v2/operator/20260925-trn22-a2-full-r1/`.
The observed BPB was independently recomputed from164,930,432nats and
83,462,186bytes. The checkpoint remains on trn22.

## Consequences

Update16:43 UTC: the bullets immediately below describe the earlier handoff.
Poojith subsequently stopped A2-r2 at16:29:30 UTC after one completed step,
no checkpoint and no eval. Its status is `human_aborted_not_quality`.
`workflow/POST_A1_PRIORITY_CORRECTION_2026-09-25.md` supersedes the old
continue-A2 policy: both hosts now run the gated dense Recursive arm, with
checked architecture and FP8 prioritized. Six staged scalar fallbacks per
host are permitted only behind ready higher-value work. No automatic table
adoption, A3 or table-base LR branch follows this negative/aborted evidence.

- A2-r2 launched16:01:54UTC /21:31:54IST, exactc5c, warm and RESEARCH_ONLY.
  Read-only snapshot16:21:05UTC showed step0 at1040.11s, proving directory
  reuse has not removed the first-step delay. No claim of effective hits.
- A1 does not qualify for table adoption; A2 remains unresolved, not negative.
- Dense Recursive can use trn21's free slot as an independently authorized
  lane while A2 continues. This does not select a table-negative branch or
  authorize an old-base flag-bank tail ahead of architecture.
- No table submission until genuine cold startup is recorded and its excess
  is charged in an actual fixed-budget replay, followed by public20M eval.
- No explicit causality receipt exists in these scorer logs; ledger field
  stays null rather than fabricated PASS. No official score is predicted.

Append-only rows: `research/v2/ledger.jsonl`.
Machine-computed verdicts: `research/v2/analysis/20260925-a1-a2-verdicts.json`.
