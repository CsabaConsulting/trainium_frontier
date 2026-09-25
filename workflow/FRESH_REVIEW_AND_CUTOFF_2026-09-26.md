# Human amendments: fresh review and moving cutoff

Recorded 2026-09-26 IST. These direct Poojith instructions amend ALL IN;
conflicting older review-family, submission-gain and priority text is overridden.
Legality, human-only actions and unchanged correctness/budget gates remain.

## Fresh Astra review after Opus refusal

- The substitute is a new GPT-6 Astra session, not a resumed/forked Implementer
  or historical Verifier. No shared context, notes or branch with Implementer.
- Its only task inputs are the exact diff, lane specification and gate receipts.
  No earlier agent conclusions or implementation notes are input.
- It must independently reproduce at least one exactness claim by executing its
  own small CPU reference test. Reading the code or rerunning only supplied tests
  is insufficient. Default verdict is REFUSE until evidence is complete.
- Archive its actual model/session, input hashes, test source/output and verdict.
  A source/CPU pass is not native correctness, speed or submission readiness.
- Final submission compliance remains Claude via Poojith. Human uploads only.
- Opus 5.5 returned `stop_reason: "refusal"` and an empty text result. There is
  no refusal sentence to quote. Do not retry or reword the request.
  Receipt: `research/v2/bedrock/20260926-opus55-refusal.json`.

## 00:30 IST leaderboard amendment

Poojith reports the top-ten cutoff is 0.9785 official, formerly 0.9824. The
0.003-0.004/day decline and approximately 0.972 deadline cutoff are planning
forecasts, not verified future scores or public-to-official calibration.

- K1 Muon and K2 embedding scatter are critical path. Only the architecture
  decision-tree follow-up may preempt their gates, native timing or full runs.
  Profiling needed to identify/validate those changes is enabling work.
- Minimum acceptable freeze result: <=0.970 public-20M/context1024. All other
  ALL IN milestones remain, including the more ambitious <=0.947 target.
  Minimum is not completion of the first-place objective.
- Prepare the first stacked 8x1024 + schedule candidate for human submission
  after its cold replay and compliance audit even if outside the top ten.
  This explicit early architecture vehicle overrides the usual 0.003 gain bar
  for that candidate only; it waives no correctness or budget gate.
- Report scored 8x1024 runs immediately: startup, charged seconds, steps,
  tokens/s, public-20M and delta against the declared same-host control.

## Current evidence, not assumed success

Original 93ae architecture: public-20M 1.017340669168314 versus same-host
0.991513958330512, delta +0.02582671083780197, 2246 steps, about270k tokens/s.
First-five durations establish startup >=1544.745 seconds, not its exact total.
Source charged estimate1785.0 seconds; independently reviewed UTC enclosing
bound through exit1763.505610 seconds after the proven900-second cap.
This is a DISCARD, not a stacked schedule result or cold submission pass.
One compile-fix rerun remains permitted by the architecture decision tree.

K1's proposed same-shape batching and four-rank ownership already exist in6ef7.
Do not manufacture a no-op improvement; profile and target remaining measured
cost. K2 scatter gains and the other team's causal explanations remain
hypotheses until our own tests reproduce them.
