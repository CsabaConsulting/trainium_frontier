# External review reconciliation R3 — 2026-09-23

Status: evidence review and controller redesign only. This document does not authorize a training
run, instance lifecycle action, or leaderboard submission.

## Accepted and implemented in the isolated loop-redesign branch

1. Fresh Researcher output now compares a shortlist of 4–6 ideas but still authorizes exactly one
   causal experiment. The artifact records mechanism class, predicted BPB delta and interval, and
   selection rationale.
2. The Researcher receives joined proposal-to-outcome memory rather than only raw recent ledger
   rows. Old mechanism labels are visibly inferred and cannot affect a verdict.
3. `proxy_calibration.py` joins smoke/full verdicts and reports clean sign agreement and Spearman
   correlation by mechanism class. It never edits gates automatically.
4. Source audits remain the first idea stream. Prediction metadata is retained so calibration can
   be measured rather than asserted later.
5. Old proposals remain resumable. Only new model proposals must satisfy the richer schema.

These changes are consistent with Rehearse's result that focused outcome memory restored late-loop
selection accuracy, while preserving the repository's one-change and independent-verifier rules.

## Accepted as proposals, not silently applied

- **Frontier baseline migration:** the autonomous control is about 1.0057 BPB, while the preserved
  width-1024 frontier result is 0.992851 public-dev (cold replay 0.993856). Migrating the pinned
  authoritative commit may be high value, but it requires a separately reviewed migration proposal,
  two same-host controls, and a new dedupe/control identity. Changing the commit in place would make
  existing comparisons invalid.
- **Table-LR scheduling:** this is already the active `--no-ngram-ve-flat-lr` experiment. Its full
  result must close before any freeze/weight-decay follow-up is proposed.
- **Context-length checkpoint evaluation, dtype/wiring diagnostics, and local-conv wiring:** useful
  only if implemented without changing the official evaluator and under a separate verified proposal.

## Rejected or deferred

- The claimed `0.978–0.988` finish range and `<2%` chance of 0.90 are speculative estimates, not
  repository measurements. They are not recorded as goals or expected outcomes.
- The review's `8–12` ideas per cycle is not supported as a magic number. Four to six gives useful
  diversity without consuming context or encouraging bundled changes.
- Compressed-schedule and branch-from-stem tiers are deferred. They alter schedule/data exposure and
  require calibration before they can serve as evidence for a 30-minute run.
- The alleged fp32 residual and conv-wiring “bugs” are hypotheses until source-level and behavioral
  diagnostics prove them. No code change is authorized from prose alone.
- A leaderboard submission solely to probe official token count is rejected until a review-safe,
  cold-replayed submission exists; submissions are scarce and human-controlled.
- Generic FP8/NKI work remains secondary to measured source-level opportunities. Existing evidence
  already shows that a kernel can be slower despite theoretical peak throughput.

## Primary-source cross-checks

- Rehearse: <https://arxiv.org/abs/2607.27687>
- Parameter Golf retrospective: <https://arxiv.org/abs/2607.01517>
- Learning-rate annealing scaling law: <https://arxiv.org/abs/2408.11029>
- Short-horizon bias: <https://openreview.net/pdf?id=H1MczcgR->
- Small-run data-curation proxies: <https://arxiv.org/abs/2512.24503>

The sources support outcome memory, prediction calibration, and caution about short horizons. They
do not directly validate compressed schedules, a specific Trainium kernel speedup, or the external
review's numerical BPB priors for this repository.
