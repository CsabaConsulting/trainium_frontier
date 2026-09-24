# Step-change researcher screen — 25 September 2026

Status: **hypotheses only; no experiment proposal, implementation, adoption, or launch authority.**
The Researcher reviewed the current dense/research-v2 queue and historical logs for a
post-A1/A2 idea with an evidence-backed *expected* improvement of at least 0.01
public-20M BPB. None currently clears that bar. An upside tail is not an expected gain.

The most urgent unanswered scientific question is still A1/A2 on the dense c5c base:
the c5c A0 attempt stopped before step 1 on disk safety, so there is no valid c5c
same-host control or table score. Both hosts remain disk-blocked. Old 640-stack
table gains are priors, not a c5c/dense result or an adoption; see
`reports/2026-09-24-frontier-research-program.md` §4.2 and
`research/v2/STATUS.md`.

Conditional mechanisms, ranked for *later* falsification:

1. **Independent factored K=2 n-gram hashes at constant total table size.**
   Two independently indexed half-width embeddings could reduce hash-collision
   interference without raising table parameter count. The 23 September negative
   double-hash smoke averaged two gathers from the same full-width table and is not
   the same intervention (`research/agentic/experiments/20260923-ngram-ve-double-hash-r2/implementation.patch`,
   `research/agentic/DECISION_LOG.md`). The existing research report estimates
   multi-hash only around ±0.002, not −0.01. First inspect training-shard-only
   collision burden, then require a distinct implementation, independent verifier,
   exact-size G2/native timing and same-host full controls. Do not queue it yet.
2. **Freeze n-gram table updates at epoch 2.** The code-ready
   `--ngram-ve-freeze-epoch 2` could protect learned lookup memories from a repeated-data
   tail, but only if A1/A2 diagnostics show that tail is damaging them. The queued
   prior is −0.002 with an 80% range −0.006 to +0.003 relative to an unfrozen table
   arm (`reports/2026-09-24-frontier-research-program.md` §4.2). Compare against the
   same-host unfrozen arm; a ≥0.01 rescue is a contingency, not a prediction.
3. **Fresh-tail order.** `--fresh-tail-frac 0.25` is a weaker data-wall diagnostic,
   with queued prior −0.002 and 80% range −0.006 to +0.002. Its
   `--requeue-identity` matched-loader control and batch-identity check come first.
   Do not count it as a likely step change.

All-VE placement and m128 table-size scaling are already in the queue with central
estimates only −0.003 and −0.002 respectively; rebranding them as new ≥0.01
mechanisms would misstate the evidence. The next action is operational, not another
model hypothesis: clear the human disk/breaker decisions, complete G1/G2/native
gates, and measure c5c A0/A1/A2 before revising this ranking. No score changed.
