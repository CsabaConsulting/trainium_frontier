# External-review synthesis — 2026-09-21

Status: evidence review only; **no experiment or implementation is authorized by this file**.

## Inputs and authority

Four independent AI reviews supplied on 2026-09-21 were compared with the current
repository handoff, decision log, and experiment evidence. The reviews are hypothesis
sources, not instructions. The competition contract and current human-approved proposal
remain authoritative.

Current verified target:

- best comparable public development BPB: `0.9928513116311253`;
- cold replay: `0.9938556748843932`;
- frozen candidate: dense 6-layer, width 1024, sequence 1024, pack factor 32,
  seed 42, no n-grams, no FP8, with exact optimizer ownership;
- accepted official score visible in the supplied record: `1.0246`; the mapping from
  that submission to an exact source/configuration is unresolved.

## Consensus worth keeping

All useful reviews converge on two linked tasks:

1. Finish the existing Source197 first-layer/last-layer offline attribution before
   spending another Trn2 run. The already measured controls are about 498–499 ms,
   first-only is 496.617 ms, last-only is 682.661 ms, and all-six is 564.512 ms.
   This disproves a simple constant per-call cost but does not yet identify the cause.
2. If that attribution exposes a viable integration boundary, implement and verify the
   saved ROW-FP8 backward with FP8 data-gradient products, BF16 weight-gradient control,
   FP32 accumulation, native evaluation, and the existing correctness gates.

Source199's saved ROW-FP8 forward result is a real component premise (roughly 9.4% on
one rank and 19.1% on four ranks, preparation and saved state charged), but it is not yet
a backward, whole-step, or BPB win. It cannot skip Source197 attribution or the 3%
whole-step promotion gate.

## Ranked direction queue

### A. Do now — no machine run

Produce one Source197 dispatch-to-NEFF attribution packet from the preserved local
captures. It must map runtime executions to binaries and surrounding compiled regions,
compare first versus last without double-counting overlapping intervals, enumerate
unmatched executions and capture warnings, and state the portion of the full-step gap
that rank-0 forward profiling cannot explain.

Decision outcome must be exactly one of:

- caller-boundary constraint identified;
- conditional ROW-FP8 backward design is supported;
- hold because attribution remains insufficient.

### B. Conditional next implementation

Only if A supports it, draft a proposal for the complete saved ROW-FP8 backward. Before
any 30-minute quality run it must pass numerical, gradient, layout/cache, checkpoint
restore, native-evaluation, causality, and same-host whole-step gates. A result below 3%
whole-step improvement is parked for combinations rather than promoted alone.

### C. Later learning-side candidates

These may deserve fresh proposals after the current systems gate, but none is presently
authorized:

- depth-tapered MLP widths at approximately fixed parameter count;
- a bounded batch-size/gradient-accumulation ramp with an explicitly defined LR rule;
- per-layer learning-rate treatment for auxiliary embeddings, if auxiliary embeddings
  are reintroduced in a future architecture;
- GQA only after confirming it is not already closed and after estimating its actual
  quality/capacity tradeoff under the 30-minute regime.

Each must first pass a ledger alias search and define one causal change, matched control,
promotion threshold, and stop condition.

## Claims rejected or parked

- **“Immediately restore pack factor 4 plus n-grams” — rejected as stale.** Pack4 and
  n-grams belong to an older regime. The current best is the later pack32/GA1 dense
  no-ngram result. Historical success does not justify reverting the frozen candidate.
- **“Add optimizer ownership / ZeRO-1” — duplicate.** Exact optimizer ownership is
  already part of the frozen best.
- **“Rerun first/last/all position tests” — duplicate.** The unprofiled Source195 timing
  evidence already exists; Source197 was created to attribute it. Analyze existing
  captures first.
- **“Insert `contiguous()` after layer 5” — unproven diagnostic idea.** It is not a
  high-confidence fix until the preserved profile identifies a matching layout/copy
  boundary.
- **PTC2/reduce-scatter overlap — parked.** This is a larger, speculative systems change
  and comes after resolving the current caller-boundary evidence.
- **Output bigram head — parked.** It adds capacity and optimizer work without evidence
  that it beats the current dense candidate within the budget.
- **Deferred Adam every N steps — closed in the historical ledger.** Prior tests kept
  `ADAM_EVERY_N=1`.
- **Evaluation-context sweep — diagnostic only.** It may help explain public/official
  score differences but cannot change the official evaluator and is not itself a
  candidate improvement.
- **Byte-weighted loss, epoch shuffle, SwiGLU, vector-square/SBUF tuning, broad width or
  parameter sweeps — do not reopen without materially new causal evidence.**

## Autonomous-loop consequence

The Bedrock team must not turn a plausible review paragraph directly into a run. Its
next loop should assign an Analyst to Source197 offline attribution, then stop for a
durable verdict. Any later code change requires a completed proposal, independent
implementation and verification roles, and one bounded human approval for the resulting
run batch. Leaderboard submission always remains a human action.

