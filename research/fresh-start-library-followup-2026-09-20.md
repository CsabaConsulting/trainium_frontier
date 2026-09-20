# Follow-up after the first NKI Library queues

Direct SSH checks at 12:23 UTC found both hosts idle, with no training/compiler
processes or accelerator holders. Source196's finite queue completed at09:05 UTC;
source195's at09:28 UTC. Backup cron preserved the results but did not dispatch
new work. Future status messages must distinguish a finite queue from continuous
agent research. No new quality run or BPB result was produced during the gap.

## Verified results

Every original frozen requirement and source fingerprint was rechecked locally.
All jobs returned zero and clean postflights. Full proofs are in
`run-logs/fs-w195-partial-position-proof-r1.json` and
`run-logs/fs-w196-vector-square-result-proof-r1.json`.

- The source196 production saved forward takes4.878602ms with Vector square,
  versus4.776949ms with the matched Scalar square. The Vector substitution is
  about2.13% slower despite exact outputs. Retain the Scalar implementation.
  NativeBF16 is5.327761ms; the existing Scalar FP8 forward advantage replicates.
- Source195 first-layer-only replacement: mean496.616637ms,
  median487.668197ms, mean microstep433.155515ms.
- Last-layer-only: mean682.661014ms, median682.696181ms,
  mean microstep627.420897ms. This is a large position effect with the same
  number of library calls and the same native custom backward in every layer.
- All six layers: mean564.512086ms, median564.248650ms,
  mean microstep507.779427ms. Cost is not proportional to replacement count.
- Bracketing zero-library controls: means499.107528/498.149224ms,
  medians488.653169/489.301529ms. Identical final loss5.126208782196045.
  Both controls and the first-only arm have a final-sample optimizer spike;
  retain every sample. Repetition at the same step warrants investigating
  schedule-dependent optimizer dispatch before calling it random system noise.

## Next finite experiments

Source197 reuses the immutable source195 checkout on trn22. It repeats first-only
and last-only timings and records the training forward of rank0 while all four
ranks execute. Each capture is followed by a separate raw-preserving Explorer
ingestion. Every model, native evaluation and source-identity check remains.
The comparison holds the number and shape of custom calls fixed and tests where
the extra surrounding work appears. Instrumented latency is not speed evidence;
match runtime dispatches to NEFFs and do not infer exact DMA traffic if metadata
is incomplete. Maximum sequence duration11000seconds; independent cleanup and
host ownership for all four requests.

Source198 is planned to measure the unchanged source194 saved FP8 forward under
one and four concurrent ranks, with all preparation and state stores charged.
The prior BF16 component's advantage disappeared under four-rank load, so this
screen is useful before committing to a complete FP8 backward. Preserve the
original source194 numerical prerequisite, exact actual-stage outputs on each
rank, paired orders and maximum-rank timing with collectives outside the timer.
No new kernel arithmetic or backward is part of this screen.

Complete ROW backward remains required work. The contraction-axis scaling
constraint in `fresh-start-library-fusion-next-2026-09-20.md` still applies.
Neither new sequence can promote a model to a quality run automatically.
