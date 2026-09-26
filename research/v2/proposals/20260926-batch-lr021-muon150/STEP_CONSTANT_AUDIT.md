# Batch-size / step-constant audit

Scope: exact adopted source8c6; candidate4b0641db changes only momentum warmup.
No new BPB result. Existing batch/LR mean0.986879270651192 remains the candidate.

| Setting / site | Current behavior | Decision |
|---|---|---|
| Muon momentum warmup, line62 /1324 |300 updates =78,643,200 tokens at batch262144, versus39,321,600 previously |Test150 updates alone; same39,321,600-token ramp |
| WARMUP_STEPS, line55 /1311 |0: LR warmup disabled |No active duration to halve |
| Ordinary LR warmdown,1313–1317 |Elapsed charged training time /1800, ratio.75 |Not step-based; unchanged in this arm |
| Muon cooldown,1326 |Last4.5% of wall-clock schedule |Not step-based; unchanged |
| Weight-decay schedule,1464 |Decay decreases with wall-clock progress |Schedule unchanged; actual shrink is applied per update, so batch/LR alters cumulative shrink |
| AdamW beta1/beta2 and Muon beta2,889/899/1465 |EMAs update once per optimizer step; bias correction counts updates |Token-scale history roughly doubles with batch. Potential separate hypothesis, NOT automatically rescaled |
| MUON_NS_STEPS=5,65/1192 |Newton–Schulz iterations inside each optimizer update |Numerical iteration count, NOT training warmup; never halve for batch |
| NUM_STEPS=1,000,000,42/1440 |Unreached safety cap; actual runs~1,920 updates |1800-second budget is still controlling; no adjustment |
| TRAIN_STARTUP_STEPS_EXCLUDED,1474 |Official imported first-five-boundary accounting |Do not alter competition accounting |
| Recent-step window20,1484 |Predictive stop safety estimate, cleared at startup boundary |Not a learning schedule; do not halve |
| Smooth loss beta.9 / log every10,1493–1499 |Reporting only, not gradient input |Keep same; compare raw/final eval rather than misleading smoothed losses |
| Initial cache refresh / GC,1445/1500 |One-time first-step actions |No duration to rescale |

At the observed1,919–1,921 updates,300 spans15.6% and150 spans7.8%.
The old3,631-update run had300/3631=8.26%. Exact future update count is unknown.
All optimizer/model/eval/timer functions remain byte-identical in the candidate;
28,308 CPU scalar/reference cases pass. These are not native or BPB evidence.

The batch/LR+Recursive result0.9887829006 is+0.0017974 worse than its same-host
batch/LR comparator. It is not stacked into either warmup arm or final package.
