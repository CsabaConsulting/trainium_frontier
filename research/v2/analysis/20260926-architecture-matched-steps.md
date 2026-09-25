# Compile-limited architecture: matched-step training-loss check

No host work or model changes. Sources are the original93ae8x1024 run and its
declared same-host c5c6x1024 control, both trn22, seed42,131072tokens/step.
This is not a comparison to the newer Recursive6ef7 schedule stack.

| Step (zero-based) | Base logged loss | 8x logged loss | 8x minus base | Base LR multiplier | 8x LR multiplier |
|---|---:|---:|---:|---:|---:|
+| 0 | 9.0112 | 9.0101 | -0.0011 | 1.000 | 1.000 |
| 4 | 7.7614 | 8.0959 | 0.3345 | 0.760 | 0.229 |
| 10 | 6.6218 | 8.2239 | 1.6021 | 1.000 | 0.830 |
| 100 | 4.0385 | 4.7997 | 0.7612 | 1.000 | 0.799 |
| 200 | 3.6527 | 3.7890 | 0.1363 | 1.000 | 0.765 |
| 300 | 3.5304 | 3.5540 | 0.0236 | 1.000 | 0.730 |
| 400 | 3.3623 | 3.3529 | -0.0094 | 1.000 | 0.696 |
| 500 | 3.3459 | 3.3193 | -0.0266 | 1.000 | 0.661 |

The55 common logged steps through500 are available. Mean logged-loss delta
over the ten410..500 samples is-0.01801. At500the architecture is0.0266lower,
but this is weak support for doing the already-authorized compile work, NOT
evidence of a quality win or an estimate of final BPB.

Important limitations:

- These are the trainer's debiased exponentially smoothed training losses,
  not individual-batch loss or validation BPB.
- Time-based LR differs strongly. Architecture drops to0.230atstep2 while
  the base is0.760atstep4, then both reset the startup accounting afterfive
  boundaries. At500the architecture uses0.661versus1.000. This schedule
  confounding is not removed by matching update count.
- Startup consumes much more architecture budget;2246/3633=61.82%of base
  steps. The final fixed-budget negative remains true but cannot isolate
  intrinsic architecture quality. No compiler crash occurred.
- Same seed and batch configuration do not by themselves prove all rank
  sample identities or stochastic states match; no such extra claim is made.

Raw log references and SHA-256:

- `research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/final-evaluation/training.log`
  `1cf3d0a6343d45a9fd2deb0c1b504efd91da91f8f8dee939d7f7f6e3b0b8c27b`
- `research/v2/operator/20260925-trn22-a0-r10/final/train.log`
  `5dfe38ee815be4ffe48cf85e351b7ee22cfcfdb6837a0e5a235aebd53fa3a741`

Method: parse each `step ... | loss ... | lrm ...`line, inner-join on numeric
step<=500, subtract rounded logged losses. No interpolation or hidden samples.
