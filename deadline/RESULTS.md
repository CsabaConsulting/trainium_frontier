# Deadline results

Frozen submission: official 0.9645, historical public20M mean 0.96433779426305065. Recorded cutoff 0.9555. All scores below are public diagnostics, not portal scores.

| Variant | Matched public20M BPB | Difference from baseline | Status |
| --- | ---: | ---: | --- |
| FP32 table second moment | 0.963995885 | -0.000341909 | Within training budget; native causality and batch independence passed. One run, gain near measured noise; not promoted. |
| Structured V/O initializer | 0.964805820 | +0.000468025 | Within training budget; native correctness checks passed. No gain. |
| Four-block half-resolution attention | 0.987961419 | +0.023623625 | Within training budget; native correctness checks passed. Reject stronger coarsening. |
| Four-block MLP sharing with NKI selection/backward | 0.978947940 | +0.014610146 | Completed valid training, 3040 steps / 602.3M tokens; no quality gain. |
| Six-block MLP sharing with NKI selection/backward | 0.990985848 | +0.026648054 | Completed valid training, 3327 steps / 659.7M tokens; no quality gain. |
| Vocabulary logit bias | 0.965380324 | +0.001042530 | Full public20M agrees exactly in LNC1/LNC2 and original independent scorer; native checks passed. No gain. |
| Two output heads, second using ReLU features | pending | pending | CPU/static gates passed; fresh native training launched 06:10 UTC. |

The PyTorch MLP-sharing prototypes were stopped because native step times were approximately three times slower. Four explicit NKI operators recovered baseline speed for four-block sharing and about 9% more throughput for six-block sharing. Native outputs and gradients matched CPU references exactly. This is a kernel implementation win, but both final quality results are worse than the frozen baseline. Both jobs completed with clean postflight receipts.

The two six-block attention-coarsening runs were stopped during compilation after the four-block quality result ruled against that allocation. Both guardians released their hosts with no failures. No final training or quality claim is made for those stopped arms.

Accelerated scoring was independently checked on the same structured-initialization checkpoint: batch1 public20M 0.9648058195 versus batch32 0.9648085852 (difference 0.0000027657). The same 2M prefix agreed within 0.0000005141 BPB, with identical bytes and token counts. Warm batch32 public20M took 177.05s versus the batch1 pass's 769.42s. Both native causality and batch independence passed; checkpoint hash stayed unchanged. Each final head experiment must also pass its own prefix agreement gate. The trusted prepare.py evaluator remains unmodified.

Source, requests, raw logs, preflight/postflight and checkpoint hash receipts are on research/deadline-hailmary-20260930. Heavy checkpoints are under ignored out/checkpoints/ and retained remotely. Frozen submission/train.py remains unchanged.

At 06:36 UTC, all four full-width models (headbias, dualhead, mlpnki, mlp6nki) passed native LNC1 batch1/batch2 finite-output, causality and batch-independence checks with deliberately nonzero MLP/table values. These use random weights and establish hardware compatibility only, not trained quality. The guardian completed with a clean postflight.

Final head checkpoints were copied locally and size/hash verified. The source-host dual-head evaluation hit its outer deadline timeout after successful full-budget training; the host cleaned successfully. Its separate full20M job on trn23 remains bounded and continues after the submission deadline. No original source-host full score is claimed.

The first three spare-score receipts used only 2621440 tokens because a generator replacement accidentally shortened both the full cap and its count assertion. They are explicitly excluded from this table. Correct full20M recovery uses unchanged prepare.py with literal --eval-tokens 20971520. The output-bias recovery agrees exactly with the original unaffected full scorer and LNC1 scorer, 0.9653803241496522. Inference-only compiled operator caches were reused after clean postflight; no training state or source was modified.
