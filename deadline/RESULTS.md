# Deadline results

Frozen submission: official 0.9645, historical public20M mean 0.96433779426305065. Recorded cutoff 0.9555. All scores below are public diagnostics, not portal scores.

| Variant | Matched public20M BPB | Difference from baseline | Status |
| --- | ---: | ---: | --- |
| FP32 table second moment | 0.963995885 | -0.000341909 | Within training budget; native causality and batch independence passed. One run, gain near measured noise; not promoted. |
| Structured V/O initializer | 0.964805820 | +0.000468025 | Within training budget; native correctness checks passed. No gain. |
| Four-block half-resolution attention | 0.987961419 | +0.023623625 | Within training budget; native correctness checks passed. Reject stronger coarsening. |
| Four-block MLP sharing with NKI selection/backward | pending | pending | Completed valid training, 3040 steps / 602.3M tokens; matched evaluation pending. |
| Six-block MLP sharing with NKI selection/backward | pending | pending | Completed valid training, 3327 steps / 659.7M tokens; matched evaluation pending. |
| Vocabulary logit bias | pending | pending | CPU/static gates passed; fresh native training launched 06:10 UTC. |
| Two output heads, second using ReLU features | pending | pending | CPU/static gates passed; fresh native training launched 06:10 UTC. |

The PyTorch MLP-sharing prototypes were stopped because native step times were approximately three times slower. Four explicit NKI operators recovered baseline speed for four-block sharing and about 9% more throughput for six-block sharing. Native outputs and gradients matched CPU references exactly. This is a kernel implementation win; final model quality is still unknown.

The two six-block attention-coarsening runs were stopped during compilation after the four-block quality result ruled against that allocation. Both guardians released their hosts with no failures. No final training or quality claim is made for those stopped arms.

Accelerated scoring was independently checked on the same structured-initialization checkpoint: batch1 public20M 0.9648058195 versus batch32 0.9648085852 (difference 0.0000027657). The same 2M prefix agreed within 0.0000005141 BPB, with identical bytes and token counts. Warm batch32 public20M took 177.05s versus the batch1 pass's 769.42s. Both native causality and batch independence passed; checkpoint hash stayed unchanged. Each final head experiment must also pass its own prefix agreement gate. The trusted prepare.py evaluator remains unmodified.

Source, requests, raw logs, preflight/postflight and checkpoint hash receipts are on research/deadline-hailmary-20260930. Heavy checkpoints are under ignored out/checkpoints/ and retained remotely. Frozen submission/train.py remains unchanged.
