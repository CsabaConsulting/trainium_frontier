# Questions for primary-source literature review

These are research prompts and hypotheses, not established conclusions. Automated
research outputs are leads; verify important claims in primary sources and the
installed implementation before promoting them to the campaign plan.

## A. Small-data training and compute/capacity allocation

Research techniques for minimizing language-model held-out NLL when training from
scratch under a strict 30-minute wall-time budget on a single accelerator with
about 300 million available training tokens and a fixed 8192-token BPE. Baseline
is a 6-layer width640 pre-norm decoder, RoPE/QK norm, ReLU-squared MLP, residual
scalars, untied embeddings, Muon for matrices and AdamW for embeddings/head,
plus large learned hashed token bigram/trigram value tables. ~2100 optimizer
steps at 131072 tokens/global batch give ~280M tokens. A wider width896 model
can reach better held-out loss at equal tokens but is slower. Throughput gains
that permit ~350M tokens have sometimes worsened quality.

Use research papers, author repositories, and original experimental records only.
Identify high-value mechanisms and counterexamples involving finite-data repetition,
critical batch size, warmup/stable/decay schedules, per-token or per-update scheduling,
optimizer/capacity interaction, matrix/embedding parameter allocation, shared layers
and progressive growth, multiscale causal context, cheap n-gram memory, and EMA.
Separate evidence at this small/short scale from speculative transfer of large-model
results. Account for the fact that validation BPB is summed token NLL divided by a
fixed total-byte denominator. No pretrained weights, external data, modified tokenizer,
validation adaptation, or scoring tricks are allowed. Do not merely list common
tricks: explain why each could work here, how previous negative results may be
confounded, smallest discriminating tests and explicit stop criteria. Include
precise primary URLs and distinguish facts from conjecture.

## B. Trainium2 systems and NKI training kernels

Investigate official AWS Neuron documentation/source and primary implementations
of high-throughput decoder training on a full Trainium2 chip (NeuronCore-v3,
4 logical cores in LNC2, NKI0.5 and native torch-neuronx2.11 stack). Target workload:
6 decoder layers, width640 or896, head_dim128, seq2048, 4 sequences/microbatch,
4 gradient-accumulation microbatches, fixed8192 vocab, ~500M BF16 n-gram embedding
parameters with RMSProp and ~56M dense parameters with Muon/AdamW.

Evaluate opportunities in model-level compilation, collective/gradient fusion,
sparse embedding gradient and optimizer paths, fused normalization/residuals,
full forward+backward attention, MLP GEMM fusion, FP8 casting/scaling/master weights,
and end-to-end host-dispatch overhead. Existing NKI RoPE/QK norm and softcap+CE
already run; a naive attention implementation is slower than built-in SDPA.
Explicitly distinguish native Trn2 float8_e4m3/e5m2 double-row nc_matmul from
Trn3 OCP e4m3fn, MXFP8 and nc_matmul_mx. Identify API versions and numerical
correctness traps. Assess per-compiled-region compiler flags and preserving
FP32 sensitive optimizer math under global BF16 model autocast. Cite usable
primary code or docs; quantify Amdahl/traffic bounds and testing costs rather
than assuming peak FLOPS translate to whole training throughput.

## C. Optimizer and low-precision state as a quality bottleneck

Research primary-source evidence for precision failures in short transformer
training with Muon/Newton-Schulz (matrix polar orthogonalization), global compiler
BF16/FP8 autocast, RMSProp on large BF16 hashed n-gram tables, dense replicated
gradient averaging, and per-rank clipping before averaging. In the target setting
there are ~2100 updates, 4 data-parallel ranks and ~300M total train tokens.

Determine when and why BF16 Newton-Schulz is acceptable versus dangerous; effects
of FP32 residual/update accumulation, scalar casting, stochastic rounding, FP32
master weights, epsilon placement, rare-row moment decay/bias correction, clipping
before/after averaging, and constant-LR table updates. Consider tail averaging of
only sparse tables versus whole-model EMA, masked/lazy sparse optimizer exactness,
and matched-token numerical diagnostics. FP8 should be scoped to selected forward
GEMMs with backward/optimizer precision retained, not assumed globally safe.
Give testable hypotheses and negative evidence, primary citations, implementation
cost, expected failure signatures, and a small orthogonal experiment matrix.
