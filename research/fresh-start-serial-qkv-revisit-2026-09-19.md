# Revisit QKV execution while preserving the trained model

Updated 2026-09-19 23:33 UTC. Source151 completed cleanly on both hosts. Both layouts
pass all9 independent FP64 forward/all-six-gradient cases and lifecycle gates.
The token layout is5.358% slower than its native complete projection control
(4.947391 versus4.695781ms); channel layout is10.478% slower
(5.208927 versus4.714912ms). No whole-model or quality promotion follows.

Two early failures were diagnosed and preserved: source149's unbound NKI
`kernel_assert` required module-global helper resolution; source150's logger
used `kind` for both marker argument and case payload. Source151 fixes only the
logger after source150 fixed symbol visibility. Neither math nor bounds changed.
All three waves have clean per-job postflights and local raw logs.

The real-attention adapter and gate are now prepared. They keep the existing
attention body, use its preprojected qkv argument and preserve contiguous
outputs, independent Parameter/cache identities and native evaluation. The gate
requires the source151 component evidence and identical primitive bytes, then
checks real nonlinear attention, causal/batch independence, cache/master
lifecycle and strict native restore. A matched real32x1024x1024 attention timing
includes refresh, forward/backward, gradients and all layout/dispatch costs.
This diagnoses whether complete caller behavior differs from the component;
it does not justify a full-budget quality trial in advance. Hardware pending.

`serial_qkv_nki.py` supplies two BF16 layouts: weight-stationary with up to512
tokens across the TensorEngine free dimension, and token-stationary with up to512
output channels. Both retain ordinary separate square weight inputs, load/transpose
those weights on chip, and return three separate contiguous token-row outputs.
They support deliberately different Q/K/V inputs; no normalization gradient paths
are combined. The custom backward keeps six independent native BF16 products;
the ordinary cast graph sends separate gradients to the original FP32 masters.

The component probe declares FP32 gamma_K plus one BF16 output-rounding bounds
against independent CPU FP64 products before hardware results are known. It gates
both native and candidate forwards/all six gradients, zero/tiny/outlier/cancellation
cases, updated weights, cache refresh, two-microbatch accumulation, nonmutation,
repeatability and stream/row independence. Real32768x1024 gates precede alternating
native/candidate timing including three cache casts, all forward/backward products,
copies and dispatch. Neither a component win nor contiguous outputs alone proves
that the complete attention caller is faster or equivalent.

The current CausalSelfAttention already accepts optional preprojected `qkv`.
A subsequent adapter can call the original attention body through this argument,
preserving its VE gates, RoPE/normalization, attention and output projection without
duplicating that code or converting the block to a parallel architecture.

The older fused-QKV trial is a valid negative for its exact implementation.
`trn2-experiment-results.md` records206 versus240 steps and worse BPB because
strided Q/K slices disabled the NKI RoPE/normalization path. The later parallel
block experiment restored contiguous Q/K, but changed when attention and MLP
consume the residual. Its quality loss does not isolate serial QKV execution.

There is another confound in the old switch: `CausalSelfAttention.__init__`
replaces three independent square Parameters with one3D-by-D Parameter.
`GPT.setup_optimizer` groups matrices by shape and Muon orthogonalizes each
complete matrix. Thus the switch changes optimization as well as execution.
This follows from local code; it is not a claim about the relative size of
that effect in the historical run. Slice-wise random initialization also does
not guarantee the original RNG order.

A narrower experiment would preserve all original Parameter objects, initial
weights, separate Muon states, model/checkpoint keys and serial block order.
Cache packing would derive from current weights inside the charged refresh;
the execution kernel would return separate contiguous Q/K/V outputs. The first
version should retain the existing RoPE/normalization kernels and assert that
their fast path is actually used. Combining projection and RoPE is a later
kernel change with its own arithmetic and rounding gates.

Two easily missed contracts matter. First, `Block.forward` captures the Q/K/V
input before applying its local-convolution residual update. A replacement
must consume the supplied attention inputs, not normalize the later residual.
Second, the baseline keeps three normalization gradient paths when
`USE_QKV_NORM_CSE=False`. Sharing a forward value must not silently collapse
three input gradients or reorder BF16 rounding. Keep distinct Q/K/V input
derivatives in the first gate. Refuse shifted or differently sourced streams
unless the kernel explicitly supports them.

The decision sequence is:

1. Gate actual32x1024x1024 inputs, three current master weights and refreshed
   caches against the existing independent projections. Preserve each master
   gradient and each input gradient, two accumulated microbatches, repeated
   calls, row independence, causality, cache refresh and native checkpoint load.
2. Charge packing, all output copies, forward/backward and accumulation in a
   complete component measurement. Exact retained values are preferred; any
   deliberate accumulation change needs independent bounds and a separate label.
3. Gate and measure the full model with the existing serial architecture,
   seed42, pack32/GA1, unchanged optimizer groups and RoPE fast path. Inspect
   its caller allocation if a component gain disappears, as it did for FP8/head.
4. Require a material whole-step gain before a30-minute quality run. Earlier
   claims that throughput becomes worthless after one data pass are not an
   advancement rule: current public best completes more than one corpus pass.

Source references: `train.py` classes `CausalSelfAttention`, `Block`,
`GPT.setup_optimizer`, and `rope_norm`; the older parallel/coalescing notes
around `USE_PARALLEL_BLOCK`; `research/trn2-experiment-results.md` Solution13c.
Current caller line2853 represents many Linear operations; its missing HLO,
framework and layer metadata cannot identify QKV as the cause of FP8 spilling.
This remains a bounded hypothesis, not a profiler-derived QKV speed prediction.
