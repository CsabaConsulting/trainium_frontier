# Revisit QKV execution while preserving the trained model

Current update,2026-09-19 23:48 UTC: source153 diagnostic is complete and clean on both
hosts. At widths256/1024, with cached and uncached weights, replacing Q or K
alone is exactly native. Replacing V alone reproduces the failure only when
VE is present; without VE all partial/full replacements are exactly native.
The FP32 V hypothesis reduces VE-case RMS from0.004142/0.004067 to
0.00004476/0.00008954 and eliminates bound failures. The stored-projection
controls omitted requires_grad on Q/K/V, introducing a compile-context
confound; their no-VE results differ from the partial replacements. The next
version fixes that diagnostic fixture. Neither observation replaces the
original failed attention gate.

Source154 adds opt-in FP32 V output from the same BF16-input dot accumulator.
Q/K storage and all backward matrix operands remain BF16. The component
probe tightens V-output bounds to FP32 rounding and adds a non-BF16 incoming
V-gradient case. Old BF16 regression, independent FP32-output/six-gradient
bounds and original unchanged attention lifecycle/causality/restore gates
must all pass. Adapter FP32 V is selected only for VE consumers. Hardware
and performance remain pending; extra output traffic is charged in timing.

The [AWS transpose API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html)
and [Trainium2 architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html)
support bit-preserving FP32 transpose on this generation. The token-stationary
layout avoids a result transpose; the other layout retains an FP32 tile
through its output transpose. No SDK files are changed.

A useful limit for the next decision: nine32768x1024x1024 forward/backward
matrix products require about618.5GFLOP. At the documented79BF16TFLOP/s per
physical core, the optimistic two-core compute floor is about3.914ms versus
our4.696/4.715ms native component observations. This is a nominal arithmetic
floor, not a whole-model attribution: data movement, scheduling and four-rank
resource contention also matter. It limits plausible BF16-only improvements
and makes broader FP8 projections/fusion a more promising follow-up than an
unbounded BF16 tiling sweep. Keep original gradients, VE precision and real
caller timing as gates when extending the precision region.

Source154 token FP32-V component PASSES all10 cases, including non-BF16
incoming V derivatives and production shape. Complete component5.727667ms
versus native FP32-V5.644936ms is1.466% slower. This is a different explicit
output contract from the earlier BF16 component, so do not mix its controls.
The corrected token frozen controls now match native exactly without VE;
with VE the BF16 materialization difference remains and FP32 V fits closely.
Its original whole-attention gate is running.

The source154 channel FP32 component fails to compile cleanly because its
final transpose scratch was left BF16 while the input is now FP32. Source155
changes that allocation to the output dtype; weight/input transpose scratch
remain BF16 and get distinct names. No SDK patch or numerical relaxation.
A fresh four-stage channel retry will preserve the failed source154 records.

Source154 token attention now has a precise remaining failure. Its B4/T128
checks at C256/1024 pass, including all gradients and cached accumulation,
causality, native evaluation/restore and padded rows. Production B32/T1024/C1024
then fails the unchanged output bound:47,860 coordinates, RMS0.0041308524.
Postflight is clean00:00:37UTC; no full attention timing was reached. The new
source156 diagnostic uses that exact seed152032/cache/shape and makes both
original and frozen inputs differentiable. It compares both V storage contracts,
partial Q/K/V replacements and frozen BF16/FP32 controls with/without VE.
This investigates possible shape/context-dependent rounding; it does not
assume the cause or fit a precision switch to one case.

Earlier preparation/results below are retained chronologically.

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

Source152's real-attention gate then failed on both layouts with identical
width256/uncached/VE output differences:190 coordinates outside the unchanged
bound, relative RMS0.0041420266. Both exited cleanly at23:35:01/23:35:18UTC.
This is a failed integrated numerical gate; no attention speed was measured.

Source153 diagnoses partial Q/K/V replacement versus native preprojection,
with/without VE and caches. Materialized BF16 and CPU-derived FP32 V controls
isolate a possible retained-precision boundary. Returning/fixing intermediates
can change fusion, so those controls are hypotheses, not new acceptance targets.
No tolerance is relaxed and the original failed gate must be rerun after a fix.

The real-attention adapter and gate are implemented. They keep the existing
attention body, use its preprojected qkv argument and preserve contiguous
outputs, independent Parameter/cache identities and native evaluation. The gate
requires the source151 component evidence and identical primitive bytes, then
checks real nonlinear attention, causal/batch independence, cache/master
lifecycle and strict native restore. A matched real32x1024x1024 attention timing
includes refresh, forward/backward, gradients and all layout/dispatch costs.
This diagnoses whether complete caller behavior differs from the component;
it does not justify a full-budget quality trial. Its original hardware gate
currently FAILS as above; the diagnostic does not bypass that result.

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
