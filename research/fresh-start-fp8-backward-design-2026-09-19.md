# FP8 backward and reuse experiments

Updated 2026-09-19. These are proposed extensions, not measured full-FP8 gains.
The packed mixed-precision component uses actual FP8 forward GEMMs and BF16
backward:15.957811 ms versus15.625856 ms native, a2.1244% isolated loss. Model
integration now passes; full-model allocation timing is in progress. The first
FP8 incoming-gradient dot is under independent stage tests, not yet autograd.

## Why the fixed input scale matters beyond one reduction

Write the quantized input as `X[t,c] = Qx[t,c] / sx`, with one fixed `sx`
from the RMS-normalization bound. For the up-projection weight gradient,

`dWup[h,c] = sum_t dPre[t,h] * X[t,c]`.

If `dPre = Qd / sd` also uses one scale, the two inverse scales factor out of
the token contraction. If instead `sx[t]` varies with every token, it stays
inside the sum. An ordinary FP8 dot followed by one rescaling cannot reproduce
that sum. This is an algebraic observation, not a benchmark. Our fixed-input
contract therefore opens a simpler weight-gradient representation even though
its measured standalone component gain was only0.519%.

The hidden activation still has per-token scales. Its down-projection weight
gradient needs a separate strategy: backward-only requantization with a common
scale, scaled partial contractions, or an independently gated fixed/delayed
hidden scale. Requantization changes the gradient approximation and must not be
described as exactly differentiating the previous saved-value surrogate.

## Bounded experiments in order

1. **Caller layout:** profile the already-gated token-row backward. Wave78 had
   an extra compiler transpose on one core and populated spill counters. Wave82 now confirms removal of
   8,704 private compiler transpose instructions and the populated spill counters;
   forward device time falls from 3.568860 to 2.801243 ms. Do not use summed instruction
   durations as additive wall time or missing DMA metadata as exact bandwidth.
2. **Packed FP8 weight cache (boundary, component and model gates passed):** prepare the GEMM layouts once per optimizer
   update, retain them in HBM, and consume them directly in both microbatches.
   The old code prepares BF16 QDQ weights, then transposes/casts them again in
   each forward. Packed consumption now avoids those repeated conversions. Charge packed-cache construction, extra storage, invalidation
   and BF16 backward copies. Verify legacy FP8 byte representation at the HBM
   boundary; a framework dtype named E4M3 does not establish identical encoding.
3. **Producer/statistic fusion:** investigate producing squared ReLU and its
   range statistic together. Ordinary `nisa.activation` reduction is sum-only
   in the skill reference. `activation_reduce` wraps the same sum-only operation. Installed SDK
   inspection confirms `activate2` is NC-v4-only, so it is unavailable on Trn2.
   A fused maximum needs another documented NC-v3 route. A maximum of unrounded
   squares must be rounded consistently with the saved BF16 hidden values.
   Retain exact metadata/quantizer and stage tests; an unsupported API is not a
   valid implementation. Moving work to Scalar Engine can also lose throughput.
4. **Backward data-gradient island:** start with one FP8 product for `G@Wdown`,
   with separate gradient scaling and pair-rounding characterization. Add the
   ReLU derivative and the input-gradient product only after the first stage
   passes. Preserve FP32 master gradient accumulation. Current-gradient scale
   reductions are training work and must be timed.
5. **Backward weight-gradient island:** exploit the fixed `sx` for `dWup` and
   separately solve the hidden-scale issue for `dWdown`. The long token
   contraction, partial accumulators, PSUM capacity, stores and transpose layout
   belong in the cost model. A forward-only gain cannot establish this result.

For genuinely FP8 backward, ordinary FP64 autograd of the old STE is a quality
reference, not the full implementation reference: quantizing the incoming
gradient changes the VJP. Specify the new algorithm and use an independent CPU
quantizer, measured pair rule and FP32 accumulation bounds. Report discrepancy
from the original BF16 gradient separately. Test zero/subnormal/outlier/sign
patterns, all gradients, cache versions, accumulated masters and restore.

## Scale policies and state

The current input policy uses scale4/D1024 and scale8/D256. It has an explicit
finite BF16 range and passes exhaustive quantizer and compiled norm tests. It
does not justify an arbitrary fixed hidden scale: squared activations depend on
the changing weights. A Cauchy-Schwarz bound using quantized weight row norms is
a candidate, but could be loose enough to sacrifice useful subnormal precision.
It would need an L2 input bound, quantization/rounding margins and device-timed
row-norm preparation; the existing infinity-bound test alone is insufficient.

Delayed hidden/gradient scales are another candidate. Advance scale state only
at explicit optimizer-update boundaries, record every required statistic in the
charged workload, and freeze it during evaluation and repeated-forward checks.
Checkpoint and rank-consistency tests are necessary. Never compute a forward
scale from the current batch's other tokens: that would couple predictions to
future or unrelated tokens. Small-batch public validation cannot calibrate a
training model or its frozen evaluation state.

## Primary research and its limits

[NVIDIA's delayed-scaling documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
explains how historical maxima remove a separate current-tensor read, while
retaining scale-history state and synchronization costs. This supports testing
the policy; its CUDA implementation and performance are not Trainium evidence.

[Perez et al., 2023](https://arxiv.org/abs/2309.17224) study per-tensor activation,
weight and gradient scales in language models from111M to70B parameters. The
small end is relevant to this campaign's capacity; their results do not prove
that our legacy-FP8 pair arithmetic or 30-minute budget will preserve BPB.

[Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433) describe complementary
E4M3/E5M2 encodings. The installed NKI skill reference documents mixed E4M3/E5M2
operands in NC-v3 double-row mode; each new operand combination still needs a
device pair-rule probe. NVIDIA's E4M3 range must not be substituted for the
legacy E4M3 maximum240 already verified in our kernels.

[Fishman et al., revised2025](https://arxiv.org/abs/2409.12517) identify late
SwiGLU outlier amplification in much longer runs. This is motivation to measure
outliers rather than assume short forward gates prove stability. Our ReLU-squared
architecture and short training horizon differ; their failure mechanism and
throughput results are not direct predictions for this model.

Whole-model allocation can change the ranking of isolated components. A
numerically gated FP8 component that loses slightly may still justify a bounded
caller-profile experiment, provided the isolated loss is recorded. A full
training promotion still needs complete-step evidence and matched-budget BPB.

## Additional algebraic candidate: move activation scales into the gradient

A common activation scale is not the only way to handle a token contraction.
For saved hidden values `H[t,h] = QH[t,h] / sH[t]`, the exact algebra is

`dWdown[c,h] = sum_t (G[t,c] / sH[t]) * QH[t,h]`.

Rescale the gradient rows by the already saved inverse hidden scales, then
quantize that rescaled gradient with one scale (or one scale per output channel).
Those new scales factor outside the contraction. This retains the original
FP8 activation bytes and avoids a second activation quantization solely to make
its scale constant. The earlier statement about per-token scales remaining
inside an *unmodified* dot product still holds; changing the other operand is
a distinct route. The same transfer applies to adaptive input scales in dWup;
fixed RMS input scaling remains the simpler special case.

This is an unimplemented algebraic candidate, not a demonstrated FP8 VJP. The
rescaled gradient can have a different dynamic range, and quantizing it changes
the gradient approximation. Charge its rescaling, maximum reduction, layout,
quantization and master output. Test extreme scale ratios, zero/subnormal and
outlier gradients, overflow/underflow, long-token pair sums and accumulation.
Use an independent CPU reference for the specified gradient algorithm; report
error against the BF16 surrogate separately. Current-gradient statistics are
training work and must not become prediction-time state or validation calibration.
