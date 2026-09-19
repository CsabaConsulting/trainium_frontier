# FP8 backward and reuse experiments

Updated 2026-09-19. These are proposed extensions, not measured full-FP8 gains.
The packed mixed-precision component uses actual FP8 forward GEMMs and BF16
backward:15.957811 ms versus15.625856 ms native, a2.1244% isolated loss. Model
integration passes but whole-model timing loses17.7363% (587.956385 versus
499.384107 ms). Caller traces/ingestions/analyses are complete and preserved;
ordinary Linear source shows the largest extra spills and engine time. Both FP8 incoming-gradient dots now
pass9 independent cases but gain only0.56–1.65% with incremental packing charged.
Fused FP8 data backward passes9 cases and matches the separate FP8 algorithm;
6.205545 ms versus5.505396 ms BF16 is12.72% slower. No autograd or FP8
weight-gradient integration follows these measurements.

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

## Compiler-scope diagnostic

A separate fullgraph-compiled MLP variant passed both real-Linear integration
widths, including master-gradient/cache/restore checks. Source95 full-model
comparison is complete: native499.641620ms, monolithic FP8586.422830ms, separate
FP8619.975017ms. All clean; separation adds5.72% over monolithic and24.08% over
native, so this strategy is rejected for speed. Last losses differ; no
full-model bitwise equivalence is claimed. The original spill lead remains an
attribution, not proof of the tensor or of a successful remedy.

## Share backward quantization using activation-scale transfer

Proposed algebraic extension, not implemented or a measured gain. Let saved
hidden values be `H[t,h] = QH[t,h] / sH[t]`, using the existing positive
power-of-two row scales. Choose one global backward scale `sG` and quantize
`Gprime[t,c] = G[t,c] / sH[t]` into `QGprime = quantize(sG * Gprime)`.
The resulting chosen gradient approximation is
`Gtilde[t,c] = sH[t] * QGprime[t,c] / sG`. Then:

```
dWdown = QGprime.T @ QH / sG
dHidden = sH[:, None] * (QGprime @ QWdown) / sG / sWdown
```

The row scales cancel inside the weight-gradient contraction, while they stay
outside the channel contraction for the data gradient. This lets the SAME
quantized gradient serve both products. Preserve derivative/BF16 boundaries
and then choose a common scale `sD` for quantized preactivation gradients `QD`:

```
dWup = QD.T @ QX / sD / sX
dX = QD @ QWup / sD / sWup
```

Here `sX` is the already gated fixed RMS-input scale. Sequential rescaling is
intentional: multiplying inverse scales first previously underflowed. Returning
raw `QX`/`QH` plus scale metadata from forward may replace the larger dequantized
backward saves. Measure those layout/save costs and all four backward products
together; separate two-product timing cannot establish the amortized benefit.

This changes the quantized backward algorithm. Quantization of an upstream
gradient is nonlinear, so do not label it an exact linear VJP or reuse the
old FP64-autograd surrogate gate without modification. Build independent CPU
quantizers and FP64 references for each chosen matrix product, then test
tiny/zero rows, signed cancellation, broad row ranges, overflow/underflow,
outliers and arbitrary upstream gradients. Global gradient scaling may lose
small but useful contributions; finite outputs alone are insufficient. Full
model speed and30-minute BPB remain required.


## First weight-gradient implementation, 2026-09-19 02:52 UTC

`fp8_weight_gradient_nki.py` now implements row-scale multiplication, row-major
legacy FP8 byte preparation and a long-token double-row weight product.
Down gradient uses BF16(G*inverse_hidden_scale), one online global calibration
and the existing hidden bytes; up gradient uses the fixed input scale. The
product pairs corresponding positions in adjacent128-token blocks and rescales
sequentially into FP32. Shared data-gradient consumers are not implemented.

The probe declares exact decoded quantizers and rescaled BF16 inputs, the
existing independent11-bit-pair/FP32 accumulation bound, repeated calls and
input integrity. Eight small cases include zero/tiny/outlier/range/cancellation
behavior; two optional real shapes exercise16384-token reductions in both
orientations. Report scale-transfer underflow, quantized-away nonzero gradients
and error against the original BF16 surrogate separately from implementation
checks. Wide per-row activation ranges can make global scale transfer lose
useful contributions; an accurate implementation would not remove that risk.

Optional timing charges current-gradient transfer/calibration, both operand
byte preparations, reduction and FP32 stores. It assumes saved forward scales;
extra forward metadata stores, activation derivative, shared data products,
master accumulation/optimizer and whole-model costs are absent. Stage gates
precede any timer; no full-FP8 or training quality result is claimed.


Source103 all10 stage gates pass clean02:55:33. Ordinary real-shape gradients
differ from the original BF16 surrogate by2.6504% RMS down and2.7312% up.
The first charged layout loses: down6.431302 versus1.125327ms native;
up7.069405 versus1.084679ms. Clean02:58:04. No autograd or quality promotion.
These timers include current gradient calibration/transfer and both byte
preparations, but not extra forward scale stores or other backward products.

Code-level next candidate: compute a256x2048 output block using eight128x512
PSUM tiles (smaller groups for smaller shapes), loading the whole256-token
operand panels once per contraction block. This reduces repeated small DMA
requests and improves contiguous free extents. The original128x512 layout
reloads the same gradient panel for each output-column tile. That is an
algorithmic traffic observation, not a profiled latency attribution. Require
exact results versus the first FP8 product and the original independent pair
bound. If successful, measure all calibration/packing costs again.

### Source105 failure and source106 bank fix

Source105 grouped operands pass seven down fixtures but fail small up shape
[512,512,256]:65,520 coordinate-bound failures and0.500913 relative RMS. No
tolerance changed. Independent accumulators were spaced by the moving width
(256), placing two logical tiles inside one512-element PSUM bank. Source106
changes only allocation/index bank spacing to `nl.tile_size.psum_bank_fmax`.
All10fixtures pass, including bit-exact grouped-versus-original products at
actual16384-token shapes. Clean03:09:16.793153. This controlled change supports
the bank-alias diagnosis; original failure logs remain. Component timing follows
with all row-scale, calibration and byte preparation costs retained.

Source106 complete component clean03:10:51.448438: grouped down1.919147ms
versus native1.095259ms; grouped up3.177483ms versus native1.093426ms. Large
improvements from source1036.431/7.069ms, but still75%/191% slower than native.
All10gates and composed-versus-stages pass. Next diagnostic separately times
current-gradient calibration, each operand conversion, prepared FP8 product,
and down-gradient row-scale transfer. Isolated timings are not additive and
do not replace the complete charged component. This distinguishes conversion/
calibration cost from the long-token matrix product before further changes.

### Source111 independently controlled scheduling/layout candidates

Source108 down prepared FP8 product0.856747ms versus native1.085396ms, but
calibration0.344010ms,row transfer0.162511ms and byte conversions0.121265/
0.371519ms erase that advantage. Up calibration1.176563ms and prepared product
1.471031/native1.074664ms both need work. These isolated times are diagnostic
and nonadditive; complete charged timings remain1.923247/3.160304ms.

Two opt-in changes preserve the current quantized algorithm. Independent
128-row tile maxima followed by a small exact max reduction remove the
sequential calibration dependency while retaining its exponent policy. Balanced
512x1024 grouped output tiles fill8PSUM banks in the up product and reduce
summed operand-panel width versus256x2048. Each switch is tested separately
before composition. All independent pair/quantizer checks and exact original
calibration/product comparisons remain; no new numerical tolerance or claimed
quality gain. No forward saved-byte or shared-gradient cost is made free.


## Source113 scheduling and panel experiments, 2026-09-19 03:44 UTC

All three component experiments pass the unchanged10cases, original FP8 product
comparison, independent quantizer/pair bounds and clean host checks. Independent
calibration tiles alone give down1.909484/up3.167645ms; native1.090103/1.108476ms.
Balanced512x1024 panels give down1.909761/up2.530159ms; native1.097864/1.083665ms.
Combining both gives down1.885640/up2.515716ms; native1.114084/1.086046ms. None
wins the full charged component. Do not add isolated stage times as if they were
the same compiled pipeline. The combined prepared products are0.841978/0.813016ms,
while current-gradient calibration is0.335195/1.167815ms. Calibration and both
byte preparations remain charged; forward scale stores and shared data gradients
are still absent. This motivates exact calibration work, not a quality claim.

Three exact amax alternatives are now prototyped: BF16 sign masking before max,
unsigned magnitude-code maximum with a final BF16 decode, and VectorEngine fused
abs/max (tensor_scalar_reduce). The installed NKI0.5 signature/docs explicitly
support the latter unary abs and maximum reduction; device compilation is still
a gate. CPU ordering certification covers65280finite encodings but cannot prove
hardware subnormal behavior. The new device probe compares original and candidate
metadata with the independent CPU scale policy, decoded quantizer outputs,
repetition and input preservation, including pure subnormal tensors. Both real
calibration shapes are timed only after every numerical case passes. No existing
FP8 path selects these opt-in modes by default.

Scale-transfer sharing still needs data-gradient-specific tests. Quantizing
G/sH can discard rows with small hidden activations; multiplying a subsequent
dHidden product by sH cannot restore those lost values. A small weight-gradient
RMS error does not establish acceptable data gradients. Delayed state must be
charged, device-resident, advanced only at optimizer boundaries, checkpointed
and frozen for evaluation. These remain separate design candidates.


## Source116 composition and source117 preparation prototype,03:53UTC

Fused abs/max retains all10full-product gates. Up projection falls from2.515716
(combined original calibration) to1.987077ms, including calibration and both
byte preparations. Native BF16 is1.080190ms in the new run, so FP8 still loses.
The source117 prototype computes per-core partial maxima, combines them in a
second kernel and quantizes there using the exact existing exponent policy.
For the down projection it recomputes BF16(G*inverse_hidden_scale) locally in
both passes, avoiding its full HBM temporary. Both kernel calls are charged.
Disjoint partial writes and one owner for final metadata avoid write races;
completion between kernels provides the necessary synchronization.

The new18case gate includes maxima on either physical core, each with/without
scale transfer, zero/sign/tiny/pure subnormal/all-declared-code cases and both
actual shapes. It checks each partial maximum against CPU, exact original scale
metadata and byte encodings, decoded independent quantization, repeats and input
preservation. This preparation-only probe must pass before integrating it with
matrix products or claiming whole-backward improvement. Saved forward byte
reuse and shared data gradients remain later explicit costs/gates.


Source117 completes clean03:54:47. All18cases pass, including exact partial
maxima, metadata and raw FP8 byte encodings. Full gradient preparation costs
0.346830ms down and0.750925ms up, versus the already improved fused-amax path
0.535401/1.035930ms (original0.683044/1.564552ms). Both kernel calls and the
recomputed BF16 row multiplication are charged. Source118 now composes this
preparation with the balanced product; repeat all10product gates and include
activation preparation before any component promotion. Full autograd/quality
remain untested; do not call these numbers full FP8 training speedups.

## Two-core preparation composition result and per-row follow-up

Source118 passed all10 gradient-product cases with unchanged original FP8 bytes,
metadata and products. Both operand preparations are charged: down1.547501ms
versus native1.094661ms; up1.695063ms versus1.074635ms. Neither clears BF16.
This does not establish a full FP8 backward path or trainable quality gain.

Next bounded test applies the already verified fused abs/max instruction to the
per-row calibrations at BOTH data-backward boundaries. This retains the existing
per-row quantizer, sequential inverse scaling, BF16 ReLU derivative and saved
BF16 dPre; it does not substitute the weight product's global scale. Eleven
fixtures include independent quantizer/FP64 pair bounds, exact original eight
stage outputs, tiny/subnormal/signed-zero gradients, row independence and the
actual shape. Timing charges transpose/packing once plus two data-gradient
chains, paired with original fused FP8 and native BF16. No quality promotion
without complete composition and whole-step evidence.

Source119 fused per-row calibration COMPLETE and clean04:07:05. All11 cases
pass exact original eight-stage comparisons, independent quantizers/pair bounds,
subnormal/signed-zero cases and actual16384x1024x4096 shape. Two data chains plus
extra weight transpose/packing:5.654671ms versus original fused6.200740ms
(8.8065% faster), still slower than native5.510375ms. No all-four-gradient
composition/autograd or full training claim. Further backward work must also
reuse quantized activations produced by forward, charging any additional saves;
repacking those from BF16 is currently an explicit cost, not a free input.
