# Next library-derived fusions

Checkpoint: 2026-09-20 07:11 UTC. Consult `RESUME.md` and final postflights for
current host state. These are conditional experiments, not accepted speed or
quality improvements. The pinned library is upstream92d11f63; bundled control
kernels, canonical training entry and frozen submission remain separate.

## Measured premise

- BF16 library training MLP: approximately2.4% faster as an isolated complete
  component, but14.6% slower in the complete model. Source188 reproduces2.1%
  faster with one rank and2.2% slower with four concurrent ranks; the four-rank
  timings also drift across rounds. All numerical/compiler/repeat gates pass.
- ROW FP8 source187: all ten original quantized-arithmetic gates pass. Production
  forward4.867048ms, versus nativeBF164.617179ms, charging online preparation.
  This forward excludes training saves/backward. Quantized-reference correctness
  does not establish equivalent training quality.
- Source189 directly reuses the library's stride-2 FP8 PSUM transpose in a fused
  weight producer. Up preparation0.180071ms vs0.256682ms; down0.137327ms vs
 0.202133ms. Both are exact in bytes/scales/metadata, with independent reciprocal
  and FP8 gates, changed-input tests and all eight fixtures. The complete ROW
  forward now passes all10 cases:4.686795ms vs4.596355msBF16, still1.968%
  slower. This is an improvement over5.412%, not an accepted forward win.

## Fuse BF16 input quantization into the library tile loader

The current forward first writes quantized token rows plus four scale bytes to
HBM, then the library reloads them. Source179 measured approximately0.913ms
for the production input producer including audit metadata. Avoiding that
materialization is a more substantial premise than further weight-only tuning.
An isolated producer duration is not an additive estimate of full-model savings.

The pinned code already distinguishes quantized and BF16 input, but ROW's
double-row projection requires explicit per-token scales. Passing BF16 input
alone is insufficient: `allocate_hidden_tensor_tile` only allocates input
scales when the input is already packed FP8. The experiment therefore needs
explicit, narrowly scoped adapters in the isolated namespace:

1. Keep `MLPParameters.hidden_tensor` as the actual BF16 tensor. Preserve all
   shape/dtype validation; do not disguise BF16 data as packed FP8.
2. Set the hidden compute tile's dtype to legacyE4M3 for the quantized
   projection. Preserve BF16 intermediate activation/output and existing
   weight/activation scaling boundaries.
3. Allocate one FP32 input-scale vector per token subtile through the existing
   SBUF manager. Allocate reusable BF16 source/absolute-value buffers and FP32
   absmax/reciprocal scratch in the same managed scope.
4. Add the actual new allocation footprint to the up-projection tiling bound.
   With128-row scratch and hidden widthC, source plus absolute-value buffers
   cost4*C bytes per partition; max/inverse cost8; scale vectors add4 per
   token subtile. Include alignment and any additional buffers actually used.
   Preserve allocator collision/underflow assertions and charge a smaller
   chosen tile if necessary.
5. Replace the packed-input loader with BF16 DMA, exact row absmax, original
   `max(absmax/240,1e-6)`, separately checked reciprocal, and FP8 conversion
   directly into the library's hidden tile. Reuse its existing transpose,
   double-row matmul, scaling, ReLU-squared and intermediate quantization.
6. In diagnostic mode only, expose actual input bytes/metadata for independent
   CPU checks. Compare the complete output exactly against the existing
   prepared-input ROW implementation. Require diagnostic/plain equality,
   repeatability, changed-input sensitivity and unaffected-row independence.

Run small zero/tiny/outlier/alphabet cases before production; retain the original
ten-case GEMM and rounding gates. The performance comparison must include current
weights, all remaining producers and plain forward. No backwards or quality claim
follows. This design is now implemented in source191
(`scratch/nki_library_row_fused_input.py`), with device validation pending.
The prepared-input control remains unchanged. The new probe retains every
original stage/arithmetic bound and additionally checks actual input bytes and
metadata, exact original intermediate stages, and independent changed-input
quantization. No input producer or quantized HBM tensor occurs in the plain
forward path. Both complete timings include current weight preparation.

Relevant pinned sources: `core/mlp/mlp_parameters.py`,
`core/mlp/mlp_cte/mlp_cte_constants.py`, and the basic CTE allocation, tensor I/O,
projection and tile-info files under `research/vendor/nki_library_92d11f63/`.
The copied ungated shard already has one separately proved allocation-cleanup
correction; new hooks must target its actual imported function bindings too.

## Address the model context before enlarging the training change

Source180 supplies same-host, same-source native/library training-forward
captures with all four ranks active. Match every runtime dispatch to its
preserved binary before comparing matmul precision, active intervals and source
attribution. Investigate ordinary Linear operations and spills surrounding the
custom MLP as well as the library itself. Source-family intervals overlap;
dropped DMA notifications prevent exact traffic/bandwidth conclusions.

If the surrounding graph is the dominant regression, test a specific changed
boundary or bounded scratch allocation. The already measured separate compiled
region was worse, so another region split needs a different, explicit premise.
Do not infer that every hand-written NKI replacement improves a compiled model.

Once a forward candidate is viable, design training saves and all four backward
products together. Preserve FP32 master parameters/accumulation, refresh each
quantized cache after updates, and state the quantization surrogate explicitly.
Storing the quantized squared activation and reconstructing ReLU through a square
root is a possible memory experiment, but changes the derivative approximation;
it must not inherit the existing BF16 backward equivalence claim.

Advance to a30-minute quality run only after full backward/cache/checkpoint gates
and a repeatable complete-step improvement of at least3%. Keep the same seed,
data order, global token batch and official public evaluation boundary. A failed
gate produces a preserved diagnosis and targeted new request, never a relaxed
old threshold. Each request uses an immutable committed source, finite timeout,
exclusive host ownership, pre/postflight cleanup and mirrored logs.

## Source190 bounded allocator experiment

Paired source180 traces are decoded and all dispatch identities match. The
library forward adds71.475086ms; ordinary Linear source train.py:2853 is a
major new attribution. Observed reload counters rise0.592 to13.467GB at that
line, while core0 Tensor active intervals rise15.574 to45.204ms. FP32 matmul
work falls. Dropped DMA notifications forbid exact traffic/bandwidth claims;
source intervals overlap. This motivates a hypothesis, not a causal conclusion.

The upstream allocator stacks upward from0 and heaps downward from212728B per
partition. Source190 restricts that upper bound to128KiB (trn21) or64KiB
(trn22). The same tile-size planner uses the actual remaining space. All
allocation/scope/collision checks stay intact. The budget is an explicit static
kernel argument. An AST proof verifies that only the function name/argument,
accepted-budget assertion and allocator bound change in the copied upstream
entry. The original training-save arithmetic and10-case probe remain unchanged.

Each host gets a bounded four-step sequence: component numerical/full-gradient
gates; real cached/uncached Linear integration; native full-step control;
budgeted joined full-step candidate.30 measured steps after5 warmups, fixed
seed42, same global batch, optlevel1, no separate compiled region. Component
speedup is not a prerequisite for this scheduling hypothesis. Stop on any
correctness/cleanup failure. Do not promote to a30-minute trial without a
repeatable complete-step improvement and all existing master/cache/eval gates.

## 07:19 UTC progress

Both source190 budgets passed all10 original training fixtures and both real
Linear/cache integration widths, including independent FP64 gradients and
exact FP32 accumulation/eval/checkpoint gates.128KiB complete component
14.884966ms vs native15.202572ms;64KiB15.648807ms vs15.196156ms. Whole-model
controls/candidates are running. No quality trial is warranted yet.

Source191 also explicitly suppresses the library's BF16 DMA-transpose shortcut
for the inline ROW path: quantization first needs channels on the free axis,
then the existing FP8 PE transpose supplies the double-row projection. Packed
input controls retain their original shortcut decision. All new allocations
use the existing manager, and the additional per-partition footprint is
4*C +8 +4*subtile_count bytes. No extra kernels or arithmetic tolerances are
introduced. Source/hash and inverse-AST checks are in
`run-logs/fs-w191-inline-input-ast-proof-r1.json`.

## Source190 final result and current follow-ups

Both fixed SBUF budgets pass all original component/cache/model gates and finish
CLEAN.128KiB full step562.878526ms versus486.986693ms native;64KiB566.748940ms
versus488.645588ms. No improvement of model integration and no quality run.
Source192 tests compiler placement separately while retaining virtual capacity
and lifetime checks; see fresh-start-library-compiler-placement-2026-09-20.md.
Source191 input-loader fusion is now running on trn21; source192 on trn22.

## Source191 forward win and the required training contract

Source191 COMPLETE/CLEAN at the production32768x1024x4096 shape. All ten
fixtures and all original frozen requirements pass. NativeBF164.602453ms,
prepared-input ROW4.687184ms, inline-input ROW3.926421ms. Inline is14.6885%
faster thanBF16 and16.2307% faster than the prepared-input control. Every
producer/master cast is charged. All three timing rounds agree on the sign.
Exact bytes/scales/stages and changed-input checks pass. Random-fixture
relative RMS versus unquantizedBF16 is still5.76445%; this is an approximate
forward, not evidence of training-quality equivalence.

Source193 captures all three unchanged complete functions in a single matched
profile, after the source191 prerequisite. Unprofiled timing is separate from
instrumentation. Runtime/compiled identities and raw hashes must be preserved.
Source192 independently tests whether compiler SBUF placement fixes BF16 model
integration. Full component and real Linear/cache gates pass; model timing
is still pending.

A full ROW training implementation cannot simply reuse the old fixed-scale
FP8 backward metadata. Define Qx, Qu, Qd and Qh as stored quantized bytes; sx
and sh as token-row scales; su and sd as output-channel weight scales. The
forward represents Xeff=diag(sx)Qx, Ueff=Qu diag(su), and
Deff=Qd diag(sd), with the retained BF16 ReLU/square/scaling boundaries.
An explicit straight-through surrogate can keep the2*ReLU derivative while
using those effective quantized operands. It must be labeled a new approximate
VJP, not the exact derivative of rounding.

For the data gradients, move sd into the incoming output-gradient columns
before the Qd-transpose product, then apply2*saved_ReLU. Move su into the
resulting hidden-gradient columns before the Qu-transpose product. Calibrating
and quantizing each scaled gradient introduces explicit new rounding stages;
include them in an independent reference and do not inherit fixed-scale bounds.
For weight gradients, move sh into the output-gradient rows for the down-weight
reduction, and sx into the hidden-gradient rows for the up-weight reduction.
This follows directly from the matrix products and differs from the old
fixed-x-scale autograd implementation.

Required implementation/gates before model timing:

1. Save the actual BF16 pre-square ReLU, quantized input bytes with sx, and
   quantized squared activation with sh. A training forward still needs these
   stores: input-loader fusion removes the immediate reload, not the future
   backward's need for data. Charge every save. Reconstructing ReLU via square
   root is a separate approximation and must not inherit this derivative.
2. Build both forward and backward weight layouts from current FP32 masters.
   Preserve raw FP8 bytes using the gated NKI transpose, not the known failing
   ordinary uint8 transpose. Measure all refresh/transpose/metadata costs.
3. Implement the four products with the row/channel scale movement above.
   Produce weight gradients and accumulate microbatches inFP32. Preserve each
   parameter's identity, RNG order, optimizer group and cache lifetime.
4. Derive a stagewise independent reference before measurement: exact decoded
   legacyE4M3 values, the characterized pair-add model where applicable, and
   every declared BF16/FP32 scale/activation rounding. Keep original failures
   visible. Include zero/tiny/outliers, changed weights, changed rows, causal
   and batch independence, repeated execution and exact master accumulation.
5. Gate real cached/uncached model integration and strict native evaluation/
   checkpoint restore. Then compare complete forward/backward/cache/optimizer/
   communication steps on one host. Require a repeatable3% whole-step win
   before spending a30-minute quality run; use the same seed and data order.

The14.7% forward result is sufficient to justify this design work. Its benefit
may shrink after training saves and scale preparation; the complete measurement
will decide. Source192/193 results should guide implementation order and layout.

## Source193 replication and source194 saved-forward gate (08:13 UTC)

The unprofiled production191 win replicated:3.908544ms inline,4.595774ms native,
4.688562ms preparedROW. Same host, same inputs, unchanged function AST and all
preparation included. Three captured binary identities match decoded profiles.
Large products areFP8; scalar/vector activation and scale work is still material.
All raw/derived files are preserved and cloud-verified. Profile warnings prevent
exact DMA traffic claims; unprofiled timing remains the speed authority.

The next finite gate implements only saved forward state, without choosing a
new backward quantizer prematurely. It returns output plus BF16 pre-squareReLU,
actual Qx/sx and Qh/sh. At32768x1024x4096 these five saves occupy436,469,760B;
all writes are included in the complete call, with no raw-up diagnostic store.
The original no-save forward, preparedROW and native controls remain. A native
forward with the current HxT ReLU-save contract is an additional measured arm.
Source191's exact decoded arithmetic/reference checks are retained and extended
to every saved tensor, changed inputs, repeats and row independence. No allocator
or tiling bound is weakened and no extra SBUF is allocated for these stores.

The later gradient producer must not reuse the forward activation floor1e-6
without justification. Multiplying early small gradients by output-channel
weight scales can make that floor erase them. Existing backward power-of-two
calibration uses a much smaller2^-100 guard. Declare/test the chosen gradient
rule, including zero-initialized down weights, early updated weights, zero and
tiny upstream gradients, before the full four-product VJP or model integration.

## Profile-directed exact activation experiment, conditional after194

Source193 provides a concrete next target for the earlier fast-function idea:
`nki_library_row_mlp.py:53` spends about0.626ms/core of Scalar active time
squaring the BF16 ReLU tile with `nisa.activation(op=nl.square)`. The preceding
scaled ReLU accounts for about0.805ms/core Scalar activity; row quantization's
conversion adds about0.493ms/core. Vector is also busy (~2.41ms/core total), so
engine intervals alone do not predict a speedup and are not additive wall shares.

A narrowly scoped candidate can replace square's scalar activation with a
Vector multiply of the same BF16 tile by itself. It must preserve both existing
BF16 scale/activation rounding boundaries, quantized bytes, saved ReLU and all
source194 exact tests. Do not assume the multiplication has identical denormal
or overflow behavior: zeros/tiny weights/outliers and actual-stage equality stay
mandatory. Measure the same complete saved forward, not just the square call.
The potential benefit is better engine scheduling, while the risk is moving
work onto an already busy Vector engine. This is a queued experiment, not an
implemented or measured improvement; it does not require a lookup table.

More aggressive fusion across the square/quantization boundary would change
rounding and therefore needs its own declared arithmetic/reference contract.
It cannot inherit source191/194 equality claims. The complete backward and
whole-step gates still determine whether any forward improvement advances.

## Source194 completed saved-forward result (08:37 UTC)

All ten fixtures and310 exact checks pass, with clean postflight08:33:51UTC.
Production native HxT saved forward5.323111ms versus ROW saved forward4.771002ms
(10.3719% lower latency). Current FP32-master casts/weight preparation and all
436,469,760B of actual forward state stores are included. The six returned
outputs match the original Scalar-stage diagnostics exactly, including repeats,
changed input and unaffected rows. Backward and whole-model performance remain
unmeasured for this ROW variant. See fs-w194-training-save-forward-proof-r1.json.

A further backward design constraint is the scale axis at weight-gradient
matmuls. After moving forward row scale sh(T) into GY(T,C), a new *per-token*
gradient calibration cannot be pulled outside the token contraction:
Qh^T diag(sh * sg_token) Qg is not Qh^T Qg times an output scale. Use a separately
gated global or per-output-channel gradient scale (equivalently quantize rows
of the transposed gradient), or explicitly retain the contraction-varying scale.
The same constraint applies to Qx and the up-weight gradient. Both weight-gradient
products need this design, including the transposed producer's two-pass/max and
layout cost. Per-token calibration remains convenient for data-gradient GEMMs,
where token rows are not the contraction dimension. A BF16 weight-gradient arm
is a useful accuracy/performance control before requiring all four products to
useFP8; charge reconstruction/layout costs and state its surrogate explicitly.
