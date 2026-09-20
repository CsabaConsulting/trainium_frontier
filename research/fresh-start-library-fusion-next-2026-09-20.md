# Next library-derived fusions

Checkpoint: 2026-09-20 06:54 UTC. Consult `RESUME.md` and final postflights for
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
  forward follow-up is running; its result must decide the next step.

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
follows. This design is researched but not implemented or dispatched yet.

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
