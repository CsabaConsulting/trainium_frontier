# Continued kernel work and research — 2026-09-20

User direction: keep both provisioned hosts doing useful experiments and use
fresh research to find new directions or revisit negatives with a changed premise.
Continue the existing pre/postflight, immutable source, host locks, log mirroring
and GitHub backup protocol. No seed sweep, quality promotion or submission is
implied by a diagnostic result. The coordinator runs only its finite prepared
list; it does not implement an autonomous idea generator.

Source156 production diagnosis finished cleanly on both hosts. Results are
identical: individual Q/K/V replacements and native preprojection are exact;
joint BF16 differs at81 coordinates with VE, joint FP32-V at47,860. This makes
another undirected tile sweep inappropriate before understanding the interaction.

Source157 keeps primitive arithmetic, adapter, train.py and all numerical bounds
unchanged. It extends only the diagnostic to replace QK, QV and KV pairs and
adds a direct all-three wrapper to distinguish the adapter's slicing context.
The original all-three adapter and native-preprojected case remain controls.
The gate can be random (original fixture) or zero (actual GPT initialization).
The output projection stays nonzero to avoid hiding upstream errors. Zero-gate
success cannot substitute for the random/trained-gate acceptance check.
Materialized controls already measured in156 are omitted from this focused run.

Initial finite queues:

- trn21: production B32/T1024/C1024 pairwise test with random gate, then zero gate,
  token layout, seed152032, cached differentiable inputs.
- trn22: B4/T128/C1024 pairwise test with random gate, then zero gate, channel
  layout, original small-fixture seed152055, cached differentiable inputs.

Both layouts already give exactly the same original production diagnosis.
These new grids isolate pairs and gate values within each shape; a conclusion
about shape alone still requires a same-layout follow-up if results differ.
Component prerequisites are reused only when primitive and probe bytes match
and the recorded result is complete, source-matched and clean. Each remote
job is bounded to1800seconds plus30seconds kill grace; each two-job sequence
has a4800second deadline and requires exact finite diagnostic record identities.
A failed process or incomplete result stops advancement. A recorded numerical
mismatch is useful diagnostic evidence and never an integration approval.

Next decision: distinguish projection fusion from gate rounding and consumer
context. Do not hardcode a precision choice by input shape. Full forward/backward,
cache refresh/accumulation, causality and checkpoint checks remain required before
a full-step timing or training experiment. Broader FP8 projection coverage and
backward-first FP8 remain open because the MLP-only integration exposed costly
surrounding compiler behavior; passing or failing BF16 QKV alone cannot decide it.

A Tavily pro research request was started with a primary-source-only prompt on
broader FP8 regions, backward-first precision, compiler resource boundaries,
optimizer/collective elimination and revisitable model/data confounds. Request
ID21d003c3-9a58-45d3-8df0-321de7fc03e6 is preserved with its prompt under
research/external/trainium-next-directions-2026-09-20.*. Treat its report as
research leads; verify API claims against official docs and local installed SDK
before implementation. Historical memory's September8 scaled_mm failure is
specific to that frontend and superseded by our working NKI FP8 primitives.
It does not close direct TensorEngine FP8. Current limitation is whole-step cost.

## Next independent screen: FP8 only for square dX

Source158 adds `scratch/fp8_square_dx_probe.py`. Keep forward and weight-gradient
products native BF16; replace only the input-gradient product for square attention
projections with the already independently characterized NKI E4M3 gradient GEMM.
This tests a different precision boundary and matrix aspect ratio from the full
MLP experiment. The old rectangular gradient pilot borrowed forward FP8 cache
work; this screen charges original master casts, current calibration and dual
packing every three-product call. Unused dual producer outputs remain charged.

Seven small cases plus32768x1024 gate independent CPU forward/weight products,
FP8 weight calibration/byte layout, per-row gradient quantization and the original
pair arithmetic bound. Repeats, input nonmutation and row independence are checked.
The native forward/weight product must also remain exactly equal. Compare complete
three-product cost only after all gates pass. This is not actual autograd/master
accumulation integration or a full-model timing. A component loss closes this
specific boundary unless profile evidence supports a concrete cost removal;
a win justifies actual autograd and cache-lifecycle gates, not a quality run.

The Tavily report is complete and preserved. One central recommendation is
inapplicable: it suggests MXFP8 hardware kernels for Trn2, but the official
[nc_matmul_mx API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul_mx.html)
explicitly restricts the instruction to NeuronCore-v4 or newer. The page's broad
Trn2 relevance tag is not the instruction support statement. The report also
uses the wrong backend name (`torch-neuron` vs our installed `neuron`) and
mixes changing Muon optimizer matrix arithmetic with ordinary network GEMMs.
Do not apply those suggestions. No SDK upgrade or optimizer precision change.
The usable hypothesis is backward-first coverage with our working NC-v3 E4M3
primitives; no documentation establishes a speed gain on this workload.

## Source157/158 outcomes and the next discriminating controls

All four source157 requests completed cleanly. At production shape, every pair
with random VE gate differs by the same81 coordinate failures; with zero gate,
all pair/full/direct replacements are exactly native. At the small shape, QK
alone is exact, but pairs containing V fail both random/zero gates (710/626
coordinates). The direct wrapper equals the ordinary adapter in these results,
so removing its redundant slice alone is not a numerical fix.

Source158 is complete/clean00:56:43 UTC with all8 declared numerical cases and
40 independent native-product audits. Forward and weight product are exactly
native. Real square dX quantization changes relative RMS by0.03772356, reported
separately from implementation correctness. Native complete three-product
median1.617033ms; FP8 dX with all preparation1.839821ms,13.78% slower. No autograd,
model, quality or speed promotion. This is a useful negative for that boundary.

Source159 runs two independent follow-ups:

- trn21: at the production fixture, freeze only the gate GEMM preactivation
  from an independent CPU FP64 product of the actual BF16 operands, in BF16
  and FP32 variants. Compare native and joint-NKI consumers to the UNMODIFIED
  native authority, with original all-NKI and native-preprojected controls.
  Freezing removes gate derivatives and may affect fusion, so it is diagnostic
  only. It distinguishes gate arithmetic from V storage without accepting a
  precision switch based on shape or zero initialization.
- trn22: reuse source158 gates and byte-identical code, then measure native/
  fresh-FP8 three-product calls, optimistic cached FP8, both isolated dX
  products, and current cache preparation. Require all corresponding outputs
  to be exact first. If cached FP8 still loses, cheaper preparation alone
  cannot rescue this implementation in the measured context. Separate scope
  timings are not additive and none replaces a full-step measurement.
