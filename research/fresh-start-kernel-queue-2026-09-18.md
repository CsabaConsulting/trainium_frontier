# Kernel queue and precision-specific experiments

Updated23:19UTC,2026-09-18. This is the executable/design queue, not a list of
claimed gains. Read `RESUME.md` and the latest postflights before using a host.
Every device request retains immutable source, cleanup and numerical gates.

## Ready requests and active work

| Item | State | Next action / evidence |
| --- | --- | --- |
| T1024/pack16 + original fused AdamW | 3282updates; small public0.992175; checkpoint local and cloud verified | Larger20.97M/T1024/O1 evaluation **0.9974687136975224**, clean; first larger public sub1.0, not private leaderboard |
| Padded optimizer ownership |3377updates/442.6Mtargets; checkpoint local and cloud verified | Larger public **0.9973302213667847**, clean; current best. T1024/pack16 ownership phase passes at515.473750ms versus528.353028ms:2.4376%, below3% gate; combine gated CE before another quality run |
| Scalar reciprocal + Newton | 34component gates,16.77% component gain; repeated optimizer/cache/restore gate now passes | Optional whole-step combination after the active quality runs; no approximation is in those runs |
| Scalar rsqrt correction | Mode5 passes34 gates but loses:1.363799vs1.262366ms; no promotion | Mode6 also passes34 gates but loses1.25%; both closed for this component; mode2 remains faster |
| Exact CE core/gather changes | Wave68 isolates discrepancy to forward outer-loop trip count1, for both input precisions; cores agree and native-lse backward is exact | Adaptive/gather passes all original/boundary/row-independence gates;5.679065vs12.339531ms FW+BW,53.9767% component gain. Compiled installed-autograd gate passes all36 checks; fresh whole-step control now active, candidate follows |
| Polynomial exponential | Nine device records pass;3.57–4.19times slower | Closed as a standalone speed replacement; retain the accurate implementation for diagnostics |

The CE experiment compares original, explicit one-core mask, two-core mask,
and two-core SBUF target gather. It retains tanh/exp/log, the target-relative
shift and512-wide reduction groups. Exact forward, saved logsumexp, backward,
repetition and row independence precede16384×8192 FW+BW timing. The one-core
arm is a kernel launch configuration within the existing LNC2 environment;
it is not a return to the previously hanging host-wide LNC1 configuration.

## FP8 can change which layout wins

The user's hypothesis is part of the plan: a layout that loses in BF16 may
win in FP8. Evaluate the implementation in its intended precision. Earlier
separate FP8 projection results reject their measured layouts and costs; they
do not reject a compressed full MLP. No evidence establishes a full FP8 gain
yet, and no BF16 timing is used as a veto on a new FP8 layout.

The concrete candidate is the full-cache MLP from `mlp_fused_bf16.py`. At
D1024/H4096/token_tile512, its source-level live estimate is16MiB for the two
BF16 weights,5MiB for input/hidden tiles, and another1MiB for grouped output,
before reserve/allocator overhead. That sits near the on-chip capacity. Its
measured caller graph spills and loses despite numerical correctness. Halving
the two weight caches alone saves8MiB; compressing activation tiles changes the
allocation again. A spill threshold can produce a nonlinear benefit. These
are live-size estimates, not proof of actual allocation or saved runtime.

Use a factorial investigation so storage and compute are distinguishable:

1. Current native BF16 forward/custom backward: the deployed control.
2. Current full-cache BF16 NKI MLP: reproduce only as the paired layout control.
3. FP8 weight storage with BF16 compute: compress the retained caches and
   dequantize only the tile consumed by the matmul. This isolates on-chip
   capacity/traffic, while charging quantization/dequantization and gradients.
4. Full FP8 forward island: FP8 up/down GEMMs, with activation/weight scaling
   and conversion inside the measured path. Keep the hidden activation on chip
   between GEMMs, and save what backward actually consumes. Initially retain
   a declared BF16 backward to obtain a complete training-component cost;
   this is mixed-precision training, not a claimed full FP8 pipeline.
5. FP8 backward island: add input-gradient and weight-gradient products with
   their own scale/overflow/accuracy gates. Charge reduction, transpose, saved
   data, master-gradient output and per-update cache refresh. Only this stage
   answers the fuller FP8-pipeline question.

The grouping of contraction elements must match on both matmul operands.
Legacy double-row packing can pair corresponding positions from two128-wide
blocks, avoiding an extra interleave. That changes pairwise rounding order:
record it and extend the already characterized empirical FP8-pair reference.
Do not silently treat its order as identical to the earlier adjacent-pair probe.
Use only the installed NC-v3 legacy E4M3/E5M2 APIs, not Trn3 MX/OCP assumptions.

Numerical gates distinguish the chosen quantized computation from its quality
effect. Check casts/scales/overflow, stage outputs against independent CPU
references, all input/weight gradients under an explicitly stated quantizer
surrogate, and error against the BF16 model. Include zero weights/activations,
saturation, outliers, small values, changing parameters/cache versions and
initial/trained weight regimes. Declare bounds before hardware results; do
not retrofit a failed BF16 tolerance into an FP8 pass. End-to-end BPB still
requires the full30-minute comparison after a complete-step gain of at least3%.

The storage-only prototype passes its full wave69 gate under the explicitly
selected FP64-propagated backward contract. Four cases include the real
16384x1024x4096 shape; compressed/uncompressed NKI outputs and gradients match
exactly. The original CPU-BF16/cross-compilation discrepancies remain recorded
as diagnostics, not erased. Online calibration, FP32 master casts, cache
preparation and two accumulated forward/backward calls are charged per update.

It **loses** isolated timing:17.898402ms FP8 storage/BF16 math versus14.989582ms
paired NKI BF16 cache and15.641984ms native original. Repeated tile
conversion is a plausible cost to investigate, not a profiled attribution.
The quantized real-shape model differs from original BF16 by4.111% forward RMS;
passing implementation gates does not establish acceptable training quality.
No optimizer, collectives or surrounding-model spill savings are included.
Do not promote this storage-only variant from these measurements. Actual FP8
GEMM forward/backward islands remain distinct candidates. A forward prototype
now exists in `fp8_mlp_island_nki.py`, with its stage gate in
`fp8_mlp_island_probe.py`. It has not yet passed hardware compilation or accuracy.
It quantizes activations per token, avoiding future-token and cross-row scale
dependence, and uses both projections in double-row mode. Token-major hidden
tiles make row calibration straightforward; explicit transposes return the
matmul layout. The per-token scales, extra transposes and backward saves all
belong in later complete timing. First gate covers seven small cases plus an
optional real shape: exact calibration/casts, FP32 projection stages against
the independently characterized11-bit pair rule in the new contraction order,
rounding, repeated calls and row independence. Backward and timing follow only
after the forward contract passes.

## Additional retro-search ideas retained

| Family | Concrete use | Stage / gate |
| --- | --- | --- |
| Newton / polynomial correction | Reciprocal update; rsqrt seed correction with fewer multiplies than two Newton steps | Reciprocal integrated gate passes; rsqrt modes5/6 pass all34 numerical gates but lose timing; no promotion |
| Exponent-bit range reduction | Power-of-two FP8 scales without log2/floor/exp2 | Implemented; calibration-only gate covers zero, underflow, outlier, endpoints and all declared finite BF16 codes |
| Compact finite-domain tables | Joint softcapped exponential and derivative, with analytic tails | CPU study passes all65280 finite BF16 codes and13 loss/gradient scenarios;4/5/6MiB alternatives. Device gather/reduction/reciprocal/log and timing pending |
| Polynomial/rational composition | Evaluate multiple required outputs together, reuse shared intermediates | Design; compare complete loss FW/BW, not standalone approximants only |
| Representation / traffic elimination | Let loss consume existing BF16 logits and return correctly rounded BF16 head gradients | Exact dtype-boundary probe after FP32 CE scheduling result; caller integration required |
| Work elimination | SBUF target gather; optimizer compute ownership; precomputed symmetry | CE component passes with53.98% gain; ownership full training/evaluation complete; RoPE trig already precomputed |

For power-of-two scaling, a useful candidate policy for positive normal amax
with biased exponent b is scale=2**(133-b), so scaled amax lies in[64,128).
Its inverse is2**(b-133). Both follow directly from exponent bits and provide
headroom below the legacy E4M3 maximum. Clamp amax away from zero before bit
extraction, handle the declared dynamic range and apply rescaling in an order
that does not prematurely underflow a product of inverse scales. Verify the
actual scale policy and quantized outputs against CPU before timing. This is
a numerical representation idea from the broad fast-function search, beyond
sqrt or a lookup table.

For a compact BF16 table, avoid allocating all65536 FP32 values on every one
of128 partitions (32MiB per function). An active exponent interval, sign handling
and analytic small/large tails could fit a few MiB, but those extra instructions
and gathers may lose to the hardware's piecewise polynomials. Preserve stable
confident-token loss evaluation; a naive logsumexp-minus-target rewrite can
reintroduce cancellation. Any table must also supply a tested backward path.

Primary sources and the13 user links are recorded in
`fresh-start-retro-numerics-2026-09-18.md`. No claim is made that a leaderboard
team uses any of these techniques.

Wave71 actual-FP8 first run fails at frontend helper lookup, before MLP device
execution. Calibration12checks pass and postflight is clean. The factory now
registers its undecorated helper names in module globals; the same forward
gates will be rerun on a new immutable source. No tolerance changes. The STE
BF16 backward and independent FP64 surrogate-reference/bounds helper are written,
but hardware backward gating and complete timing remain outstanding.

Wave72 resolves helper lookup and reaches backend compilation. The first FP8
transpose reports that its PSUM output must have element step2. The next source
allocates128x256 FP8 temporary storage and views every second element as the
logical128x128 transpose, then compacts into SBUF. Numerical gates are unchanged.
The BF16 STE backward/full-component driver is prepared, but dispatch waits
for the small forward-stage gate. It includes FP32 master gradients, two
microbatch accumulation and all calibration/cast/save costs; no full-FP8
backward, optimizer, collectives or whole-model timing is claimed.
