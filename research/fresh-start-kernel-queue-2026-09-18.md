# Kernel queue and precision-specific experiments

Updated 2026-09-19, 00:41 UTC. This is the executable/design queue, not a list of
claimed gains. Read `RESUME.md` and the latest postflights before using a host.
Every device request retains immutable source, cleanup and numerical gates.

## Ready requests and active work

| Item | State | Next action / evidence |
| --- | --- | --- |
| T1024/pack16 + original fused AdamW | 3282updates; small public0.992175; checkpoint local and cloud verified | Larger20.97M/T1024/O1 evaluation **0.9974687136975224**, clean; first larger public sub1.0, not private leaderboard |
| Padded optimizer ownership |3377updates/442.6Mtargets; checkpoint local and cloud verified | Larger public **0.9973302213667847**, clean; earlier best, now superseded by0.9938106631483421. T1024/pack16 ownership phase passes at515.473750ms versus528.353028ms:2.4376%, below3% gate; combine gated CE before another quality run |
| Scalar reciprocal + Newton | 34component gates,16.77% component gain; repeated optimizer/cache/restore gate now passes | Optional whole-step combination after the active quality runs; no approximation is in those runs |
| Scalar rsqrt correction | Mode5 passes34 gates but loses:1.363799vs1.262366ms; no promotion | Mode6 also passes34 gates but loses1.25%; both closed for this component; mode2 remains faster |
| Exact CE core/gather changes | Wave68 isolates discrepancy to forward outer-loop trip count1, for both input precisions; cores agree and native-lse backward is exact | Adaptive/gather passes all original/boundary/row-independence gates;5.679065vs12.339531ms FW+BW,53.9767% component gain. Compiled installed-autograd gate passes all36 checks; whole step500.372595vs519.395419ms,3.6625% faster. Fresh 30-minute T1024/pack16/ownership/CE run completed clean: 3,548 updates / 465.04M targets, small public 0.988605; checkpoint local/cloud verified. Matched larger public **0.9938106631483421**, new best; no private leaderboard result |
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
`fp8_mlp_island_probe.py`. It now passes hardware compilation, small/real forward and declared STE backward gates (wave73).
It quantizes activations per token, avoiding future-token and cross-row scale
dependence, and uses both projections in double-row mode. Token-major hidden
tiles make row calibration straightforward; explicit transposes return the
matmul layout. The per-token scales, extra transposes and backward saves all
belong in later complete timing. First gate covers seven small cases plus an
optional real shape: exact calibration/casts, FP32 projection stages against
the independently characterized11-bit pair rule in the new contraction order,
rounding, repeated calls and row independence. The complete-component result is18.181170ms versus15.662730ms native and
14.653363ms NKI BF16 cache. All scale/cast/save and two FW/BW costs are charged.
It loses16.0792% versus native; no promotion. Profiling preparation, forward and
backward separately is next. Actual FP8 backward is still unimplemented/ungated.

## Additional retro-search ideas retained

| Family | Concrete use | Stage / gate |
| --- | --- | --- |
| Newton / polynomial correction | Reciprocal update; rsqrt seed correction with fewer multiplies than two Newton steps | Reciprocal integrated gate passes; rsqrt modes5/6 pass all34 numerical gates but lose timing; no promotion |
| Exponent-bit range reduction | Power-of-two FP8 scales without log2/floor/exp2 | Implemented; calibration-only gate covers zero, underflow, outlier, endpoints and all declared finite BF16 codes |
| Compact finite-domain tables | Joint softcapped exponential and derivative, with analytic tails | CPU study passes all65280 finite BF16 codes and13 loss/gradient scenarios;4/5/6MiB alternatives. Device joint primitive passes but loses 19.04x; this full-value gather layout is closed. Full loss not integrated |
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

Wave73 resolves the FP8 transpose constraint without changing numerical gates.
Seven small cases, real-shape forward, six backward cases and two-microbatch
accumulation pass. Real-shape original-BF16 forward drift is5.7677% RMS; surrogate
correctness is not a quality result. Keep the earlier failures as compiler
diagnostics, not evidence that FP8 itself is unavailable.

A later scale-policy alternative is fixed or delayed, per-layer activation
scales with explicit frozen evaluation state. That can avoid per-token reduction
and transpose overhead, but current-batch/global activation calibration would
break token independence. It needs its own state/causality/overflow/quality gates
and is not part of the current implementation. Weight-gradient FP8 products
also need a scale policy compatible with their token contraction dimension.

The actual-FP8 profile (wave74) completed clean and exported its compiled and
eager binaries. Ingestion/cost attribution is pending. The compact table now
has an executable `compact_ce_table_nki.py` and `compact_ce_table_probe.py`: a
5MiB joint exp/derivative table, exponent-bit indexing and linear tiny tail.
Clamping the large tail at BF16 magnitude255 passes an additional exhaustive
CPU gate. Device numerical gates precede real16384x8192 primitive timing;
full-loss integration remains a later stage.

Wave75 compact-table first device attempt fails at frontend scalar-source
validation in tensor_copy_predicated, before execution. Installed source
confirms unconditional src.shape access despite scalar support in its docstring.
The next version uses one prefilled FP32 ones tile outside the loops. Clean
postflight, original numerical bounds retained, small gates must rerun.

Wave76 compact table passes exhaustive finite BF16, boundary, row-independence
and real16384x8192 gates. Joint primitive47.200750ms versus2.478831ms Scalar
composition:19.04x slower. Close this replicated full-value/GpSIMD-gather layout;
this does not test every possible coarse table/interpolation representation.

FP8 wave74 trace is local/ingested with every dispatch/binary matched. It shows
forward3.68ms vs2.08ms NKI BF16 cache, approximately equal4.8ms backward, and
SBUF peak0.5080 vs0.7734. The extra Scalar/Vector work and roughly0.98ms/core
transpose activity motivate a token-row GEMM orientation. Prototype is written:
`fp8_mlp_token_rows_nki.py`. It keeps per-token scaling/causality, saves ReLU in
token rows, removes the unnecessary minimum reduction for squared ReLU, and
keeps sequential inverse rescaling. Forward/BF16-backward/accumulation/full
component gates must pass again. Full FP8 backward remains a later design.

Historical memory expansion and raw logs confirm that depth8/width768 WAS tried
under earlier recipes; the implicit depth-to-width rule confounded some labels.
Any revisit requires the new kernel/ownership/packing stack and verified actual
parameter dimensions. It is a reserve capacity experiment, not the next kernel
trial or an assumption of untapped TensorEngine FLOPs.

## Wave77 result and next invariant experiment (00:08 UTC)

The token-row FP8 layout passes all small and real forward, declared STE
backward and two-microbatch master-accumulation gates. Complete component
medians: native BF16 15.673194 ms, NKI BF16 cache 14.643596 ms, FP8 storage
with BF16 math 17.935752 ms, token-row FP8 forward/BF16 backward 17.979725 ms,
original FP8 layout 18.161799 ms. All calibration/cast/save costs are charged.
The new layout improves only 1.0025% over original FP8 and remains 14.7164%
slower than native. No full-model or quality promotion. Attribute both FP8
layouts in a fresh trace; their saved-ReLU/backward layout differs.

**Queued invariant-based reduction elimination:** current MLP input is
`norm(x)` without a learned gain. In exact arithmetic RMS normalization gives
`||x||_infinity <= sqrt(D)`. Investigate a fixed power-of-two input FP8 scale
(4 for D=1024, 8 for D=256), with explicit FP32/BF16 rounding headroom, to
remove per-token input maximum/minimum reductions. This is a new restricted
input contract, not a pass for arbitrary unnormalized inputs. Test the actual
norm boundary, zero/tiny/spike inputs, all finite BF16 values in the declared
range, clipping/overflow and row independence. Quantization may change near
FP8 subnormals; retain an independent CPU quantizer reference and report error
against both adaptive FP8 and original BF16. Hidden-activation calibration
remains adaptive. Training promotion still requires backward/accumulation,
complete-step speed and full-budget BPB gates. This experiment is not yet run.

A weight-norm-derived fixed hidden scale is a later design only: its conservative
bound could be much looser and damage subnormal precision. Current-batch global
activation scales remain disallowed because predictions would depend on other
tokens. Full FP8 backward and BF16 CE input/output boundaries remain distinct
queued candidates. Historical `_scaled_mm` failure does not veto these NKI paths.

### Fixed RMS input scale: executable contract before hardware (00:12 UTC)

The candidate now exists behind `--token-rows --rms-fixed-input`; default
behavior is unchanged. D256 uses scale8 and input bound16.25; D1024 uses scale4
and bound32.5. The scaled maximum is at most130, below legacy E4M3's240. This
removes input max/min reductions; it retains per-token hidden calibration,
weight calibration, all saved values and the previous FP32 master STE.

`fp8_rms_scale_contract.py` declares exhaustive in-domain BF16 quantizer checks
and the analytic nearest-rounding bound `abs(error) <= abs(x)/16 + 2^-10/scale`.
Its compiled actual RMS boundary uses protected epsilon2^-23, FP64 norm truth,
a predeclared FP32 gamma_(2D+8) plus BF16 rounding bound, invalid-domain and
half-output negative controls, and zero/spike/exponent-range/row-independence
cases. Small MLP gates add normalized spikes and tiny activations. These are
restricted-input tests; the previous arbitrary-input tests retain their old
quantizer and are not relabeled as fixed-scale passes.

The later complete benchmark pairs native BF16, NKI BF16 cache, FP8 storage,
original FP8, adaptive token-row FP8 and fixed-input token-row FP8 on identical
normalized inputs. Shared caller normalization is excluded from all component
arms; full-model speed must charge the actual caller. Small numerical gates
must pass before the real-shape forward/backward/accumulation timing request.

### BF16 head/loss memory boundary: executable exact gate (00:17 UTC)

The protected model already rounds head output to BF16, then explicitly widens
it to FP32 before custom CE. The new scratch candidate accepts BF16 logits,
widens them in SBUF, performs the same FP32 loss/gradient arithmetic and rounds
only the returned gradient at the existing BF16 head boundary. Original FP32
input behavior remains available. This is not lower-precision softmax math.

At16384x8192, the three full-matrix kernel transfers (forward input, backward
input and backward output) have a source-level saving of768MiB per FW/BW.
This does not prove caller traffic or speed; dtype conversion costs are charged.
`softcap_ce_bf16_boundary_probe.py` compares protected original, adaptive-FP32
and adaptive-BF16 paths, both eager and compiled, on boundary row counts,
random/tiny/saturation/uniform/confident-correct/confident-wrong cases and
arbitrary signed or zero upstream gradients. Loss, saved logsumexp and BF16
head gradient must match exactly. Real-shape timing follows those gates.

No GPT caller patch is installed: the protected training/scoring files remain
unchanged. Caller integration must remove the redundant cast only after this
probe passes and must preserve prediction dtype/semantics and optional branches.
Do not combine this candidate into a quality run from component evidence alone.

Wave79 fixed-input FP8 now passes33,542/D256 and33,798/D1024 finite BF16 input
codes, both compiled norm contracts and all9 small forward cases. Real forward
also passes; full backward and complete timing are currently in progress.
Tiny normalized inputs can quantize to zero under this policy, with the change
explicitly retained in diagnostics rather than called BF16-model equivalence.

### Caller-layout investigation after wave78/79 (00:21 UTC)

The token-row FP8 profile has44 matched runtime dispatches. Its first FW is
3.603983ms versus3.686925ms old FP8; BF16 BW stays approximately4.8ms.
Outside the custom FP8 kernel, compiler private_nkl/transpose.py contributes
8,704 Tensor transpose instructions on core0. Reported spill saves67,108,864B
and reloads134,217,728B are now populated. NKI source-level buffer reductions
did not eliminate caller allocation/layout costs. Missing dynamic DMA metadata
means these counters do not establish exact bandwidth. Saved ReLU layout is a
plausible cause, not a proven attribution to a specific tensor.

`make_autograd(..., token_row_backward=True)` is the next test: compute
`dH=G@Wdown`, `dPre=dH*(2*ReLU)` in token rows, then `dX=dPre@Wup` and
`dWup=dPre.T@X`. This keeps the existing BF16 rounding boundaries and the
independent FP64 propagated-error contract; it changes logical GEMM orientation.
Gate all gradients and accumulated FP32 masters, compare a fresh same-forward
old-backward control, and profile any gain before whole-model promotion.

Wave79 fixed-input component passes but only improves0.519% versus adaptive
input (17.926315vs18.019795ms). Its native control is15.672957ms. Do not call
the fixed scale a training speed win. Full FP8 backward and a full-model
allocation probe remain distinct: an isolated loss need not veto a justified
caller-allocation experiment, but must be recorded and all numerical/state/
causality gates must precede it. No such caller probe is implemented yet.

### Wave80/81 outcome and ready integration (00:33 UTC)

BF16 CE boundary passes72 eager/compiled loss-and-gradient comparisons plus
exact saved statistics. At16384x8192, complete loss FW/BW including original
cast boundaries is10.071585ms adaptive FP32 versus6.233179ms BF16 boundary;
protected one-core original is16.252417ms. This is a38.11% component gain, not
a whole-step gain. The next scratch adapter removes exactly the known head
`.float()` assignment through an asserted AST change; prediction/evaluation,
CPU and output-bigram paths call the original GPT forward. No protected source
or checkpoint structure changes. A real Linear/cache/head gradient gate comes
before same-source full-step control and candidate requests.

Token-row FP8 backward passes all7 small and real forward/backward/accumulation
gates. Complete component16.515157ms versus17.920212ms old backward with the
same fixed-input forward (7.8406% gain), and15.675626ms native (5.36% slower).
Capture a fresh trace with both backward orientations and the same forward to
verify whether the compiler transpose and spill traffic disappeared.

The follow-on precision/cache/gradient contracts are now detailed in
`fresh-start-fp8-backward-design-2026-09-19.md`, with primary research. Fixed
input scaling can factor out of the token contraction in weight gradients;
per-token scales cannot. Packed FP8 caches, producer/statistic fusion, data
gradients and weight gradients remain separately gated design candidates.

### Compiled head boundary mismatch (00:41 UTC)

Wave82's exact Linear/head integration gate fails on the first nonzero compiled
case (256 rows, 256 channels, weight std 0.001). Eager loss and gradients agree;
compiled maximum differences are 0.000121117 in loss, 0.00000762939 in input
gradient and 0.001953125 in FP32 master gradient. The host is clean. Both
prepared whole-step requests are blocked by this gate, not queued for automatic
execution. The component's pre-rounded-logit result remains valid in its domain.

A separate rounding diagnostic captures compiler-input MLIR and compares
loss-only, exposed-logit, standalone-head and frozen-logit CE paths. Fresh heads
and checked cache values distinguish arithmetic changes from stale weights.
Exposing logits can itself change compilation, so its effect is measured.
No numerical bound is loosened and no training promotion follows from this test.

Installed SDK inspection also establishes that `activate2` is NC-v4-only; it
cannot implement a fused max reduction on these NC-v3 instances. Ordinary
`activation` has a sum reduction, and `activation_reduce` wraps it. Producer/max
fusion needs a different documented NC-v3 route or remains a design idea.

### Root cause and two narrower experiments (00:47 UTC)

Wave83 isolates the head mismatch across six width/weight-scale cases. Compiled
original logits retain FP32 accumulator values; the candidate equals their BF16
rounding exactly. Exposing logits changes neither loss, and frozen-logit CE
reproduces each result exactly. The full BF16 forward boundary is therefore
closed as an exact optimization of this compiled head. Its standalone component
speed result remains limited to pre-rounded inputs. Do not promote that result.

The new **gradient-only boundary** retains the original FP32 logits and forward
loss, but emits a BF16 CE derivative for the head's existing backward cast.
An arbitrary FP32-logit leaf would have a different gradient, so this is a
head-consumer contract requiring exact input and FP32-master gradients under
eager and compiled execution. It is separately gated; the earlier failed gate
is unchanged. The ready probe charges the real head/loss FW/BW after exact gates.

The new **packed FP8 cache** prototype produces both the existing BF16 backward
cache and a raw-byte legacy-FP8 GEMM layout once per update. Its first probe checks
CPU-decoded values, contraction layout, repeated bytes, cache versions, finite
BF16 alphabet, tiny/outlier/zero values and real dimensions. This is only the
cache boundary; full MLP stages, backward and charged update timing come next.
PyTorch E4M3FN encoding is not assumed to match these raw legacy-E4M3 bytes.

### Wave84 gates and wave82 profile attribution (00:54 UTC)

The gradient-only head boundary passes all 32 eager/compiled loss/input/master
gradient checks and the fresh timing-input check. Complete real head/loss FW/BW
is 14.992966 ms versus 18.200297 ms (17.62% gain). It preserves FP32 head logits;
the original BF16 forward-boundary failure remains recorded. A restricted public
loss adapter now rejects non-BF16 final norm, output bigram, weighted loss and
the incompatible forward adapter. Its installed-head gate precedes fresh
whole-step control/candidate phases; no full-model speed claim yet.

Packed weight preparation passes eight cases, including finite BF16 codes in
the declared range and both real matrices. Independent legacy decoding and
contraction layout agree exactly, as do the BF16 backward caches. The new MLP
consumer loads the packed bytes directly into its FP8 SBUF layouts. Forward
stage/row-independence gates precede compiled gradient/accumulation gates and
complete update timing. The paired control still builds layouts in each forward.
All online preparation, both caches and two microbatches are charged.

The new paired FP8 trace verifies removal of 8,704 private compiler transpose
instructions, and the earlier 64 MiB spill-save / 128 MiB reload counters are no
longer populated in the new forward interval. Device forward drops 3.568860 to
2.801243 ms; backward stays about 4.84 ms. All 57 dispatches map to binaries.
This attributes wave81's unprofiled 7.84% component gain; it is not a second
speed benchmark. Dynamic DMA metadata warnings still limit bandwidth claims.

### Packed MLP outcome and fair precision controls (01:07 UTC)

Wave85 passes ten packed-cache forward cases and eight training cases. All
paired saved values and gradients are exact, the independent FP64 surrogate
bounds pass, and two-microbatch accumulation reaches FP32 masters exactly once.
Complete update: packed FP8 15.957811 ms, previous FP8 16.442287 ms, native BF16
15.625856 ms. Packing gains 2.95% within FP8, but the candidate still loses 2.12%
to native. Preparation, both caches and all saved values are charged.

That small component loss permits a justified surrounding-model allocation
probe after cache/restore/causality gates. It does not justify a training quality
run yet. First, a fair BF16 control tests the same saved-activation/backward
orientation. It includes both the deployed contiguous HxT save and the component
benchmark's HxT view, so those layouts are not silently conflated. The existing
independent FP64/BF16 backward bounds and a derived forward bound are declared
before measurement; paired exactness is reported separately, not fitted.

The installed gradient-only loss adapter also passes its 32 exact head cases.
A fresh full-step control completed clean, and the paired candidate is active.
No step-speed or quality conclusion is available until that comparison finishes.

## 2026-09-19 01:12 UTC: full-step CE decision and packed FP8 attribution

Wave85 head-gradient-only CE keeps original FP32 logits and passed all32 exact
head gates. Its same-source four-rank phase finished clean: median complete step
497.864910ms versus500.517682ms, a0.5300% improvement. All four finite/causality/
batch/repetition postchecks pass and last loss is identical5.0649232864. This
misses the predeclared3% advancement gate; park for a later compatible combination,
with no dedicated quality run and no complete parameter-equality claim.

Wave86 native BF16 token-row component gate is active on trn22, source03a9d902.
On trn21, add a trace of the already-gated packed FP8 consumer/cache refresh.
This is attribution, not a replacement for the unprofiled wave85 timing. The
next model wrapper must preserve FP32 master identities, charge all cache work,
pass updated-cache/accumulation/restore tests, and check FP8 training row
independence even though checkpoint evaluation intentionally uses native BF16.

Source03a9d902 and recoveryfbd7f48a were verified on GitHub at01:10:40 UTC.
All41 inventoried heavy artifacts verified by01:05:25, with no errors.

## 2026-09-19 01:15 UTC: BF16 layout reversal and FP8 model-integration contract

Wave86 native BF16 gate passes five small/real cases, independent FP64 bounds
and exact master accumulation. Component HxT view15.605782ms, deployed contiguous
HxT15.618884ms, token-row save/backward17.434858ms (11.6268% slower). Thus the
layout that helped FP8 is a loss for BF16 at identical geometry. Keep the
deployed BF16 control; do not use this BF16 loss to veto the FP8 candidate.

Wave87 packed FP8 six-arm trace is active on trn21, sourcee1173d13. trn22 is clean
and ready for wave88 integration. New fp8_mlp_integration keeps native FP32
parameters/checkpoints, uses explicitly quantized training with native BF16
evaluation, and prepares packed/QDQ caches in each ordinary timed refresh.
Initial uncached calls also charge preparation. All short training shapes
zero-pad *inside the per-token MLP* to a multiple of512; there is no token-count
fallback to a different training algorithm. No attention sequence is extended.

Before phase timing, require actual Linear master/cache accumulation and refresh
against the prior FP8 algorithm with independent CPU-prepared cache values,
FP8 training future/batch/repeat/padding independence, and exact native checkpoint
restore. The preexisting wave85 stages remain the numerical authority. No full
FP8 backward, complete-step gain, or quality claim is established yet.
