# Kernel queue and precision-specific experiments

Updated 2026-09-19 08:20 UTC. This is the executable/design queue, not a list of
claimed gains. Read `RESUME.md` and the latest postflights before using a host.
Every device request retains immutable source, cleanup and numerical gates.

## Current priority after full FP8 integration

Source137's exact dual producer improved the complete isolated FP8 MLP by
3.006%. Source139 passed real-Linear/master/cache lifecycle gates. Source140's
complete model nevertheless takes 567.095 ms/step versus 488.702 ms for native
BF16. All four numerical postchecks pass; the final disk-reserve failure is
retained and needs maintenance before another job. No FP8 speed/BPB promotion.

1. Capture the source140 native and full-FP8 caller at pack32/GA1. Attribute the
   extra 74.667 ms in forward/backward using matched binary identities; inspect
   unused BF16 outputs and model-wide precision/layout/allocation effects.
2. If unused writes survive, implement a full-VJP-only output mode. Preserve
   exact outputs, saved bytes/scales and all four derivatives; rerun cache and
   master lifecycle gates before measuring the complete step.
3. Correct the native width guard and measure new optimizer groups before a
   bounded capacity comparison with the current kernel stack. See the capacity
   revisit note; old generic ownership costs are not trustworthy at new widths.
4. Keep reciprocal optimizer composition, reduced calibration work and compact
   function approximations available for measured bottlenecks. The standalone
   polynomial/full-value-table losses remain closed for their tested layouts.

Compiler scratch archives now have a staged preservation/retirement procedure,
so disk reserve does not repeatedly interrupt these comparisons. Every original
file must exist in a hash-verified local archive and verified GitHub asset before
its remote copy is retired. Immutable shared compilation caches remain in use.

## Ready requests and active work

New matched larger public best: **0.9928513116311253**, source98 pack32/GA1,
3660updates in1784.6training seconds. Same global targets/update as the previous
0.9938106631483421 pack16/GA2 checkpoint. No ngrams, seed sweep or FP8. Both
checkpoints are local/cloud verified. Training's disk-reserve-only postflight
failure and clean recovery are preserved; the larger evaluation is clean.

| Idea | Measured state | Next action |
| --- | --- | --- |
| Compile unchanged online FP8 cache preparation | Source135 pack32 complete MLP15.1108ms vs compiled native15.3822ms:1.76% faster; independent32768-token product/gradient gates pass | Prepared fused calibration plus both packed orientations; gate exact cache bytes/complete gradients, then time against previous compiled producer |
| FP8-specific packing | One layout improves FP8 component about2.95% while worsening BF16 by11.63%; FP8 whole model still17.74% slower | Continue FP8-specific development; BF16 ranking is not a veto |
| FP8 grouped weight-gradient products | Balanced banks + fused abs + two-core preparation pass10cases: down1.548/up1.695ms vs native1.095/1.075ms | Complete full FP8 MLP passes11cases but15.938ms vs15.486ms BF16; profile full derivative costs |
| Reuse forward-produced FP8 bytes | All10cases pass; online cache work +2forward producers6.899→5.919ms (14.20%); preserves existing saves | Composed in complete FP8 MLP; saves do not yet make it faster than BF16 |
| Fused per-row FP8 data-backward calibration |11cases pass;5.655ms vs original6.201ms and native5.510ms |8.81% same-algorithm gain; compose shared producers and weight gradients before full-model use |
| Cached vocabulary head and exact CE | Component14.095440vs18.202054ms; whole step37.79% slower; caller profile local/cloud verified | Split-head196 exact same-scope checks pass, but681.026ms vs501.265ms original; boundary trial closed for promotion |
| Save exact softcap intermediate | All20component checks pass;13.850659ms vs native18.219967ms | Retain for composition after solving caller allocation; unchanged save bytes |
| Reciprocal refinement + optimizer ownership |24composition checks; component16.77% faster, whole step1.128% faster | Retain for independently gated combinations; below3% advancement threshold |
| BF16 magnitude maximum / fused abs | All3alternatives pass13cases; fused abs fastest, calibration0.197/0.650ms vs0.344/1.184ms | Composed in weight/data gradient prototypes; integer-code and sign-mask versions are correct but slower |
| Smaller tables / rational composed functions | Design candidates; full-value LUT and standalone polynomial timings lose | Test full producer/consumer costs with independent range/gradient gates |

Source135 now gates **32768tokens/one microbatch**, matching the current
pack32/GA1 winner. Independent forward/byte/all-four-gradient-product checks,
updated master caches and the native FP64 rounding-bound control pass. Compiled
FP8 preparation yields15.110784ms per complete MLP versus15.382169ms for compiled
native:1.76% faster. At the former16384x2shape FP8 was0.485% slower. This is a
component result, not a full-model speed or BPB result. All original records
remain; the shape dependence is measured rather than extrapolated.

Source137 now gates a producer that computes the same calibration and writes both packed
orientations from each quantized tile, eliminating the BF16 HBM transpose and
second quantization. A resident variant retains at most8MiB of BF16 weights in
SBUF per physical core. Both cores repeat the global maximum before writing
disjoint packed row blocks; that duplicated work is charged. It passes all
CPU metadata/QDQ/layout references and exact complete gradients. The prior
compiled producer and compiled native control remain in the comparison. Resident
preparation0.294823ms vs previous compiled0.541435ms; complete FP8 MLP14.886892ms
vs native15.348333ms (~3.01% component gain). Streaming14.906892ms also passes.
Next: gate real Linear masters/cache refresh/eval restoration with the full FP8
VJP, then measure the complete four-rank model. No whole-model/BPB promotion.

The following ideas from the retro-computing search have already reached device
experiments. This distinguishes a negative measured layout from an untested idea.

| Family | Result | Status |
| --- | --- | --- |
| Exponent-bit range reduction for FP8 scales | Replaces log2/floor/exp2, extensive exact calibration/quantizer gates | Implemented in FP8 prototypes; no full FP8 quality gain |
| Reciprocal/Newton optimizer update |16.77% component gain;1.13% whole-step gain with ownership | Keep for composition |
| rsqrt correction variants |One Newton step fails large-gradient gate; two steps and quadratic variants pass but lose speed | Numerical failures and slower variants retained |
| Polynomial exponential |Nine device records pass;3.57–4.19times slower than hardware exp | Closed as standalone replacement |
| Compact full-value softcap/exp/derivative table |65280CPU codes and device gate pass;47.20vs2.48ms (19.04times slower) | Close this gather layout; smaller range-reduced tables remain distinct |
| Exact CE work/traffic elimination |53.98% component and3.66% whole-step gains | In current best checkpoint |
| Saved intermediate and on-chip weight reuse |Cached-head component improves22.56%, whole caller loses37.79%; saved-softcap passes | Isolate caller allocation with normalization-aware boundary; FP8 forward-byte reuse prototype ready |
| Precomputed trigonometry/symmetry |RoPE already precomputed | Do not duplicate existing work |

Original failures, subsequent root-cause diagnoses and incremental FP8 results
remain below and in the design/result logs. New requests must retain immutable
source, finite/causal/repeated-call/master-gradient checks and pre/post cleanup.
Whole-step advancement normally requires at least3% before a30minute quality run.

The official [nc_matmul reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html)
confirms that NC-v3 FP8 double-row mode has a different contraction layout:
partition dimension plus a first free dimension of2. BF16 cannot use that mode.
This supports precision-specific layout searches; measured whole-step cost still
includes scale selection, packing, saved activations and backward work. The
installed SDK remains the authority for supported legacy FP8 formats.

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

## 2026-09-19 01:19 UTC: integration harness limit, retained failure

Wave88 passes every D256 integration gate and D1024 cached/uncached forward,
input/master gradients, accumulated updates and fresh/unchanged cache checks.
It then hits Dynamo's default eight-specialization limit at the first D1024
no-grad prediction call: both widths share Probe.forward's code object, and
each deliberately exercises multiple shapes/cache states. Postflight is clean.
The gate remains incomplete and cannot authorize FP8 phase timing.

Wave89 resets only process-local inactive Dynamo guards *between* independent
width tests. Fullgraph=True, the default specialization limit, all numerical
bounds and every prediction/restore check remain. No shared device cache purge.
The same-source native BF16 control can run independently on trn21 while this
rerun runs on trn22. FP8 phase remains dependent on a complete passing gate.
Wave87 raw trace32files/499268795B is local and every remote/local SHA matches;
local parquet ingestion is underway, with bounded memory/time.

## 2026-09-19 01:21 UTC: integration passes; first true FP8 gradient stage

Wave89 integration passes all D256/D1024 gates, including FP8 training row,
batch, repeat and padding independence, exact master accumulation/cache refresh,
and native BF16 checkpoint restore. The isolated Python guard reset resolves
the identified harness limit; no tolerance/check/default-limit change. Clean
postflight01:20:21. Whole-step FP8 timing is now authorized by the numerical gate
once its same-source BF16 control finishes cleanly on trn21.

Next on trn22: independently gate a packed FP8 incoming-gradient GEMM for
G@Wdown and dPre@Wup geometries. Per-row gradient calibration and double-row
E4M3 dot use the existing CPU quantizer/pair reference. Cases include zeros,
signed cancellation, tiny2^-110 gradients, row exponents-80..30 and outliers.
Implementation error and error versus the old BF16 surrogate are distinct.
Timing charges the EXTRA transposed-weight packing per update plus two dots;
existing forward-prepared weight metadata/cache is shared, not magically free
new work. No derivative fusion, weight-gradient products or autograd install
yet; this is one backward stage, not full FP8 training.

## 2026-09-19 01:23 UTC: gradient probe frontend correction

Wave90 stopped before gradient arithmetic: the new validation helper contained
a Python raise, which this NKI frontend does not support. The compiler points
directly to that statement and recommends assert. Use the installed skill's
assertion-helper template in wave91; retain the same shapes, math and checks.
The failed label/log is preserved. This is not evidence against FP8 gradients.

## 2026-09-19 01:26 UTC: trace attribution and repack diagnosis

Wave87 raw32files/499268795B and parquet27tables/304690666B are local and
hash-verified. All70 runtime dispatches match binaries. Packed forward2.535229ms
versus previous token-row FP82.803872ms; sampled SBUF peak0.130903 versus
0.364328, no populated spill counters in either; backward about4.84ms. Missing
dynamic DMA metadata still prevents exact bandwidth claims. Device profile
times are attribution; wave85 unprofiled complete component remains authority.

Wave91's corrected validation helper compiles; seven small gradient GEMM cases
and real G@Wdown arithmetic pass. It fails exact raw-byte equality when packing
an existing BF16 QDQ cache: maximum uint8 difference128 suggests signed zero.
The original packed-cache contract already treats0/128 as equivalent. Wave92
will count/classify every mismatch, reject any changed nonzero encoding/layout,
and require independently decoded values exactly equal. No pair/gamma bound
changes. Keep the failed attempt and require clean postflight before rerun.

## 2026-09-19 01:32 UTC: standalone FP8 gradient stages pass; fuse the chain

Wave92 passes all nine small/real arithmetic gates, exact repeat/other-row tests
and original signed-zero repack contract. Every changed byte is0/128:107 for
G@Wdown's weight,103 for dPre@Wup; zero nonzero changes and decoded values exact.
Complete incremental cost (one extra transpose/pack plus two dots) is2.091248
versus2.126371ms for G@Wdown, and2.092029 versus2.103713ms for dPre@Wup. Gains
1.65175% and0.55538% are too small for standalone promotion. No autograd/quality
gain follows. The failed frontend/byte-equality attempts remain preserved.

The next kernel fuses both data-gradient dots and the BF16 ReLU-squared derivative.
It keeps the first dot in SBUF, saves BF16 dPre for the later weight gradient,
and retains per-row signed gradient calibration at BOTH boundaries. The forward
nonnegative calibration shortcut must not be used for signed dPre. Gate both
FP32 dot stages with the independent quantizer/pair model, each BF16 rounding
and derivative stage exactly, paired separate-vs-fused output, and row
independence. A compiled BF16 chain has its own independent FP64 propagated
rounding bound before timing. Charge both extra transposed caches and two data
backward calls per update. Weight-gradient products/master accumulation and
model integration are still excluded; this is not full FP8 backward.

## 2026-09-19 01:40 UTC — Caller regression and backward-stage result

Wave89 clean paired phase runs establish a17.7363% complete-step regression,
mostly in microstep compute (+85.0014 ms), not just refresh (+2.52845 ms).
All actual wrapper/cache/FP32 accumulation and FP8 training-independence gates
passed before this comparison. Native BF16 evaluation remains the declared
prediction path. Keep the independently correct candidate as diagnostic source;
no quality run until a new complete-step improvement crosses3%.

Wave92 proves both gradient GEMMs against independent quantizers/FP8 pair
bounds, including tiny, cancelling and outlier gradients. The signed-zero
repack mismatch is fully classified, not numerically relaxed. Wave93 fuses
G@Wdown, BF16 ReLU-square derivative and dPre@Wup. It passes9 cases, exact
agreement with separate FP8 stages, and saves9.54% versus those separate calls;
it still loses12.72% to the BF16 chain. Weight-gradient products and autograd
integration remain ungated. Diagnose complete caller dataflow before continuing
small FP8 product variants.

Both caller trace jobs use source dcd6b80 and rank0 device/runtime recording
with all ranks active. Same-host non-profiled timing is already preserved;
cross-host trace durations do not replace it. The older wave5 remote duplicate
was pruned under the host lock only after local, remote and GitHub preservation
verification; exact maintenance manifest retained.

## Additional fusion candidate: vocabulary head and loss

`fresh-start-fused-head-loss-design-2026-09-19.md` records the primary-source
review and staged gates for eliminating the FP32 logits intermediate. Apple
CCE offers the algorithmic precedent; AWS experimental CE offers an LNC2 tiling
reference but takes materialized logits and does not preserve our default
softcap/precision contract. Start without gradient filtering and retain the
FP32 head accumulators that the failed wave83 boundary exposed. Separate simple
forward fusion from backward recomputation and charge every extra GEMM.
Priority depends on the current native caller profile; no speed or quality
gain is established. This is work reuse/traffic elimination from the broad
retro-computing search, not another standalone polynomial benchmark.

## Prepared diagnostic: compile the packed FP8 MLP separately

`--fp8-mlp-separate-region` exposes an explicit caller boundary while the same
complete FP8-forward/BF16-surrogate-backward callable remains compiled with
`fullgraph=True`. The shared callable is created once across all six layers;
all graph-boundary/dispatch and cache costs remain timed. This tests compiler
allocation scope, not a new scale, dtype, GEMM or backward approximation.

`fp8_mlp_integration_check.py --real-width --separate-region` retains every
exact output/input-gradient/FP32-master-gradient/cache/update/padding/restore
check. Only the caller is permitted to break its graph; the test verifies a
compiled autograd node remains reachable through any outer view nodes. Default
monolithic mode still requires the original exact outer-node assertion.
AST checks pass locally; hardware gate pending. Prioritize a whole-model paired
comparison only if the caller trace supports a compilation/layout cost.


## 2026-09-19 02:29 UTC — larger microbatch under the new exact kernel stack

Prepared `fs-w98-native-owned-context1024-pack32-phase-r1` for trn22 after the
reciprocal candidate is clean. Source1da6c07, T1024/pack32/GA1, native BF16 MLP,
exact ownership/adaptive CE, seed42 and131072 global targets per update. Control
is the same-host/source pack16/GA2 phase503.328809ms, clean02:23:38. No matching
pack32 test was found in the reviewed run logs/research notes or memory search;
older memories cover pack2/4 and earlier contexts. This is not a seed sweep or
changed global batch. It trades fewer repeated microsteps for larger activation
and scratch allocations; do not assume it fits or wins. Full-step numerical
postchecks, memory/cleanup and the3% gate precede any quality experiment.

The phase still clips before averaging, like its control. Quality must keep
the best run's clip-after-average policy. Changed reduction grouping is not
claimed bitwise equivalent, even though token count and context are preserved.

Latest dispatch state is at the top of RESUME.md. Pack32 quality has completed
within budget, checkpoint local/cloud verified, matched larger evaluation active.
New exact-work reuse candidate: save FP32 softcap unit values for backward at
the same byte count, removing its repeated tanh. Source113 gate prepared.
Further design candidate: mask the BF16 sign bit and reduce ordered unsigned
magnitudes for exact amax. NKI tensor_reduce arithmetic widens integer values
toFP32; all16-bit codes fit exactly. Compare original metadata and zero/tiny/
subnormal behavior explicitly before use. Reference: installed skill's primary
API copy `neuron-nki-docs/references/programming/api/api-nki-isa-tensor.md`,
tensor_reduce and tensor_scalar dtype rules. Not implemented or timed.


Source115 amax update,03:48UTC: fused Vector abs/max passes all13CPU/original/
quantizer/repeat/nonmutation cases. Exact scale policy retained, including pure
subnormal values. Median calibration falls42.82% at16384x1024 and45.12% at
16384x4096. Sign-bit masking also passes13cases; large-shape0.755972ms is slower
than fused abs0.649855ms. Integer magnitude-code reduction is running. The
source116 composition flag reruns the original10FP8 gradient-product cases and
charges rescaling, calibration, both byte preparations and the product. No
speed benefit from these calibration components is yet claimed for a full model.

## Complete FP8 component and next precision-specific design (04:48 UTC)

Source124 gates all six MLP matrix products (two forward, four backward), BF16
ReLU derivative and FP32 master gradients. All11 cases pass; exact current
compiled/staged outputs and accumulated gradients match. Independent quantizer
and empirical paired-arithmetic bounds are retained. Current per-row data
quantization and matrix-wide weight-gradient calibration define a new VJP;
this is not an exact replacement for the original BF16 derivative.

The charged component15.937871ms still loses to native15.485907ms by2.9186%.
Saving forward FP8 bytes and fused magnitude maxima are real local gains but do
not establish a full-model benefit. Profile the complete native and full FP8
arms, with online cache work, to measure the residual preparation/layout costs.
Cross-host profile intervals are for attribution; the same-host component
benchmark remains the speed authority. No30-minute FP8 quality trial yet.

A bounded next DESIGN, conditional on those profiles: fuse per-output-channel
weight-gradient calibration with its long-token matrix product. A core owns
output rows, so it can load a contiguous transposed gradient tile, reduce its
maximum across tokens and reuse the on-chip quantized tile without a separate
cross-core global-maximum pass. This changes the quantization policy and needs
independent per-channel CPU scales, byte decoding and pair-product bounds.
Down-gradient hidden inverse scales must still be transferred at the declared
BF16 boundary. Charge the gradient transpose/scale producer and any duplicated
loads. Only afterwards consider producing that layout inside data-backward;
never treat an unmeasured transpose or saved output as free. Compare all4
products and complete FW/BW again before model integration. Not implemented.

### Profile-driven priority before another precision change

Source125/126 full-path attribution matches all8/16 native/FP8 runtime
executions to their binaries. FP8 backward TensorEngine activity is much lower,
but Scalar/synchronization activity and current-weight preparation are higher.
Cache preparation has8 extra dispatches and0.757ms of instrumented device time
vs0.112ms native. Cross-host/phase-synchronized traces are attribution only;
source124 still loses2.9186% under matched unprofiled timing.

Prioritize an exactly gated compiled cache-preparation graph, with a compiled
native-cast control. Then test a fused NKI current-weight calibration/dual-layout
producer if needed. Charge all work and gate metadata/bytes/cache refresh after
updates. Both are DESIGN, not implemented. Keep the per-output-gradient
calibration design independent. See fresh-start-full-fp8-mlp-diagnosis-2026-09-19.md.
