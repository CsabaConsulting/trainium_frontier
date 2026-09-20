# Continued kernel work and research — 2026-09-20

Current checkpoint 2026-09-20 21:53 UTC: BOTH HOSTS IDLE, direct SSH verified;
199/197r2 queues and profile preservation are complete. Source199 saved FP8
forward reduces latency9.46%/19.14%/9.43% in the one/four/one-rank comparison,
with all preparation and saves charged. Full ROW backward and197 first/last
profile analysis remain pending. No new BPB result. User is considering a Claude
Code handoff; no further experiments dispatched. Read HANDOFF-CLAUDE-2026-09-20.md
and RESUME.md. Backup cron continues without launching or merging experiments.

Previous checkpoint 2026-09-20 12:43 UTC: source199 runs the corrected, unchanged-math saved-FP8 contention screen on trn21. Source197 continuation r2 runs on trn22 after verified12GiB archival cleanup; the original ingestion reserve failure is retained. Both queues use independent pre/postflight and finite limits. A separate finite, committed local sidecar preserves/uploads the four accepted profiles after completion. No new BPB result. See RESUME.md and fresh-start-library-followup-2026-09-20.md for exact ownership, plans, failure boundaries and backups.

Checkpoint2026-09-20 08:48UTC: source194 saved forward wins10.3719% with all310
exact checks and10fixtures, clean. Trn21 source196 Vector-square comparison
RUNNING, first9cases exact; production pending. Trn22 source195 position-r2
RUNNING: first/last/all-six plus final zero-library control. Original zero mean
499.107528ms/median488.653169ms includes a preserved optimizer outlier; final
control brackets the comparison. Never-launched position-r1 is SUPERSEDED.
See RESUME.md for exact sources, bounds, active labels and backward-scale design.
No new BPB result or submission promotion.


Checkpoint2026-09-20 08:32UTC: both hosts active.194 training saves has all ten
initial exact saved-tensor checks, final production timing/postflight pending.
195 four-selection routing gate COMPLETE/CLEAN,256 native gradient comparisons
and20exact checks verified; zero-library model control running. Start frozen
195 position sequence only after both gate/control complete cleanly. Trn22
scratch maintenance r4 restored19.56GB free after local/cloud verification.
See RESUME.md for exact source hashes, labels, bounds and pending work.


Checkpoint2026-09-20 08:13UTC:192/193 COMPLETE/CLEAN, both hosts idle.
Compiler SBUF full model563.005751ms vs488.528818ms (+15.245%), all gates pass.
Source191 forward win replicated193:3.908544ms vs4.595774msBF16, all preparation
included. Raw/derived profiles local/cloud verified. Source194 training-forward
saves implemented and locally AST checked, NOT DISPATCHED. Partial-layer model
causality is next independent diagnostic; preserve native HxT in untouched blocks.
See RESUME.md and final receipts for current status. No new BPB result.


## 2026-09-20 07:59 UTC — library input fusion wins production forward

Source191 COMPLETE/CLEAN with all ten cases and frozen requirements verified.
InlineFP83.926421ms vsBF164.602453ms (14.6885% faster), and16.2307% below the
prepared-input ROW control. All online preparation is charged. No backward or
quality claim; FP8/BF16 random-output RMS difference remains5.76445%.
Source193 now profiles the three unchanged complete forwards on trn21, then
ingests the preserved trace. Source192 on trn22 passed all component/cache
gates and runs native/candidate model timing. Full ROW training scale algebra
and required saved-state costs are recorded in library-fusion-next. The
partial-replacement compiler counterfactual is conditional on192's outcome.
See RESUME.md for identities, cleanup and restart conditions.


## 2026-09-20 07:43 UTC — scratch budgets negative; both new library variants running

Source190 full-model tests both COMPLETE/CLEAN:128KiB562.878526ms vs486.986693ms
(+15.584%);64KiB566.748940ms vs488.645588ms (+15.984%). All original numerical
and cleanup gates pass. Fixed budget reduction does not fix the model regression.
Trn21 now runs source191 inline input quantization; trn22 source192 automatic
SBUF placement through its full-gradient/cache/model sequence. Neither has a
speed or quality claim yet. Source192 leaves logical allocator checks active.
See RESUME.md for source/plan/coordinator identities and restart conditions.


## 2026-09-20 07:39 UTC — concrete compiler-placement and input-loader tests

Source190 both native controls COMPLETE/CLEAN (486.986693/488.645588ms);
final candidate arms are running. Both full-gradient and cache gates pass.
Source191 inline input quantization is frozen at153c4c13 and awaits trn21.
Source192 compiler-managed SBUF placement is frozen at8b1a3fc0 and awaits
trn22; its four-step plan retains all original math and correctness gates.
Detailed surrounding-Linear instruction analysis and reviewed primary-source
research are in fresh-start-library-compiler-placement-2026-09-20.md. No BPB
promotion. See RESUME.md and final postflights before any dispatch.


## 2026-09-20 07:19 UTC — bounded scratch gates pass; input-loader fusion prepared

Source190 both10-case full-gradient and2-width Linear/cache integration gates
COMPLETE/CLEAN.128KiB component14.884966ms vs15.202572ms;64KiB15.648807ms
vs15.196156ms. Both same-host native/model sequences are running; no quality
claim. Source191 inline BF16-to-FP8 input loader implemented, original arithmetic
and prepared-input control retained, additional actual-byte/stage/change checks
AST-verified. No191 dispatch yet. Trn22 archive/cloud-verify/retire maintenance
restored free space to17.524GB; all source180 profile artifacts are cloud-verified.
See RESUME.md for exact identities.


## 2026-09-20 07:11 UTC — library fusions and caller evidence

Source189 COMPLETE/CLEAN: exact fused weight preparation29.85%/32.06% faster;
complete ROW forward4.686795ms vs4.596355msBF16, still1.968% slower, all10 gates
pass. Source188 one/four-rank complete component−2.102%/+2.163%, all gates pass.
Source180 paired traces COMPLETE/CLEAN and preserved; surrounding ordinary
Linear activity/spills increase, not FP32 matmul work. DMA counters are incomplete.
Source19064/128KiB allocator budgets implemented and statically verified,
NOT YET DISPATCHED: full gradient/cache gates then native/candidate model steps.
Trn22 verified archive-before-retirement maintenance r3 prepared for disk headroom.
See RESUME.md, the NKI library audit and library-fusion-next plan for details.
No quality/BPB gain or submission change. Source159 T512 remains prohibited.


Latest host status and failure recovery: RESUME.md, updated06:40UTC. Trn21
source187 finishes the ROW FP8 forward gate after targeted compatibility fixes;
trn22 source180 r3 captures the library caller. Native caller analysis is ready.
Source188 prepares matched one/four-rank MLP contention timing. Concrete library
transfers and subsequent fusion ideas are in the library audit. No BPB promotion.

User direction: keep both provisioned hosts doing useful experiments and use
fresh research to find new directions or revisit negatives with a changed premise.
Continue the existing pre/postflight, immutable source, host locks, log mirroring
and GitHub backup protocol. No seed sweep, quality promotion or submission is
implied by a diagnostic result. The coordinator runs only its finite prepared
list; it does not implement an autonomous idea generator.

Latest checkpoint06:13UTC: both BF16 library model integrations regress despite
the component gain. Trn22 source180 captures paired caller traces. Trn21's
source181 ROW FP8 preparation fails cleanly before MLP execution; source182
isolates the byte transpose while preserving the original code and failure.
See RESUME.md and fresh-start-nki-library-audit-2026-09-20.md for current status;
earlier host-status paragraphs below are historical.

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

## Source159 results and next work — 2026-09-20 01:14 UTC

Both source159 diagnostic requests completed cleanly. The FP8 cost split is
1.582102499ms native three products, 1.796937233ms fresh FP8, and
1.781838535ms cached FP8 with preparation optimistically free. Isolated dX is
0.549018034ms native and 0.689475099ms FP8; preparation alone is0.118491700ms.
Separate scope times are not additive. Cheaper preparation alone cannot rescue
this measured implementation. A source160 cached-product device capture is
prepared to distinguish TensorEngine, quantization, transpose and DMA costs;
its two-arm unprofiled comparison remains separate from profiled attribution.

At the production QKV fixture, the CPU-FP64-product/BF16 frozen gate gives the
same81 failures as joint NKI, with both native and NKI projection consumers.
The FP32 frozen gate gives6085 failures for both. This localizes the remaining
production discrepancy to gate/consumer compilation, without proving its exact
lowering cause. Frozen-gate derivatives are invalid for learned-gate training.
No integration promotion; pause further BF16 QKV rewrites without new evidence.

trn22 coordinator fs-w159-context512-sequence-r1 PID30002 started01:10:46 UTC,
first clean preflight01:10:52. It compares current native dense6x1024 at
T1024/pack32 then T512/pack64, keeping131072global tokens/update, GA1, real token
stream, seed42, five warmups and thirty measured updates. All four finite,
causality, repeat and batch-independence checks are mandatory. This is a geometry
screen, not a BPB comparison; shorter context can hurt learning. No full-budget
training is automatically authorized by this finite two-step coordinator.
Search of fresh-start notes/request/summary files found no512-context experiment;
this does not establish that the older campaign never tried it.

The Tavily report is complete, not running. Its broad MXFP8/Trn2 recommendation
was rejected against the instruction-level NC-v4 restriction. Existing NC-v3
E4M3 kernels remain the applicable FP8 path. The frozen public best and canonical
train.py/prepare.py are unchanged.

Source16094e93a7 is deployed on trn21. Profile/ingest coordinator
fs-w160-fp8-square-dx-profile-sequence-r1 PID31475 started01:13:36 UTC;
plan commit f12a760. Its profile uses the unchanged source158 component evidence
as present in immutable wave159, checks all prerequisite hashes, and captures
both cached product arms with current installed native profiler APIs. Ingestion
is a separate guarded CPU job after a clean postflight. Raw artifacts need local
hash verification and GitHub artifact backup before any remote retirement.

## Source160 profile and source161 operation-fusion hypothesis

The paired cached dX capture and CPU ingestion completed cleanly at01:14:12
and01:14:56 UTC. All13 raw files (9,850,819 bytes) and25 derived files
(4,737,546 bytes) are locally SHA256-verified. Both runtime dispatches match
their ingested binary identities. Unprofiled medians are0.544250ms native and
0.693195ms FP8 on this host; source159 remains the separate complete-component
cost authority. The profile shows native/FP8 device intervals0.524307/0.670392ms.
FP8 regular matrix products have the expected0.5 BF16 cost multiplier, but
Scalar activity is0.6343/0.6350ms per core and Vector0.5765/0.5771ms. Native
Tensor activity is about0.5047ms; FP8 Tensor activity about0.45ms. Source lines
94/95 (the two FP32 output scale passes) account for roughly0.194 and0.209ms
of Scalar activity per core; max/min reductions each take about0.155ms of
Vector activity. These overlapping intervals are not additive execution shares.

The installed Explorer2.32.0.498 rejects --show-profile-schema. Use the matching
SchemaFields.parquet instead. Missing dynamic DMA metadata means traffic and
memory-bandwidth bounds are provisional; compiler/HLO metrics are absent too.
No compiler/runtime upgrade, cache purge or unsupported MXFP8 instruction is
needed. The instruction/source intervals provide a concrete compute hypothesis.

Source161 adds a separate experimental gradient kernel, leaving the original
byte-identical. Two independent flags: (1) abs plus max in one Vector reduction,
using the already-working dual-weight producer's operation pattern; (2) retain
both sequential FP32 scaling operations in one Vector instruction with BF16
output, avoiding extra scale/cast passes. Never precombine inverse scales,
which could overflow or underflow before multiplication. Test each flag and
their combination using the original seven-domain gradient suite and eight
square cases, independent CPU bounds, and exact old-kernel square outputs.
Only after all checks measure complete freshly prepared three-product calls
and cached dX controls. No full model or quality is promoted by this screen.

API support was verified against the installed NKI0.5 signatures/docstrings
and the primary [tensor_scalar API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_scalar.html).
The two-operation API supports FP32 arithmetic and final dtype conversion;
the two-multiply choice requires Vector on Trn2. Actual bitwise/numerical
and speed behavior remains subject to the hardware gates above.

## Source161 results and source162 continuation — 2026-09-20 01:31 UTC

- Source161 three-arm FP8 dX sequence COMPLETE/clean01:26:18 UTC on trn21.
  All45 declared cases across abs/scale/both pass unchanged CPU bounds and
  exact original square outputs. Both-fusion cached dX0.487894ms beats native
  0.544186ms by10.34%; freshly prepared three products1.625796ms still loses
  to native1.579398ms by2.94%. No autograd/model/quality promotion.
- trn21 is clean between jobs. Preparing source162 paired native/original/
  fused dX profiling with DMA notifications, followed by CPU ingestion.
  Source161 performance is the uninstrumented authority; trace is attribution.
- trn22 fs-w159-context512-sequence-r1 PID30002 is in the T512/pack64 phase,
  same source64f548e, four ranks,131072global tokens/update, seed42. Control
  T1024/pack32 completed clean01:20:52, median489.128874ms and all four model
  checks. Candidate has reached measured steps around490–496ms; wait for
  complete checks/postflight. Prepared30-minute request is NOT dispatched;
  its runtime gate requires at least3% complete-step saving.
- Source160 original paired dX profile/ingestion is complete and preserved.
  Raw13files9,850,819bytes and derived25files4,737,546bytes have local hashes
  and verified GitHub backup digests at01:21:24/01:21:32 UTC. Scalar output
  rescalings and Vector max/min reductions motivated source161. Dynamic DMA
  metadata was missing; do not treat traffic estimates as exact bounds.
- Next useful transfer: fuse the sequential FP32 output-scale operations in
  the existing two-product MLP data backward. Abs reduction is already fused
  there. Keep scale order (no precombined scale product), original independent
  eleven-case gates including subnormal/signed-zero inputs, and charge caches.
- Last verified GitHub cron01:30:52 UTC: source4a107d6 and snapshot7dbce907;
  both remote refs verified. This completion/profile source follows it.
- Best matched public BPB remains0.9928513116311253; cold replay0.993855675.
  No new BPB or competition submission. Finite coordinators sync every45s;
  ten-minute cron preserves/pushes but selects or launches no experiments.


Source159 context screen completed clean01:30:57 UTC: T512/pack64 median
492.366241ms versus matched T1024/pack32 control489.128874ms,0.662% slower.
All four model checks pass in both arms. The prepared30-minute request is
rejected by its predeclared3% speed gate and is not dispatched. Next trn22
work transfers the profile-supported scale fusion to MLP data backward.

## Transfer to MLP and separate DMA tracing from compute attribution

Source163 keeps the original MLP kernel/probe unchanged, adding a separate
variant that fuses the two sequential FP32 multiplies at BOTH data-gradient
outputs into one Vector instruction. It retains original scale order and BF16
rounding before the ReLU-squared derivative. Magnitude reduction was already
fused in this MLP and is not a new claim. Reuse all eleven independent original
stage cases and add the current32768-token shape. Original-kernel diagnostic
and normal outputs must match exactly, including subnormal/signed-zero cases,
with no tolerance change. Timing compares old fused-abs against the candidate,
charging extra backward transpose/packing for one32768-token chain. Shared
forward calibration is explicitly excluded as already paid, so this is not a
complete four-product MLP, autograd or whole-model speed measurement.

Source162 DGE-enabled three-arm capture completed clean01:33:37 UTC. All exact
controls passed. However, enabling DMA notifications stretches the FP8 calls
from below1ms to around9ms even outside the active profiler context. The output
explicitly marks those timings as instrumented, not speed evidence. Keep this
trace for traffic analysis, and capture the same three arms without DGE in
source163 for compute-engine attribution. Source161's ordinary-runtime timing
remains the performance authority. Never infer a kernel regression from the
DGE-enabled timing or splice its intervals into uninstrumented wall time.

## Backward-only cache after the dX kernel win

Source164 removes the dual producer's unused restored BF16 and forward-packed
outputs only for square dX. Keep its matrix calibration, row ownership, legacy
quantizer and backward byte layout unchanged, and prepare from current master
weights on every call. Source159's preparation-only rejection applied to the
old slower dX. Source161 changes that premise: cached dX now wins10.34%, while
fresh three-product work loses2.94%. Reprice the narrower preparation before
considering autograd integration. Eleven independent metadata/layout cases
include signed zero, subnormal, tiny, large and production weights; eight
unchanged square-product CPU gates and exact original FP8 outputs follow.
Every master cast, calibration and required packing is charged; no forward
FP8 cache is borrowed. Stop at the component result, not a model promotion.

Source163 non-DGE compute trace independently reproduces the cached result:
0.489108ms candidate versus0.539251ms native and0.691124ms old FP8 before
profiling. Three runtime dispatches match their binary identities. Device
intervals native/old/new are0.529872/0.666426/0.469006ms. New Scalar activity
is0.3222ms/core versus old0.6345ms; new Vector0.4303ms versus old0.5760ms.
Useful matmul FLOPs and FP8 double-mode multipliers are unchanged. These
engine intervals overlap; do not add them as wall-time shares. All source162
DMA and source163 compute raw/parquet files are now locally SHA256-verified;
GitHub artifact upload is still pending at this checkpoint.

## Source163/164 decisions and source165 model controls

Source163 MLP data-gradient fusion completed clean01:40:57 UTC,12cases including
both16384/32768token shapes and subnormal/signed-zero gradients. All original
outputs and CPU stage bounds pass. One32768-token chain plus extra backward
cache work: old FP85.548443ms, new4.706015ms (15.18% faster); native same-QDQ-
weight surrogate5.396273ms. Shared forward calibration is still borrowed.
Source164 backward-only square weight producer completed clean01:40:50UTC with
19cases. Fresh complete projection1.631882ms versus existing fused-dX/dual
producer1.628419ms and native1.576031ms. This is a negative, not promotion.

Source165 explicitly selects no scale fusion (default), backward-only, or
forward+backward inside the existing full-FP8 MLP research integration. Default
math, checkpoint format and canonical train.py are unchanged. The new forward
keeps two sequential FP32 multiplies and final BF16 rounding, with resident
compact token-row outputs only. Original forward/backward kernels remain the
independent numerical authority; original complete-autograd, cache update,
two-microbatch FP32 accumulation and CPU-stage references are unchanged.

Prepared next queues (not yet running at this source commit): trn22 checks
actual Linear integration with backward-only scale fusion, then fresh native
and candidate full-step timings. trn21 runs eleven complete forward+backward
component cases with original references and all preparation/gradients charged,
then actual Linear integration and matched native/candidate whole-step timings.
Every request has its own host lock, bounded runtime and pre/post cleanup;
advancement requires complete declared gates and clean postflight. Neither
queue launches quality training. Compiler scope/spilling may erase the component
win, which is why these controls are required. No concurrent host experiments.

Source162/163 raw and parquet artifacts now all have verified GitHub asset
digests (01:40:10–01:40:35 UTC). The01:40:50cron pushed source6ac8358 and its
recovery snapshot with verified refs. Source165 and new launch receipts follow.

Before source165 deployment, trn22's clean source163 postflight had only
9,026,797,568bytes free. Maintenancefs-maintenance-preserved-wave140-trn22-
20260920T014925Z rehashed both wave140 native profile trees locally and
remotely, revalidated live GitHub asset digests, acquired the host lock and
verified the exact preceding clean experiment. It retired only those redundant
raw/derived remote files and staging hardlinks whose hashes matched preserved
raw data. Unmatched staging files are retained; sources, checkpoints and
compilation caches are untouched. It completed clean01:49:39UTC, freeing
5,976,428,544bytes and leaving15,003,205,632bytes. Local intent/result/three
progress records and maintenance-script SHA256 are preserved. Both hosts now
have immutable source500ba34 in wave165. Source500ba34 was pushed/ref-verified
by the01:50:55 cron; new plans/maintenance receipts still need their commit.

Both source165 queues STARTED01:52:07UTC under plan commit03ede7a. trn21:
fs-w165-fp8-all-scale-model-sequence-r1 PID5712, four steps/10000-second bound;
first guard1615520/clean preflight01:52:13. trn22:
fs-w165-fp8-backward-scale-model-sequence-r1 PID5676, three steps/6000-second
bound; first guard1931233/clean preflight01:52:13. Source16312case prerequisite
passed on both. Requests have exact hashes and immutable source500ba34, plus
separate pre/postflight for every component, integration and full-model run.
Phase comparisons require all four finite/causality/batch/repetition checks;
FP8 candidates must show35refreshes for each of12MLP weights. Coordinators
mirror every45seconds and stop after the finite list. No automatic quality
training, seed sweep or competition submission. Canonical train.py,prepare.py
and the frozen public0992851 submission remain unchanged.

## NKI Library lead — 05:29 UTC

The user's Re:Invent pointer led to a pinned audit and two hardware tracks.
Details and exact claims are in fresh-start-nki-library-audit-2026-09-20.md.
Main attention changes alone are not a useful timing gain; the installed SDPA
already uses older library attention. The CTE MLP forward is21.2% faster at our
production shape but training saves/backward remove most of this benefit.
Reusing its required intermediate transpose now yields2.404% complete-component
improvement with ten original numerical fixtures and clean postflight. Trn21
source176 performs real-Linear cache/gradient gates before matched full steps.
The current complete-step/quality threshold remains unchanged. Native/compiler
master weights, eval and the frozen submission are preserved.

Trn22 source171 separate-region full-FP8 r1 completed all numerical checks but
failed disk reserve. Verified local/GitHub preservation of12.98GB old scratch
then remote retirement restored20.59GB, with independent clean health. The
unchanged native/candidate r2 repeat is active under a new frozen finite plan;
r1 remains a failed postflight, never relabeled as clean. No seed sweep.
