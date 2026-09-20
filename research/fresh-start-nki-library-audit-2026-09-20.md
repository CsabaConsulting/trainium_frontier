# Pinned NKI Library investigation — 2026-09-20

The user's Re:Invent lead is [AWS Neuron's NKI Library](https://github.com/aws-neuron/nki-library).
The clean local clone is pinned at `92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`
(August 28). Source references below use that commit, rather than a moving main.

Current result at06:13UTC: the best BF16 complete-component candidate is2.404%
faster, but its joined/separate whole-model steps are14.619%/25.146% slower.
Neither advances to a quality run. Source180 is capturing paired native/library
training-forward traces to attribute the regression. The online ROW FP8 producer
passes all14 fixtures; source181's composed preparation fails before the MLP on
invalid weight bytes. Source182 preserves that original failure and isolates
ordinary byte transpose, float carrier and explicit NKI transpose, including
all256 codes. No new training-speed or BPB gain is established.

## What is new, and what we already use

Read-only inspection at 04:06 UTC confirmed both hosts use Python3.12.3,
NKI0.5.0, neuronx-cc2.26.6360 and native torch-neuronx2.11.3. Neither has the
standalone nki-library package installed. Both have compiler-bundled `nkilib`.
No device holders remained after the source165 queues. Free space was17.45GB
on trn21 and11.53GB on trn22; keep the existing8GiB pre/postflight floor.

The compiler's `nkilib` contains442 Python files; the clone has627. There are
204 byte-identical files,227 changed common files,196 new files and11 installed
files absent from main. The selected installed attention forward/backward and
MLP files are byte-identical to this clone's `origin/2.31_release`, not its
`origin/2.32_release`. This is file identity, not a claim that every installed
file matches a release tag or that the compiler version equals the SDK version.

| Candidate | Finding | Action |
| --- | --- | --- |
| Attention CTE + backward | Our native SDPA already calls the bundled pair. Main has changed code, including layout handling and forward tiling/masking. | Compare the pinned pair directly with bundled, charging both passes and transformations. |
| CTE MLP | Newer explicitly managed SBUF/PSUM tiling, buffered weight loads and BF16/legacyFP8 paths. | Screen the BF16 path at our actual32×1024×1024 shape, then inspect resources in the caller if useful. |
| Fused Adam | Main file is byte-identical to installed. Our optimizer already has separate, more stringent Adam numerical probes. | No claim of a new kernel from installing this package. Revisit only a specific scheduling/math difference. |
| Cross entropy | Main file is byte-identical to installed and lacks our fused softcap boundary. Already consulted during the fused-head investigation. | No blind replacement of our faster adaptive softcap/CE path. |
| `matmul_loop_nest`, `TiledTensor`, NeuroTile | Useful examples of bank management, rotating buffers and layout-aware DMA. | Transfer a measured scheduling idea into the failing surrounding projection region; abstraction alone is not a speedup. |
| MXFP8/MXFP4 MLP/matmul | Use `nc_matmul_mx`, requiring NeuronCore-v4; our Trn2 is v3. | Not a runnable Trn2 FP8 pipeline. Study scheduling separately from incompatible arithmetic. |
| TKG / GPT-OSS megakernels | Primarily inference/decode workloads. | Reference for fusion and memory management; our objective includes a full backward and updates. |

The [README's integration section](https://github.com/aws-neuron/nki-library/blob/92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e/README.md)
explains that installing the standalone package redirects compiler-bundled
`nkilib` imports. It explicitly does not guarantee compatibility with a released
compiler. We retain the installed stack and compare selected kernels in an
isolated namespace, with `NKILIB_FORCE_BUNDLED_LIBRARY=true` for the control.

## Source preservation and one necessary model adapter

`scratch/nki_library_vendor.py` extracted the66-file relative-import closure
of attention forward/backward and MLP:1,184,694bytes of unmodified upstream
Python. `vendor/nki_library_92d11f63/manifest.json` records every file hash,
upstream commit, external imports and the installed comparison. Apache2.0
LICENSE and original copyright notices are retained. An intentionally empty
package root prevents upstream's installation/namespace replacement mechanism.
The user-owned clone is unchanged. Each probe verifies the entire closure and
selected installed file hashes before testing.

Source inspection found a concrete activation gap: `SquaredReLU` exists in
`ActFnType` and `apply_activation`, but CTE's `_evict_gate_res_tile` still calls
`get_nl_act_fn_from_type`, which rejects SquaredReLU. A direct call would fail
before producing an output. `scratch/nki_library_relu2_adapter.py` installs a
strict BF16/ungated/no-bias/no-quantization hook only inside our isolated CTE
module. It calls upstream's own ReLU-then-square helper. The BF16 intermediate
rounding matches our native MLP. Original vendored sources stay byte-identical;
the separate adapter is explicitly part of the candidate, never the control.

## Initial bounded experiments

Trn21: `scratch/nki_library_mlp_probe.py`, six cases. Random, zero, outlier and
changed-weight fixtures at2×512×256; random2×1024×1024 and32×1024×1024.
Both native and adapted library outputs are checked against independent FP64
ReLU-squared algebra with propagated FP32 accumulation/BF16-rounding envelopes,
all output channels for33 rows spanning the complete batch/sequence. Full
outputs must be finite and repeat exactly; changing another batch cannot alter
batch0. Timing includes ordinary Linear weight orientation and both layout
conversions. This is a forward-only screen: the library does not return the
saved ReLU needed for training. Add/charge saved activations or recomputation
and validate all derivatives before any training speed claim.

Trn22: `scratch/nki_library_attention_probe.py`, six cases. Random, zero-QK,
small-QK and outlier fixtures at2×2×512×128; random2×8×1024×128 and production
32×8×1024×128. The paired wrapper reproduces the installed SDPA's LSE layout,
FP32 mixed reduction setting, scale and gradient transposes. Both versions
must pass independent FP64 output/dQ/dK/dV checks on full selected head
sequences. The unchanged campaign native comparison additionally limits the
new-vs-bundled drift. Repetition, prefix causality, zero future K/V gradients
and independence of another batch are exact gates. Timing charges both passes,
saved statistics, transposes and gradient accumulation. It is not a whole-model
or BPB result. No global backend decomposition is patched.

Each job uses a new committed immutable checkout, explicit fingerprint list,
exclusive host lock, clean preflight, finite timeout and postflight. A finite
coordinator mirrors logs every45seconds and stops on any failed check. No quality
run is automatically promoted. Local syntax checks precede deployment; device
compilation and numerical results are pending until actual receipts exist.

## Source165 completed results, preceding this lead

All seven jobs and all original numerical gates finished cleanly. The complete
isolated FP8 MLP improved to13.110648ms, versus oldFP814.589666ms and native
BF1615.425906ms, including current weight preparation and all four backward
products. Nevertheless the complete model remains slower:

| Same-host pair | Native step | Candidate step | Regression |
| --- | ---: | ---: | ---: |
| trn21, fused forward+backward scaling |488.587103ms|550.129347ms|12.596%|
| trn22, fused backward scaling |488.616140ms|556.930141ms|13.981%|

Both candidates refreshed all12 MLP caches35times and passed all four rank
postchecks. The prior component improvement is real, but does not remove the
surrounding model regression. No30-minute quality run is warranted from these
timings alone. Exact source/postflight/log identities are in
`run-logs/fs-w165-scale-fusion-final-comparison-r1.json`. The established public
best0.9928513116311253 and frozen submission remain unchanged.

The useful new direction is to compare proven library tiling/allocation and
attention implementations with our measured integration gap. A wholesale
package switch would change many unrelated kernels at once and obscure that
comparison.

## Source166 interface failures and source167 corrections

Both jobs launched clean04:17:04UTC from29b90bb9 and stopped cleanly before
timing. Preserve their failed logs/postflights; neither is a negative speed
result and neither demonstrated a numerical failure in a candidate output.

- MLP04:17:21: `_validate_mlp_required_arguments` requires a gate tensor even
  with `skip_gate_proj=True`. Supply the same transposed up-weight operand in
  both slots. The disabled gate remains disabled; this adds no cache, weight
  conversion or matrix product. The original parameter validator stays intact.
- Attention04:17:22: bundled forward/backward and repetition pass, but newer
  CTE compilation stops at `min(tile.num_valid_k, _K_TILE_SZ)`, because the
  NKI0.5 frontend treats the property as a function rather than an integer.
  `scratch/nki_library_attention_compat.py` replaces only the isolated
  `KTileInfo` constructor with a dataclass factory whose stored field is the
  identical `seqlen_k - k_start_pos` expression. A128-case CPU factory check
  precedes compilation. Tile classification, masks and reduction schedule
  stay in unchanged upstream code. This is a parser compatibility adapter,
  not a hardware workaround or numerical tolerance change.

The source167 retries use new immutable checkouts and labels. Existing source166
requests/plans and the stopped coordinators are not edited or restarted. Check
the latest source167 launch/postflight/sequence receipts for hardware status.

## Source167 compiled; diagnostic follow-up source168

Both source167 jobs stopped with clean postflight by04:21:54UTC. The attention
property workaround passed128 CPU factory cases and compiled successfully.
For the first2×2×512×128 random fixture, the new forward and all three gradients
are EXACTLY the bundled control. Both versions repeat exactly. The independent
FP64 check then rejected the BUNDLED control at one output coordinate:
relative RMS0.001999714, maximum coordinate error/bound1.265144. This is not a
candidate-versus-control discrepancy and no timing or promotion was accepted.

The original coordinate envelope `.03*abs(reference)+.008*global_RMS` was an
empirical screen, not a propagated floating-point error bound. A small result
from cancellation can have BF16 probability-rounding error proportional to
the sum of absolute weighted values, rather than the small signed output.
Source168 adds a separate diagnostic that retains/reports every original
failure and independently estimates the BF16 exp/P/V and gradient rounding
boundaries using dense FP64 CPU algebra. It does not change the original probe,
replace its acceptance test, time kernels or allow quality promotion. The
unchanged new-versus-bundled comparison and exact repeats still assert.

The MLP reached both native and candidate execution, then failed the output
shape/dtype assertion. Its adapter indexed `[0]` unconditionally, despite the
native wrapper unwrapping a single output. Source168 handles Tensor versus
collection explicitly and reports expected/actual shape and dtype. Original
FP64 bounds, all six fixtures and full-tensor repeat/batch gates stay intact.
This is an interface fix, not a new kernel or relaxed numerical check.

## Source168 small-case evidence and source169 shape/API adapters

The corrected MLP passes all four small cases with EXACT native outputs,
original FP64 bounds, repetitions and batch independence. The small random
forward is0.065846ms versus native0.059723ms; this is not a production speed
claim. At2×1024×1024, upstream switches to intermediate sharding. Its loop
uses B×S while the DMA-transpose helper only flattens the input for token
sharding, producing an out-of-bounds sequence index1024:1152. Source169 gives
the row-independent MLP an explicit `[1,B*S,C]` view and reshapes its result
back. Both views remain inside the measured function and batch checks remain.

All16 new-versus-bundled attention output/gradient comparisons across four
small cases are EXACT. The separate diagnostic preserves355 coordinate
failures per version across32 CPU field checks; maximum relative RMS0.003404.
At the first rejected output coordinate, query4/channel51, FP64 gives
0.00736431948 while both kernels give0.00927734375. The independent staged
BF16 calculation gives EXACTLY0.00927734375. The absolute weighted-value sum
is1.998044895 (signed result0.007364), demonstrating cancellation sensitivity.
This explains the inadequacy of the global-RMS coordinate heuristic; it does
not redefine the original check as passing. All failures are preserved.

At T1024, main's chained Vector row-max reduction passes `reduce_res=None`
for intermediate tiles, which installed NKI0.5 rejects. Source169 retains
the reset/accumulate commands and writes each prefix maximum into its already
allocated per-tile column. The later maximum over these columns is unchanged:
max(all prefix maxima)=max(all tiles). The scratch compatibility helper is a
licensed281-line copy of the pinned function with only this assignment
changed; all extra readout instructions are charged. No extra SBUF buffer,
global ISA patch, disabled reduction chain or compiler replacement is used.
The pinned66 files and bundled control remain untouched. Source169 continues
the same diagnostic without permitting promotion or changing thresholds.

Both source168 postflights are clean, commandrc1, no residual holders or health
failures. Compact counts/source/log/postflight hashes are in
`run-logs/fs-w168-nkilib-small-shape-diagnostic-summary-r1.json`. Cron04:30:52
pushed sourcec36a1d14 and snapshotca6bc01f, including the mirrored failures.

## Source169 production results and bounded follow-ups

Both source169 jobs completed with commandrc0, clean postflight and every
declared requirement; independently checked against the frozen plans locally.
The MLP passes all six cases, including the production32×1024×1024 shape:
native5.129884652ms versus adapted library4.049636101ms (21.058% lower time).
The three alternating timing rounds are tightly grouped. Input/output reshape
and both weight-layout conversions are charged. The library/native output
relative RMS is0.00286148; original independent FP64 and exact repeat/batch
checks pass. This remains a FORWARD-ONLY result, with no saved training
activation, backward, optimizer or whole-model gain claimed.

Attention completes all six diagnostic cases, including production shape.
Every declared unchanged native comparison passes, but the retained CPU
coordinate heuristic still reports1514 failures across both versions/heads/
fields. This is explicitly a completed diagnostic, not successful passage
of that heuristic. Both postflights are clean; free16.77GB/10.26GB.

Source170 preparations: trn21 compare token-row sharding, eliminating
intermediate-channel summation between cores. All original MLP checks remain,
plus the campaign's unchanged native coordinate/RMS comparison before timing.
Trn22 perform paired attention timing with exact causal/batch/repeat checks
and the unchanged native comparison, while retaining/reporting every original
CPU screen failure. Its result is explicitly non-promoting: timing can tell
whether formalizing the corrected independent rounding contract is worthwhile.
No quality run is queued by either follow-up.

If an MLP forward win survives, the training candidate must save ReLU values
inside the forward kernel and charge that HBM traffic, all backward products,
FP32 master-gradient accumulation and actual weight refresh. Recomputing the
whole up projection would likely consume the1.08ms forward saving. Prefer
an explicit saved-activation output to hiding recomputation outside timing.

## Source170: exact token-row output, attention essentially unchanged

Source170 token-row MLP passes all six original CPU/repeat/batch cases AND
all six retained native-equivalence comparisons. All outputs are exactly
native, including production. Production forward4.042806849ms versus fresh
native5.128247148ms (21.166% less time), including casts/layouts specified by
the forward probe (weights themselves are BF16 inputs). It keeps the previous
forward gain while avoiding intermediate partial-sum BF16 rounding.

The paired attention diagnostic completed with exact repeat, causal-prefix,
zero future K/V-gradient and other-batch checks. At production, new10.703351ms
versus bundled10.745843ms, only0.395% lower; at the small batch it is1.297%
slower. These forward+backward timings do not justify a whole-model change.
The original CPU heuristic failures remain visible and non-promoting. Prioritize
the MLP training extension; preserve attention adapters/diagnosis as references.

Source171 implements an explicit saved-ReLU HBM output during the library
forward, followed by the existing native BF16 backward into FP32 masters.
Eight proposed component cases include zeros, tiny gradients, outliers,
changed weights, real width and32768 production tokens. Require retained
native output/saved/gradient comparisons, stagewise FP64 saved/output checks,
full-coordinate independent FP64 autograd gradient bounds, exact repeats,
independent rows and exact two-microbatch FP32 accumulation. Master BF16 casts,
contiguous transposes, activation stores and all backward work are timed.
Optimizer, communication and the surrounding model are excluded and must be
tested separately if the complete component remains faster.

Trn22 has9.60GB free after source170, close to the8GiB floor for larger jobs.
Before another model compilation, a separate maintenance script may retire
four named old profile copies (waves36/39/78/82). It requires a clean idle host,
the exclusive host lock, fresh local SHA256 checks, live GitHub asset-digest
checks and all remote-file hashes. Sources, checkpoints, compiler caches,
staging directories and all local/cloud evidence remain intact. Check its
actual intent/result receipts; preparation is not evidence of deletion.

### 04:56 UTC recovery and independent queue

Source171 saved-ReLU/full-backward component started on trn21 at04:52UTC.
The first seven fixtures pass; production32768-token fixture is in progress.
No complete timing/quality claim yet. Trn22 has source171 deployed and a finite
separate-compiled-region full-FP8 queue prepared. Source165 full component was
faster but its whole model was slower; the separate-region mechanism was only
previously tested with source95's older forward-only FP8/surrogate backward,
not the current compact full-FP8 four-product backward and all scale fusion.
The new queue requires byte identity for43 primitives versus source165, all
original component/numerical gates, actual separate-region Linear integration,
a fresh samehost native control and all charged full-step candidate costs.
No quality run is automatically promoted.

Trn22 profile retirement failed safely before mutation: waves36/39/78/82 have
empty directory remnants, despite is_dir() being true. All expected files are
missing; no unexpected files. Wave5-r3 is also empty. Preserve the intent and
failure; do not delete or retry merely to mark maintenance successful. The next
maintenance selects four old losing checkpoint copies, verified against local
full bytes and current GitHub asset digests, while excluding best and replay.

### 05:00 UTC: distinguish forward, plain-autograd and existing-trainer controls

Both source171 training component queues completed cleanly and every frozen
requirement was independently rechecked locally. The original eight fixtures
include full-coordinate independent FP64 gradient bounds, original native
comparisons, independent forward GEMM gates, exact repeated calls, causal row
independence and exact two-microbatch FP32 master accumulation. At production,
15.444630ms library versus18.311820ms plain native is15.657% lower. However, a
matched three-shape follow-up uses the CURRENT native HxT saved-activation /
custom backward control: library15.361039 versus native15.216359ms, **0.951%
slower**. This is not a win over the current trainer. The plain-autograd speedup
alone would be an invalid promotion argument. Machine-readable source/log/
postflight identities: `run-logs/fs-w171-nkilib-matched-component-comparison-r1.json`.

Next source172 adaptation writes the saved ReLU directly as HxT through a DMA
access pattern, retaining the library forward's row sharding and the current
backward layout. All original8case gates will be rerun. The new integration
module and phase flags are only prepared code until component and actual
Linear/cache gates pass. No quality run is queued. The documented DMA transpose
requires SBUF destination, so this first candidate uses an ordinary DMA store
with HBM strides rather than invoking DMA transpose on an unsupported HBM
output. Source: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_transpose.html

Trn22 separate-region full-FP8 actual Linear integration passed both widths;
the fresh fullmodel native control is compiling, with candidate timing next.
The verified checkpoint retirement freed1,879,330,816B and left11.350GB free.
All four local/cloud full checkpoint copies remain, including current best and
cold replay; only selected old EC2 losing copies were unlinked. The earlier
empty-profile failure is preserved as a failure and caused no deletion.

### 05:05 UTC: saved-layout stores need coalescing

Source172's direct HxT store passes numerical checks through production full
FP64 gradients, but the2048x1024 complete component took94.988537ms versus
1.107599ms native. The strided destination is mathematically correct and
extremely inefficient. Production timing is still running; no promotion.
Next source173 uses128x128 TensorEngine transposes and contiguous HxT stores.
It explicitly reserves32KiB SBUF below the library allocator's lower bound;
there is no untracked temporary allowed to overlap its manual scratch space.
The current projection's PSUM bank is reused only after ReLU eviction. All
upstream source stays byte-identical and all8 original numerical cases run.
The prepared integration hooks follow the new store implementation only after
its gate passes. This is an implementation hypothesis, not a speed claim.

Additional library lead: STATIC and ROW quantization are explicitly Trn2 FP8
paths in common_types.py and mlp_cte_basic_projection.py. The latter has
separate double-row up/down implementations. This is different from the MX
Trn3 path; revisit its tiling only after the BF16 saved-layout investigation.
The fresh baseline phase breakdown is dominated by the training microstep:
433.109ms of488.734ms mean total. Sync gradients16.549ms, optimizer26.465ms,
weight caches2.871ms. These source165 numbers argue for continuing forward/
backward fusion instead of assuming communication is the main bottleneck.

### 05:13 UTC: complete saved-layout comparisons

Both source172 and173 finished cleanly and all original8fixture gates passed.
Strided HxT1579.535094ms versus15.243443ms native is rejected. Coalesced,
on-chip-transposed HxT17.918935ms versus15.200161ms native is also rejected.
The latter retains exact saved ReLU and exact production dx/dup/ddown; its
forward differs by only2.08e-5 relative RMS, passing the original independent
and native bounds. No tolerance changed. Neither variant enters the trainer.
Source174 will test the simpler compiler-managed bulk transpose after the
original source171 TxH forward. It retains the8 fixtures and adds zero-up and
zero-down to exercise zero-initialized projection paths. All HBM transpose
read/write and four backward products are charged.

Trn22's fresh matched native fullmodel control488.541417ms reproduces the
previous488.616ms. Its separated full-FP8 candidate has started compilation.

## Source174 result and source175 follow-up (05:20 UTC)

Source174 passed the frozen ten-case FP64/native/stage/accumulation screen and
clean postflight. Production complete component: bulk transpose15.992353ms
versus matched current native15.251666ms (+4.86%). No model promotion.

Source175 prepares a lower-cost ordering: evict BF16 ReLU, run the original
library intermediate transpose, save its already-transposed HxT tile, then
square in place before down projection. BF16 rounding order is unchanged;
elementwise square commutes with the bit-preserving transpose. No extra SBUF
allocation, activation transpose, or up-projection recomputation. The same
ten-case screen and native HxT control apply before any integration.

Trn22 separate-region all-scale FP8 completed with rc0 and finite rank checks,
but postflight correctly rejected free disk7,574,847,488B against8GiB reserve.
Keep that failure unmodified. The descriptive timings are slower than native;
no promotion. Source171 exact receipts and full timing summary are preserved
in fs-w171-fp8-separated-diagnostic-comparison-r1.json. A guarded12GiB capture
of old NKI scratch is in progress; no remote deletion until archive hashes and
current GitHub asset digests have independently verified preservation.

At05:24UTC scratch maintenance completed cleanly. All4,991 original files
(12,978,670,751B) are verified inside the local archive and in live GitHub
assets before remote retirement. Free disk increased7.57GB ->20.59GB.
Prepare the source171 native/FP8 clean repeat under new r2 labels, requiring
the exact recovery receipt and unchanged original numeric/rank gates.
The r1 failure remains intact and cannot serve as a successful prerequisite.

## Source175 completed: reuse the required transpose

All ten fixtures and frozen requirements pass, clean05:24:45UTC. Production
complete component14.845040ms versus native HxT15.210640ms:2.403582% lower.
Small2048x1024 remains slower1.165296ms versus1.085126ms. The save is exactly
native at production; no thresholds or arithmetic bounds were changed.

Source176 changes the previously untested integration hook to use this gated
source175 kernel. Proceed to actual ordinary Linear/FP32-master/cache refresh,
accumulation, training row independence and native evaluation/checkpoint gates,
then a fresh same-host native/candidate whole-step comparison. Component
improvement is too small to establish a useful whole-step gain; only actual
measurements can decide whether compiler effects enlarge or erase it. No
quality trial is authorized by the component alone.

Source176 r1 was rejected LOCALLY by fresh_start_launch before SSH because
the generated integration argv omitted the mandatory timeout prefix. There
is no dispatch receipt, remote launch, kernel result or cleanup to relabel.
The immutable r1 plan/stopped coordinator are preserved. New r2 labels retain
the exact kernel/source/gates and add timeout2400s/TERM plus30s kill grace to
every job. The host lock, pre/postflight, fingerprint gates are unchanged.

## Clean FP8 repeat and separate BF16 region follow-up (05:33 UTC)

Source171 r2 native/candidate BOTH complete and clean, all unchanged frozen
checks independently verified. Full-step medians488.469301/592.811543ms:
FP8 separate region21.361% slower. This confirms the descriptive r1 slowdown
after restoring disk headroom. No quality promotion. Exact hashes, phase
contracts and medians are in fs-w171-fp8-separated-clean-comparison-r2.json.

Source177 prepares a separate compiled region for the source175 BF16 library
MLP, a different activation-buffer/forward contract from the FP8 path. Retain
all cache/update/eval gates; additionally walk the derivative graph to verify
compiled execution in the separate-region integration (the prior draft only
asserted the direct joined node). No changed kernel arithmetic, source175
component gate reuse requires byte identity. Measure its own matched native
control on trn22; do not infer gains from cross-host or component timings.

## Source177 separate-region harness failure and source178 diagnosis

Source177 stopped cleanly05:35:45UTC after two passing output/dX checks. At
second microbatch, the NATIVE reference root was no longer named
CompiledFunctionBackward. No numerical bound failed. This may be a wrapper
or a shared-code-object Dynamo cache/skip interaction; inspect actual graph
nodes before claiming the cause. The candidate and reference both originate
from Probe.forward, but only the candidate permits a graph break.

Source178 retains the original native-root assertion and every numeric bound.
It records both derivative graphs at the original failure. Its explicit
--diagnose-probe-roots mode must reproduce that same assertion first, without
counting the failed run as a gate pass. Then clear Dynamo state and give the
candidate a distinct root function that calls the same MLP. Require all48
original output/dX/master-gradient comparisons,10 exact training/eval/restore
checks, two widths and compiled execution of BOTH control and candidate. If
isolation fails, stop again; do not waive compilation or numeric gates.

## Next FP8 direction grounded in the pinned library

The ROW/STATIC CTE branches use ordinary `nc_matmul(perf_mode="double_row")`
with two 128-element contraction tiles on Trn2. They do not require MX
instructions. Source: core/mlp/mlp_cte/basic/mlp_cte_basic_projection.py,
_perform_doublerow_gate_up_projection and _perform_doublerow_down_projection.
The compiler control already uses older NKI attention, so importing the library
is not equivalent to newly enabling NKI everywhere.

ROW is not a free BF16 input conversion. Its input contract is legacy E4M3
bytes with an appended four-byte FP32 scale per token, shape[B,S,C+4]. Weight
scales are broadcast per output channel (up[128,I], down[128,C]); its loader
reads the appended input scale through an FP32 access pattern. See
mlp_parameters.py::mlpp_input_has_packed_scale and mlp_cte_basic_tensor_io.py::
load_packed_hidden_scales. Our current resident producer instead uses per-matrix
power-of-two scales and two packed weight orientations. The format conversion,
calibration and every cache refresh must be timed. No precomputed weights or
untimed corpus calibration should be credited as a training gain.

The library intermediate row quantizer computes max-absolute / dtype maximum,
clamps the dequantization scale to1e-6 before reciprocal, then casts the scaled
intermediate. Its zero handling is explicit. Legacy E4M3 maximum is240; PyTorch
E4M3FN's448 encoding is different. Preserve raw uint8 bytes and reinterpret only
inside NKI, as the existing independently verified producers do. Do not silently
cast those bytes through torch.float8_e4m3fn. The original ROW eviction rounds
weight-scaled PSUM to BF16 before applying the input-row scale and activation;
a CPU reference must account for that intermediate rounding.

A bounded next implementation would (1) gate online per-row/input and
per-output-channel/weight byte producers, including appended-scale packing;
(2) extend the original ROW ReLU eviction with BF16 square in the isolated
namespace; (3) screen forward against independently decoded FP8 values and a
staged FP64/BF16 reference, reporting BF16 approximation error separately;
(4) add saved ReLU/quantized activations and all four derivative products;
(5) gate master/cache/update behavior and measure complete steps. Require zero,
tiny, changed and outlier weights/rows, repetition and batch independence. A
forward-only or free-cache result cannot authorize a quality run. This is a
concrete design lead, not an implemented or queued FP8 experiment yet.

## Source178 diagnosis resolved (05:46 UTC)

The original failure reproduces with native_root=LinearBackward0. Its graph has
only Linear/ToCopy/Pow/Relu/AccumulateGrad nodes and NO CompiledFunctionBackward;
it was actual eager fallback, not a harmless outer view. Candidate graph still
has two compiled nodes. A distinct candidate root then passes the unchanged
native-root assertion on every microbatch, all48 original numeric comparisons,
10 exact checks, both widths and full cache/update/eval/restore checks. Frozen
requirements (including the two original diagnostic comparisons) are verified;
postflight is clean05:44:08UTC. The kernel and numeric bounds are unchanged.
The first root-cause record is fs-w178-nkilib-shared-root-diagnosis-r1.json.

Both queues are now in full-model measurements: trn21 source176 joined compiler
region and trn22 source178 separate region. Native whole-model controls run in
independent processes. Source176 native completed cleanly05:42:28UTC with13.32GB
free before the candidate; source178 recovered host has18.09GB after integration.
No full-step/BPB gain has been established for the library yet.

## Packed ROW quantizer prerequisite (prepared 05:55 UTC)

`nki_library_row_quantize.py` implements online BF16 absmax, the pinned
library's max(absmax/240,1e-6) dequantization scale, FP32 reciprocal and raw
legacy-E4M3 output. Its optional C+4 layout writes the per-token FP32 scale
through the same typed HBM access pattern used by the library consumer. All
metadata stores remain charged. This is new code awaiting hardware validation.

The independent screen has14 fixtures: zero, tiny, outlier, wide row-scale
range and random values in both ordinary weight-row and packed token layouts;
the finite BF16 alphabet; and all three production activation/weight shapes.
Every fixture also changes a row and requires all other rows to stay exact.
Absmax, dequantization scale and decoded quantized values must match their
independent CPU references exactly. The reciprocal is separately screened
against FP64 division at a predeclared4 FP32 unit-roundoff relative bound;
this bound is an experimental requirement, not an AWS precision promise.
Signed zero byte encodings are equivalent, but every decoded value is checked.
Only a passing, clean producer run can support a later ROW MLP forward screen;
neither its standalone timing nor a forward-only result qualifies a model.

API references: [reciprocal](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.reciprocal.html)
performs VectorEngine FP32 arithmetic; [activation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.activation.html)
performs multiply/add/activation in FP32 before casting to the destination.
The producer deliberately avoids any PyTorch E4M3FN reinterpretation.

## Complete model outcomes and next attribution (05:59 UTC)

Both original frozen whole-model sequences completed CLEAN and passed every
integration/rank/numeric requirement. Joined library MLP561.502963ms versus
489.888026ms native on trn21 (+14.619%). Separate library MLP611.722864ms
versus488.809250ms native on trn22 (+25.146%). No quality promotion. The
extra cost is in microstep forward/backward; optimizer/communication/tail
medians remain close. The isolated2.4% complete-component improvement does
not overcome compiler integration costs. This is additional evidence that
low-precision arithmetic alone does not explain earlier FP8 caller regression.

A paired training-forward device capture can now compare ordinary Linear
array geometry, executed NKI/library regions and observed spill/reload counts.
Use the same source, host, seed and35 preceding updates; include the ordinary
30-step measurements before capture. Trace durations are attribution, never
headline speed evidence; record dropped notifications and preserve all NEFFs.
Only after interpreting this comparison choose a new compiler-boundary change.

An inherited PROF_REPO still named wave171 in both timing arms. Read-only SSH
hashes confirmed its train.py and prepare.py match both actual checkouts
exactly on both hosts; scratch adapters resolve beside the actual phase script.
No algorithm difference follows from that path in these runs. Source180 will
default the timer to its own checkout, reject a different PROF_REPO, assert
resolved imports and emit their hashes per rank. The audit JSON preserves the
old path rather than silently rewriting any finished request.

Source179 ROW producer passed all14 fixtures,28 original/changed quantizer
gates and154 exact checks, CLEAN. Maximum reciprocal relative error5.2824e-8
is below the frozen2.3842e-7 screen. Production input pack0.913098ms; up/down
weight row quantization0.117969/0.082215ms, before layout/broadcast work. A
ROW MLP still requires separate stagewise and full-training validation.

## ROW FP8 forward screen prepared (06:05 UTC)

Source181 uses the gated179 online producer and pinned library ordinary
double-row ROW projections. The isolated hook adds BF16 square after the
original scaled BF16 ReLU. Diagnostic mode stores up-projection FP32 sums,
rounded ReLU and intermediate FP8 bytes/scales; the timed mode omits those
stores, and their final outputs must agree exactly. No upstream file changes.

Ten fixtures retain zeros, outliers, changed/tiny weights, row-scale variation
and both real shapes. Decode every packed input/weight and intermediate byte.
Audit raw up GEMM against the independently characterized11-bit pair-sum
reference and FP32 accumulation bound; verify original scaling/ReLU rounding
exactly; check intermediate scale exactly and decoded bytes against a frozen
FP32 reciprocal/multiply interval (8 unit roundoffs), then independently audit
down projection with both BF16 scaling stores in its bound. Small cases use
all GEMM coordinates; production uses documented row/channel samples plus
full-output finite/repeat/row-independence checks. The pair rule is empirical,
not an AWS guarantee; approximation versus unquantized BF16 is diagnostic.

The headline component arms charge online activation/weight calibration,
FP32 master casts, transposes and128-row scale broadcasts. Saved training
activations and the full backward remain absent; this screen can authorize
further implementation only, never a whole-model or BPB claim.

## Source182 transpose diagnosis and source180 record boundary

Source182 reproduced the original prepared-weight corruption. Omitting only the
two transposes yields exact bytes and identical other six outputs. The framework
float-carrier workaround produces the same corrupt up-weight tensor as ordinary
transpose (235,495/262,144 mismatches); source182 fails cleanly before its NKI
arm. The failed request/plan/source are retained. Source183 is a separate exact
NKI gate; no float-carrier success or MLP result is claimed.

Source180 native forward capture completed rc0/CLEAN with seven compiled binary
exports, no missing exports, and no unresolved trace keys. Its coordinator
stopped because the substring matcher counts NEFF_ARTIFACT again inside
TRACE_NEFF_ARTIFACT. Exact record-boundary parsing passes every unchanged payload
and model check. The original stopped sequence is retained; its proof is
fs-w180-native-record-boundary-diagnosis-r1.json. A subsequent request will use
record-boundary markers. Full UUID coverage still requires Explorer ingestion.
Both helper code and the original frozen requirements remain unchanged.

Source183 passes all frozen gates/CLEAN: the NKI integer-carrier transpose is
byte-exact in composed preparation and all256-code fixtures, including both
production weight orientations. Ordinary transpose fails3,391,488/4,194,304
bytes per production fixture; float-cast workaround was already rejected. NKI
layout costs0.134614/0.135399ms, charged in future complete-forward timings.
Source184 uses only this verified layout substitution and clears all original
input/up/down producer gates. It then fails before MLP execution on NKI0.5's
parser: tuple assignment to object attributes is unsupported. Source185 splits
only those two statements into four assignments. Normalizing the old AST this
way exactly equals the new AST; original arithmetic and all bounds remain.
The parser failure and its clean postflight are preserved.

Source185 clears the parser but rejects our diagnostic store from PSUM directly
to HBM. This is an instrumentation error, not a library numerical result.
Source186 routes the same FP32 values through SBUF using tensor_copy before
DMA. The store remains conditional on diagnostics; stripping diagnostic branches
from old/new ASTs gives identical plain paths. Original184/185 failures, every
fixture, byte/numerical bound and exact plain-versus-diagnostic check remain.

Source186 reaches an upstream ROW cleanup bug with skip_gate_proj=True: only
the up-scale allocation exists, but cleanup pops both up and gate allocations.
This underflows when the constants are released. Source187's isolated shard
copy changes one AST branch to pop the gate scale only if allocated. Strict
allocator checks stay enabled. Replaying the original/corrected cleanup AST
with gated/ungated mock heaps reproduces the original underflow and verifies
balanced cleanup in both corrected cases. Upstream vendored files and numerical
requirements remain unchanged. Hardware validation is still required.

Caller r2 retains another nested copy of the ambiguous NEFF marker. It stops
before torchrun. Caller r3 fixes that one nested marker constant and preserves
all payload fields. Its first dispatch preceded its plan commit because the
shell continued after a cron-lock failure. The immutable kernel and recorded
plan/request hashes were unchanged. A tracked starter now refuses uncommitted
or changed plan/request blobs; positive and negative cases are checked locally.
The original event is retained in fs-w180-r3-precommit-dispatch-diagnosis-r1.json.
