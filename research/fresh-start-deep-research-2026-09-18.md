# Fresh-start synthesis: the most credible routes below 1.0 BPB

This is a living research argument, with falsifiable experiments. It separates
results in existing logs, facts verified in source/numerical probes, and proposals
whose BPB benefit is unknown. The completed width896 direct-attention control
now reaches **0.987568 BPB at context 2048** and **0.999880 at context 1024**
on four-rank, 2,097,152-token public diagnostics. It completed 2,188 updates
within the 30-minute allowance. The earlier width896 checkpoint's larger
20.97M-token standalone result at context 1024 was **1.009889**; the new
checkpoint still needs that larger check. The newer fused-FP32-table-optimizer
checkpoint has now completed the identical 20.97M-token standalone check:
**1.004452862 BPB**, versus 1.009889395 for the first wave. The small-sample
sub-1.0 result therefore does not establish robust sub-1.0 quality at context1024.
These are public measurements, not a private leaderboard score. See the [measured results](fresh-start-results-2026-09-18.md)
for source commits, timing, numerical gates and limitations.

The strongest opportunity is to recover useful capacity and optimization quality
from the existing transformer. A wider version now learns below 1.0 on the public
context 2048 diagnostic within the budget. Meanwhile the current code and
measurement history contain concrete changes that previous high-level conclusions
do not account for. A complete replacement with an SSM or all-FP8 stack would make
these mechanisms harder to isolate.

**12:58 UTC update:** the dense6×1024 path, without hashed n-gram tables,
now completes2,977updates/390.2M target presentations in budget at **0.981710
BPB@2048 /0.996689@1024**, versus1.012886/1.027777 for the1,805-update dense
control. Custom NKI convolution plus BF16 normalized branch inputs reduces the
gated update from0.986s to0.588s; residuals stay FP32. The combined change
therefore improves real full-budget quality, not just throughput. Its checkpoint
is locally hash-verified. These are four-rank2.1M-token public diagnostics. The
identical20.97M-token standalone check now gives **1.002066687 BPB**, versus
**1.004452862** for the table model, with the same83,462,186bytes scored. The
dense implementation is a competitive full-budget route; it is still above1.0
on the larger sample and has no independent-seed confirmation. Eight-layer dense capacity passed a
0.744s phase gate but finishes at1.004083 on the small1024 diagnostic, worse than
six layers'0.996689 because the extra capacity comes with fewer budgeted updates.
That mechanism is an interpretation, not a matched-update ablation. There is still no verified information
about the leading teams' architectures.

The real-shape cached-weight FP8 probe passes its numerical gates but does not
beat native BF16 after layout preparation; the MLP output projection is the
clear negative. The B8 microbatch phase saves only1.9%, below its full-run gate;
B16 regresses to615ms, so this packing sweep is closed. A fresh current-graph device capture is preserved
locally, and its unprofiled585ms update agrees with the earlier588ms result.
The next full-budget algorithm control changes only clipping order to average
gradients before clipping. These are measured decisions and a pending test,
not evidence that the remaining0.00207BPB gap has been solved.


Supporting audits:

- [Experiment evidence and source-history audit](fresh-start-evidence-2026-09-18.md)
- [Training and native-backend code audit](fresh-start-code-audit-2026-09-18.md)
- [Rules, metric and hardware audit](fresh-start-rules-research-2026-09-18.md)
- [Operational plan and run protocol](fresh-start-plan-2026-09-18.md)

## 1. Reconstruct the actual problem

The reference is now a six-layer, width-640 model with roughly 56M dense parameters
and four learned hashed n-gram tables totaling 536,870,912 parameters. Describing
it as merely a tiny NanoGPT misses where its memory and optimizer traffic go.
Table capacity is also not equivalent to dense transformer capacity: a token only
activates a few rows, and collisions share statistics between unrelated n-grams.

The reliable historical public result is approximately 1.004–1.006 BPB at context
2048 and about two million evaluation tokens. The strongest raw-log value located
is 1.004355; an official-style standalone public evaluation reports 1.003045.
These are neither private leaderboard scores nor interchangeable with context1024
results. Historical larger-sample evaluation actually made an older model's score
worse, so we cannot assume the public evaluation size explains the leaderboard gap.
See `trn2-experiment-results.md:1560–1645` for that calibration.

The useful counterexample is `tc-sh-896x6-2135`: **0.987716 BPB**, 2,137 steps,
approximately 280M token presentations, but **2,488.7 charged seconds**. It is
over budget, not a valid win. It does prove that the dataset and broad architecture
can support better validation loss. Recovering approximately 28% of that runtime
would bring the same number of steps within 30 minutes; smaller improvements might
still suffice if fewer steps retain enough of the capacity benefit. Both statements
are hypotheses until a budgeted run finishes.

The campaign must optimize three interacting quantities:

1. Useful token presentations per second, including optimization and communication.
2. Generalization per update for the actual model, table precision and schedule.
3. Capacity that fits into the resulting number of updates before the timer ends.

An improvement to only the first quantity can lose if it changes arithmetic or
shifts the final cooldown onto a different corpus prefix. A short run that ranks
architectures at five minutes can select the wrong capacity for thirty minutes.

## 2. Three assumptions that the audit overturns

### Current ordinary-embedding gradient: a specific new kernel hypothesis

The successfully ingested NKI/BF16 dense capture now attributes 587.204 GFLOPs
of regular Tensor work to input and unigram value embeddings, even with all
hashed tables disabled. The former FP32 attention-backward matrix bottleneck
is gone: total regular FP32 work in the current backward graph is only
0.199 GFLOPs. These are instruction FLOPs, not exclusive runtime fractions.
The unchanged model uses one input table and three value tables at vocabulary
8192 and width1024; all four see the same token IDs in a microbatch.

This was not wholly missed in earlier work. The September11 benchmark at
8192×640 and BF16 weights measured native embedding forward/backward at3.18ms
versus advanced indexing5.20ms and index_select5.12ms. That negative result
justified retaining native ordinary embeddings. It used uniform IDs and a
best-of-five timer. The current width, table precision and device trace justify
a focused retest, including a different algorithm, rather than blindly enabling
the already rejected indexing change.

The proposed algorithm stable-sorts IDs once, shares compact metadata across
the four tables, and uses NKI's FP32 scan with reset flags to accumulate each
ID's gradient. Per-ID end indices select the sums and absent vocabulary rows
are explicitly zero. A scan with resets avoids subtracting two large prefix
sums. Each output row/channel has one writer, so duplicate-ID scatter races do
not arise. The initial kernel includes gradient load/transposition, local
gathers and the final standard [vocabulary,channels] output layout. CPU sorting
and 81,920 bytes of metadata upload at the actual shape must be charged.
This training-only metadata never changes inference or the tokenizer.

The installed NKI0.5 API and AWS's
[scan documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor_scan.html)
support the recurrence. The
[per-partition gather](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html)
selects input order and group ends. API availability establishes feasibility,
not speed; vector scan or layout costs may still lose to dense Tensor work.
FP32 accumulation may also change gradients relative to compiler BF16
autocasting and therefore requires a full-budget quality comparison.

`embedding_segmented_probe.py --device cpu --real-shape --training-row`
passes eleven FP64-reference/error-bound cases: small/actual-shape uniform,
Zipf-like, one repeated ID, zero gradients, cancellation, and a real shard0
training row. That row has1,867 unique IDs and a maximum count279 among8,192
tokens. All absent rows are exactly zero. The first CPU attempt passed synthetic
checks but failed to locate shard0 in the default cache; the second uses a
dedicated cache linking the already preserved training shard/tokenizer. Both
logs are retained. This validates the mathematical reference, not the unrun
NKI kernel.

Advance only after device numerical checks and a measured component saving
of20% over current native backward, including an explicitly labeled estimate
for preparation shared across four tables. Compare native embedding and
advanced-index autograd at the real shape as controls. Then require an actual
four-rank whole-step saving with all metadata preparation charged, updated-model
contract checks, and full-budget BPB. Do not claim a whole-model gain from the
FLOP count, a kernel-only timing, or a host-side preparation estimate.

**Device outcome, 13:08:** the scan passes the small cases but fails the first
actual T8192/C1024/V8192 case (relative RMS0.507686, over1.28M error-bound
violations). It stops before timing that shape. Do not integrate it. The next
diagnostic separates load/transpose, permutation gather, scan and group-end
gather, varying channel-loop count and token length. Instrumentation can change
buffer lifetimes, so the original kernel remains a separate check.

A distinct algorithm keeps native BF16 matrix accumulation but remaps present
IDs to a compact local vocabulary. It then clears the ordinary dense output
and scatters each unique accumulated row exactly once with NKI. Channel-based
program ownership is shared by clearing and scattering, avoiding a cross-core
clear/scatter race. All preparation, upload, zeroing and scatter costs belong
in the comparison. The tokenizer, output vocabulary and model parameters are
unchanged. CPU checks cover duplicates, zero gradients, cancellation, the full
8192-ID fallback and a real training row. Device checks are still pending.

The first row's1,867 unique IDs is not representative enough to set workspace
size. A fixed-loader audit of2,048 rows /16,777,216 shard0 training tokens finds
median2,186, p99=2,457.12, maximum2,544 unique IDs. Of those rows,1,633 need a
4096-row bucket,414 fit2048, and one fits1024. None needs8192 in this sample;
this is not a guarantee for the whole corpus or arbitrary training input.
Use a fixed4096 ordinary workspace for the component test, with a full8192
fallback for overflow. A future integration must handle that fallback without
silently losing rows and must charge any recompilation. Local CPU metadata
construction median0.730ms is not an EC2 timing. The audit changes no data.

### Current default attention is not the historical compiled graph

The newly introduced `nki_attention` helper is unconditionally Dynamo-disabled.
The default model calls it even when the custom kernel flag is off. CPU tracing
of the actual attention module confirms two graphs/one break, versus one graph/no
break with a direct SDPA call. Sixteen output and gradient cases match bit exactly
on CPU. The opt-in `--compile-sdpa-direct` makes the difference reviewable.

Both the compiled backend **and** native eager SDPA can select the installed NKI
flash attention. Therefore the proven issue is graph partitioning; it is not
proof that the default is using a slow unfused attention implementation. The
effect could be small, or compiler fusion/resharding could make it substantial.
Measure it at full four-rank geometry before promoting it.

The first 60-step four-rank smoke measured approximately 0.66–0.67 s/step with
direct compiled SDPA versus about 0.70s in the first width640 BF16 run. This is a
throughput lead, not a completed quality comparison: the smoke intentionally
stopped before cooldown. Outputs passed the in-run causality check.

Git history places the disabled wrapper after the historical record, width896,
and BF16-autocast comparisons. The first new wave inherited it. Those runs remain
useful results for their exact source, but historical comparisons are confounded
by graph boundaries as well as width, duration and precision. This correction is
central to interpreting the evidence, not a reason to discard inconvenient runs.

### The baseline optimizer was not entirely FP32

Dense weights and Muon state have FP32 components, but the Newton–Schulz input is
explicitly cast to BF16 in the baseline. Thus global BF16 autocast cannot be blamed
on a previously-FP32 Newton–Schulz calculation without additional evidence.

One plausible place to investigate is SDPA's fallback backward, whose installed
decomposition explicitly upcasts operands to FP32. First determine whether that
fallback actually executes. The package has no supported per-region compiler-flag
option: arbitrary `torch.compile(options={...})` keys do not implement a precision
override. Explicit math or NKI islands would be needed after localization.

### BF16 table moments do not implement their nominal decay

The table optimizer allocates second moments with `zeros_like(parameter)`, hence
BF16. Its nominal update is

`v_next = v + 0.001 * (g² - v)`.

For a zero-gradient row, the desired decrement is 0.1%. That is smaller than half
the downward BF16 spacing of normal positive values. Round-to-nearest can return
exactly `v` forever. The CPU test used 1,001 positive values over many orders of
magnitude: every BF16 value stayed fixed after the decay, whereas FP32 values
changed. After 1,000 zero-gradient updates, 1.0 remained 1.0 in BF16 but became
approximately 0.367695 in FP32. The native-device probe has now reproduced this.
The two-core fused FP32 update measured 10.28ms per full table including copy-back,
versus 11.28ms for the legacy BF16 path and 23.38ms for ordinary FP32 arithmetic.
Repeated updates through the actual native optimizer also pass CPU-reference
checks, including varying LR, sparse/zero/None gradients and two decay constants.
The full width896 run now completes 2,204 steps versus 2,188 for the legacy
control, with 0.999735 versus 0.999880 BPB at context1024 on the same four-rank
2.1M-token sample. It retains whole-model throughput. The quality difference
is too small, and exposure differs, to conclude that optimizer fidelity improves
generalization. Larger fixed evaluation and a repeat remain necessary.

This is a numerical fact, **not yet an explanation for poor BPB**. The accidental
history dependence may act like useful regularization or AdaGrad-like damping.
Making the arithmetic more faithful increases some late effective learning rates
and can worsen generalization with the inherited hyperparameters. The right first
comparison changes state/arithmetic precision while retaining BF16 table storage;
it is not an all-FP32-table experiment with doubled gradient communication.

## 3. First experimental ladder

| Priority | Hypothesis and isolated change | Gate before full run | Evidence that closes or advances it |
| --- | --- | --- | --- |
| A | Restore direct compiled SDPA | CPU outputs/gradients and graph capture; short full-chip timing/causality | Same-source no-autocast and BF16 controls; speed without loss divergence advances |
| A | Use recovered throughput for width896 or768 | Short stable run, memory, compile/startup margin | Full 1800s BPB at both contexts, charged time and saved-model evaluation |
| A | Independent transformer track without hashed n-gram tables | Reprofile current BF16/direct-SDPA graph; do not assume removing tables speeds it up | Spend measured savings on dense capacity; compare full-budget BPB and then matched-exposure table ablation |
| A | FP32 n-gram second moments and update math, BF16 weights/grads | CPU and native sparse/zero/outlier update tests; memory and time | Full-budget quality; if promising, separate state decay from weight rounding and retune table LR narrowly |
| A | Fuse table RMSProp in NKI | CPU reference, native output/state checks, runtime scalar changes | Include copy-back/layout cost and all-rank whole-step time; reject microkernel-only wins |
| B | Clip after gradient averaging | Real distributed CPU reference already passes | Paired full budget at the same graph/precision, then another seed if useful |
| B | Average only current flat-LR n-gram tables late in training | In-budget update and finalization; deterministic saved model | Compare one predeclared tail window to current last iterate; old dense-model EMA negatives are not the control |
| B | Mask zero-byte targets in training CE | Match trusted scoring numerator; CPU/NKI loss/gradient checks | Measure affected target fraction, then full run; positive-byte tokens keep ordinary CE |
| C | Optimizer-only stochastic rounding or compensation for table weights | Quantify canceled nonzero updates; deterministic eval | Use only if cancellation is material and cheaper alternatives fail |
| C | Native scaled FP8 MLP forward | Actual installed API, error checks, cast-inclusive speed | Advance to training only if this island lowers full-step time without loss instability |
| C | Sparse exact table communication/update | Trace identifies sufficient dense traffic; duplicate-safe algorithm | Exact all-rank aggregation and lazy-state reference before any quality experiment |
| C | Restarted Gram Newton–Schulz for rectangular Muon groups | BF16 stability and aspect-ratio-specific benchmark | Whole-step benefit, not just orthogonalization throughput |

Do not combine all attractive options immediately. Start with a trusted control
and one difference; combine only independently promising changes. Some interactions
deserve deliberate pairs: capacity×throughput, table state precision×table LR,
and flat table LR×tail averaging. Preserve the winning source revision, not just
the flag list.

## 4. NKI should target measured cost, not operator coverage

NKI RoPE/QK normalization and softcap/CE already exist. The compiled Muon path
supersedes the old NKI Muon flag. The custom attention implementation has accurate
results but poor integrated speed. Replacing all PyTorch operators would duplicate
working code and add launch/layout boundaries without a demonstrated benefit.

For a component occupying fraction `f` of step time, acceleration by `s` gives
at most `1 / (1 - f + f/s)` whole-step speedup. Eliminating half the time in a 10%
component saves only 5% overall. FP8's peak GEMM throughput is not a forecast of
training throughput. This bound should be written beside every kernel proposal.

The next profile must use the full four-table reference geometry. Old
`phase_timer.py` defaults disable the n-grams and do not represent this model.
Separate forward/backward, gradient clipping/reduction, table optimizer, Muon,
weight-cache refresh and exposed host time. A synchronized phase profile changes
overlap; follow it with ordinary unsynchronized whole-step timings.

The separate dense 6×1024 track has now been profiled at four ranks and the real
loader geometry. Its 1.417s unprofiled step spends 94% in model microbatches;
the optimizer accounts for only 3.9%. The device trace localizes the problem
further: a 218.6ms forward versus 97.8ms backward, with only about 17% forward
TensorEngine activity per physical core. Repeated small-DMA regions map to the
depthwise convolution. Another 84ms region has roughly 262K tiny transfers tied
to a `multiply.21`/transpose intermediate, but no high-level source mapping yet.
Missing DMA notifications prevent an exact byte-bound claim. These facts favor
layout and convolution work ahead of a wholesale FP8 training rewrite.

Those two probes have now separated sharply. Replacing the trained convolution
with static PyTorch shifts makes the step **3.083s**, about 2.18× slower.
Removing only the frozen zero Q/K shift makes it **0.986s**, saving 30.39% of
step time with the optimizer/tail nearly unchanged. Both retain the intended
trainable function, although finite-precision paths can change. This is concrete
evidence against generalizing a compiler perturbation's width640 benefit to
width1024. That dense control completed1,805updates, at1.012886BPB@2048 and
1.027777@1024. Custom NKI convolution passes separate native FP32 and BF16
forward, gradient and causality gates through B4×T2048×C1024, including zero
initial weights. The BF16-norm/NKI composition measures0.588s and passes updated
four-rank model gates. However, the NKI-only FP32 model produces NaN on its first
compiled forward, even with one kernel program. The eager full model and a
compiled graph retaining intermediate tensors are finite. Therefore the earlier
713ms FP32 timing is withdrawn as usable training evidence; the graph interaction
must be resolved before that arm is promoted. The BF16 composition's full-budget
run is separately in progress.

A distinct mixed-precision candidate is to keep the residual accumulation in
FP32 but store normalized branch inputs in BF16. The historical dtype trace
already established that FP32 residual scalars promote the block stream; the
initial version of our dtype table missed that implication and has been corrected.
This is a real arithmetic change, unlike the frozen-zero graph probe. It may
activate the BF16 weight cache and reduce activation traffic without using FP8.
RMSNorm epsilon/reduction, attention input dtypes and the final loss must be
audited explicitly; a variable named `COMPUTE_DTYPE` is insufficient evidence.

### Attention precision is an observed kernel issue, not only a dtype hypothesis

The installed profile schema defines FP32 work as four adjusted FLOPs per raw
FLOP. Counting **regular MATMUL** instructions, excluding transposes, spills and
weight loads, the captured backward graph contains **656.091 GFLOPs of FP32
work** across its two physical cores. Approximately **613.97 GFLOPs (94%)** map
to five source lines in the installed native `attention_bwd.py`: dQ, dK, dV,
recomputed QK and the probability-gradient matmul. The instruction operands also
identify FP32 inputs. The remaining regular backward matrix work totals
3,270.316 GFLOPs at the schema's BF16/FP16/TF32 cost ratio. This is one rank's
profiled microbatch, not a complete four-rank optimizer update.

The installed native decomposition passes the original query dtype to NKI
attention backward with `mixed_precision=True`. In that kernel, `kernel_dtype`
follows `q_ref.dtype`, while the statistics dtype stays FP32. This explains why
compiler-wide BF16 matmult autocast alone leaves this work in FP32. Installed
file hashes and version identity are in the
[source audit](run-logs/fs-native-attention-precision-source-audit-r1.json);
the expanded [profile summary](run-logs/fs-w5-dense1024-device-micro-r3.summary.json)
retains the missing-DMA/HLO-metadata warnings. FLOP counts do not require summing
overlapping engine durations, and do not establish a speedup by themselves.

Two independent experiments now isolate the options:

1. Cast Q/K/V only at the SDPA boundary to BF16, returning its output to the
   original query dtype. Keep FP32 normalization outputs, residuals, MLP inputs
   and Linear input/cache behavior otherwise unchanged.
2. Use FP32 RMSNorm statistics with explicit FP32 epsilon, then store normalized
   branch inputs in BF16, including the final head. Keep residual accumulation
   FP32. This also activates the existing BF16 Linear weight cache and changes
   more of the forward/backward arithmetic than the attention-only arm.

CPU gates exercise the real model with nonzero attention/MLP/conv branches,
three accumulation microbatches per update and two updates. Cache-on/off losses,
master gradients and updates match exactly for each policy. Future and batch
perturbations also match exactly. The normalization policy separately passes
15 FP64-reference cases, including tiny and large inputs. Its initial checker's
cancellation bound was corrected from `abs(sum(products))` to
`sum(abs(products))`; the unchanged RMS-error threshold still passes. These are
CPU gates only. Native phase probes additionally check finite parameters and
gradients and exact future/batch/repeated-call independence after real updates.
Full-budget trained quality remains the final promotion gate.

### Compute estimates put precision work in perspective

The no-table 6×1024 model has 117.454M parameters, but only 83.887M are Linear
weights; 33.554M are lookup embeddings. Applying `6*N*tokens` to every parameter
would overstate dense arithmetic. At 131,072 targets/update the linear work is
approximately 65.971 TFLOPs. Ideal causal versus full-square attention adds
9.900–19.791 TFLOPs, before FlashAttention recomputation and other operations.
At the measured 0.9865s step, that is **76.9–86.9 TFLOPs/s of useful model
arithmetic**, an analytical estimate, not measured hardware MFU.

AWS lists one Trainium2 chip at **667 TFLOPs/s dense BF16/FP16/TF32** and
**1,299 TFLOPs/s FP8**, plus 181 TFLOPs/s FP32. Do not mistake the FP8 number for
the BF16 peak. The roughly 11.5–13% useful-arithmetic/advertised-BF16-peak ratio
is only context: the present workload mixes precisions, recomputation, padding,
layout movement and non-matrix work. It is not a promise of the remaining
speedup. Sources: [Trainium2 chip specifications](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html),
[NKI hardware guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html),
[model arithmetic estimate](run-logs/fs-dense1024-analytical-flops-r1.json).

### Fused table optimizer

The four tables contain 536.9M elements. BF16 gradients alone total about 1.07GB
per rank. A naive sequence of full-table square, lerp, divide, square root, epsilon,
gradient divide and subtraction moves many multiples of that through memory.
Changing moments to FP32 adds state bandwidth but does not require changing the
BF16 table gradients sent between ranks.

An elementwise NKI tile can load weight/gradient/moment once, retain arithmetic
intermediates in SBUF, and store weight/moment once. At BF16 weight+gradient and
FP32 moment, the ideal reads+writes are 14 bytes per element, before aliasing or
copy-back costs. A returned-buffer prototype needs additional copy-back traffic,
which must be counted. It is a sensible first custom kernel because the mathematical
reference is simple and its effect on precision is explicit.

The prototype now passes the native repeated-update gate and is opt-in in training.
It uses runtime scalar tensors for learning rate and bias correction to avoid
recompilation every step. Tests include zero moments, untouched rows, rare gradients,
outliers, changed scalar values and multiple consecutive updates. The full run
completed 2,204 updates versus 2,188 for the legacy direct-attention control,
scoring 0.999735 versus 0.999880 at the small context1024 diagnostic. This near
tie does not establish a quality win. Its larger standalone result is 1.004453.
Fusion with different rounding points is not bit-exact equivalence to the old
BF16 algorithm.

### Sparse tables: promising but substantially harder

Each rank processes only about 32,768 token positions per optimizer step, versus
1,048,576 rows per table. The number of distinct touched rows is smaller still.
This creates a real sparsity opportunity. It does not establish the current
all-reduce is the bottleneck or that a sparse implementation will be faster.

An exact design needs all-rank index/value aggregation, duplicate-key summation,
correct clipping order, and lazy decay using last-touched timestamps. Hash collisions
are part of the model and must accumulate rather than overwrite. Fixed-size padded
buffers may keep the compiled shapes stable, but sorting/deduplication and launch
costs belong in the comparison. Every rank must end with consistent weights, and
the final checkpoint must be a consolidated model.

Installed NKI documentation exposes DMA read-modify-write for indirect access,
but requires unique indices for that mode. An old blanket statement that no
scatter operation exists is too broad; the missing duplicate-safe scatter is
still a real issue. Do not implement concurrent read-add-store as if it were atomic.

### Scaled FP8 as an island

Trainium2 native legacy E4M3/E5M2 and double-row matmul are different from newer
Trn3 MX/OCP interfaces. Start with BF16 inputs, scale/cast internally, accumulate
in FP32, and return BF16. Retain BF16 backward initially. Explicit scales and
layouts must be charged, and the BF16 control must have comparable boundaries.
See the [official Trainium2 architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html)
and [matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html).

The native numerical probe found an additional nuance: the tested double-row
outputs match 11-significand-bit rounding of each two-product partial followed
by FP32 accumulation. Exact ties and exponent-boundary patterns, plus a held-out
seed family, support this empirical model; it is not documented as an AWS
specification. Naively demanding a pure FP32 dot of the quantized inputs failed
for a reason distinct from input quantization or output BF16 rounding.

The first honest throughput test is negative: 0.361ms for the FP8 prototype
versus 0.0956ms for native BF16 at 8192×768×512, including transpose/casts but
excluding online amax. Even pretransposed FP8 takes 0.220ms. Its repeated weight
loads/conversions and limited tiling leave engineering room, but peak FP8 FLOPs
alone do not justify integrating it. Full-sequence adaptive scaling at evaluation
could also introduce future-token dependence; the scale policy needs a separate
causality check.

The second implementation removes repeated weight loading within a GEMM and
tests the actual dense MLP shapes. At M8192/K1024/N4096 and M8192/K4096/N1024,
native BF16 measures0.556/0.548ms, while cached FP8 including both operand
layouts and casts measures0.633/1.096ms. Even the activation-layout-only arm is
0.550/1.002ms. All numerical gates pass, but no training integration is justified.
Static scales and excluded backward make these component bounds, not a full FP8
pipeline test. The [results ledger](fresh-start-results-2026-09-18.md) retains
all controls, including native F.linear and pretransposed lower bounds.

DeepSeek-V3 demonstrates carefully scaled low-precision training, but its large
model and GPU-specific engineering do not transfer as a recipe. It is a source
of hypotheses about scaling granularity and sensitive paths, not evidence that
compiler-wide FP8 should work here. [Technical report](https://arxiv.org/html/2412.19437v1).

## 5. Reopen data and optimizer conclusions carefully

Historical exclusions should retain their actual scope. The paired 900s
1024/PF8 versus 2048/PF4 trial held global micro-token count fixed and tied at
the 1024 scorer (difference 0.000035); that is good evidence against that exact
short-context switch, not a bound on every position weighting or length curriculum.
Likewise the expensive one-hot study used eager tiled lookups and primarily
million-row tables. Its negative result is strong for that geometry, but its
claim that an untiled 8192-row one-hot cannot fit is inconsistent with its own
0.12GiB footprint. An older compiled 8192×640 lookup also lost in isolation.
Reopen a lookup spelling only if a current whole-model trace identifies a
layout cost it can remove; no such attribution is established for `multiply.21`.

### A second pass is not universally wasted compute

Large controlled studies of data-constrained language modeling found that a few
passes can remain useful. That does not prove the present model benefits from a
second pass; it rules out treating a one-pass ceiling as a universal law.
[Scaling Data-Constrained Language Models](https://www.jmlr.org/papers/v26/24-1000.html).

Here, repeating the initial corpus prefix during final cooldown, flat n-gram LR,
BF16 state history, and limited dense capacity could interact. The actual epoch
counter is tied to document ingestion and buffering, not exact unique-token
coverage. The fixed packer can discard tails. An audit can count these effects
without changing the pipeline. Replacing that pipeline is outside this campaign's
current run slate because the rules explicitly prohibit data-pipeline changes.

The audit is now complete: raw training text has 315.5M tokens. At 2,800 optimizer
steps, only 288.5M first-pass text tokens have been retained; 24.7M encoded tokens
remain buffered in 1,570 long documents, while a second pass over shorter
documents has begun. At 2,026 steps, first-pass retained text is 265.2M tokens.
See the results ledger and full count JSON before making an epoch-based claim.

The width640 BF16 shortened run approximately matches the old exposure, but its
actual step count and schedule must be checked; wall time is not exact step matching.
Also account for the wrapper source change. A decisive numerical attribution needs
same-source same-input updates plus controlled schedule/exposure, followed by a
separate full-budget quality comparison.

### Averaging and compensated updates target different failures

Late averaging reduces variation across iterates. Preserving sub-ULP updates fixes
numerical cancellation. Neither implies the other is unnecessary. The current
tables retain flat LR throughout the dense model's cooldown, so old negative EMA
results on a different dense model are weak evidence against table-only averaging.

Recent 150M/300M language-model work finds that averaging can make simple schedules
competitive across training horizons. Our table-only proposal is an extrapolation,
not a tested result from that paper. The inspected setup uses3.3B/6.6B minimum
tokens, explicitly no data repetition, and AdamW without weight decay; this is
not direct evidence for a ~315M-token corpus and our Muon/table optimizer.
Predeclare one short tail window and count
all averaging/finalization within the budget. [Anytime Pretraining](https://arxiv.org/html/2602.03702v1).

BF16 round-to-nearest can also cancel nonzero parameter updates. Work on BF16
training supports investigating stochastic rounding or compensated updates at
the weight update, rather than making every forward operation stochastic.
The prior global-SR repeatability failure does not close that narrower direction.
We should first measure cancellation on real table updates; compensation costs
additional memory and bandwidth. [Revisiting BFloat16 Training](https://arxiv.org/html/2010.06192v2).

### Muon acceleration without changing the model

Gram Newton–Schulz shifts repeated work toward a smaller Gram matrix for rectangular
updates. The authors explicitly show that the naive low-precision reformulation
is unstable and introduce restarts. Our MLP aspect ratios make it worth a bounded
benchmark if Muon is still a significant fraction of step time. Square groups
may be better left unchanged. Neither the CUDA kernels nor their published
speedups apply directly to this installed native backend.
[Algorithm and stability discussion](https://tridao.me/blog/2026/gram-newton-schulz/).

The reference code uses FP16, FP32 normalization, and a restart after two
iterations. Our optimizer uses BF16 normalization with a1.01 safety factor and
different polynomial coefficients. A transplant would therefore change several
algorithms at once. First derive the recurrence with our existing coefficients,
check exact-arithmetic equivalence and low-rank/ill-conditioned cases, then
measure the actual native optimizer. Its present55ms of a588ms update bounds
the total upside even before engineering costs. No Gram implementation has been
promoted. [Author implementation](https://github.com/Dao-AILab/gram-newton-schulz/blob/main/gram_newton_schulz/gram_newton_schulz.py).

### New bounded candidate: canonical-token continuation constraints

Deterministic BPE leaves some token continuations impossible. A causal output
constraint could stop allocating probability to them without adding learned
memory. The ICML2025 paper proves useful BPE locality properties but explicitly
warns that pretokenization can create false negatives. Our fixed regex differs
from GPT-2, so a generic table or isolated decode/re-encode test is insufficient.
[Canonical BPE paper](https://arxiv.org/abs/2506.07956).

The first local probe considers only an ASCII-letter previous token at least
three bytes long followed by an ASCII-letter candidate. Such a seam cannot end
one of the regex's contractions, whose letter suffix has length at most two.
Within this restricted letter run, use the fixed tokenizer's re-encoding test;
leave every uncertain Unicode, numeric, whitespace, punctuation, short-left,
and special-token case unrestricted. This is deliberately incomplete.

`canonical_token_mask_probe.py` tests4,854,956 eligible pairs and marks343,128
impossible. The derived mask rejects **zero** observed canonical pairs over
63,336,106 training tokens in86,016 provided shard0 documents and50,012 synthetic
context cases. It uses no validation observations, learned table, or external
training text. This empirical falsification check supports the restricted seam
argument; it does not prove a general pretokenizer mask correct. Build time was
11.52s locally; that narrower first version was not quality-tested. Generated matrices stay in
`out/`; source, counts, tokenizer hash and training-shard hash are preserved.

The `ascii-word` extension permits one ASCII non-letter/non-number prefix before
at least three letters, excluding apostrophe and CR/LF. It tests16,852,392pairs
and marks1,166,945, again with zero violations on the same63.3M training tokens
and200,012 synthetic contexts. Build time is41.75s locally. A frozen dense
checkpoint passes CPU finite-output, exact mask-reference, future-token, other-
batch, repeated-call and batch-size checks. The native result is recorded below.
The mask uses the current input token to constrain the next-token distribution;
it never reads a future token. Special-token rows and columns remain unrestricted.

The native paired check now passes exact future/batch/repetition/CPU-reference
checks and gives **0.996184154→0.996108324 BPB**, an improvement of only
**0.0000758305** on the same world1/B1/context1024/2.1M-token prefix. It used the
new dense checkpoint and a freshly tokenizer-derived `ascii-word` mask. This
conservative version is deprioritized; there is no training integration or
larger-sample projection. The raw log and summary in the measured-results ledger
preserve both scorer outputs and the20.425s construction cost.

The separate modded-nanogpt PR reports roughly0.0017 nats/token improvement from
canonical masking across20 paired runs, with about3.15s CPU table construction.
That establishes a practical precedent on its GPT-2/GPU workload, not an expected
gain for our8K vocabulary. Its current implementation handles regex boundary
exceptions explicitly. [Author's experiment](https://github.com/KellerJordan/modded-nanogpt/pull/350).

### Regime checks on the latest outside research

The fourth automated report (`small-corpus-research-status.json`) supplied useful
leads but no accepted recipe. Primary-source inspection changes its conclusions:

| Source | What transfers as a hypothesis | Why its headline is not a recipe here |
|---|---|---|
| [One Epoch Is All You Need](https://arxiv.org/abs/1906.06669) | model size and token supply must be optimized jointly | its one-pass argument replaces repeated data with more fresh data; our corpus is fixed |
| [Best Sequence Length for BabyLM](https://arxiv.org/abs/2510.19493) | context length can change optimization and inductive bias |125M models and100M words, downstream tasks and changed batch/LR; not our paired BPB objective |
| [Pre-training under infinite compute](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79682ef54677a2cb3dfb8383b372037f-Abstract-Conference.html) | finite-data regularization and averaging may have more headroom than one pass | compute-rich repetition, ensembles and distillation do not fit a30-minute budget without an explicit costed design |
| [modded-nanogpt PR344](https://github.com/KellerJordan/modded-nanogpt/pull/344) | amortize FP8 weight conversion/layouts once per update; consider separate QK/V dimensions | its ~5.15% measured total speed gain is on H100; Trn2's128-wide tiles can erase QK96 savings, and our first cast-inclusive FP8 GEMM is slower than native BF16 |

The report's suggested BatchNorm update is irrelevant to this RMSNorm model;
its reshuffling proposal conflicts with the immutable-pipeline constraint.
No architectural claim about Trainium leaderboard teams was verified.

## 6. Keep exotic ideas behind cheaper tests

SSM/RWKV/Mamba, SLAY/YAT and broad depth/width sweeps already consumed considerable
effort. Revisit only with a newly demonstrated implementation or causal explanation
of the earlier failure. Multi-token prediction and Engram-like gates also have
local notes/implementations; they are not untouched discoveries. Parameter-golf
leaderboards can inspire training ideas, but techniques relying on test-time
adaptation, extra data or a different tokenizer are not compatible here.

A later moonshot could combine a smaller dense trunk with a better structured
learned memory. First measure table collision/update statistics and marginal gain
per byte or per millisecond. More table parameters alone do not imply more usable
capacity. Another plausible direction is a fused linear-plus-loss backward, but
the existing fused CE means its upside must come from head/logit traffic; it needs
both dX and dW, not just an attractive forward benchmark.

## 7. Research quality and stopping rules

### Fixed reserved-output mask and a matching objective

The unchanged scorer masks all targets with zero token-byte count. The fixed,
SHA-verified tokenizer has exactly four such IDs:8188–8191. The trainer still
includes these targets in ordinary CE. A fixed output mask derived only from
that metadata provides a bounded diagnostic before changing training.

For a positive-byte target, removing zero-byte-class mass q and renormalizing
changes its NLL by ln(1-q), which is nonpositive. This predicts the direction
of a frozen-checkpoint diagnostic, not its magnitude or the quality of a newly
trained model. Keep masked logits finite: the trusted scorer computes all CE
values before multiplying by its target mask, so an infinite loss would risk
0*infinity producing NaN.

`research/scratch/zero_byte_mask_eval.py` uses a fixed-10000logit sentinel after
softcap and two unchanged scorer calls. It neither fits a scalar to validation
nor selects forbidden tokens from validation observations. Its CPU gate on the
preserved dense checkpoint passes finite, exact future/other-row/repetition,
B1-versus-B2 and CPU-reference checks. Synthetic positive-target NLL never
increases; excluded-target CE remains finite and becomes zero after masking.
Native checks and a paired BPB measurement are pending. No training/submission
change is promoted. A promising training arm would both mask these output
classes and omit their targets from the training loss, with an explicitly
defined reduction and checked NKI backward.

The existing [rules/metric audit](fresh-start-rules-research-2026-09-18.md)
already proposed zero-byte target masking. The fixed output-normalizer probe
is an additional way to measure the current wasted probability mass without
changing the data loader, tokenizer or trusted scoring implementation.

### Next systems hypotheses after the current quality checks

These proposals are conditional experiments, not measured wins:

| Hypothesis | Smallest useful test | Advancement/stop criterion |
|---|---|---|
| The new dense graph benefits from fewer, larger microbatches | B8/GA2 versus B4/GA4 at fixed131,072 global targets andT2048, then B16/GA1 only if justified | full-step timing and updated-model gates; real-loader geometry changes packing/order, so quality needs a full run |
| The first FP8 negative is partly a layout/shape problem | completed cached-weight real-MLP-shape probe; all numerical checks pass | still loses after preparation, especially the output projection; no training integration |
| Four LNC2 ranks are no longer the best full-chip mapping after removing hashed gathers | deferred: the earlier LNC1 audit records initialization/synchronization hangs even without model compilation on this native stack | no training retry merely because the architecture changed; first require evidence of a working runtime mapping and compatible custom kernels, without SDK/driver changes |
| NKI convolution changes a compiled buffer/graph assumption in the FP32 path | preserve the failing uninstrumented graph, then a minimal eager/compiled pair and owned NEFFs; instrument only one boundary at a time | finite hooked/eager execution is evidence to localize, not a production workaround; never promote a fast NaN graph |

For an FP8 MLP island, RMS-normalized inputs make a causal static or per-token
scale plausible; a scale derived from the whole future-containing sequence is
not acceptable. Keep residuals/statistics/optimizer in FP32 and initially retain
BF16 backward. Precisely specify the intended surrogate gradient before an
arithmetic gate: quantized forward plus higher-precision backward is a numerical
algorithm change, not bit-exact operator replacement. Cached row/transpose weight
layouts must be refreshed once per update and paid for. The first prototype's
M8192,K768,N512 negative remains valid for that implementation and shape.

Larger packing bins can also change which long documents fit. The old corpus
audit found24.7M first-pass tokens still buffered at2,800 updates. Thus an apparent
microbatch quality effect may combine execution efficiency, reduction rounding,
and changed document presentation. Reuse the existing immutable loader and report
actual geometry/exposure rather than calling every such trial an exact arithmetic
equivalence test.

The older sequence-length result also needs a precise scope. The900s pair in
`trn2-experiment-results.md:2663` compared T1024/pack8 with T2048/pack4, matching
loader length8192 and global batch, and found1.096998 versus1.097033BPB when both
were scored at1024. The shorter context was1.6% slower in that implementation;
at2048 evaluation it suffered the expected untrained-context extrapolation loss.
This supports retaining2048 for that implementation. A single near-tie does not
bound every curriculum's potential gain to0.000035. On the changed dense graph,
an optional phase gate can ask whether halving attention length now buys useful
time while keeping the same8192-token loader rows. Advance only with a material
whole-step saving and then a full-budget common-context quality comparison;
do not extrapolate a1024-only model's score to longer contexts. This is behind
the current B8/T2048 gate and current-graph profile, not a new broad sweep.

The LNC1 exclusion above is stronger than an unexplored parallelism proposal:
the prior audit at `trn2-experiment-results.md:6465` records hangs with one worker,
without compilation, with explicit`--lnc=1`, and even without distributed init.
It names the same compiler/native-framework/runtime versions now installed
(driver patch versions differ in the records). Removing hashed tables does not
address those failures. Do not spend either productive instance on another
unbounded runtime-hang investigation or alter its driver configuration.

Three asynchronous external research reports were collected under
`fresh-start-literature-2026-09-18/`. They are discovery aids, not accepted evidence.
The independent source/code checks already rejected several recommendations:

- XLA `mark_step` and old XLA flags are inapplicable to the native backend.
- Inference-only kernels do not establish a training forward/backward win.
- Muon's deliberately short polynomial is not an exact-SVD orthogonalizer; an
  arbitrary orthogonality-error threshold is not a universal correctness gate.
- A few repeated epochs are not inherently useless; conclusions must match the
  model, training horizon and optimizer.
- Trainium3 format support is not proof of Trainium2 support.

For each trial, record a hypothesis, one primary comparison, exact source and
compiler flags, and a predeclared advancement gate. A failure to compile or pass
causality closes that implementation, not all implementations of its abstract idea.
A 50-step smoke establishes executability and throughput, not final BPB ranking.
Near-tied scores need a second seed and a larger fixed evaluation before selection.
Avoid searching seeds as a substitute for a reproducible improvement.

Keep one host available for bounded mechanism/kernel tests while the other runs
the best full-budget candidate. When mechanisms are settled, use paired full runs.
Never overlap device work on the same chip. All launches use clean, committed
source, preserve live logs, and retrieve checkpoints before cache/disk maintenance.
The conclusion of this work will be supported by completed measurements, including
negative results, rather than the number of proposed ideas.
