# Fresh-start synthesis: the most credible routes below 1.0 BPB

This is a living research argument, with falsifiable experiments. It separates
results in existing logs, facts verified in source/numerical probes, and proposals
whose BPB benefit is unknown. The first two new training runs are still in flight
at this revision. A sub-1.0 result has **not** been established by this campaign.

The strongest opportunity is to recover useful capacity and optimization quality
from the existing transformer. There is already direct evidence that a wider
version learns below 1.0 when given too much time. Meanwhile the current code and
measurement history contain concrete changes that previous high-level conclusions
do not account for. A complete replacement with an SSM or all-FP8 stack would make
these mechanisms harder to isolate.

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
approximately 0.367695 in FP32. Native-device confirmation is a separate gate.

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

The prototype is deliberately separate from training. It uses runtime scalar
tensors for learning rate and bias correction to avoid recompilation every step.
Correctness must include zero moments, untouched rows, rare gradients, outliers,
changed scalar values, and multiple consecutive updates. Fusion with different
rounding points is not bit-exact equivalence to the old BF16 algorithm.

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

DeepSeek-V3 demonstrates carefully scaled low-precision training, but its large
model and GPU-specific engineering do not transfer as a recipe. It is a source
of hypotheses about scaling granularity and sensitive paths, not evidence that
compiler-wide FP8 should work here. [Technical report](https://arxiv.org/html/2412.19437v1).

## 5. Reopen data and optimizer conclusions carefully

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
not a tested result from that paper. Predeclare one short tail window and count
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
