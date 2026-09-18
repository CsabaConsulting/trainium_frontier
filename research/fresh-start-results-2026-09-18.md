# Fresh-start measured results — 2026-09-18

This ledger separates full training, standalone evaluation, numerical tests and
microbenchmarks. A successful process is not automatically a promoted model.
Public diagnostics do not establish a private competition score.

**Larger-sample result, 11:57 UTC:** the optimized dense6×1024 checkpoint,
without hashed n-gram tables, scores **1.0020666867349943 BPB** at context1024
on20,971,520 public tokens. The best similarly evaluated width896 table model
scores1.004452861885313. The dense result improves by0.002386175 under the
same evaluation geometry; it does not establish robust sub-1.0 quality or
independent-seed repeatability. Its earlier four-rank2.1M-token diagnostic was
0.996689. Prefer the larger fixed-sample result for current model selection.

## First completed training wave

Both runs use depth 6, four learned n-gram tables of 1,048,576×128 elements,
layer sites 1/5, compiled Muon, frozen Q/K shift, seed 42, four ranks, LNC2,
pack factor 4, global batch 131,072 tokens, and compiler flags
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
Source commit: `ca635670f67b067201fb9b6167eccf414b70bacd`.
The newly introduced disabled attention wrapper is present in this source.

| Run | Width | Steps | Training seconds | Charged estimate | Public BPB @2048 | Public BPB @1024 |
|---|---:|---:|---:|---:|---:|---:|
| `fs-w896-bf16-1800-r1` |896|2,026|1,784.3|1,785.4|0.992245|1.005110|
| `fs-w640-bf16-exposure1395-r1` |640|1,966|1,379.6|1,380.7|1.009678|1.024660|

Evaluation uses 2,097,152 public tokens per context. Both causal checks passed
with maximum absolute difference 0; both runs report `over_budget=False`.
The width 896 run has 631.9M total parameters, of which 536.9M are n-gram tables.
Its startup allowance estimate was 166.8s, below the 900s allowance.

The width 640 run is a shortened exposure diagnostic. It finished about 140 steps
short of a historical 2,100-step control; graph partitioning also differs across
those source revisions. It does not isolate a BF16 numerical penalty at matched
data exposure. Do not compare it to historical records as a clean precision A/B.

Raw logs: [width896](run-logs/fs-w896-bf16-1800-r1.log),
[width640](run-logs/fs-w640-bf16-exposure1395-r1.log). Launch manifests alongside
the logs contain complete commands, source hashes and environment settings.

## Trusted standalone checkpoint evaluation

The saved width 896 model loaded successfully through unchanged `prepare.py`:

```
NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES=0 NEURON_CC_FLAGS='--optlevel=1'
python prepare.py eval-public --train-py train.py \
  --checkpoint out/fs-w896-bf16-1800-r1/final.pt \
  --seq-len 2048 --eval-tokens 2097152
```

Result: **0.9905338826086015 BPB**, 2,097,152 tokens, 8,418,678 bytes,
5,780,144.5 total nats, 68.851 evaluation seconds, no timeout.
See [standalone log](run-logs/fs-eval-w896-r1-s2048-o1.log).
This evaluation omits training's global BF16 matmult autocast. It is a separate
evaluation configuration, not additional training progress or an independent seed
replication. **The 0.001711 difference is not an isolated precision effect:**
in-process evaluation uses four ranks, whereas this standalone command uses one.
`prepare.py` assigns row groups by rank and caps tokens per rank, so a partial
evaluation samples different document prefixes when world size changes. Precision,
sample membership and packing boundaries can all contribute. Future precision
comparisons must hold rank count, batch size, token limit and context constant.

A **20,971,520-token** standalone evaluation at context **1024** has completed:
**1.009889394576897 BPB**, 83,462,186 bytes, 58,423,696 total nats,
852.391 evaluation seconds, no timeout. It uses the same O1 inference flags.
See [larger evaluation log](run-logs/fs-eval-w896-r1-s1024-20m-o1.log).
This is closer to the formal rules' approximate validation size, but remains
public data. The shorter-context result is above 1.0 and is not interchangeable
with the context 2048 diagnostic. Both standalone commands use world size 1 and
batch size 1. Both in-process first-wave evaluations use world size 4, batch size
1 per rank. Neither is claimed to reproduce the private validation data.
The checkpoint has been copied locally and its SHA256 verified against the remote
file; see [preservation record](fresh-start-preservation-2026-09-18.md).

The newer `fs-w4-w896-nki-rms-fp32-r1` checkpoint was evaluated through the same
unchanged `prepare.py`, world size1, batch size1, context1024, O1 inference flags
and 20,971,520-token cap. Result: **1.004452861885313 BPB**, **83,462,186 bytes**,
**58,109,184 total nats**, 839.739 evaluation seconds, no timeout. The byte count
matches the first-wave evaluation exactly. This improves BPB by 0.005436533;
training changed direct SDPA, table-optimizer arithmetic and completed updates,
so this is not an isolated optimizer comparison. See
[larger newer-checkpoint evaluation](run-logs/fs-w6-eval-w896-nki-rms-s1024-20m-o1-r1.log).

The optimized dense `fs-w12-dense1024-nki-norm-1800-r1` checkpoint now completes
the identical standalone configuration: **1.0020666867349943 BPB**,
**20,971,520 tokens**, **83,462,186 bytes**, **57,971,140 total nats**,
709.243 evaluation seconds, no timeout. Source`c61bfca`; unchanged scorer,
world1/B1/context1024/O1 without global autocast. Its checkpoint SHA256 is
`d05929eddd9a0dc313ebb750b01008f6c3f83386f371114532291a42c5e48755`.
Both model families were freshly trained within the same1,800s budget, seed42;
architecture, update count and precision differ, so this is a budgeted candidate
comparison, not an isolated n-gram ablation. The dense checkpoint still has
ordinary unigram value embeddings and trained causal local convolutions.
See [dense larger evaluation](run-logs/fs-w12-eval-dense1024-nki-norm-s1024-20m-o1-r1.log)
and its structured summary. No private leaderboard result is inferred.

## Direct compiled SDPA: completed 30-minute width896 control

`fs-w3-w896-direct-control-r1`, source `00f82d8`, completed **2,188 steps**
and 286,785,536 target-token presentations. The four-rank 2,097,152-token
public evaluations give **0.987568 BPB at context 2048** and **0.999880 at
context 1024**. Causality passed with maximum difference 0. Training took
1,784.5s, charged estimate 1,785.6s, startup allowance 463.4s, total 2,263.7s;
`over_budget=False`. Steady steps take approximately 0.81–0.82s.

Width, tables, seed, batch geometry and BF16 matmult flags match the first
width896 wave. This run uses the direct compiled attention route and the legacy
BF16 table optimizer. It reaches 162 additional updates and improves the two
in-process context scores by 0.004677 / 0.005230. This is a practical budgeted
improvement, **not a step-matched numerical comparison or an independent seed**.
The 1024 result is only barely below 1.0 on this partial public sample; a larger
standalone evaluation is still required. No private leaderboard claim follows.

The checkpoint is copied locally with an independently matched SHA256. See
[raw training/evaluation log](run-logs/fs-w3-w896-direct-control-r1.log) and the
[preservation record](fresh-start-preservation-2026-09-18.md).

### Earlier width640 smoke

`fs-w2-sdpa-direct-bf16-smoke-r1`, source `ad3060e`, trains width 640 for 60 steps
with `--compile-sdpa-direct`. Steps after cold compilation took about 0.66–0.67s,
compared with about 0.70s in the first width 640 BF16 run. First and second steps
took 214.16s and 171.60s. The causality check passed with difference 0.

This is a roughly 5–6% scheduling/fusion opportunity at full four-rank geometry.
It is not evidence of switching NKI attention on: installed native eager and
compiled SDPA paths can both call NKI. Its 1.641925 BPB is from an intentionally
uncooled 60-step smoke and must not enter model rankings.

## N-gram RMSProp precision and fusion

Native hardware reproduces the BF16 second-moment problem: after 1,000 zero-gradient
updates with beta2=.999, a moment initialized to 1 stays 1 in the legacy BF16 path;
the FP32 path reaches 0.36769542. This is an arithmetic observation, not proof that
restoring decay improves generalization. The old behavior may regularize training.

The fused NKI prototype keeps BF16 parameters/gradients, promotes gradient products
and updates in SBUF, stores FP32 moments, and rounds only the final parameter
write. This changes state precision **and** update arithmetic. At two tested
shapes, four gradient patterns and one/two program variants, all 24 cases passed
independent CPU-reference checks. Moment errors were at most 9.1e-13 in those
cases. Rare BF16 output differences reached one ULP; no bit-exact claim is made.

One full 134,217,728-element table, LNC2, interleaved warmed repetitions:

| Implementation | Median ms/table |
|---|---:|
| Legacy native BF16 foreach |11.2842|
| Native Torch FP32 arithmetic/state |23.3837|
| NKI FP32, one program, explicit copy-back |17.8291|
| **NKI FP32, two programs, explicit copy-back** |**10.2830**|
| NKI FP32, one program, mutable inputs |24.9807|
| NKI FP32, two programs, mutable inputs |17.5685|

The fastest version includes copying updated parameters/moments back. Native
mutable-input handling was slower, so the integrated implementation chooses
out-of-place outputs. Measurements exclude full-model synchronization/contention
and do not establish the cost of updating all four tables on all four ranks.
Logs: `fs-w2-rms-nki-probe-r{1,2,3}.log`; r1 is a preserved compilation failure
fixed by exposing NKI language symbols in module globals.

Source `00f82d8` adds opt-in `--ngram-rms-fp32 --nki-ngram-rms`; both default off.
Its CPU regression preserves every tested default result bit exactly. The actual
native optimizer integration gate passed 20 updates at beta2=.999 and .9, using
two shapes, changing learning rates and sparse/zero/None/outlier gradients.
The independent reference uses FP64 moment recurrence, rounded to FP32 state.
Maximum absolute moment differences were 5.30e-5 / 9.16e-5 on the outlier cases
(within the predeclared relative tolerance); maximum one-step BF16 weight
difference was 0.000244140625. Original gradients and skipped parameters remained
unchanged. This broader test has different scales from the small-gradient probe
above. See [actual optimizer check](run-logs/fs-w3-nki-rms-integration-r1.log).

The **full-budget** counterpart `fs-w4-w896-nki-rms-fp32-r1` has completed,
source `a62df27`, with the same training-source SHA256 as its direct-SDPA control.
It reached **2,204 steps / 288,882,688 target positions**, training 1,784.8s,
charged estimate 1,785.9s, startup allowance 475.2s, total 2,267.2s, and
`over_budget=False`. Causality passed exactly. Four-rank public evaluation over
2,097,152 tokens gives **0.986947 BPB @2048 / 0.999735 @1024**.

Relative to the legacy BF16-state control, that is 16 more steps and BPB changes
of -0.000621 / -0.000145. State/update arithmetic and token exposure both change;
this is not step-matched, and the small advantage does not establish a reliable
quality improvement. The useful confirmed result is that faithful FP32 optimizer
state is feasible without losing the full-model throughput. Its checkpoint is
locally preserved with matching independent SHA256. See
[completed run](run-logs/fs-w4-w896-nki-rms-fp32-r1.log).

## Exact corpus and exposure audit

The diagnostic reads only the five allowed training shards and the fixed tokenizer.
It does not modify `prepare.py`, loader order or training examples. Its length
simulator is checked against the actual loader body on synthetic four-rank streams,
with token-conservation assertions; the production audit tokenized all 424,960
documents once and replayed the exact packing decisions per rank.

Raw corpus: **315,522,229 text tokens**, 315,947,189 encoded tokens including BOS,
and 1,262,079,205 UTF8 bytes. The prior roughly 288M number describes effective
first-pass exposure more closely than raw corpus size.

| Steps | All-pass target positions | First-pass retained text tokens | First-pass pending encoded tokens | First-pass not yet loaded encoded tokens |
|---|---:|---:|---:|---:|
|1,966|257,687,552|257,341,502|22,572,095|33,778,968|
|2,026|265,551,872|265,195,300|23,407,608|25,013,131|
|2,100|275,251,200|274,881,512|24,250,302|14,397,036|
|2,135|279,838,720|279,462,726|24,675,048|9,349,750|
|2,195|287,703,040|286,982,910|25,443,815|993,413|
|2,800|367,001,600|288,466,566|24,681,432|0|

At 2,800 steps, 1,570 first-pass documents remain buffered; every one is longer
than the 8,193-token loader bin. Another 2,375,801 first-pass text tokens were
dropped as truncated suffixes. Second-pass retained documents first appear at
loader rows 8,740–8,848 across ranks, roughly optimizer steps 2,185–2,212. A printed
refill epoch is therefore not a coverage certificate. More steps can mix repeated
shorter documents with still-unconsumed long ones.

At 2,026 steps, 356,572 targets are BOS, about 0.1343% of all targets. This limits the
plausible gain from eliminating a zero-byte-target training/scoring mismatch; it
does not motivate byte-weighting ordinary tokens.

Files: [audit script](scratch/corpus_exposure_audit.py),
[method and invariants](scratch/corpus_exposure_audit.md),
[complete counts](run-logs/fs-corpus-exposure-20260918.json),
[audit log](run-logs/fs-corpus-exposure-20260918.log).

## Dense transformer without hashed n-grams: measured bottleneck

`fs-w3-dense1024-phase-r1`, source `00f82d8`, uses depth 6, width 1024,
117,459,724 parameters, direct compiled SDPA, compiled Muon and the same BF16
matmult flags. Four ranks, LNC2, four 8192-token microbatches per rank, the real
allowed loader and all training control/gradient collectives are included.
Unigram value embeddings and trained local convolutions remain enabled.

After three warmups, eight unprofiled steps give these medians:

| Phase | Milliseconds |
|---|---:|
| Complete step |1,417.096|
| Four forward/backward microbatches |1,331.151|
| Optimizer |55.292|
| Gradient all-reduce |18.362|
| Gradient clipping |6.750|
| Weight-cache refresh |2.619|
| Control collective |2.214|

The model microbatches account for about 94% of step time. Even eliminating the
entire optimizer would save only about 3.9% at this geometry. This is a concrete
reason to investigate the model kernels rather than prioritize optimizer work
for the dense track. This short timing probe does not measure trained quality.

The subsequent native runtime trace, `fs-w4-dense1024-native-trace-r1`, identifies
the forward and backward graphs by correlating driver events' `External id`
with CPU `CompiledFunction` / `CompiledFunctionBackward` events. Across four ranks,
median execution spans are **222.76–223.82 ms forward** and **100.58–101.11 ms
backward**, four calls each per step. CPU dispatch is much shorter. These are
profiled spans for attribution; the unprofiled timings above establish throughput.
They do not yet identify the operations inside the slow forward graph.

Do not sum the CPU/runtime/driver views: they overlap, and large apparent
`aten::add.out` or copy durations include queued execution waits. The
[trace summary](run-logs/fs-w4-dense1024-native-trace-r1.summary.json) records NEFF
identities, correlations and SHA256 hashes of all 20 raw profile artifacts
(25,547,468 bytes). Reproduce with [the summarizer](scratch/summarize_native_trace.py).
The raw traces are preserved locally under
`out/fresh-start-profiles/fs-w4-dense1024-native-trace-r1/`, outside Git.

Instruction capture `fs-w4-dense1024-device-micro-r1` failed during warmup with
NRT status 1204, software notification queue overflow. Optional
`NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` was enabled. The failure is preserved, not
a training result; r2 retries the same source without that optional setting.
This retry can still expose device instructions, but missing DMA packet tables
will limit memory-bound analysis. It is not evidence of model numerical failure.

Instruction capture r3 has now ingested successfully with the **same-process
compiled NEFFs** plus two eager NEFFs named by the runtime trace's persistent
cache keys. The installed profiler alone omitted these pre-warmed binaries.
One rank captured instructions while all four ranks performed the same synchronized
microbatch. Metadata confirms two physical cores per logical core.

The forward and backward execution windows are **218.575 ms** and **97.757 ms**.
Independent unions of `ActiveTime`, clipped to each window and normalized per
physical core, give forward TensorEngine activity **17.18% / 17.21%** and backward
**56.63% / 56.57%**. Forward sync-engine activity is **67.19% / 65.66%**.
Engine times overlap and must not be summed. These are profiled attribution
measurements, not replacements for unprofiled step timings.

Explorer marks six roughly 13–15ms regions and one **84.35ms** region for many
small DMA transfers; the latter contains roughly 262K `DMA_DIRECT2D` instructions.
Several repeated regions have source attribution to `train.py:1338`, the ordinary
depthwise `F.conv1d`. The large 84ms region has mostly absent source labels, so
its operator identity is not inferred from transfer count. This justifies a
same-architecture static-shift convolution probe before an FP8 training rewrite.

Data-quality limits: optional DGE metadata is missing, DMA notifications were
dropped, and HLO/compiler metrics are absent. Consequently no exact HBM byte
bound or HLO MFU is claimed. NKI/source locations are partially available; the
nonempty layer field is often only `Unknown`. The
[device summary](run-logs/fs-w5-dense1024-device-micro-r3.summary.json) retains
warnings, physical-core metrics, execution identities and copied-artifact hashes.
Reproduce it with [the summarizer](scratch/summarize_dense_device_profile.py).

## FP8 feasibility: empirical arithmetic understood, prototype slower than BF16

The native Trainium2 legacy-E4M3 double-row primitive compiles and runs. The first
128×256×128 test failed the strict quantized-reference comparison: 98/16,384
outputs mismatched the selected tolerance, with maximum absolute difference
0.000244140625 among mismatches. An explicit conversion round-trip diagnostic now
separates device FP8 conversion from accumulation/layout. No speed or training
claim is accepted before resolving this check. Online scaling, backward matmuls,
gradient accuracy and end-to-end time remain additional gates.

`fs-w4-fp8-characterization-r2`, source `65a70a5`, completed three shapes:
128×256×128, 256×768×512 and 8192×768×512. The latter two use both LNC2 cores.
Device E4M3 conversions match the independent quantizer **exactly** for both
operands at every shape, including 6,291,456 activation elements in the largest
case. Nonzero subnormals survive conversion. A predeclared flush-to-zero matmul
reference is slightly worse, providing no support for that proposed explanation.

FP8 output RMS error against the unrounded quantized reference is
0.001644–0.001666 relative; the BF16 control is 0.001662–0.001676 against FP32.
These aggregate errors resemble the BF16 output-rounding floor, but that alone
did not explain the failed elementwise numerical check. Returning FP32 outputs
revealed an additional relative RMS error of about 0.000106 against an FP64 dot
product of the quantized operands; the BF16 control was around 7e-8. At K=256,
2,216/16,384 outputs exceeded the declared FP32 sequential-accumulation bound.

Saved operands distinguish the cause: rounding each adjacent **pair of FP8
products to 11 significand bits**, followed by FP32 accumulation, explains the
observed outputs. This is an **empirical arithmetic model, not an AWS hardware
specification**. Held-out seed-offset 53 passes all declared bounds at all three
shapes, including 4,194,304 outputs in the largest. Exact boundary patterns show
round-to-even behavior and preservation of `240*240 + 240*240 = 115200`, so the
pair adder has FP16-like precision but a wider exponent range. Treating it as a
literal FP16 intermediate would incorrectly predict overflow. The reference is
explicitly named; the original output tolerance was not enlarged.

`fs-w5-fp8-11bit-bench-r1`, source `4c6d258`, includes activation transpose and
BF16→FP8 casts, using both LNC2 cores where the shape permits. Interleaved warmed
median measurements for 8192×768×512 are:

| Implementation | ms |
|---|---:|
| Native Torch BF16 |0.09558|
| Prototype NKI BF16, layout included |0.33685|
| Prototype NKI FP8, layout/casts included |0.36113|
| Prototype NKI FP8, already-transposed input |0.21976|

Online amax/scaling calibration is **excluded**, so even these numbers are
optimistic for a full FP8 pipeline. The prototype is not a speed win and has not
entered training. Its tile structure repeatedly reloads/converts weights;
better reuse and output tiling are still valid engineering questions. Backward
accuracy, scaling policy and BPB remain untested. Against an unquantized FP32
matmul, FP8 relative RMS error is about 3.77% at the large shape.

Scaling must also preserve the evaluation contract: a current whole-sequence
activation amax can let future tokens affect earlier predictions. A training
implementation needs an explicit causal/stateless evaluation policy, such as
frozen scales or appropriate per-token scales, with perturbation checks.

## Next controlled full runs

The user explicitly requested keeping a serious route without n-grams. Both
width896 table-model runs above are complete. Trn22 is training the dense
width1024 candidate with the frozen-zero Q/K shift removed, under the same
30-minute limit (`fs-w6-dense1024-noqk-1800-r2`, source `c333adb`). Existing unigram value
embeddings and trained local convolutions are retained initially; removing all
three together is a distinct architectural experiment.

Old no-table measurements were paradoxically slower on their earlier compiler
graph. That is a reason to inspect actual phases, not a permanent ban on removing
tables. The FP32-state NKI optimizer is available for a later table-model arm;
it is not a commitment to that architecture. Record both 1024/2048 evaluation
contexts, then step-match any promising difference that changes throughput.

## Dense model: two compiler-graph ablations

Both completed probes use source `fe038e8`, the same unchanged `train.py`, four
ranks/LNC2, BF16 matmult flags, depth6/width1024, no hashed tables, the real
loader, B4×T2048 per rank, four accumulation microbatches and global 131,072
targets/update. Eight unprofiled steady steps follow three warmup steps.

| Convolution | Frozen-zero Q/K shift | Median step ms | Microbatches ms | Optimizer ms |
|---|---|---:|---:|---:|
| Native `F.conv1d`, earlier control |present|1,417.096|1,331.151|55.292|
| Static PyTorch shifts |present|3,083.145|2,998.892|55.162|
| Native `F.conv1d` |removed|986.485|900.022|55.629|

The static-shift version is **2.18× slower**, despite passing an independent
FP64 CPU forward/gradient check. Reject its current lowering. Removing the zero
Q/K shift saves **30.39% step time / 43.65% throughput**, with an approximately
unchanged optimizer/tail. It removes 6,144 frozen zero parameters, leaving
117,453,580 parameters and the same intended trainable function. Native compiled
rounding can still differ. The older width640 result favoring the zero-shift
graph does not transfer to this shape.

Logs and median records:
[static shift](run-logs/fs-w5-dense1024-shift-conv-phase-r1.log),
[zero-shift removal](run-logs/fs-w5-dense1024-noqkshift-phase-r1.log).
The ablation does not by itself identify the trace's 84ms region with certainty;
that attribution would require a new trace. Full-budget BPB remains pending.

Full-run r1 was stopped by the duplicate-log guard before spawning training;
its artifacts are retained. R2 uses a distinct outer `.driver.log`. During r2,
an 8 MiB/s capped read-only transfer recovered the older raw device profile
(10:03–10:14 UTC); this is recorded background I/O, not part of an accepted
microbenchmark. All 17 raw artifacts and same-process binaries are locally
preserved and their hashes match the remote originals.

### Same-host dense control and native convolution gate

`fs-w7-dense1024-noqk-control-phase-r2`, source `7262c89`, repeats the no-QK
control on trn21 before the NKI candidate. Eight steady steps after three warmup
updates give **979.203ms median**, range 976.070–986.621ms; microbatches
894.632ms, optimizer54.519ms, tail84.433ms. The earlier 986.485ms result was on
trn22. Use the trn21 repeat for its NKI comparison instead of claiming a cross-host
speed difference.

`fs-w7-conv2-nki-native-gate-r5`, same source, passes independent CPU-FP64
forward/dX/dW references and exact future-token/cross-batch perturbations on the
native compiled device. Shapes include B1T128C128, B2T128C128, B2T256C256 and
B4T2048C1024, all FP32 residual inputs. At the largest shape, relative RMS errors
are 3.67e-8 (forward), 3.67e-8 (dX), and 5.50e-8 (dW). No FP32 rounding-bound
failures or nonfinite values occur. This is not yet a BF16-input kernel gate.

The prior four failures remain recorded. The working compiler fix zeros the
whole next-gradient SBUF tile before loading its first 127 rows at the sequence
boundary. A partition127-only memset failed even at the tiny shape; its generic
MemoryLocation error did not establish a large-tensor SBUF-capacity problem.
Native `wrap_nki` integration, rather than a raw grid-specialized kernel inside
Dynamo, also proved necessary on this installed stack. Full-model NKI timing is
completed as `fs-w7-dense1024-noqk-nki-conv-phase-r1`: **713.340ms median**,
range709.777–719.308ms, microbatches632.614ms, optimizer52.580ms, tail81.110ms.
The raw timing is27.15% lower than its same-host979.203ms control and includes
forward, both gradients, weight conversion and partial-gradient reduction.
**Promotion withdrawn:** this first scratch run lacked the updated-model finite
gate; later integrated FP32 runs produce NaNs. It cannot establish a usable
whole-model speedup. Independent component gates pass but do not supersede the
whole-model failure. The kernel remains opt-in, with separate BF16 integration
checks described below.

### Completed full dense control, without hashed tables

`fs-w6-dense1024-noqk-1800-r2`, source `c333adb`, completed **1,805 updates**
and **236,584,960 targets**. Public BPB is **1.012886 @2048 / 1.027777 @1024**,
four ranks, 2,097,152 tokens per context. Causality is exact. Training1784.9s,
charged estimate1785.3s, startup237.7s, total2041.6s, `over_budget=False`.
Steady steps were about0.99s. The checkpoint is locally preserved with matching
remote SHA256; its receipt is beside the raw log. This establishes the control
for the faster dense candidates, not a conclusion that every no-table model loses.

### Native precision probes and composition

All rows use dense6×1024, no hashed tables or QK shift, the same real loader,
four ranks/LNC2, B4T2048/rank, GA4, BF16 matmult compiler flags and seed42.
Eight unprofiled steady steps follow three warmup updates. Precision policies
are intentional arithmetic changes, not exact replacements of the baseline.

| Host | Convolution | Precision policy | Median step ms | Microbatches ms | Whole-model gate |
|---|---|---|---:|---:|---|
|trn21|native|baseline|979.203|894.632|control|
|trn21|NKI|baseline|713.340|632.614|ungated initially; later FP32 integration fails|
|trn21|native|BF16 at SDPA only|907.584|823.416|four ranks pass|
|trn21|NKI|BF16 at SDPA only|670.093|585.410|four ranks pass|
|trn22|native|baseline, earlier same-host control|986.485|900.022|full budget completed|
|trn22|native|BF16 norm outputs|785.120|701.778|four ranks pass|
|trn22|NKI scratch|BF16 norm outputs|588.743|502.273|four ranks pass|
|trn22|NKI integrated, corrected|BF16 norm outputs|587.999|504.238|four ranks pass|

The norm/convolution combination saves **40.32%** of step time against the earlier
same-host no-QK control. Every precision row passes checks on all four ranks after
real updates: finite parameters/available gradients/logits, nonzero attention
output branches, exact future-token, cross-batch and repeated-call independence.
These evaluation checks use B2T256 and are outside timing; native kernel arithmetic
was initially checked at B4T2048C1024 **for FP32 convolution inputs only**. In the
BF16 norm combination, the convolution receives `norm(x)` in BF16 even though the
residual stream is FP32. The later independent BF16 gate passes, as recorded
below. Full-training BPB is still pending.

The NKI-only full-budget run launched on trn21 at10:45 UTC as
`fs-w9-dense1024-nki-conv-1800-r1`, source`ccf22c0`, **failed before step0 was
logged**, with NaN loss reported by all four ranks. It has no quality result.
The failure is preserved and promotion is blocked pending localization of the
first bad value. The component arithmetic and scratch-model gates do not overrule
this integration failure.

`fs-w10-dense1024-integrated-nki-norm-phase-r1` at`18ab710` passes the four-rank
updated-model checks but measures about782ms, not589ms. Inspection found the
production convolution guard accepts only FP32 inputs, so BF16 norm makes it
fall back to the original convolution. Its attention branch weight norms exactly
match the norm-only probe. This is a dispatch gap, not a successful integrated
NKI measurement. The BF16 dtype must be independently gated before opening that
guard. The subsequent integrated FP32 full-model probe also fails, as recorded
below; the full trainer's optimizer ordering is not responsible.

### Follow-up: first-forward localization and corrected BF16 promotion

`fs-w10-dense1024-integrated-nki-fp32-phase-r1` completes its timing region at
713.998ms but fails the finite-parameter/gradient gate on all four ranks. Those
timings are invalid as evidence of usable training. The phase harness now rejects
a nonfinite loss immediately instead of merely timing its health collective.

`fs-w12-dense1024-nki-nan-diagnostic-r1` adds explicit stage synchronizations to
the real trainer. Every rank reports NaN on **micro0 forward**, before backward,
clipping, reduction, optimizer or cache refresh. Initially created CPU parameters
were not changed by an optimizer at that point. Thus learning-rate or optimizer
instability and loss-buffer corruption *after* the forward cannot explain this
failure. This diagnostic changes scheduling and makes no timing claim.

`fs-w11-integrated-conv2-bf16-zero-gate-r1` and
`fs-w12-integrated-conv2-fp32-zero-gate-r1` pass compiled component arithmetic and
future/cross-batch checks at four shapes, with random and zero weights. The BF16
forward and dX match rounded independent FP64 references exactly in these cases.
Its compiled master-weight gradients retain FP32-level accuracy, despite the
source's BF16-to-FP32 cast pair; do not describe the compiled gradient as a
bit-exact BF16-rounded reference. The original native BF16 convolution has larger
rounding differences. This is a numerical-policy change requiring quality checks.

Opening the production guard to the now-gated BF16 dtype yields
`fs-w12-dense1024-integrated-nki-norm-phase-r2`, source`c61bfca`: **587.999ms**
median, range584.964–591.660ms, microbatches504.238ms, tail84.152ms. All four
updated-model gates pass; branch weight-square sums exactly reproduce the scratch
norm/NKI probe. Final rank0 training loss is5.793033. The corrected integration
actually executes the NKI convolution, whose input is BF16 while residuals stay
FP32. This full-budget candidate launched11:06 UTC on trn22 as
`fs-w12-dense1024-nki-norm-1800-r1`. Its BPB is pending.

### Compiled-forward localization, wave14

All three probes below use the actual dense6×1024 model, seed42, the real first
training microbatch B4T2048, original FP32 normalized activations, and integrated
NKI convolution. Initial CPU parameters are asserted finite. They do not execute
an optimizer and make no timing or training-quality claim.

| Probe at source `e6f6139` | Result | Meaning |
|---|---|---|
| `fs-w14-dense1024-single-program-fwd-r2` | loss NaN on both identical forwards | the failure does not require a two-program kernel |
| `fs-w14-dense1024-forward-capture-r1` | finite loss and all retained module inputs/outputs | hooks change lifetimes and graph outputs; not a fix or proof the normal graph is correct |
| `fs-w14-dense1024-eager-fwd-r1` | finite loss9.010475159 on both identical forwards | the uninstrumented eager full model is finite; compiled integration is implicated |

The preceding r1 one-program wrapper failed in Dynamo while directly calling a
different autograd class's backward; no device arithmetic verdict came from it.
Inlining the same tensor body repairs that diagnostic, not the model NaNs.
At11:21 the bounded FP32 investigation was paused in favor of a useful BF16/NKI
depth8 full-chip phase gate. The original failure remains preserved and unresolved.

### Checkpoint configuration repair

`load_for_eval` previously rebuilt from live source defaults, ignoring saved
architecture/arithmetic toggles. That can fail strict loading when a CLI disables
tables/QK parameters, or silently change same-shaped behavior. It now restores
only the explicit `ARCH_TOGGLE_NAMES` allowlist before model construction, using
import-time defaults for legacy missing keys. Unknown keys cannot replace code
globals. `prepare.py` and its two-argument scoring hook are unchanged.

CPU gates deliberately reverse live table/QK/precision flags and change softcap
before loading; saved logits are reproduced exactly, with and without tables.
Legacy missing flags use import defaults. The BF16 norm policy also needed to
align V with Q's dtype after CPU FP32 value-embedding addition; the native tested
BF16 path already had matching dtypes. Failed test logs remain preserved: the
first table-count assertion assumed every layer carried VE, and the first norm
checkpoint check caught that CPU mixed-dtype SDPA issue. The first integrated
native kernel checker failed on constructing `SimpleNamespace` inside Dynamo;
the eager native check passes, and the compiled checker now calls the production
autograd operation directly. These were not kernel accuracy failures.


### Dense depth8 gate and full-budget promotion

`fs-w14-dense1024-depth8-nki-norm-phase-r1`, source`e6f6139`, uses151,012,368
parameters, no hashed tables, BF16 norm outputs and NKI convolution. The real
four-rank B4T2048/GA4 phase measures **743.742ms** median, range740.304–748.787ms;
microbatches638.517ms and optimizer67.912ms. All four updated-model gates pass
with finite parameters/gradients/logits and exact future/batch/repetition checks.
Rank0 loss after the short probe is7.295557; this is not a quality ranking.

The predeclared <=950ms/finite/causal gate is satisfied. The full30-minute run
launched on trn21 at11:27 as `fs-w14-dense1024-depth8-nki-norm-1800-r1`. Its
training source hash matches the active six-layer run; the depth and host differ.
Initial loss9.0118 is finite. The completed run reaches **2,365updates** and
**309,985,280target presentations**, **0.990702BPB@2048 /1.004083@1024**,
charged1785.3s, training1784.8s, excluded startup279.8s, total2090.5s.
Causality_max_abs is0 and over_budget isFalse. The same four-rank2.1M-token
diagnostic is0.008992/0.007394 worse than six layers. Keep six layers as the
reference; the larger capacity does not win this full-budget comparison.
The8-layer checkpoint is locally preserved (604,075,069bytes, matched SHA256).
This negative does not establish a general depth optimum or isolate the cause:
both capacity and update count change, and the standard depth setting also
adds an ordinary value-embedding table. No large standalone8-layer evaluation
is prioritized given the current gap.


### Optimized dense full-budget result: public small-sample sub-1.0 without hashed tables

`fs-w12-dense1024-nki-norm-1800-r1`, source`c61bfca`, completes **2,977updates**
and **390,201,344target presentations**, versus1,805updates/236,584,960targets for
the earlier dense no-QK control. Both are dense6×1024 without hashed tables; the
new run uses integrated NKI convolution and BF16 normalized branch inputs while
retaining FP32 residuals. This is a combined precision/implementation/exposure
comparison, not evidence of identical numerical updates.

| Full-budget dense model | Updates | BPB@2048 | BPB@1024 | Estimated charged seconds |
|---|---:|---:|---:|---:|
| No-QK control, native convolution/original norm outputs |1,805|1.012886|1.027777|1,785.3|
| NKI convolution + BF16 norm outputs |2,977|**0.981710**|**0.996689**|1,785.3|

Both rows use four ranks and2,097,152public evaluation tokens at each context.
The new run has64.93% more updates and improves the context1024 diagnostic by
0.031088BPB. In-run causality_max_abs is0; over_budget isFalse. Training1784.9s,
excluded startup estimate289.1s, model-reported total2091.0s. The checkpoint
contains117,453,580parameters and is locally preserved with matching SHA256
`d05929eddd9a0dc313ebb750b01008f6c3f83386f371114532291a42c5e48755`.

The standalone20,971,520-token/context1024/B1/O1 evaluation started11:43 UTC as
`fs-w12-eval-dense1024-nki-norm-s1024-20m-o1-r1`. It completes at **1.002066687BPB**;
see the standalone section above for exact counts and comparison. The small
sub-1.0 diagnostic is not a private leaderboard score or a repeatability claim.

### Conservative canonical-token constraint: native paired quality test

`fs-w15-dense1024-canonical-paired-eval-r1`, source`4f0e77a`, evaluates the frozen
optimized dense checkpoint with a freshly built, tokenizer-only `ascii-word`
constraint. Native finite/base-finite checks pass; CPU mask-reference, future,
other-row, repeated-call and B1-versus-B2 differences are all exactly0.
The table contains1,166,945forbidden pairs and builds in20.425s on trn22; its SHA
matches the independently built/falsified CPU artifact. No validation data
enters construction, no model parameters are updated, and `prepare.py` is unchanged.

| Same frozen model and world1/B1/context1024/2,097,152-token prefix | BPB | Total nats | Evaluation seconds |
|---|---:|---:|---:|
| Original logits |0.996184154485097|5,813,116.0|70.592|
| Conservative impossible-pair mask |0.9961083239858673|5,812,673.5|72.022|

Both calls score8,418,678bytes. The paired improvement is **0.0000758304992BPB**,
or442.5total nats on this prefix. Deprioritize this conservative mask: its measured
gain is small relative to the campaign's remaining gap, and a consistent training
implementation would need its own cost/gradient/full-budget gates. This is not a
claim that a broader constraint cannot help, or a projection onto20.97M tokens.
The one fixed-order timing pair is not an interleaved warmed speed benchmark.
Raw log and structured result are retained beside the source manifest.

### Cached-weight FP8 at actual MLP shapes: numerical pass, no promotion

`fs-w16-fp8-tiled-real-mlp-r1`, source`8704139`, caches the complete weight tile
in SBUF once per GEMM call and reuses each quantized activation tile across
output tiles. It includes a BF16 kernel with the same tiling, native matmul and
F.linear controls, and separate operand-layout arms. Three interleaved timing
repetitions each contain12calls after warmup. All four shape gates pass finite
checks, quantized/unquantized error bounds and sampled pair-reference checks.
The roughly3.77% relative RMS difference from unquantized matmul is a real
FP8 quantization change; the empirical pair-rounding reference is not an AWS
hardware specification.

| Median milliseconds, M=8192 | K1024/N4096 | K4096/N1024 |
|---|---:|---:|
| Native BF16 matmul |0.556214|0.548434|
| Native BF16 F.linear, original NK weights |0.556833|0.588618|
| Weight layout refresh alone |0.083793|0.085504|
| Cached NKI BF16, activation layout included |0.705372|1.286571|
| Cached NKI FP8, activation layout and casts included |0.550498|1.001536|
| Cached NKI FP8, both layouts and casts included |0.633388|1.096418|
| FP8 pretransposed operands, lower bound |0.384483|0.550813|

Static CPU-calibrated scales are held fixed during timing; online amax and
backward are excluded. Weight caching here lasts one GEMM invocation, not an
optimizer step. Amortizing the separately measured weight layout over four
calls would estimate0.571/1.023ms, but this is not a measured training result.
Do not integrate this implementation: one projection only ties the native
path before weight-layout cost, and the other loses. Fused layout production
or a persistent cache would require a new implementation and new evidence.

### Dense microbatch packing with the successful NKI/BF16 path

`fs-w15-dense1024-pack8-nki-norm-phase-r1`, source`4f0e77a`, uses B8/T2048/GA2
and the same131,072global targets/update as the B4/T2048/GA4 reference. All four
updated-model finite/causal/repetition checks pass. Across eight warmed updates,
the median is **576.865ms**, range574.518–579.311ms, versus587.999ms for B4.
The roughly1.9% saving misses the predeclared570ms full-budget promotion gate;
the prepared B8 full run is not launched. One B16/GA1 phase is warranted by
the small positive trend and is active. Packing changes loader row geometry
and token order, so eventual full-budget quality would need measurement.

### Current dense profile preservation and throughput control

`fs-w16-dense1024-nki-norm-device-micro-r1`, source`8704139`, captures one
rank0 microbatch of the current six-layer NKI/BF16-normalization graph at
B4/T2048/GA4 on four ranks. Its unprofiled update median is **585.183ms** and
all four updated-model gates pass. This agrees with the587.999ms earlier gate;
it does not show the large throughput deterioration reported for some older
campaign runs. Profiled execution duration is not used as throughput evidence.
All63raw files, including the eight compiled graph binaries, are locally
SHA256-verified (1,198,602,679bytes). Ingestion/attribution is pending.
