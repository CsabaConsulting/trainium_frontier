# Fresh-start measured results — 2026-09-18

This ledger separates full training, standalone evaluation, numerical tests and
microbenchmarks. A successful process is not automatically a promoted model.
Public diagnostics do not establish a private competition score.

**Larger-sample result, 13:35 UTC:** the optimized dense6×1024 checkpoint,
without hashed n-gram tables and clipping after gradient averaging, scores
**1.0015512989000874 BPB** at context1024 on20,971,520 public tokens. The earlier
clipping-before dense reference scores1.0020666867349943; the difference is
-0.0005153878349069, with28 more budgeted updates and the same training seed42.
This remains above1.0 and does not establish an independent-seed algorithmic
advantage. The best similarly evaluated width896 table model scores
1.004452861885313. Prefer the larger fixed-sample results over the small-prefix
diagnostics. One predeclared seed43 repeat is running, not a seed sweep.

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
the prepared B8 full run is not launched. One B16/GA1 phase was warranted by
the small positive trend. `fs-w15-dense1024-pack16-nki-norm-phase-r1` now
completes at **614.578ms** median, range600.769–627.696ms, with all four
updated-model gates passing. The rank0 microbatch total remains486–492ms,
but the measured synchronization interval grows to43–72ms. This localizes the
observed regression to the update tail; it does not isolate a communication
or rank-imbalance cause. Do not run full B16 training. Keep B4 and close this
packing sweep. Packing changes loader row geometry and token order, so even
a phase speedup would not establish a quality win.

### Current dense profile preservation and throughput control

`fs-w16-dense1024-nki-norm-device-micro-r1`, source`8704139`, captures one
rank0 microbatch of the current six-layer NKI/BF16-normalization graph at
B4/T2048/GA4 on four ranks. Its unprofiled update median is **585.183ms** and
all four updated-model gates pass. This agrees with the587.999ms earlier gate;
it does not show the large throughput deterioration reported for some older
campaign runs. Profiled execution duration is not used as throughput evidence.
All63raw files, including the eight compiled graph binaries, are locally
SHA256-verified (1,198,602,679bytes). Seven additional warmed eager cast/scalar
NEFFs (82,442bytes) have been recovered with body hashes matching exactly the
seven UUIDs reported missing by Explorer. The main compiled graphs were already
present: the earlier missing-compiled-graph diagnosis was incorrect. The compiled
events' trace persistent keys differ from their exported keys, while their
logical cache keys match. Keep these identifiers separate. Full UUID validation
still belongs to ingestion.

Local Explorer2.32 ingestion was killed with signal9 during severe memory
pressure (31GiB used,149MiB available), before completing its Parquet output.
The raw capture and recovered binaries remain intact. Ingestion completes
successfully on idle trn22, with GOMAXPROCS4/GOMEMLIMIT40GiB, outside device timing. The
runtime trace independently attributes38.734ms to the forward graph and77.557ms
to backward; those are profiled attribution spans, not new throughput results.

The successfully ingested device windows are36.514939ms forward and74.308579ms
backward. Tensor Engine activity covers approximately51% of each window on each
physical core; engine overlaps must not be summed into wall time. The backward
graph's regular FP32 matrix work is only0.199166720 GFLOPs across the two physical
cores, versus approximately614 GFLOPs in the older FP32 attention capture.
The current attention precision change has removed that particular bottleneck.

Regular Tensor instructions attributed to input embedding `train.py:4043`
account for154.174226432 GFLOPs and the three unigram value embeddings at
`train.py:4060` for433.029906432 GFLOPs. General linear layers at2853 account
for3503.427321856 GFLOPs. These are hardware instruction/source attributions,
not exclusive runtime shares or proof of a particular speedup. The profile
contains dropped DMA notifications and lacks dynamic DMA metadata/HLO stats,
so it cannot support an exact HBM bound or HLO-based MFU claim. Layer labels
are empty/Unknown; NKI source paths are the usable attribution. See the
separate device, runtime and embedding-attribution JSON receipts.

### Shorter attention and narrower precision gates

`fs-w15-dense1024-context1024-pack8-nki-norm-phase-r1`, source`4f0e77a`, keeps
the8192-token loader row and131,072global targets/update while changing from
B4/T2048/GA4 toB8/T1024/GA4. All four updated-model gates pass. Its eight-update
median is **577.148ms**, range571.877–581.206ms; this misses the predeclared
558.6ms/5%-saving threshold. The small timing difference cannot justify assuming
shorter-context quality transfers. No full run is promoted.

`fs-w15-dense1024-nki-sdpa-only-phase-r1`, same source, retains FP32 normalized
branch outputs/residuals and casts only the SDPA boundary to BF16, using NKI
convolution. All four updated-model gates pass. Its median is **678.258ms**,
range673.746–682.743ms, consistent with the earlier670ms scratch arm. It is
about15% slower per update than broad BF16 normalized outputs. This is a valid
precision/throughput candidate for one full-budget quality comparison; it is
not an operator speedup or a measured BPB improvement.

### Clipping after gradient averaging: modest small-prefix improvement

`fs-w16-dense1024-nki-norm-clipafter-1800-r1`, source `8704139`, completes
3,005 updates / 393,871,360 target presentations in 1,784.9 estimated charged
seconds. It scores **0.981174 BPB @2048 / 0.996049 @1024**, with four ranks,
2,097,152 evaluation tokens and zero measured causality difference. The
six-layer dense NKI/BF16 reference with clipping before averaging scored
0.981710 / 0.996689 with 2,977 updates. The new differences are -0.000536 /
-0.000640, with 28 additional updates; this is not a step-matched comparison
or sufficient evidence of a reproducible algorithmic quality improvement.
The 469,834,849-byte checkpoint is locally hash-verified. Its fixed 20.97M-token
evaluation completed at **1.0015512989000874 BPB**:20,971,520tokens,83,462,186bytes,
57,941,324nats,694.709s,world1/B1/context1024/O1. Relative to the similarly
evaluated before-clipping reference, the difference is-0.000515387835. The
checkpoint is preserved and one independent seed43 repeat is running.

### Ordinary-embedding segmented scan: real-shape numerical failure

`fs-w17-embedding-segmented-native-r1`, source `f9f4196`, passes all five
small-shape device checks: uniform, Zipf-like, one repeated ID, zeros and
cancellation, at T256/C256/V256. At T8192/C1024/V8192, the first uniform-ID
case fails: maximum absolute difference0.09873, relative RMS0.507686, and
1,283,056 FP32 error-bound failures. Absent vocabulary rows remain exactly zero.
The run stops before timing the actual shape. **No kernel speedup or training
integration is promoted.** Stage and shape isolation is the next diagnostic;
the cause is not yet established. Small native-control results confirm that
production autocasting rounds gradient inputs to BF16, so a working FP32 scan
would also be an arithmetic change. Raw eager-NKI small-shape call timings do
not establish what a captured training graph would cost.


### Full-budget precision control: FP32 normalization loses

`fs-w15-dense1024-nki-sdpa-only-1800-r1`, source `4f0e77a`, finishes2,600updates/
340,787,200targets in1,785.1 charged seconds, at0.989128BPB@2048/1.004102@1024.
The four-rank2.1M-token diagnostics are0.007418/0.007413 worse than the broad
BF16-normalization reference, with377 fewer updates. Causality difference0,
in budget. Preserve the469,834,849-byte checkpoint, but do not promote a larger
evaluation of this weaker control. This is a combined precision/throughput
comparison, not a matched-update numerical study.

### Fixed zero-byte output classes: small paired gain

`fs-w17-dense1024-zero-byte-paired-eval-r1`, source `f9f4196`, evaluates the
frozen earlier dense baseline, SHA256 `d05929eddd9a0dc313ebb750b01008f6c3f83386f371114532291a42c5e48755`.
The constant mask removes tokenizer IDs8188–8191 after softcap with finite-10000
logits. It is derived from tokenizer byte lengths, not validation observations.
All finite/reference/future/other-row/repeated/batch-size gates pass, with exact
zero differences; no positive-byte target's loss increases. At identical
world1/B1/context1024,2,097,152tokens/8,418,678bytes, the unchanged scorer gives
0.996184154485097 raw versus0.9958271085073683 masked: **-0.0003570459777287 BPB**.
The two calls take145.0795s combined. This is a modest diagnostic, not a full
20.97M-token result or evidence that the new clipping checkpoint crosses1.0.

### Embedding stage isolation and compact scatter: correctness negatives

`fs-w18-embedding-segmented-stage-debug-r1`, source `e1f120f`, confirms that the
actual-shape load/transpose is exact but permutation-gather errors appear in
later channel blocks. The original two-program kernel fails in channels512–767;
the instrumented permutation stage fails in channels256–1023. One-program and
shorter-shape failures also occur; FP32 continuation flags do not fix them.
Instrumentation changes allocation/lifetimes. These observations motivate
reloading indices inside each block, but do not prove an API mutation contract
or compiler defect. No actual-shape speed is established.

`fs-w18-embedding-compact-native-r1`, same source, fails its first independent
FP32 scatter check atT256/C256/V256: maxabs0.0854112,relativeRMS0.628752,
16,384 bound violations, absent rows zero. It stops before compact native
matmul or timing. Clear/scatter ordering and indirect DMA remain unisolated;
no speedup or training integration is promoted. Wave19 will instead gather
compact rows into each output row once, with valid-row masking. CPU tests pass
all11cases, including actual-shape distributions and a provided-training row.
The current fixed4096 bucket has a correct8192 fallback. Captured NKI timing
and all host metadata/upload costs are required before any performance claim.


### Compact gather succeeds at the component gate

`fs-w19-embedding-compact-gather-native-r1`, source `9f455b6`, passes all44
numerical gates, including one/two-program FP32 output expansion, actual shapes,
repeated IDs, zeros, cancellation and the first provided-training row. The
compiled compact BF16 gradient matches the native control exactly on that row.
Median native backward is3.115153ms, compact prepared1.802317ms, complete
metadata creation/upload2.552759ms. Shared across four tables, the estimated
component cost is1.989927ms. The latter is an estimate, not an integrated timing;
input-ID D2H and graph interactions remain for the next four-rank phase gate.

A scratch autograd wrapper now shares metadata across ordinary WTE/VE tables,
preserves parameter identities/checkpoint names and falls back to native backward
when more than4096 IDs are present. CPU actual-shape checks pass changed inputs,
native overflow and return to compact mode, repeated IDs, zero gradients,
evaluation with a different shape and loading into an unmodified model. Native
compiled-autograd checks and the timed full model are the next gates. The phase
probe places D2H metadata acquisition and preparation inside every microstep.

The user explicitly prioritizes sustained kernel/root-cause work over seed
selection and sets a competitive target near0.9749. Preserve failing artifacts,
locate the first invalid stage, distinguish eager/composed execution and test
hypotheses. A failed implementation is not evidence that its whole approach is
impossible. The running seed43 repeat is a single variability observation; no
seed sweep will follow it.


### Shared compact embedding: full-model gain is modest

`fs-w21-embedding-compact-autograd-native-r1`, source `0db0db3`, passes all12
small/actual-shape native autograd cases. Every tested forward and gradient
matches its native control exactly, including overflow and return to compact
mode; checkpoint loading into the ordinary model passes.

`fs-w21-dense1024-compact-embedding-phase-r1`, same source, passes all four
updated-model finite/causal/batch/repetition gates. All176 microbatches across
four ranks use the4096 bucket, with observed maximum distinct-ID count2529.
Input-ID D2H, metadata construction/upload and output expansion are timed. The
8-update median is **581.098882ms**, range576.183356–583.387608ms,
versus587.999ms reference. This misses the predeclared570ms full-run gate. The
component improvement only partially transfers to the full graph; do not promote
a30-minute run yet. Improving the exact backward, metadata costs or bucket size
remains a concrete kernel question, not a closed architecture hypothesis.

The separate scan split-call test, source `83d5cae`, passes actual-shape eager
checks but fails compiled composition. Later-block wrong values can be matched
to different original rows. A dependent-index-read and scratch-lifetime probe
is prepared; the cause is not yet established. No unstable scan enters training.

### Automatic host cleanup validated; gather failure narrowed

The version2 guardian is deployed on both hosts at `cae3cae`. The trn21 smoke
test deliberately leaves a harmless child after its parent exits. The guardian
terminates exactly that marked child, leaves no descendants or device holders,
and records a healthy postflight. Local ownership/exclusive-receipt tests also
pass. The previous compact phase and seed43 repeat receive explicitly labeled
legacy postflights, without retroactively claiming automatic preflights.

`fs-w22-embedding-gather-lifetime-r1` finds unchanged dependent index reads
despite incorrect gather values at8192 tokens. Explicit reuse of scratch buffers
does not repair the failure. Smaller256/512/1024-token cases pass. The harness
stops at Dynamo's eighth specialization before all final configurations run;
resetting Dynamo between independent diagnostics addresses that harness limit,
not the incorrect gather. Automatic failure cleanup completes and host checks
pass. The captured initialization TPB reset request completes after seven seconds;
there is no unfinished reset or fatal/OOM message in the captured interval.

The next compact-backward candidate uses3072 rows instead of4096, retaining
native overflow fallback. The fixed workspace need only be a multiple of128;
the old power-of-two rounding unnecessarily expanded3072 back to4096. All16 CPU
autograd cases pass, including exact-bucket and one-ID-overflow boundaries,
changed inputs, ordinary checkpoint loading and evaluation. Native gates and
charged full-model timing remain required before promotion.

`fs-w23-embedding-compact-b3072-autograd-r1`, source `7b08884`, passes all16
compiled device cases. Tested outputs and gradients match native exactly,
including3072 distinct IDs and3073-ID fallback. Automatic postflight is healthy.
The full-chip phase is running; a nearby same-host native control is required
because the prior4096-row phase and baseline were measured on the other host.

`fs-w23-embedding-gather-lifetime-r2` completes all20 diagnostics. At8192 tokens,
gathered values remain incorrect in both eager and compiled modes and both
scratch-allocation variants, while every dependent index read is unchanged.
This narrows the reproducer but does not establish a particular compiler defect.

### Seed43 repeat preserved, without seed selection

`fs-w18-dense1024-nki-norm-clipafter-seed43-1800-r1`, source `e1f120f`, completes
2,959 updates/387,842,048 targets in1,785.0 charged seconds. The four-rank small
public checks give0.986019BPB@2048/1.000259@1024, versus the predeclared seed42
run's0.981174/0.996049. Differences are+0.004845/+0.004210, with46 fewer updates
on a different host. This is a full-budget repeatability observation, not an
isolated seed or hardware effect. No further seed search or selected-seed claim
follows. The469,834,849-byte checkpoint is independently hash-verified locally
and remotely: `2245d799b6f39dd5f8ef8adb2f5f77f263c0c1ae967b966c3426adfed02192b7`.
The legacy postflight is healthy. No larger public score was measured for this
repeat, so its small score must not be compared with20.97M-token results.

### BF16 MLP fusion enters component development

The next larger target fuses the ordinary up projection, BF16 output rounding,
ReLU-square and down projection. Normal FP32 weight layouts enter the kernel;
weight reads/casts and transposed access costs are included. Both weight matrices
remain in SBUF across512-token tiles. The kernel also saves BF16 ReLU activations
to HBM for backward; this cost cannot be omitted from a training comparison.
Backward initially uses native BF16 GEMMs with FP32 parameter gradients.

All18 small CPU output/gradient checks pass on random, changed-weight and zero
inputs. The NKI forward is not yet validated. Device gates will check both fused
GEMMs against FP64 references, compiled native gradients, and actual8192×1024
inputs with4096 hidden channels before any component speed claim. Full-model
timing and training remain separate later gates.

The first fused MLP device probe, `fs-w24-mlp-fused-bf16-component-r1` at
`1fa9165`, passes all16 eager/compiled GEMM stage checks and24 native comparisons.
At the actual shape, forward, saved ReLU, input gradient and up-weight gradient
are bit-exact to native; down-weight relative RMS difference is1.73e-7.
However, forward costs785.810ms versus1.383ms native, and forward/backward costs
785.853ms versus4.331ms. The small shape also regresses9.070ms versus0.123ms,
so large-shape cache pressure alone cannot explain the pathology. Do not promote.
Capture a small device trace first to locate transfer/layout overhead and spills.

A separate segmented embedding implementation replaces the suspect on-chip
gathers with indirect DMA: BF16 permutation, FP32 segmented scan, FP32 dense
group-end expansion. This matches the native input-rounding policy and charges
all intermediate HBM traffic. All11 CPU distribution/training-row gates pass;
native stage and composition checks are still required.

### Gather root-cause progress and nearby compact control

`fs-w26-embedding-gather-manual-address-r1`, source `da8c7b8`, replaces automatic
scratch placement with five disjoint explicit addresses. All four actual-shape
eager/compiled, low/high-source-address cases are exact: zero wrong values and
zero changed dependent index reads. The installed allocator passes byte offsets
to `bind_memloc`; units were verified before use. This implicates automatic
allocation/scheduling, not bad input indices or simply a high-address limit.
It does not yet validate the uninstrumented segmented-gradient composition;
the next kernel assigns every one of its scratch buffers explicitly and reruns
the original gates.

The same-host compact3072 phase gives582.432747ms versus the nearby native
control's588.844843ms: **1.088928% saving**. Both automatic postflights and all
updated-model checks pass. This still misses570ms, so no full training run is
promoted. Shrinking the workspace has not produced the hoped-for substantial
whole-model gain.

### MLP profile identifies tiny DMA packets

The small paired device trace is hash-preserved locally:11 files/25,932,677bytes.
Its derived Parquet contains524,288 two-byte DMA packets for the fused graph,
while the native graph uses coarser transfers. Recorded spill save/reload bytes
are zero. This points to transfer granularity rather than capacity spilling in
the small reproducer. Runtime descriptor construction is a plausible additional
source of host/driver delay, not yet a measured attribution. Dynamic DMA metadata
and compiler metrics are incomplete, so do not claim exact bandwidth or MFU.
The next MLP version must use contiguous transfers and explicit transposes,
then repeat correctness and unprofiled timing with layout conversion included.

Packet attribution isolates the tiny transfers to the two weight inputs:
262,400 packets each, including262,144 two-byte packets per matrix. The input
activation and two outputs use substantially fewer packets. The revised kernel
loads ordinary contiguous FP32 weight tiles, casts in SBUF and transposes via
TensorEngine before caching. The old direct-transpose/cast DMA remains an
explicit failed control. Native gates and complete timings are required again.

`fs-w25-embedding-segmented-dma-r1`, source `0aeeb54`, passes all11 CPU, eager,
per-position scan and compiled-composition cases. The real training-row dense
gradient differs from native by at most2.98e-8. However, prepared DMA/scan costs
18.920ms versus3.218ms native, and metadata-included cost is19.909ms. This is a
correct reference/workaround, not a training candidate. Its cast/layout costs
remain a target if the manual-scratch original kernel also misses the speed gate.

### Two failed implementations narrowed, neither silently promoted

`fs-w27-embedding-segmented-manual-sbuf-r1`, source `7af6de9`, still fails the
full actual-shape gradient despite the passing standalone gather. Next check:
the standalone passing arm reused the same scratch tensor objects outside its
loop, while the full kernel creates different objects at the same addresses
inside loops. Address overlap does not necessarily establish compiler dependency
identity. A composed diagnostic now returns raw/permuted/scanned intermediates
and compares allocating inside versus outside loops, with an uninstrumented
control retained. This is a hypothesis under test, not a confirmed compiler bug.

`fs-w28-mlp-fused-bf16-contiguous-weights-r1`, source `3ec4f93`, stops at compile
validation: gen3 transpose source/destination dtypes must match. It attempted
BF16-to-FP32 transpose. The correction transposes FP32-to-FP32 first, then casts
the cache value to BF16. There is no device timing for the failed version.
Both failure postflights are healthy and all evidence is retained.


### 14:35 UTC — wave29 narrows gather and MLP failures

All four completed wave29 jobs have healthy automatic pre/postflight receipts,
no leftover device holders, and no cleanup kills. Kernel reset messages are
paired initiation/completion during ordinary process attachment; no unfinished
reset or ECC error was observed. Their exact source is06215d3.

- Full manual scan: reusing tensor identities outside loops does **not** fix the
  result. Both eager/compiled and instrumented/uninstrumented variants reproduce
  1,283,061 final violations. Raw loads are exact; permutation gathering first
  fails for channels768–1023. The first three paired channel blocks are exact.
- Independent gather tiling:512 outputs per call is **exact**, including all
  dependent index reads, in eager and compiled actual-shape probes. Width1024
  produces57,344 wrong values,2048 produces190,976,8192 produces1,966,080. The
  source retains its full8192-row index range. This is a concrete boundary to
  test in the full computation, not validation of that computation.
- Revised contiguous-weight MLP passes all16 FP64 GEMM stage bounds and24 native
  forward/gradient comparisons. Actual shape8192×1024×4096 forward is554.088ms
  versus native1.36410ms; forward/backward555.364ms versus4.34585ms. This is an
  improvement on the flawed785ms kernel, but still a large regression. A second
  small device profile was captured to attribute remaining costs.

Next controlled variants preserve the failed controls: (1) constrain permutation
  gathers to512 outputs in the complete scan; (2) reorder mixed-dtype explicit
  scratch so every FP32/uint32 tile starts at a4*T-byte boundary, as in the
  passing manual-address reproducer; (3) replace transposed activation DMA in
  the MLP with contiguous128×128 loads/stores plus on-chip transpose. Address
  alignment and compiler dependency behavior remain hypotheses. All variants
  need original, uninstrumented correctness gates before timing/promotion.

The evaluation contract audit confirms public CLI defaults B1/T1024, with the
  exact trusted metric. The private runner configuration is not completely
  exposed here. Continue reporting context length, token count and public shard
  explicitly; do not equate the better T2048 score with the private leaderboard.


### 14:41 UTC — gather repaired; MLP transfer regression removed

Wave30 source3e08b9f passes healthy postflight for all three completed jobs.

- Explicit512-output permutation gathers make every one of20 complete-kernel
  stage/final checks exact. This includes uninstrumented eager and compiled
  outputs, both loop-local and reused scratch. The follow-up component suite
  passes all eleven distributions/shapes, including actual training IDs,
  repeated calls, zero input after nonzero input and all absent vocabulary rows.
  FP32 accumulation is closer to the FP64 reference than the native operation
  which rounds gradients to BF16. No model-quality claim follows from this.
- The repaired embedding is currently slow: actual training row native3.1621ms
  versus151.124ms with prepared metadata. Metadata creation/upload arm median
 46.744ms has large18.55–47.23ms variation and is unexpectedly faster than the
  prepared arm. Do not infer a preparation speed benefit or trustworthy additive
  decomposition from this. Investigate transfer geometry and queued execution.
- MLP contiguous activation transfers cut actual forward from554.088ms to
 1.69012ms. All16 stage bounds and24 native gradient/forward gates pass. Native
  forward is still faster at1.35026ms; full forward/backward4.24574ms versus
 4.33462ms is only a2.05% component improvement. No full-model promotion yet.
- The preserved intermediate MLP profile confirms scalar weight packets were
  eliminated: previous524,288 two-byte packets are absent. Remaining activation
  transfers are4,096 packets of128 bytes in the small shape. Its device window
  is635,344ns, with only~11us TensorEngine active per physical core and waits
  dominating. No spills are recorded. Missing dynamic DMA metadata still
  precludes exact bandwidth/MFU claims. New raw profile799,229bytes is local
  and hash-verified; derived Parquet/analysis receipts are retained.

Next: apply contiguous FP32 I/O transposes to the now-correct embedding scan.
For the MLP, measure ordinary BF16 cached-weight inputs, which training already
refreshes each optimizer step, with corresponding native controls. Cache
refresh/master updates remain excluded from this component arm and must be
charged by any eventual full-model gate. A full-shape profile option is ready
if the corrected kernel still misses the native bound.


### Wave31: quantify remaining descriptor work and address-dependent gather behavior

- Tiled FP32 scan transfers pass the full component suite. Actual training row:
  native3.17135ms, prepared4.60680ms, including metadata5.52020ms. The gross
  transfer regression is repaired; speed still fails the advancement gate.
  Next input-precision arm rounds to BF16 before gathering, matching native
  arithmetic, and keeps FP32 accumulation. Reference bounds use those same
  rounded inputs, including the long single-ID accumulation case.
- Address-only full scan control is mixed: aligned scratch with8192-output
  gathers passes the uninstrumented eager/compiled final result, but adding
  intermediate HBM stores reproduces1,283,061 final violations. Both tensor
  identity variants behave alike. Alignment alone is therefore not a robust
  fix. The512-output variant passed both instrumented and uninstrumented arms.
- BF16 cached-weight MLP numerical gates pass, yet forward1.67965ms versus
  native1.34390ms and forward/backward4.37851ms versus4.25577ms remain negative.
  Its full-shape profile is local/hash-verified,22,670,829bytes. Native device
  window1.286966ms versus fused1.604594ms; GpSIMD active~0.181ms versus~1.341ms
  per physical core. Fused emits4,096 DMA_DIRECT2D instructions versus512, and
  TensorEngine active~1.217ms versus native~1.124ms. These are attribution
  metrics, not a replacement for unprofiled timing; overlapping instruction
  sums cannot be added, and missing DMA/compile metadata prevents exact MFU.
  Next: preserve saved ReLU in the kernel's channel-major HxT layout, reducing
  its transpose/store count, and include view consumption in backward timing.
- Current official README/prepare.py were fetched read-only through authenticated
  GitHub; both match our local files. prepare.py hash remains5807c46e. The first
  Tavily research invocation hit a CLI syntax mismatch (stdin needs research run);
  the corrected request is running. All failed invocation records are preserved.

All completed device jobs have automatic healthy postflights; the address-only
  control failure is numerical, not an unclean host or driver reset failure.


### Wave32: precision and saved-layout controls remain below advancement gates

Source52f9dae. Both completed jobs are numerically valid and have healthy
automatic postflights; no owned cleanup kills or unfinished resets.

- Segmented scan with BF16-rounded gradient inputs passes independent FP64
  bounds, including long repeated-ID accumulation, and the native control's
  corresponding bounds. Actual training-row prepared4.59307ms versus native
 3.16469ms; complete metadata arm5.62263ms. This is effectively unchanged from
  the FP32-input4.60680ms result. Merely halving these buffers does not solve
  the remaining cost. Revisit the already-correct three-kernel DMA alternative
  with casts in SBUF and contiguous transpose tiles; it bypasses local gather.
- Saved HxT ReLU with immediate wrapper transpose back to TxH passes all
  component gates but regresses: forward1.80912ms/native1.33358ms, complete
  forward/backward4.47432ms/native4.23321ms. The follow-up preserves HxT through
  the autograd boundary and forms hidden derivatives in that layout. All18
  independent small CPU output/gradient controls pass, including changed weights
  and zero input. Native actual-shape gating remains required.
- The Tavily Pro report completed and is audited. Several old-allocation and
  hardware-generation suggestions are unsuitable; no documented cause of our
  gather corruption was established. The reviewed research note links current
  primary APIs and ties accepted directions to measured evidence. Raw provider
  JSON is hash-preserved locally.


### Wave33: both corrected transfer/layout designs earn full-model gates

Source0428ecb; three completed device jobs have healthy automatic postflights.
All component comparisons include kernel calls, intermediate HBM traffic and
layout movement; no scored training claim is made from component timing.

- DMA permutation plus segmented scan and expansion passes every eager stage,
  compiled composition, FP64 bound, zero/repeated-call and native-control gate.
  Actual training row native3.132663ms versus prepared0.855589ms and including
  CPU metadata creation/upload1.303211ms. Four-table amortization estimates
 0.967494ms each, but excludes input-ID D2H; only whole-model timing can establish
  the real saving. The revision separates DMA dtype conversion into SBUF and
  uses contiguous128x128 scan transfers. Largest real-row difference from native
  is2.98e-8. Scratch shared-metadata autograd now also has CPU accumulation,
  outstanding-forward metadata lifetime, unsupported-shape and checkpoint gates.
- Raw HxT saved ReLU consumed directly by layout-aware backward passes all16
  independent stage bounds and24 native forward/gradient controls. Actual shape
  native forward1.342891ms/fused1.419571ms; complete forward/backward4.242960ms
  versus3.960757ms (6.65% component saving). The previous immediate TxH conversion
  is removed from the timed forward. An actual-native-module integration preserves
  FP32 master Parameters, differentiable BF16 caches and native evaluation; CPU
  cached/uncached accumulated-gradient, cache-refresh and checkpoint gates pass.
- A new full-shape raw-layout device profile is complete; preservation/analysis
  follows independently of the unprofiled full-model gate.

Next: native autograd integration checks on both hosts, then separate full-chip
  real-data phase tests with metadata/caches/optimizer charged. Embedding keeps
  the predeclared570ms advancement threshold; the MLP needs a reproducible whole-
  model saving before combining it with another change. No seed sweep.


### Wave34: integration gains do not equal full-model gains

Source b0128d1. Both actual-shape integrations pass output/gradient checks,
accumulation, cache or metadata lifetime, native evaluation, and strict native
checkpoint restore. Completed phase jobs have healthy automatic postflights.

- Segmented DMA: full-step median **578.1483 ms**, microsteps total 491.8118 ms.
  All four ranks use the segmented path on all 44 measured/warmup preparations,
  including input-ID D2H, CPU sort, uploads and ordinary gradient accumulation.
  Nearby same-host native control: **586.4605 ms**, microsteps 502.1890 ms.
  The improvement is **1.417%**, below the predeclared 570 ms advancement gate.
  No 30-minute run is justified yet. Isolated per-table speedups overestimate
  the full-model benefit; shared native lowering, transfers and synchronization
  remain suspects, not established attribution.
- Fused MLP: full-step median **681.9695 ms**, microsteps 597.5696 ms. This is a
  clear regression despite the isolated forward/backward advantage. Finite and
  causal/batch/repeated-call checks pass on every rank. A runtime trace is being
  captured. A targeted alternative forms dX in ordinary TxC directly, preserving
  saved HxT activations; the current formulation returns a transposed result
  that the upstream normalization must consume. This layout explanation is a
  hypothesis. Independent CPU master/cache/accumulation gates for the alternate
  derivative pass; native and full-model gates remain required.
- Raw HxT MLP forward profile is locally hash-verified (19,046,466 bytes). Native
  device window 1.281745 ms; fused 1.343830 ms. GpSIMD DMA_DIRECT2D count falls
  from 4,096 in the prior TxH kernel to 2,560, with GpSIMD active about 0.836 ms
  per physical core versus the prior 1.341 ms. TensorEngine active about 1.096 ms
  versus native 1.12 ms. These are diagnostic active windows, not additive costs
  or throughput evidence; missing DMA metadata precludes exact bandwidth claims.

Next bounded designs, varied separately: group MLP row transfers across channel
tiles; test normal-layout dX without that forward change; fuse embedding indirect
loads, FP32 segmented scan and 512-output end gathers into one kernel, eliminating
intermediate HBM tensors. The new fused embedding must pass all original staged
FP64, instrumented/uninstrumented, compiled, zero/repeat and native-control gates.


### Wave35: runtime attribution changes the MLP diagnosis

Source 357b785. Fused embedding passes all 11 cases including every FP64 scan
position, instrumented and uninstrumented eager outputs, compiled composition,
zero/repeated calls and native controls. It nevertheless regresses: real-row
prepared median 31.479 ms (10.91–31.72 ms samples) and metadata-inclusive
52.851 ms versus native 3.157 ms. Small 256×256 case also reproduces the slowdown
(0.404 ms prepared versus native 0.071 ms). An on-device small-shape trace will
distinguish the narrow indirect reads, interleaved metadata loads/casts and local
gather cost. No model integration follows this failed speed gate.

MLP ordinary-layout dX passes the actual-native-module master/cache/accumulation
and checkpoint tests. However, the preceding full-model runtime trace weakens
the original upstream-gradient hypothesis: the fused model forward is 61.353 ms
versus the earlier baseline capture's 38.734 ms, while backward is 76.210 versus
77.557 ms. Each is still one compiled graph, not many extra host launches.
These diagnostic captures are different runs and are not paired throughput
evidence, but they locate the large regression in forward. Capture that forward
on device before spending another full-step test on dX alone. The runtime trace
and 22 associated files are locally hash-verified, totaling 63,183,126 bytes.

The new forward-only capture leaves the real timed-step gradients in place for
postchecks, performs no additional update, and reports a forward count separately
from microsteps. All completed wave35 jobs have healthy automatic postflights.


### Wave36: scalar metadata DMA explains the fused embedding regression

The small-shape capture is locally hash-verified (6,706,919 bytes). Its fused
window is 0.202802 ms versus native 0.026004 ms. Within the fused window,
metadata input3 produces **65,600 four-byte and 65,536 two-byte DMA packets**.
The aggregated transfers identify the strided interleaved metadata loads and
int32-to-BF16 presence cast. This is direct packet evidence, not an inference
from peak bandwidth. Missing dynamic DMA metadata still limits total-bandwidth
claims. Next: transpose the CPU metadata once, preserving preparation cost,
load each contiguous column at its original32-bit width, cast presence in SBUF,
and repeat every instrumented/uninstrumented numerical and performance gate.

Grouped MLP row transfers pass all16 stage and24 native numerical/gradient
gates. Real-shape forward is 1.133791 ms versus native 1.332775 ms; complete
forward/backward 3.651568 ms versus 4.219335 ms (13.46% component saving). The
full-model forward regression still needs attribution before any promotion.
The forward-only device capture is running on the other host. Completed jobs
have healthy automatic postflights; neither changes the best BPB checkpoint.


### Wave37: contiguous metadata repairs the gross regression, not the final speed

Source a1729bc. All 11 fused embedding shape/distribution cases, full scan
positions, instrumented/uninstrumented eager calls, compiled calls, native
controls and zero/repeat checks pass. Actual training row: prepared **3.878788 ms**
and including metadata **4.797906 ms**, versus native **3.135013 ms**. This is
much faster than the 31.479 ms strided-metadata kernel but still loses to both
native and the three-kernel 0.856 ms implementation. Fusion alone is not a win;
remaining local gather and narrow indirect-DMA costs must be charged. Automatic
postflight is healthy. The prior profile count is corrected to 65,600 four-byte
packets and 65,536 two-byte packets within the fused execution window.

A four-rank synchronized MLP component test is prepared to separate whole-chip
contention from surrounding-graph effects. It reports per-rank times, starts
all ranks in the same arm, and excludes the closing barrier from timing. It
does not change optimizer/model arithmetic or constitute a whole-model result.


### Wave38 and GUI crash recovery: whole-chip contention is not the main MLP regression

Source052c839; all four ranks pass actual-shape numerical and gradient gates.
Concurrent native forward/backward medians are4.7006–4.7367ms, versus
fused4.3795–4.3942ms. Native forward medians1.5136–1.5275ms, fused1.5712–1.5962ms.
The component advantage survives four-rank contention, while the full-model
forward regression is much larger. Surrounding graph layout and scratchpad
pressure remain hypotheses requiring the full device trace. All jobs completed
with clean automatic postflights; both hosts were idle when recovered17:07UTC.

The wave36 full-forward raw capture is locally verified (1,020,828,335 bytes),
but local ingestion hit its300-second timeout at15:39:32UTC, returncode124.
Its52 partially written Parquet files are not a valid analysis; the main
Instruction table is zero bytes. Preserve the raw capture and ingest elsewhere
with adequate resources. The wave37 corrected-metadata small capture also
completed cleanly; its remote raw data still needs retrieval and analysis.

The surviving two-kernel embedding edit is preserved as a prototype, not a
correctness/speed claim. It fuses wide DMA permutation and scan while retaining
the separately validated DMA expansion. The crash recovery and next actions
are recorded in `RESUME.md`; regular GitHub backup is now explicitly authorized.


### Wave39: recovered kernel gate and completed MLP profile ingestion

The two-kernel wide-DMA permutation/scan prototype passes all11 stage/final
cases, including the real training row, full FP64 scan recurrence, debug and
uninstrumented eager outputs, compiled composition, native controls and
zero/repeat checks. Real-row prepared0.902076ms, metadata-inclusive1.734378ms,
native3.150630ms. It has no demonstrated advantage over the three-kernel
0.855589/1.303211ms result from the earlier run and is not promoted. Automatic
postflight is healthy. The prior wave37 small profile is now locally verified:
11files/420,881bytes.

Local full-model MLP ingestion had timed out. The first remote retry failed
before analysis because Explorer is installed under `/opt/aws/neuron/bin`,
not the native Python venv. The failure was preserved and cleanup passed.
Retry2, fresh output and GOMEMLIMIT24GiB, succeeds in95.28s;32 Parquet files,
299,505,407bytes, independently hash-matched locally. The summary job also
completes cleanly. Device forward58.761415ms versus baseline36.514939ms;
Tensor active~27.37ms/core versus~18.8ms, with increased Scalar and GpSIMD
activity. The leading DMA groups include a `SpillSave` instruction. Quantify
spill bytes and attribution against the baseline before claiming a mechanism
or implementing a lower-scratchpad kernel. Dropped/missing DMA metadata still
prevents reliable total bandwidth claims. These are diagnostic captures,
not a fresh paired throughput experiment or a BPB improvement.

The2026-09-18 deliberate-restart checkpoint is documented in `RESUME.md`.
Code/results are now regularly pushed, and the best model and key raw captures
have verified off-laptop backups. No experiment is active on either accelerator
at the checkpoint; the remaining local artifact-upload queue is resumable.


### Wave40: full-graph spill evidence and two negative integration controls

The CPU-only trace comparison completes cleanly: recorded SBUF reloads rise
from2.3395GB to7.2645GB in the fused forward. Most candidate reload bytes are
attributed to ordinary Linear operations, while regular matrix FLOPs remain
nearly unchanged. The isolated kernel's timing therefore misses a substantial
surrounding-graph effect. Names containing `spill` alone do not reveal it.
See `fresh-start-mlp-spill-investigation-2026-09-18.md` for graph windows,
source attribution, population counts, compute bounds and data-quality limits.

Actual native-module gates pass for grouped I/O and tile256. Fresh same-host
full-chip controls measure590.315ms/trn21 and599.575ms/trn22. Grouped I/O gives
665.457ms (+12.73%); tile256 gives686.590ms (+14.51%). All numerical independence
postchecks and automatic cleanups pass. Neither reaches the3percent improvement
gate; neither enters training. Reducing token tiles alone does not repair the
integration regression. These tests do not invalidate the separate weight
residency hypothesis.

### Wave41: streamed weights pass correctness but lose the whole-model gate

Mode1 retains only the up matrix; mode2 streams both weight matrices in
contiguous blocks. All16 stage and24 native gradient gates pass for both.
Mode1 real-shape forward1.304021ms/native1.338066ms; forward/backward
3.817119ms/native4.224694ms (9.65% component saving). Mode2 forward
1.493931ms/native1.335086ms; forward/backward4.013717ms/native4.239976ms
(5.34% component saving). Repeated weight reads and transposes are included.
Actual FP32-master/cache/accumulation/evaluation/checkpoint gates pass for mode1;
mode2 integration is running. Mode1 full-chip phase is now running on trn21.
No training throughput or BPB improvement is claimed yet.


Wave41 final full-step results: mode1 on trn21 is **673.016550 ms**, 14.01%
slower than its 590.315111 ms native control; mode2 on trn22 is **693.390713 ms**,
15.65% slower than its 599.574587 ms control. All four updated-model checks and
automatic postflights pass. Neither variant advances. Source-level reductions
in declared weight residency have not established a reduction in caller spills.

### Wave42: prepacked BF16 weights improve the component, not the model

Source `df6fa725868ea36d460d3fef97b453eb7ac1bd32`. Mode3 caches each BF16
weight in stationary-ready blocks once per optimizer update; ordinary master
weights and gradient paths remain unchanged. Exact pack/unpack, all16 FP64
stage gates,24 native output/gradient gates and actual master/cache/accumulation/
checkpoint gates pass. At8192x1024x4096, component forward/backward is3.541410 ms
versus native4.229880 ms; this excludes the separately measured0.287952 ms
layout refresh per MLP.

The full-chip phase includes all cache preparation and measures **678.041824 ms**
versus the same-host590.315111 ms control. Median microbatch total593.152004 ms;
cache refresh4.385946 ms. All four updated-model independence/finite checks and
automatic postflight pass. **Reject for full-budget training.** Faster on-chip
weight handling alone does not fix the surrounding graph regression.
Receipt: `run-logs/fs-w42-prepacked-phase-comparison.json`.

### Wave43: explicit SBUF reuse passes its component gate

Source `7c99d57437c2ce6d2de5acbca44581fc6f431234`. Mode4 uses five disjoint
explicitly placed SBUF tensors, reusing each logical tensor in sequential outer
loops; declared extent7.125MiB per physical core at the real shape. All FP64 and
native stage/output/gradient gates pass, including changed and zero inputs.
Forward2.013316 ms versus native1.335363 ms; forward/backward4.555887 ms versus
native4.237448 ms; packed-layout refresh0.303469 ms is separate here. The
actual cache/master gate and then a bounded full-model test decide whether the
larger component cost buys a reduction in graph-wide allocation pressure.

### Wave44 diagnostic plan: separate the autograd boundary from NKI forward

The new native-forward control retains input flattening, contiguous HxT saved
ReLU and the exact custom backward, while executing the two projections in
ordinary PyTorch. Gate master gradients/cache refresh first, then measure the
same full model. If it is fast, the regression needs the opaque NKI forward;
if it is also slow, investigate the saved layout/backward boundary before more
kernel tiling work. A separate layer-selection option allows replacing only
one MLP to distinguish local per-layer cost from a graph-wide discontinuity.
These options are scratch-only and have no quality result yet.


Wave43 final: actual master/cache gate passes both shapes and all accumulated
gradients, updates, evaluation and checkpoint checks. Full-chip phase is
**705.225310 ms** versus trn22 native599.574587 ms,17.62% slower. All four updated
model checks and automatic postflight pass. Explicit7.125MiB placement did not
repair the full-model regression; reject for training. Receipt
`run-logs/fs-w43-fixed-sbuf-phase-comparison.json`.

Wave44 native-forward/custom-backward gate passes at both shapes, including
cached and uncached FP32 master gradients. Its full-model phase is running.

Wave45 adds a deliberate compiler boundary around the same packed NKI MLP and
compiled native backward. Shared tensor-only inner graph; fullgraph remains
required inside, while the outer graph permits the explicit boundary. Dispatch
and cache preparation remain timed. Numerical/master/cache gates precede the
full-model test. This is an integration experiment, not a kernel arithmetic
change or a reason to suppress compiler errors.


Wave44 native-forward/custom-backward full model: **574.075248 ms** versus
earlier same-host590.315111 ms,2.751% faster. All four updated-model checks and
automatic postflight pass. This does not yet clear the3% advancement threshold.
A fresh native control runs next, followed by a bounded combination with the
already validated three-kernel embedding backward (previous full-model1.42%).
No claim that the gains add linearly or that a quality gain is established.

Wave45 separate-region gate **fails** on changed microbatch1 at the small shape:
input-gradient9 elements exceed the original bound, max abs6.103515625e-5,
relative RMS0.003586663. The same failure appears with native forward inside
the separated region. Thus it does not require NKI forward. Both failed jobs
have clean automatic postflights; no full-model phase was launched.

Wave46 diagnostic keeps calls uninstrumented and compares complete outputs and
gradients after return. It uses FP64 matmul references with explicit BF16 stage
rounding, alternate rounding hypotheses, and repeated base/changed/zero inputs
to distinguish rounding from saved-data reuse. Gate tolerances are unchanged;
the diagnostic cannot promote a kernel even if its process exits successfully.
