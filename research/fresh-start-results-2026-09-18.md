# Fresh-start measured results — 2026-09-18

This ledger separates full training, standalone evaluation, numerical tests and
microbenchmarks. A successful process is not automatically a promoted model.
Public diagnostics do not establish a private competition score.

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

## Direct compiled SDPA: promising throughput, quality comparison pending

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

## FP8 feasibility: conversion verified, accumulation gate unresolved

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
does **not** close the failed elementwise numerical check. The next diagnostic
returns FP32 outputs and compares them to independent FP64 dot products under
the declared FP32 sequential-accumulation bound. It will separate accumulation
from final BF16 rounding without quietly relaxing the original tolerance.

## Next controlled full runs

The user explicitly requested keeping a serious route without n-grams. The next
full control is width 896 with direct compiled SDPA and the existing tables
(`fs-w3-w896-direct-control-r1`, trn21). Trn22 first profiles width 1024, depth 6
without hashed n-gram tables (`fs-w3-dense1024-phase-r1`), then will train a
table-free candidate under the same 30-minute limit. Existing unigram value
embeddings and trained local convolutions are retained initially; removing all
three together is a distinct architectural experiment.

Old no-table measurements were paradoxically slower on their earlier compiler
graph. That is a reason to inspect actual phases, not a permanent ban on removing
tables. The FP32-state NKI optimizer is available for a later table-model arm;
it is not a commitment to that architecture. Record both 1024/2048 evaluation
contexts, then step-match any promising difference that changes throughput.
