# Complete FP8 MLP: component gains and full-model validation

Updated 2026-09-19 08:20 UTC. The original profile below identified online
weight preparation as a concrete cost. Both proposed preparation changes have
now been implemented and gated; the original profile is retained as evidence.

- Source132 compiled the existing preparation: 1.006624 -> 0.549580ms. At the
  original 16,384-token/two-microbatch geometry, complete FP8 remained 0.485%
  slower than compiled native. Source135 independently gated the current
  32,768-token/one-microbatch shape and measured a 1.764% component gain.
- Source137's fused dual-orientation producer quantizes each weight once and
  avoids the intermediate BF16 HBM transpose. Resident preparation measured
  0.294823ms; complete MLP 14.886892ms versus native 15.348333ms (3.006% gain).
  The streamed variant also passed and gained 2.920%. All preparation is timed;
  the CPU metadata/bytes references and full derivative gates remain intact.
- Source139 passed the real-Linear integration at widths 256 and 1024, including
  FP32 masters, four updates, two microbatches, cache lifecycle, short-row
  padding, row independence and original native evaluation/checkpoint restore.
  Its per-Linear preparation dispatch differs from the paired component probe.
- Source140 compared the complete model at T1024/pack32/GA1 on trn21.
  Native control and candidate use the same source, data, optimizer ownership
  and clean host protocol. The phase timer clips before reduction in both arms;
  the production quality loop clips after reduction. A component gain is not
  a whole-model gain or a BPB result.

The full FP8 candidate has now completed all 35 updates and four numerical
postchecks. Its median step is **567.094872 ms**, versus the native control's
**488.701909 ms** (16.041% slower). Median forward/backward rises from
433.514570 to 508.181582 ms; cache refresh rises from 2.856698 to 4.927452 ms.
The dominant gap is therefore inside the model call, not weight preparation.
This locates the next measurement; it does not yet establish a compiler cause.
The candidate's postflight failed only the 8 GiB disk reserve after compilation.
It left no device holders or compiler/training workers. Preserve that failure,
restore headroom and collect a clean caller diagnostic before further changes.
Records: `run-logs/fs-w140-full-fp8-phase-comparison-r1.json`.

Native and full-FP8 microstep profiles use the same source140, pack32 geometry,
data, seed and 35 preceding updates. Different hosts are acceptable for source,
engine and allocation attribution; the same-host unprofiled phase remains the
speed comparison. The profile requests retain all four numerical postchecks,
actual runtime binaries and pre/post cleanup. The next source change remains
conditional on those traces, particularly the dead BF16 outputs below.

The native source140 control-r1 passed all numerical gates but failed the final
8GiB disk reserve. Its 489.273ms median remains a diagnostic. After verified
retirement of five old checkpoint copies (all preserved locally/cloud), a fresh
control-r2 finished clean at 07:35:36: **488.702ms**, last loss5.126208782196045,
all four postchecks passed. The paired full-FP8 candidate started 07:37:54 on
the same source and host; use control-r2 when its result arrives. The cleanup's
first attempt hit a no-overwrite audit-log error after two retirements; the
failure and original intent are retained. Explicit recovery completed the
remaining three with uniquely numbered progress records and clean health.

Evidence: run-logs/fs-w132-fp8-compiled-cache-component-r1.log,
run-logs/fs-w135-fp8-compiled-cache-pack32-component-r1.log,
run-logs/fs-w137-fp8-dual-weight-cache-resident-r1.analysis.json,
run-logs/fs-w137-fp8-dual-weight-cache-streamed-r1.analysis.json, and
run-logs/fs-w139-fp8-full-mlp-real-linear-gate-r1.analysis.json.

## Next bounded investigation: retained BF16 outputs

The source140 full-VJP adapter saves ReLU, raw FP8 activation bytes and metadata.
It discards the forward kernel's BF16 QDQ-X and QDQ-hidden outputs. The NKI
source still allocates and writes those outputs. At 32,768 tokens, width1024
and hidden4096, those two arrays contain 320MiB per layer, or 1.875GiB across
six layers per rank. These are source-level byte counts, not measured DMA
traffic: compiler elimination must be checked before predicting a gain.

The packed forward branch also uses BF16 up/down cache arguments only for
shape/dtype checks; its values come from the packed bytes. The full backward
uses both packed orientations. The dual producer still returns BF16 QDQ weight
arrays to preserve the old ABI. That is another 16MiB per MLP, separate from
the ordinary native weight cache used by evaluation.

After the current full-model comparison, inspect the actual compiled caller
before changing these paths. If the dead writes remain, introduce one explicit
full-VJP-only output mode while preserving the existing reference path. Require
exact logits, ReLU, saved bytes, metadata and all four products against the
current gated implementation, then repeat the master/cache lifecycle gate.
Charge all layouts and refresh costs in the full-model timing. This is an
elimination of unused work; it does not require a different quantizer or a
weaker error bound. Leave the frozen BF16 submission unchanged.

## Original source124/source125 diagnosis

The complete FP8 MLP passes all11 cases, including the actual16384x1024x4096
shape, independent CPU scales/quantizers/pair arithmetic, exact compiled/staged
outputs and gradients, two distinct accumulated microbatches and updated/restored
weight caches. Forward and all4 backward matrix products use FP8; nonlinear
math remains BF16 and master gradients FP32. This is a new approximate VJP,
not the original BF16 derivative, and is not integrated into the full model.

The same-host, unprofiled, fully charged result is15.937871ms vs15.485907ms
native BF16, a2.9186% loss. Existing FP8-forward/BF16-backward is15.756617ms.
Timings include current casts, calibration, both packed weight orientations,
forward saves, both distinct FW/BW microbatches and FP32 gradient accumulation.
No optimizer, collectives or whole-model quality is included.

Both source125 single-arm captures and CPU ingestions completed cleanly. All
raw and parquet hashes match local copies. Runtime-to-binary checks match8
native and16 full-FP8 dispatches, with no unmatched execution. The extra8
FP8 dispatches belong to current weight-cache preparation.

| Instrumented quantity | Native BF16 | Complete FP8 |
|---|---:|---:|
| Cache preparation device intervals, ms |0.112367|0.757307|
| Micro0 forward device interval, ms |2.654745|2.608986|
| Micro0 backward device interval, ms |4.868031|4.663893|
| Micro0 backward TensorEngine active, core0 ms |4.059482|2.181706|
| Micro0 backward ScalarEngine active, core0 ms |0.987601|2.436987|
| Micro0 backward VectorEngine active, core0 ms |1.226756|1.515005|
| Micro0 backward sync active, core0 ms |0.038757|0.815935|

These captures came from different hosts and synchronize at phase boundaries.
Use them for attribution, not another speed claim; source124's matched timing
is authoritative. Engine activity intervals overlap and cannot be added.
Dynamic DMA metadata and some compiler/HLO metrics are absent, so no exact
bandwidth or MFU inference is warranted. Cached binaries may reference earlier
immutable source paths; binary identities were matched and the current
algorithm passed independent gates.

Inference: faster matrix products are offset by Scalar/Vector work, barriers
and preparation of weight layouts. First try compiling the unchanged online
cache-preparation calls together: reduce launches while requiring exact current
metadata, both packed byte orientations, BF16 caches, master immutability and
refresh after updates. Include a similarly compiled native-cast control.
Then consider one NKI producer that calibrates current weights and emits both
orientations, charging its own reductions, reads and stores. Neither is yet
implemented or a proven win. Keep the per-output-channel fused gradient design
as the subsequent precision/layout experiment, with its distinct quantization
gates; do not change both at once.

The source124 original postflight failed only the8GiB disk reserve. Explicit
verified cleanup restored clean health before these profiles. Source125
captures/ingestions each have clean postflights. Raw captures are GitHub-digest
verified (35,690,507B native and33,010,935B full tar assets). Derived parquet
is locally verified and included in automatic artifact discovery.

Evidence: run-logs/fs-w124-fp8-complete-mlp-component-r1.summary.json,
run-logs/fs-w126-complete-mlp-profile-comparison-r1.json, and its linked analyses.
