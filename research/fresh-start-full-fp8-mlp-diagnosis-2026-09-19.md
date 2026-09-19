# Complete FP8 MLP: correctness passes, preparation still costs too much

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
