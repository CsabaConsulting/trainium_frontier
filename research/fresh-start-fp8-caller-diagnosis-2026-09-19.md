# FP8 caller regression: attribution and controlled boundary test

Updated 2026-09-19 02:09 UTC. No FP8 speed or quality promotion.

The packed FP8 MLP is competitive in isolation but loses17.7363% in the whole
model:587.956385ms versus499.384107ms. This same-source/same-host wave89 pair
charges two microbatches, gradient synchronization, optimizer and cache refresh.
Two-microbatch computation adds85.001439ms and cache refresh adds2.528454ms.
Numerical, cache freshness and all four native-evaluation postchecks pass.
The whole-step result is the performance evidence; the profiles below explain
where to investigate. They were captured on different hosts and are not a
substitute for the unprofiled paired benchmark.

## Complete caller profiles

Both captures use source `dcd6b80a4199339333d93a8e005aa746d3fd28d2`, six layers,
width1024, T1024/pack16/GA2, seed42, real data, ownership and exact adaptive CE.
One warmed rank0 microstep is captured while all four ranks participate.

| Metric | Native BF16 | Packed FP8 forward / BF16 backward |
| --- | ---: | ---: |
| Forward device interval |68.001178ms|109.520190ms|
| Backward device interval |138.211172ms|137.589812ms|
| Forward spill-save counter |4,788,875,264B|3,142,467,584B|
| Forward spill-reload counter |6,997,243,392B|10,391,957,504B|
| Ordinary Linear source spill-reload |1,835,008B|6,492,786,688B|
| Ordinary Linear Tensor activity, core0/core1 |7.800816/8.012658ms|22.569284/22.758825ms|
| Ordinary Linear Scalar activity, core0/core1 |0.867826/0.787777ms|11.564346/11.621143ms|

Ordinary Linear source is `train.py:2853`, `F.linear(x,w,None)`. It is shared
by several projections, so source attribution alone does not identify a
particular attention projection or the vocabulary head. The ingested `layer`
and `hlo_name` metadata at this source are Unknown/empty; do not infer a
specific projection from them. Private transpose activity is negligible in
forward. The FP8 kernel source itself has about101MB of reported spill reloads,
far less than the ordinary Linear source's6.49GB. More allocator fragmentation,
changed surrounding layouts or an unfavorable compiled matmul schedule are
plausible hypotheses. These counters alone do not establish which caused it.

Engine/source interval unions overlap and must not be added into exclusive
wall-time shares. Sampled SBUF peaks do not measure every live allocation.
Explorer warnings about dynamic DMA metadata prevent claims about exact
physical HBM traffic or bandwidth. The regression cannot be described simply
as a slower FP8 backward: the observed backward intervals are nearly equal.

Source96 analysis matches all33 FP8 and45 BF16 ordinary runtime dispatches to
preserved binaries. Each also has one explicit startup c10d barrier without
an ordinary persistent-key dispatch. The barrier is checked by ordering,
annotation, device interval, binary identity and SHA256. Source94's original
count failure is preserved; no unknown execution is dropped.

Source96 precision reports have an integer-division bug: adjusted/raw FLOPs
prints0 for FP8 instead of0.5. Actual FP8 forward counts are
825,694,879,744/826,944,782,336 raw FLOPs and
412,847,439,872/413,472,391,168 adjusted FLOPs on the two cores. Timings,
counters, raw totals, BF16(1) and FP32(4) classifications are unaffected.
Both reporting scripts now cast to float64 before division. Preserve the
original report and produce a new labeled analysis; this is not a kernel fix.

## Controlled next experiment

Source95 `d14298ea22ab33ff8be47afc9cba4a81ca8967ac` adds one shared compiled
region for the FP8 MLP, reached through a Dynamo-disabled boundary. The FP8
algorithm, BF16 surrogate gradients, FP32 masters, caches and evaluation
policy remain unchanged. The real-Linear gate passes widths256 and1024,
including repeated updates, exact reference gradients, changed caches,
padding and row independence, and exact native checkpoint restoration.

Run these same-host/same-source phases sequentially after clean postflight:

1. `fs-w95-native-owned-fp8-control-phase-r1` (active02:07:45).
2. `fs-w95-packed-fp8-monolithic-control-phase-r1`.
3. `fs-w95-packed-fp8-separate-candidate-phase-r1`.

All use five warmup and30 timed complete updates with unchanged real data and
geometry. A new graph boundary may add launches or change surrounding fusion;
that cost is deliberately included. Do not call the hypothesis confirmed merely
because the integration gate passes. The predeclared3% complete-step gain gate
still applies before a30-minute FP8 quality run. Even a speed win would need
quality validation because FP8 training followed by native BF16 evaluation is
an explicitly approximate training policy.

Related designs: shared gradient quantization in
`fresh-start-fp8-backward-design-2026-09-19.md`; exact fused head/loss in
`fresh-start-fused-head-loss-design-2026-09-19.md`. Neither is implemented or
promoted by this caller diagnosis.
