# FP8 packed-weight residency experiment

Updated2026-09-19 13:23UTC; hardware correctness gates PASS, streamed speed pending.
Default OFF. The source145 native/legacy/compact comparison is complete and
clean:488.702/568.354/562.902ms. Source147ac4f789 is deployed on trn22;
`fs-w147-fp8-streamed-weight-gates-r1` started09:50:09 with clean preflight.
It completed clean09:59:18 with all11 cases including32768tokens and both
real-Linear widths/cache/master/restore gates. The same-source resident
compact whole-model control completed clean13:20:28, command0 and all four
model postchecks. Median562.968046ms, within0.012% of source145 compact,
with identical diagnostic loss5.120891094207764. Streamed compact is ACTIVE,
guard1808156, started13:22:11.276554 after clean preflight and explicit gates.

The complete FP8 MLP has a small isolated advantage but the source140 full
model loses16.04%. Matched caller identities put most of the added device time
in the forward graph. Ordinary Linear operations show much higher observed
spill/reload counters. Missing notifications and HLO/layer metadata prevent
exact traffic measurements or identifying a particular Linear. This motivates
an allocation experiment; it does not establish that resident FP8 weights
cause those spills.

The current forward keeps both packed FP8 matrices resident in SBUF:4MiB each
at1024x4096,8MiB in total per physical core. The proposed optional
`stream_packed_weights` path loads one output group into reusable storage:
0.5MiB for the up group and2MiB for the down group. Sequential output-group
loops bound their simultaneous weight storage at the source level. Contraction
loops remain affine and preserve their exact paired products, accumulation,
quantization, BF16 rounding, two ordered inverse rescalings, outputs and full
backward. Five compact forward saves are required. Every packed cache is still
prepared from current master weights inside the charged update.

This is a deliberate tradeoff, not a free memory reduction. At32768tokens,
there are256 token tiles across the two cores. A straightforward execution
reads8MiB of weights per tile,2GiB per layer/rank, versus16MiB total when each
of two cores loads the full resident cache once. Those are source-level traffic
estimates; compiler hoisting/allocation may change actual execution. Sequential
groups can also reduce overlap. A smaller isolated source buffer does not prove
a smaller caller allocation, and the isolated kernel may become slower.

The gate reuses the original independent CPU quantizers and all four FP8
product bounds. It compares all five retained tensors exactly against both
the original seven-output path and compact resident path, including repeated
calls; then checks compiled FP32 master gradients, two-microbatch accumulation,
refreshed weights, ten edge fixtures and32768x1024x4096. A separate real-Linear
gate checks cached/uncached masters, training padding/row independence and exact
native evaluation/checkpoint restore. No tolerance is relaxed.

If those checks pass, compare resident compact and streamed compact in the
same full model on one host/source with all preparation, communication and
optimizer costs included. Native BF16 remains the advancement control. This
specific caller-memory hypothesis merits a bounded full-model test even if
an isolated component slows down; only the whole-model result can justify
promotion or a30-minute quality trial. Profile changed caller allocation if
the full-step result is promising or contradicts the predicted tradeoff.

This is distinct from source137's dual-cache producer residency: that kernel
prepares packed bytes. This experiment changes the consumer's weight lifetime.
The frozen submission and established BPB remain unchanged.

Earlier BF16 streaming/prepacking trials remain valid negatives: source40–48
whole models were slower despite passing gates, including a fixed-SBUF variant
with7.125MiB declared storage. This FP8 experiment is not a claim that streaming
is untried or guaranteed to cure composition. It tests the current FP8-specific
consumer and compact-output combination, consistent with ranking layouts in
the precision in which they will run.

The later source140 operand query found ordinary-Linear BF16 MATMUL geometry
changing from128x128 to128x32/32x128 with512 free elements for65,536 observed
operations on one physical core. Shared-source/lost-notification limits prevent
identifying the layer or establishing a memory cause. If streaming does not
restore whole-step performance, investigate the opaque kernel output layout
and actual caller tiling before pursuing another resident-size variant blindly.
