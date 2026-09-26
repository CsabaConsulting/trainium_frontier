# Real-shape BF16 ordered consumer: supplied-metadata component only

Scope is the separately authorized first portion of gate2 in
`research/v2/proposals/20260926-k2-bf16-production-domain/SPEC.md`.
This is a source prototype for a future synthetic component, default disabled.
It is not connected to train.py, autograd, the loader, or a launcher.

## Source and exact call contract

New file `consumer_nki.py`, derived from protected diagnostic kernel SHA256
2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae.
The protected file is unchanged. Frozen scientific source is7419927c,
train.py SHA25636a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
Neither parent nor new consumer is a production metadata producer.

`build_kernel()` raises EXPERIMENT_DISABLED before importing native modules.
An explicit `enabled=True` constructs the NKI callable with grid2; constructing
or importing source is not launch approval. There is no executable main/host path.

Inputs, all resident tensors:

| Input | Shape | Type |
|---|---|---|
| gradient | [32768,1024] | BF16 only |
| counts | [8192,1] | int32 |
| starts | [8192,1] | int32 |
| positions | [32768,1] | int32 |
| group_maximums | [64,1] | int32 |

Output is fresh dense FP32 [8192,1024]. The callable rejects FP32 tangents by
its NKI specialization assertion; it never casts an FP32 tangent down.
Shape/dtype assertions are specialization guards, not runtime-data validators.

Precondition for the future synthetic component: an external independent
checker must fully authenticate the immutable metadata before upload, verify
transport/actual values, and retain those tensors through every consumer call.
counts count actual synthetic IDs; starts are exclusive row offsets; positions
is a permutation of0..32767 with each row's positions in strict ascending token
order and matching its ID; maxima are exact maxima in consecutive128row groups.
No invalid/stale metadata is permitted under this diagnostic precondition.
In-range corruption can silently produce a wrong result: this kernel does not
validate it. Full producer invalid/stale tests remain pending. No CPU routing
or supplied metadata may be used to claim production integration or speed.

## Recurrence and ownership

For each row/channel, initialize positive FP32 zero. Visit that row's original
token subsequence in ascending order. Widen its BF16 element and add once into
resident FP32 state. The intended result is one correctly rounded FP32 addition
per occurrence. There are no partial sums, grouped sums, gradient reductions,
atomics, clamps, exceptional-value filters or tolerance checks.

Program0 owns channels0..511, program1 owns512..1023. Both programs traverse
all64groups with identical supplied maximum counts. The64group bodies are
static; each contains a `nl.dynamic_range(limit)` occurrence loop, so32768
occurrences are not statically unrolled. Padded inactive rows use skip-mode
out-of-bounds sentinels and retain accumulator bits through selection. Sentinel
token/value scratch is reinitialized every iteration to prevent stale reuse.
Each owned output row/channel is written once, including untouched positive-zero
rows. Each call has fresh output/state; separate tables/calls/microsteps are not
concatenated. Existing ordinary accumulation boundaries would remain outside
this component; there is no training integration yet.

Raw finite bits must eventually match both a prescribed ordered recurrence and
the actual ordinary native path. Both BF16-minimum signs, reachable FP32
subnormal cancellation and the two BF16-representable order witnesses remain
strict. NaN/infinity/overflow are diagnostics: classification/infinity sign and
raw payload differences are reported, and every nonfinite output fails the
finite training gate. No equal-NaN PASS is defined.

## Scratch, work and unresolved SDK behavior

Accumulator, FP32 value, updated and selected tiles are each128x512x4bytes,
so four obvious live tiles total1MiB per owner. Seven128x1 address/active/count
tiles plus one maximum scalar add3,588bytes. Explicit main buffers are allocated
outside the group/occurrence loops; selection/address expressions remain loop
temporaries. The logical1,052,164byte estimate excludes compiler staging,
descriptor storage, registers and allocator overlap/spill. Native allocation
and liveness are unknown, not proven by source reuse.

For Q=sum(group maxima),256<=Q<=32768. Padded FP32 adds/table range33,554,432
to4,294,967,296. Useful BF16 reads are64MiB/table, dense output32MiB/table;
skip effectiveness and descriptor overhead are unmeasured. Two tables would
double these numbers. This component excludes producer costs, so its timing
cannot establish the required production or whole-step gain.

Official documentation supports the source constructs:

- [SDK2.31 DMA](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html)
  supports HBM vector-indirect row loads and skip semantics. The new tangent
  descriptor uses row stride1024,128partitions,512contiguous channels and a
  static owner offset. No simultaneous scalar/vector offset is used.
- [Dynamic ranges](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.dynamic_range.html)
  and register-load bounds express the variable occurrence count.
- [Memset](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.0/nki/api/generated/nki.isa.memset.html)
  permits constant initialization of reused scratch.
- [Tensor arithmetic](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.tensor_tensor.html)
  provides elementwise add; native rounding/underflow and compiler dependency
  preservation still need exact native evidence.

The local environment has no NKI/Neuron compiler. No native/API incompatibility
has been demonstrated. Specifically unresolved: actual512channel descriptor
legality and performance, SBUF allocation/spills, runtime-loop liveness and
symmetric two-program lowering, BF16 DMA widening/subnormals, exact addition,
output writes and worst-skew32768-round time. Cold compile time is unknown.
64static bodies may cost compilation; this is explicitly part of the next gate.

## Local evidence and next boundary

Run `test_consumer.py` with existing Python normal/-O. It AST-replaces ONLY the
six geometry constants with V8/T16/C8/P4/F4/G2, then interprets the actual body
through a small fake instruction layer using CPU Torch. It separately computes
an ordinary token-order CPU oracle, checks owner write masks and loop schedules,
and reuses the same callable with changed runtime IDs/tangents and fresh outputs.
Raw inputs/outputs are recorded. This is an Implementer test, NOT NKI simulation,
native legality, full-shape execution, independent G2, or a production gate.

After fresh independent source/diff/spec/raw-receipts review, a separate complete
real-shape synthetic component proposal may test supplied-metadata correctness
first. It must preserve all required real-shape/skew/subnormal/exceptional/replay
fixtures and retrieve four-rank raw artifacts. Mismatch, compile error, spill
explosion, stale output, wrong dtype, timeout or missing artifacts is failure.
No automatic retry, producer, whole-model, timing, quality or submission follows.
