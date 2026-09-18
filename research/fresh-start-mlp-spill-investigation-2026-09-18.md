# Full-model MLP spill investigation — 2026-09-18

The fused BF16 MLP is correct and faster as an isolated forward/backward pair,
but the previous full-model integration regressed from586.46 to681.97ms per
step. Four concurrent ranks preserved the component advantage, and a runtime
trace put the regression in forward. This investigation compares the saved
full-forward device traces; neither profile is a paired throughput result.

## Evidence from wave40

Analysis source8860a7f, job `fs-w40-mlp-spill-comparison-r1`, clean pre/postflight.
Its `.analysis.json` contains input hashes, the installed Explorer schema,
per-source results, counter population counts and interval-merged activity.
Reference graph `e496117f74af27424b6c982ea962b625.neff`; candidate graph
`6ec5dd5b3e70cf667b3952ff25804a5c.neff`. Each uses its own ExecutionInfo window.

| Recorded quantity | Native forward | Fused MLP forward |
|---|---:|---:|
| Profiled graph wall time |36.514939ms|58.761415ms|
| Spill-save counter sum |1,440,841,728bytes|1,492,484,096bytes|
| Spill-reload counter sum |2,339,500,032bytes|7,264,536,576bytes|
| Instructions with a positive spill counter |13,393|37,262|
| Tensor transpose active, physical core0 |4.248451ms|6.561954ms|
| Tensor transpose active, physical core1 |4.281677ms|6.584066ms|
| Names containing `spill` |3,712|3,658|

The installed schema defines these spill counters as bytes moved between SBUF
and HBM due to lack of SBUF space. Counters are sparse, and dropped/missing DMA
metadata remains in the warnings. These are **recorded instruction-counter
sums**, not an independently complete bus-traffic measurement or an exact
memory-bandwidth bound. The naming heuristic alone would miss the regression:
there are slightly fewer explicitly named spill instructions in the candidate.

Of the candidate's recorded7.265GB reloads,5.386GB is attributed to
`train.py:2853`, the ordinary `Linear.forward`/`F.linear` call. Only57.77MB is
attributed directly to `mlp_fused_bf16.py`. Layer strings are `Unknown`, so the
source line does not yet distinguish the LM head from attention projections.
The custom kernel can therefore be inexpensive in isolation while changing
allocation/scheduling of the surrounding graph. This is supporting evidence
for a scratchpad-pressure hypothesis, not proof of an exclusive causal cost.
Instruction durations overlap within and across engines; they must not be added.
Regular FP32 matmul work is small in both traces and is lower in the candidate;
a return to widespread FP32 matmul is not supported by this evidence.

## Bounded experiments

Wave40 actual-native-module gates pass at8192tokens/1024channels for both
full-row grouped I/O at tile512 and the original I/O at tile256. FP32 master
parameter gradients, cached/uncached paths, two accumulated microbatches,
refresh after a reference update, exact native evaluation and strict checkpoint
load all pass. Both jobs have clean automatic postflights.

Each host now runs a nearby native full-chip control before its candidate.
Keep the loader, optimizer, accumulation, precision,8 measured steps/3 warmups
and postchecks identical. The predeclared advancement gate is at least3percent
total-step saving, not an isolated component win. No30-minute run is authorized
by these correctness results alone.

Wave41 prototype adds two weight residency modes behind scratch-only flags:
mode1 keeps the up matrix and streams down blocks; mode2 streams both matrices.
All transfers read ordinary BF16 rows and transpose on chip, retaining the
previous arithmetic and saved HxT ReLU/backward. At the real shape, mode0 keeps
16MiB of weight caches; mode1 keeps8MiB plus a temporary block; mode2 keeps no
full weight matrix. The tradeoff is repeated weight reads/transposes per token
tile. Actual allocation, spill behavior and end-to-end performance must be
measured; source-level live-byte estimates are only capacity checks.

Gate the streaming variants against FP64 stage references and native compiled
gradients, then actual master/cache/accumulation tests before full-step timing.
A slower isolated kernel may still merit one bounded full-model test when the
explicit hypothesis is reduced caller spill cost. Reject it if that whole-model
measurement fails; do not infer success from a smaller declared buffer alone.

## Primary-source cross-check

AWS explicitly describes the tradeoff between keeping inputs in SBUF and
spilling other live intermediates, and recommends checking oversized blocks and
buffer scope. This supports the experiment design; it does not diagnose our
specific caller graph. [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html).

The current compiler overview describes separately compiled NKI artifacts
inserted late into the graph. Allocation interactions remain a plausible
mechanism, but the page alone does not establish the reservation behavior of
our installed NKI0.5 build. We retain the installed API/SDK and measure the
model; older `nki.compiler.sbuf.alloc` search results are not a reason to use
removed APIs. [NKI compiler overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-compiler.html).

## Compute bounds from the same captured instructions

The recorded128x128 TensorEngine and2.4GHz metadata imply78.643TFLOP/s per
physical core for the BF16-equivalent bound. On core0, native Tensor active time
is18.862ms versus a12.952ms ideal including transposes; fused is27.382ms versus
11.486ms ideal. On core1 the corresponding values are18.710/12.952ms and
27.360/11.486ms. Exact values are in
`run-logs/fs-w40-mlp-compute-bounds.json`.

The candidate has less adjusted transpose work but more transpose active time,
and roughly the same regular matmul work. The Tensor active-minus-ideal gap
therefore grows substantially. This is an efficiency/scheduling lead in
addition to the increased recorded reloads, not an arithmetic-work increase.
Exact memory and complete pipeline bounds are explicitly unavailable because
DMA notifications are incomplete. The maximum measured compute-engine activity
is not substituted for a complete DMA-inclusive pipeline bound.

## Next isolated change: prepack weight layout once per optimizer update

Streaming changes the weight-read/SBUF tradeoff but repeats an on-chip transpose
for every token tile. Mode3 prepares each BF16 weight as
`[contract128, out_blocks, in_blocks, out128]` and streams contiguous stationary
blocks directly. Normal FP32 Parameters and ordinary BF16 caches remain intact;
custom backward uses those ordinary weights and routes their gradients once.
The packed copies are auxiliary values, refreshed after every update, with no
new checkpoint keys. Exact pack/unpack and stale-cache tests are mandatory.

The component probe reports packing cost separately and labels its fused timing
as excluding packing. Actual full-step timing includes each packed-cache refresh
inside `wcache_ms`. An isolated packed-input time cannot authorize promotion.
This layout work may later be reusable in an FP8 island, but mode3 itself does
not change numerical precision or training/evaluation arithmetic.


## Results through18:44UTC and the next causal controls

Streaming-down673.017 ms, streaming-both693.391 ms and prepacked678.042 ms all
lose to the same-host590.315/599.575 ms controls despite passed numerical and
postflight gates. No variant is promoted. Prepacking substantially improves
isolated forward/backward but leaves the full model slow. It weakens the claim
that repeated weight transposes inside the kernel explain the full regression.

Wave43 explicitly places five reusable SBUF buffers (7.125MiB declared extent)
with sequential outer loops. It passes the component gate but is slower there.
A full-model test is justified only as a direct allocation hypothesis test.

The shared Linear source line was additionally grouped by Penguin ID. Three
large groups span most of the graph, so this does not identify a single LM-head
problem. Do not label the5.386GB as LM-head reloads or add overlapping durations.
The source/counter association remains incomplete.

Wave44 will hold the custom autograd, flat inputs and contiguous HxT saved ReLU
constant while replacing only NKI forward with native PyTorch projections. It
also supports selecting individual MLP layers. These controls distinguish a
saved-layout/autograd issue from an opaque-kernel integration issue and a local
per-layer effect from a graph-wide layout/allocation change. No claimed SDK bug
or promotion follows until those measurements exist.
