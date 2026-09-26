# P3 component results — 26 September 2026

Recommendation: investigate the existing **BF16 MLP backward activation/layout
boundary** (saved-ReLU square, `2*relu` derivative and materialization alongside
the matrix products). This is the best-supported allowed kernel direction from
these records, not a measured optimization. Do not select a flash-attention
rewrite on the anomalous isolated SDPA forward number alone. No component is
an exclusive production-graph cost, and no whole-step saving is predicted.

## Evidence and integrity

Operator result commit `c968d7e3`, label `fs-20260926-p3-component-trn22-r1`.
Archive: `research/v2/operator/20260926-p3-component-timing/trn22-completed/results-bundle.json`,
SHA256 `e1ac184482c58da07b7ed7053810b82f5110e18ab37c1c66df8a5fab62294d64`.
All 30 captured text-file hashes and sizes match. Archived and local source
both hash to `3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99`.
All four ranks report `COMPLETE_COMPONENT_ONLY`, six records each, and finite
checked outputs/gradients. Both diagnostic and outer process exit codes are 0.
Per-rank diagnostic elapsed time is 321.4718–321.4725 seconds; the outer job
exited at 09:20:05 UTC. Retrieved postflight at 09:23:35 UTC reports all observed
processes absent, device idle and original lock inode 832589 available.

Geometry is BF16 B32/T1024/H8/D128, 32768 tokens/rank, C1024, hidden4096,
world4/LNC2, seed42. SDPA Q/K/V and upstream strides are
`(1048576,128,1024,1)`, matching the transposed BTHD boundary. Compiler flags
are `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.

## Recomputed rank medians

Milliseconds per invocation; each rank median uses four block means, each of
three calls after two warmup blocks. Slowest rank is the appropriate contention
summary. For explicit backward decompositions, the JSON `forward_*` field
times invocation of that backward-stage callable; it does not mean model forward.

| Component/stage | Rank 0 | Rank 1 | Rank 2 | Rank 3 | Slowest |
|---|---:|---:|---:|---:|---:|
| SDPA forward | 5642.281 | 5671.667 | 4186.421 | 4118.859 | 5671.667 |
| SDPA backward | 7.247 | 7.192 | 7.178 | 7.241 | 7.247 |
| Complete original MLP forward | 6.572 | 6.603 | 6.631 | 6.620 | 6.631 |
| Complete original MLP backward | 11.040 | 11.007 | 11.051 | 11.028 | 11.051 |
| ReLU/square/saved transpose | 3.309 | 3.304 | 3.344 | 3.326 | 3.344 |
| Saved square/derivative | 3.640 | 3.658 | 3.653 | 3.656 | 3.658 |
| Two forward matrix products | 5.583 | 5.647 | 5.644 | 5.611 | 5.647 |
| Four backward matrix products | 10.489 | 10.435 | 10.504 | 10.473 | 10.504 |

## The SDPA anomaly and timer interpretation

There is **no demonstrated units, median, or stale-output timer bug**. Source
lines 1406–1452 synchronize/barrier before each block, start `perf_counter`
after that boundary, synchronize at the end, and record
`1000 * elapsed_seconds / 3`. First forward/backward calls are separate and
excluded. The backward closure refers to the current output, not the initial
output; each backward block reuses the last forward graph with `retain_graph=True`.
Thus the large forward entries really denote 4.12–5.67 seconds per isolated
invocation, not 4.12–5.67 milliseconds or a block total mislabeled per call.

The four SDPA forward block means are stable within each rank: full ranges
are only 0.17–0.41% of their rank median. Ranks 0/1 are persistently slower
than 2/3. First SDPA calls took 73.15–74.82 seconds forward and 27.99–28.85
seconds backward, including compilation and execution. Removing a single cold
sample cannot explain the remaining forward latency.

This cannot be treated as attention's cost inside the successful e1e2 training
graph: one isolated forward call is already much longer than the parent's
entire observed 0.491588-second optimizer-step interval. P3 compiles standalone
callables with synthetic leaf/view inputs and retained-graph autograd; the
production model compiles a larger graph. Graph partitioning, lowering,
host dispatch/graph management, allocation or transfer behavior can differ.
Those are explanations to investigate, not identified causes.

The archive contains no graph identities, compiler counters, device-event
durations or per-call compile traces that identify the cause. Its log has no
explicit recompile alarm; one 09:17 live snapshot saw no compiler process.
Neither establishes that every timed call was compilation-free. Similarly,
the source calls the inherited conditional `torch.neuron.synchronize` helper,
but does not record an independent synchronization capability check. No
observed failure of that helper is established. Torch's `2.11.0+cpu` version
suffix alone is not evidence of CPU fallback on this Neuron stack.

Retain SDPA forward as an observed standalone host-latency anomaly and exclude
it from production-hotspot ranking. SDPA backward remains a repeated-saved-
graph component measurement, not an independently proved device-only cost.
The anomaly does not establish a source defect that invalidates the five MLP
component records.

## Kernel direction and limits

The MLP records are stable across ranks and distinguish a substantial isolated
activation/materialization path from the matrix products. They support a
focused exact BF16 saved-ReLU/derivative/layout kernel investigation in the
existing MLP backward. Include its interaction with matrix consumers; an
extra standalone activation kernel could lose compiler fusion. The complete
MLP is already faster than adding its separately materialized matrix and
activation measurements, demonstrating why subtracting these numbers or
multiplying by eight layers does not yield recoverable time.

No new component, source change, benchmark, review gate or launch is introduced
here. Existing CPU/source review plus finite native outputs do not establish
native bit-exact gradients, integrated speedup, cold full-run feasibility or
BPB improvement. P3 neither repairs nor qualifies K2: the separate K2 retry's
unbound-`tokens` failure remains a failed native gate, with no timing/BPB pass.
