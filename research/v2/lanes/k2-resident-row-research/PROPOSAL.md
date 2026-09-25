# K2 resident-row routing: conditional feasibility, not a candidate

Research decision, 26 September 2026 IST: **mathematically feasible; native
legality and speed unproved.** Keeping each row's running FP32 accumulator in
SBUF removes ordered-wave accumulator HBM traffic. It does **not** remove the
strict serial dependency for repeated occurrences of that row. Propose only a
small component feasibility test after Supervisor selection, not production
implementation or launch. Frozen K2-A e4d93 remains **REFUSE**; no tolerance,
source, data, evaluator, optimizer, gate, queue or submission changes here.

## Exact source and integration boundary

Base: a4aa4351789115a2d0aa947d9ca00ff689f9bd0b,
`workflow/proposals/research-v2/train.py`, SHA256
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
Its lines 236–238 select VE layers 1/3/5 for six layers; lines 1912–1914
construct WTE plus those three separate FP32 tables. All four look up the
**same idx** at 2230/2239. Metadata can therefore be built once per forward
and shared, but the four gradients/accumulators cannot be combined.
Production is T=32768 tokens/rank/microstep, V=8192, C=1024, pack32/world4.
The existing dense optimizer updates and cross-rank reduction remain dense.

At 3894–3901 the unmodified loader already returns device tensors; fetch only
reshapes them. Immutable `prepare.py:479–482,514–518` keeps the CPU buffers
private and copies to device before yielding. There is **no existing exposed
CPU-ID buffer** available to this train.py-only proposal. Do not introspect the
generator or change its device/data path. Proposed metadata is built from
device idx inside the counted forward: no `.cpu()`, NumPy, `.item()`, Python
data-dependent branches, variable output sizes, or duplicate atomic scatter.
Save fresh metadata tensors for each backward; no mutable cross-batch cache.
Keep original evaluation and checkpoint keys/Parameter identities unchanged.

## One design: stable counting route, then row-owned ordered additions

This is an algorithm specification, not runnable NKI code.

1. With token tiles b of128, produce fixed-size prefix table
   `P[v,b] = count(t < 128*b and idx[t] == v)` and counts `n[v]`.
   Row groups of128 can compare each tile's IDs with their row IDs, reduce
   Boolean counts, and carry integer-valued counts through tiles. Counts never
   exceed32768. Retain a group's128x256 prefixes locally, then store a block;
   avoid a separate four-byte HBM transaction for every row/tile.
2. Form exclusive row offsets `O[v] = sum(u<v, n[u])`, with O[V]=T.
   Scan only bounded nonnegative integer metadata. FP32 can represent every
   intermediate count/offset exactly here, or use exact integer operations.
3. For each token t in tile b, find its within-tile ordinal
   `q[t] = count(k<t within b and idx[k] == idx[t])`.
   Store t at `positions[O[idx[t]] + P[idx[t],b] + q[t]]`.
   These destinations are globally unique and fill exactly0..T-1. Each row's
   position slice is strictly increasing. Equality/ordinal work concerns IDs,
   **never gradient sums**. Use separate device stages/dependencies; no assumed
   global barrier inside a multi-program kernel.
4. One SBUF partition owns one row within a128-row group. For each channel
   half, initialize A[128,512] to positive FP32 zero. Loop k from0 to that
   group's maximum count. For active rows, gather positions[O[v]+k], then
   gather that token's gradient row and perform one FP32 addition into A.
   Keep A resident until all occurrences are consumed; finally store each
   row/channel half once to a fresh dense gradient, including unused zero rows.

For each row/channel the executed additions are exactly its original token
subsequence. Independent rows may interleave; additions within a row may not.
This preserves the reference recurrence by induction, including the retained
`2**24,1,-2**24` and `1,2**-24,-1` cross-tile counterexamples. No tile sums,
tree reductions, higher-precision alternative oracle, or reassociation.
For inactive lanes preserve the old A bits by selection; do not add artificial
zeros to real rows. Native rounding, subnormals, signed zero and compiler
transformations still require the unchanged exactness tests.

## What primary AWS documentation supports—and what it does not

- [DMA copy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_copy.html)
  documents runtime HBM row gather/scatter with per-partition SBUF offsets
  shaped[P,1], and an out-of-bounds skip mode. These match unique routing
  stores and two-stage position/gradient gathers. Inactive indices can use
  a deliberate out-of-range sentinel, with initialized scratch and explicit
  selection. Prove skipped lanes remain safe; never skip actual invalid IDs.
  Descriptor cost and whether this intended path compiles are unmeasured.
- [Dynamic ranges](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.dynamic_range.html)
  allow device runtime bounds. The [language guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html#dynamic-control-flow)
  documents register-driven loops/indexing and loading a1x1 SBUF bound into a
  register. This permits a proposed max-count loop without reading it on CPU.
  Installed-version support, loop-carried dependencies and scratch reuse need
  compilation proof. Do not silently replace it with32768 unrolled bodies.
- [nc_n_gather](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html)
  works within each SBUF partition, not as an arbitrary HBM gather. Do not use
  it to pretend the whole gradient is resident. [tensor_tensor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html)
  gives FP32 elementwise arithmetic, but its tensor-indirection facility is
  NeuronCore-v4+; this Trn2/v3 proposal does not rely on that facility.
- The proposed accumulator is256KiB per channel owner, with about640KiB total
  for A, BF16 gradient scratch and a separate FP32 updated tile, before small
  metadata/compiler overhead. This is about5KiB/partition, versus the documented
  [Trn2 SBUF224KiB/partition](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html).
  Capacity arithmetic is favorable, **not** an allocation/liveness proof.

Documentation consulted25September2026 UTC; no installed NKI import,
compiler invocation or remote capability check. Current public docs do not
authenticate installed SDK support.

### Do not substitute a naive argsort shortcut

The official [argsort API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/library/api/argsort-unstable.html)
is explicitly unstable. Sorting IDs alone can reorder duplicates. Composite
`id*T+t` keys would be unique integers below2**28, but the
[published implementation](https://raw.githubusercontent.com/aws-neuron/nki-library/main/src/nkilib_src/nkilib/experimental/subkernels/argsort_unstable.py)
uses N/8 repeated max8/match passes. [max8](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.max8.html)
converts to FP32 and supports at most16384 source elements/partition. Thus a
direct32768-element call violates its documented bound, and FP32 conversion
collapses some composite keys:268402688 and268402689 become equal. The API's
int32 input label does not solve either issue. A new radix/merge sorter would
be another substantial design; it is not smuggled into this proposal.

## Work, memory and straggler bounds

These are logical work bounds, **not timing, bandwidth or training-ID statistics**.
Let Q=sum over64 fixed row groups of max(n[v]) in that group. Then
256 <= Q <=32768; active fraction is T/(128Q), from100% down to0.78125%.

| Per table/rank/backward | Balanced four occurrences/row | All tokens one row |
|---|---:|---:|
| Ordered rounds/channel owner Q | 256 | 32768 |
| Padded FP32 additions | 33,554,432 | 4,294,967,296 |
| Accumulator HBM traffic | one32MiB final store | one32MiB final store |
| BF16 gradient reads without inactive skip | 64MiB | 8GiB |

With correct out-of-bounds skipping, useful BF16 gradient payload is64MiB in
both cases, but descriptor/control overhead and padded arithmetic remain.
If the actual compiled upstream tensor is FP32, gradient-byte figures double
to128MiB–16GiB without skip /128MiB useful with skip. No dtype/cast change or
BF16 observation is claimed here. The forward gather/cast contract and actual
upstream dtype must be checked before selecting any payload optimization.

By contrast, the earlier ordered-wave proposal needs256MiB–32GiB of logical
FP32 accumulator gather/store **per table**, before its gradients and initial
zeroing. Resident rows remove this term, not all costs. The four dense outputs
still total128MiB/rank; ordinary all-reduce/optimizer work is unchanged.
Each active per-half gradient gather is only1KiB BF16 (2KiB FP32); AWS's
[DMA guidance](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/get-started/about/nki-dma-overview.html)
favors larger transfers, so narrow-packet overhead is a live concern.

Shared routing costs once per forward:8MiB prefix scratch,192KiB+4 bytes
persistent counts/offsets/positions; straightforward counting requires
V*T=268,435,456 ID comparisons plus4,194,304 within-tile comparisons.
This is16384 row-group/token-tile count bodies. Buffering prefix blocks and
runtime loops must prevent scalar-DMA or compile-unroll explosions.
Three metadata stages add launch/dependency/compiler cost. Metadata reuse
avoids paying it four times; it cannot make it free.

A frequent row needs n[v] serial additions even if all others finish. Sorting
rows by frequency may reduce padding but does not remove that dependency;
it is not an additional selected variant. There is no defensible >=5%
whole-step forecast, cold compile estimate or quality prediction from this note.

## Prior attempt is a warning, not a disproof of this design

At retained Git052c83966456e37335dfe42d14fa016e4290944f:
`research/scratch/embedding_segmented_nki.py:17–44` uses stable **CPU** sort
and a scan; :61–74 limits tokens to8192 and allocates token-wide scratch.
`embedding_segmented_dma_autograd.py:59–70` explicitly performs D2H, metadata
creation and uploads, shared by four tables; larger shapes fall back.
Current32768-token support is therefore not inherited.

`research/fresh-start-results-2026-09-18.md:1182–1189` records578.1483ms versus
586.4605ms whole-step (1.417%), below its then gate. The fused versions at
:1214–1219 and :1260–1268 lost despite reduced intermediates; :1240–1248
attributes many tiny metadata DMA packets. Those receipts use other source,
shapes and correctness criteria, not today's byte-exact or5% gate. Resident
rows differ in device-only routing and row-wise persistent state; old results
neither validate nor rule them out. Prior ordered-wave8260806c remains a
separate research proposal, not an implemented baseline.

## Executed evidence and the smallest useful falsifier

Sibling `routing_recurrence_probe.py` is stdlib-only mathematics, T<=384,
C8/V256, ten synthetic cases, seed42, no Torch/model/data/native imports.
It checks route uniqueness/stability, every row's occurrence order, bytewise
FP32 scalar recurrence and both retained failures of grouped sums. Empty,
unused rows, partial tiles, signed zeros and BF16 subnormals are included.
Normal and -O outputs are archived separately. This is **not** actual dense
Torch backward parity, source/CPU G2, independent verification or native PASS.
Peak tracked Python allocation remains below4MiB; actual receipt gives bytes.

After explicit selection, first falsify the consumer alone with supplied
synthetic device metadata at V256/C128/T256: counts0/1/3/253, both cancellation
cases, inactive skip and repeated calls. Require exact native and compiled
autograd results against the retained dense CPU oracle; inspect actual dtype,
loop-carried A, output writes, cold cost and graph count. Any rounding,
unsupported-loop or spill failure parks this form; no tolerance waiver.
Then test device metadata separately for identical positions/counts/offsets,
including boundary IDs and partial/empty cases, before one end-to-end component.
This split distinguishes arithmetic/indexing from routing/compiler failure.

Only normal independent source/default-OFF, CPU/native/actual-model gates can
precede a matched whole-step comparison with **all routing charged** and the
unchanged >=5% rule. Keep cold accounting/CT limits and fixed1800 quality
budget. No new gate, launch assignment or automatic full run is created.
Sole Operator retains shared flock plus in-lock ps/neuron-ls preflight.

Night-mode log: research-only decision; reason is a genuine traffic reduction
with unresolved routing/straggler/compiler cost. Undo: decline the proposal;
no production or remote state was changed. Supervisor owns the canonical log.
