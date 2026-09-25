# K2 redesign: ordered unique-ID waves — research only

Decision at 2026-09-25 20:05 UTC / 26 September 01:35 IST: **retain e4d93
REFUSE; propose one arithmetic-feasible replacement, not implementation or
launch approval.** Exact dense CPU FP32 token-order accumulation is compatible
with parallelism across different rows, but incompatible with the rejected
one-sum-per-tile mechanism. There is no measured or predicted >=5% whole-step
win. Wait for the actual P bottleneck evidence before spending a native slot.

## Evidence binding and scope

- Adopted base: commit `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`,
  `workflow/proposals/research-v2/train.py` SHA256
  `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
- Rejected candidate, kept frozen: commit
  `25b731f7638672aa8d03a7cd15b6a02b8f5776ea`, source SHA256
  `e4d93babb4cec8ce53ee1ef9f9d4cc4238c40b93dff0245b66fbcf6bae158b03`.
- Fresh independent review:
  `research/v2/verification/20260926-fresh-astra-k2a/session/reviewer-output/REVIEW.md`,
  SHA256 `1cc7d41568e1bef6a047de1c4e3f062e9b8439fc3dc1ab29ead62901100c72c8`;
  sibling `reference_exactness_test.py`, SHA256
  `24422d083a090a3d92759bbda935a24a13dc73b909f003d91350f4ac4f9d6d84`.
  Both were read completely. The reviewer found six exactness failures in
  native-algorithm CPU emulations, not six hardware failures. The actual CPU
  Function's `index_add_` tests passed and do not validate the NKI algorithm.
- This branch changes no production source, candidate, evaluator, data,
  optimizer, launch machinery, queue, ledger, thresholds or submission.
  The only executable addition is an independently written tiny CPU research
  recurrence probe. No candidate or native implementation exists here.

## Why an FP32 grouped sum cannot fix the current candidate

In rejected source lines 1945–1978, each tile computes `equal.T @ values`
(line 1970), then adds that FP32 tile result to the incoming row (line 1977).
The CPU branch at line 2008 instead adds individual tokens into the row.
These are different floating-point evaluation orders.

For one row, let consecutive nonzero BF16 inputs at token positions 0, 128,
129 be `2**24, 1, -2**24`. Sequential FP32 gives
`round(round(2**24 + 1) - 2**24) = 0`. Summing tile 1 from zero gives
`-16777215`, then adding tile 0's carry gives `1`. With inputs
`1, 2**-24, -1`, the corresponding wrong result is `2**-24` instead of zero.
The independent reviewer also used a sequential sum *within* each tile; it
still fails, isolating the cross-tile carry problem from matmul reduction.

Thus a tile's zero-initialized FP32 sum is not a sufficient representation of
its effect on arbitrary incoming FP32 state. Promoting inputs or storing the
tile sum in FP32 does not repair it. A float64/exact sum rounded once would
also compute a different contract. Do not change the oracle or tolerance.

## The one proposed design: ordinal waves, carrying the real row each time

Retain 128-token tiles and two disjoint channel owners, but remove gradient
coalescing entirely. For tile positions `j = 0..127`, compute the integer
occurrence ordinal

`ordinal[j] = count(k < j and id[k] == id[j])`.

This counts Boolean metadata, not gradients; values 0..127 are exactly
representable. Let `M = 1 + max(ordinal)`. Process tiles in original order and
waves `w = 0..M-1` in order:

1. Active positions have `ordinal[j] == w`. Their real destination IDs are
   unique: a given ID has only one occurrence at a given ordinal.
2. Send inactive positions to distinct private dummy rows `V+j`; select zero
   updates for them. Use selection, not multiplication by zero. Do not add
   synthetic zero updates to inactive *real* rows.
3. Gather each active row's **current FP32 state**, add just that token's
   BF16-upstream value promoted to FP32, and write FP32 back before the next
   wave. Do not form a per-tile sum. Dummy rows remain separate from real rows.
4. The next tile starts from the final real rows of the preceding tile.

For a particular row, this performs exactly the same ordered list of FP32
additions as the dense token-order reference. Moving another row's independent
addition earlier or later cannot affect it. This is an induction over the
ordered occurrences of each row, including tile boundaries. It does not
assume FP32 associativity. It preserves the intended arithmetic *provided the
native addition's rounding/denormal behavior matches the reference and its
read/write dependencies really serialize*. Neither condition has been proven
on hardware here.

Empty input returns fresh FP32 zeros; unseen rows remain positive zero.
Each backward returns a fresh dense gradient; ordinary autograd accumulation
into `.grad` and optimizer/ownership code stay unchanged in any future port.
The forward remains the existing FP32 embedding gather followed by its BF16
output cast. This proposal is only for the four ordinary WTE/VE tables in
the exact 6x1024 base, not tables, architecture, FP8 or new optimizer math.

## NKI feasibility versus unproved compiler/runtime details

AWS's immutable [unique-index scatter example](https://github.com/aws-neuron/nki-library/blob/f6db353a2cc5b10f355ab5f36725e65034aab527/src/nkilib_src/nkilib/experimental/misc/scatter_add.py)
provides a gather/add/scatter construction. The proposed waves satisfy its
essential unique-destination condition *per round* without summing duplicate
gradients. This is a plausible implementation basis, not an installed-kernel
or performance claim.

AWS documents FP32 arithmetic for [elementwise tensor addition](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_tensor.html).
That alone does not promise byte parity with CPU for subnormals, overflow,
signed zero, NaN payloads, compiler transformations or an incorrectly ordered
memory dependency. Those are explicit native correctness risks. No NaN or
nonfinite acceptance is inferred from the finite CPU probe.

[Dynamic loops](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.dynamic_range.html)
and the [register-controlled loop mechanism](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html#dynamic-control-flow)
offer a way to execute `M` rounds on device without specializing on each input
ID distribution. A max-ordinal reduction, a 1x1 value transferred to a scalar
register, and a runtime loop are the intended route. Compatibility with the
installed compiler, indirect-DMA dependency tracking, layouts and two-program
execution is **unknown**. Do not assume a 32,768-body static unroll is an
acceptable fallback if that route is unsupported; stop and report it.

The proposed baseline uses explicit gather/add/store, not an assumed atomic
duplicate scatter. AWS's [DMA read-modify-write documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.dma_compute.html)
requires unique indices for indirect RMW. It does not provide an unordered
duplicate-update primitive that would satisfy token-order semantics anyway.
Public documentation was checked on 25 September 2026; this is not a host
capability check, and no API package was installed or imported.

## Honest work/compile bound; why this may be too slow

For production `T=32768, C=1024, V=8192`, there are 256 tiles per table and
`1 <= M_tile <= 128`. Bounds below describe the straightforward *padded*
wave design, combining both disjoint channel programs, not measured timings:

| Quantity per table / rank / backward | Best distribution | Worst distribution |
|---|---:|---:|
| Ordered round count per channel program | 256 | 32768 |
| Logical FP32 gather+store payload, including dummy rows | 256 MiB | 32 GiB |
| Padded elementwise additions | 33,554,432 | 4,294,967,296 |

Each round reads and writes 128x1024 FP32 elements: 1 MiB combined. Across
the four ordinary tables the nominal gather/store volume is therefore
1–128 GiB per rank, excluding the 64 MiB BF16 gradient read per table, result
initialization, IDs, metadata and runtime overhead. This is logical requested
traffic, not a hardware-bus measurement; no cache behavior is assumed.
Within a tile the loaded gradients/metadata can be reused across rounds.

All-unique IDs need one round; all-same IDs need 128. The CPU probe has 5/4/5
rounds for three synthetic uniform-random tiles, 12 each for an 11-ID cycle,
and 127/126 in the reviewer's cancellation layout. These are synthetic examples,
**not estimates of training-token statistics**. No training data was read.

The rejected tensor-engine grouping's one-round advantage is lost whenever
duplicates occur. Dynamic loops avoid intentional input-dependent graph
specializations but add control and dependency cost. Real kernel compile time,
steady latency, native graph count, SBUF pressure and full-step savings are all
unknown. There is no defensible >=5% forecast. The time saved is bounded by
the old embedding backward's contribution to the critical whole step, even if
its replacement were free. Use the existing gate's exact time/throughput
comparator, not an isolated component percentage, to judge the 5% target.

## Why not relaunch the older stable-sort/scan path

The prior code is retained in Git at
`052c83966456e37335dfe42d14fa016e4290944f` (not present in the current tree):

- `research/scratch/embedding_segmented_nki.py:17–44` uses stable CPU ID sorting
  and a literal recurrence; lines 61–63 restrict its kernel to `T <= 8192`.
- `research/scratch/embedding_segmented_dma_autograd.py:59–70` counts D2H,
  CPU metadata and uploads in each timed preparation, but silently selects
  the original path beyond 8192 tokens. The production 32768-token shape is
  not covered by that implementation.
- `research/fresh-start-results-2026-09-18.md:1182–1189` records whole-step
  578.1483 ms versus 586.4605 ms (1.417%), including metadata preparation;
  it failed its then-declared advancement gate. Lines 1214–1219 and 1260–1268
  retain the fused-path regressions and narrow-DMA explanation.

AWS [scan documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor_scan.html)
describes a recurrence with a supplied initial state and FP32 math. A correctly
ordered scan is not ruled out mathematically, but a new tiled implementation
must feed the incoming carry *into* the recurrence, not add it after a
zero-initialized scan. The old code/receipts do not establish today's exact
native parity, production-shape support or a >=5% whole-step gain. Reusing it
would be a separate researched implementation, not a small repair accepted by
this proposal. It is not a second proposed candidate here.

## Executed bounded evidence and next falsifier

The fresh batch reviewer explicitly released the CPU slot at 19:59:28 UTC.
One single-thread process then ran `recurrence_probe.py`; it exited 0, and
the slot was released to the Verifier immediately after exit. Exact command:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/poojith-devan/venv/bin/python research/v2/lanes/k2-ordered-research/recurrence_probe.py --out research/v2/lanes/k2-ordered-research/recurrence_result.json
```

PyTorch 2.13.0+cpu, Python 3.14.4, seed42, T<=384/C=8/V<=512. Ten cases
compare byte-for-byte against an explicit token loop and actual CPU
`F.embedding(...).bfloat16().backward(...)`. They also check every tile-prefix
state, each row's occurrence sequence and unique round addresses. Cases include
both retained reviewer counterexamples, all-same/unique/interleaved/random
IDs, signed zero, BF16 subnormal inputs, noncontiguous inputs and empty input.
Both old grouped-sum counterexamples are deliberately retained as unequal.
Raw structured results and exact probe hash are in `recurrence_result.json`.

This is **not** a production source check, autograd-wrapper check, independent
review, full117M CPU G2 or native result. No candidate was modified or imported.

After explicit proposal acceptance, the smallest useful native falsifier would
be one tiny two-tile compiled component covering the two retained cancellation
cases, loop bounds 1/2/128 and the real initial-state dependency. Require the
existing zero-tolerance contract, correct gradients, unique addresses and
two-program channel ownership. Capture compiler diagnostics/cold cost and fail
closed if loops or dependencies are unsupported. It must not be relabeled a
full-model or speed PASS. This is a proposed use of the existing component gate,
not a new gate or a launch request. Only after its normal independent source,
CPU/native and actual-model gates could matched whole-step timing test the
unchanged >=5% criterion. Every native action remains the sole Operator's,
under the existing shared host lock, fresh idle checks and official accounting.

Night-mode decision: preserve the refusal, record one reversible research
proposal and stop before production implementation. Reason: exactness failure
has an identified arithmetic cause, while speed viability is unknown. Undo:
decline this proposal; no production or remote state needs reverting. Supervisor
may summarize this timestamped decision in the canonical decision log.
