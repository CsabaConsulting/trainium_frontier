# Source197 offline attribution

Date: 2026-09-22  
Role: Analyst  
Verdict: **caller-boundary constraint identified**  
Run status: **No run**

This packet analyzes preserved Source197 evidence only. It authorizes no code change,
remote operation, training launch, candidate merge, or leaderboard submission.

## Question

Why does replacing exactly one native MLP with the same NKI-library MLP produce a
near-control result in layer 0 but a severe regression in layer 5?

The immutable Source195 timing evidence is:

| Placement | Step mean (ms) | Microstep mean (ms) |
|---|---:|---:|
| Native controls | 499.108 / 498.149 | — |
| Library layer 0 only | 496.617 | 433.156 |
| Library layer 5 only | 682.661 | 627.421 |
| Library all six layers | 564.512 | 507.779 |

All variants use source commit `282935ca9b1e30f6c21c397143770b03034363f9`,
four ranks/LNC2, seed 42, sequence 1024, pack factor 32, one selected library module
of the same shape, and the original BF16 custom backward in all six layers.

## Evidence integrity and method

- Raw first/last profiles and derived Parquet archives were downloaded from the
  hash-verified private research backup. All four archive SHA-256 values match their
  committed backup receipts.
- Every selected Parquet file matches its preservation receipt.
- Every retained NEFF matches the SHA-256 in its captured `neffs/manifest.json`.
- Analyzer: `.worktrees/fresh-start/research/scratch/model_caller_profile_analysis.py`
  at authoritative research commit `9a4caa7b`; SHA-256
  `0b7c1bc7ec4313b79b789ca95f8ec589e0ea1108cca85fda82fe0118a0b0255d`.
- Derived reports were produced locally with no remote query or machine operation:
  first report SHA-256 `7e5019745560d2ddfb9085625f9511d2d968ff2d84fd8274304c4b91cb562b37`;
  last report SHA-256 `2043cbf76ec73756d040e4de85dbadef57912f6ee303d6102dfa2bd83503511c`.

Both captures account for exactly two device executions: one preserved startup
`c10d::barrier` and one `model_default` execution enclosed by
`fresh_start_single_training_forward`. The runtime name, sequence ID, and two-subgraph
count match. The main compiled NEFF identity differs, as expected for the different
layer selection; there is no extra runtime dispatch in the slow case.

## Main graph comparison

| Metric | Library at layer 0 | Library at layer 5 | Change |
|---|---:|---:|---:|
| Forward device wall | 135.871 ms | 329.259 ms | **+193.388 ms** |
| Main NEFF size | 27,498,742 B | 44,664,789 B | +62.4% |
| Instruction rows | 1,961,096 | 3,773,438 | +92.4% |
| Spill saves | 9.137 GB | 10.002 GB | +0.865 GB |
| Spill reloads | 13.675 GB | 30.019 GB | **+16.344 GB** |
| HBM reads | 23.242 GB | 38.860 GB | +15.617 GB |
| HBM writes | 22.025 GB | 22.856 GB | +0.831 GB |
| Transpose instruction rows | 289,985 | 866,287 | +198.7% |

The forward-device delta explains **99.55%** of the unprofiled microstep delta:
`193.387852 / (627.420897 - 433.155515)`. This is not an exact causal accounting of
the full step—the capture is rank-0 forward under four-rank execution—but it localizes
the material regression to the compiled forward graph.

## Custom work versus surrounding work

The custom library family is not large enough to explain the regression:

| Source family | Layer 0 placement | Layer 5 placement |
|---|---:|---:|
| Library MLP instruction rows | 80,117 | 104,703 |
| Library MLP spill reloads | 305.922 MB | 309.854 MB |
| Library MLP Tensor activity, interval unions across cores | 9.309 ms | 12.431 ms |

In contrast, ordinary compiled work expands sharply:

- Native BF16 MLP-source instruction rows increase from 371,025 to 1,145,204.
- Its Tensor-engine interval unions across cores increase from 44.943 ms to
  293.139 ms.
- `mlp_fused_bf16.py:290` alone changes from 22.511 ms to 227.625 ms of Tensor
  interval unions across cores.
- The shared `F.linear` site at `train.py:2853` changes from 0.585 GB to 13.502 GB
  of spill reloads and from 30.522 ms to 90.149 ms of Tensor interval unions.
- Total transpose rows nearly triple, while private NKI transpose activity remains
  tiny (0.059 ms versus 0.096 ms). The extra transpose work is therefore not explained
  by the private custom transpose alone.
- Sampled peak SBUF utilization is slightly lower in the slow graph, so the samples do
  not support a simple global-SBUF-saturation claim.

Source/engine interval unions overlap and are not additive wall-time shares. The shared
`train.py:2853` location also covers multiple linear calls and does not by itself name a
specific layer. Nevertheless, the scale separation is decisive: the selected custom
family changes by only a few milliseconds, while ordinary BF16 MLP/linear scheduling,
spills, and transposes expand by hundreds of milliseconds.

## Verdict

**Caller-boundary constraint identified.** Moving the same library call from layer 0 to
layer 5 causes the compiler to produce a much larger and slower surrounding forward
NEFF. The slow placement does not add a dispatch, and the direct custom-library work is
not the dominant delta. The regression is a position-dependent whole-graph compilation
and layout/spill specialization effect affecting ordinary native MLP/linear regions.

This identifies the class and location of the constraint, not a proven source-level
repair. Exact physical traffic cannot be claimed because both captures report dropped
DMA notifications and missing dynamic-DMA metadata.

## Decision consequence

1. Keep complete saved ROW-FP8 backward implementation **on hold**. A faster backward
   would not remove the demonstrated forward caller-boundary penalty.
2. The next proposal should be one bounded, compile-only/correctness-first boundary
   experiment derived from this packet. It must keep one causal factor and determine
   whether a stable graph boundary/layout can prevent recompilation of ordinary native
   MLP regions.
3. Do not assume that adding `contiguous()` is the fix. The proposal must connect its
   exact boundary transformation to the affected graph and define a same-host complete
   forward/step gate.
4. If no boundary variant restores ordinary BF16 MLP/linear behavior, abandon the
   library integration path and return to candidates that do not perturb the monolithic
   compiled graph.

## Capture limitations

- Rank-0 training forward was captured while all four ranks executed; backward,
  optimizer, and collective attribution are outside this packet.
- Profiled latency is diagnostic, not a replacement for the unprofiled speed result.
- Dropped DMA notifications prohibit exact bandwidth or physical-traffic conclusions.
- Source families and engine intervals overlap; no family totals were added to claim a
  wall-time decomposition.

