# Library buffer placement and surrounding Linear work

Prepared 2026-09-20. This is a falsifiable compiler-placement hypothesis, not
an accepted optimization. Source192 preserves the training MLP arithmetic,
all ten source175 fixtures and every original numerical bound.

The paired source180 profiles match preserved binaries to runtime dispatches.
For regular MATMUL instructions at train.py:2853 on core0, native executes
55,059 instructions / 0.921T FLOPs, with 15.320ms active time. The joined
library graph executes 110,286 / 1.003T, with 44.897ms active time. The dominant
native work unit is 16,777,216 FLOPs; the largest candidate count is 66,548
instructions of 4,194,304 FLOPs. Candidate Scalar COPY adds 20.977ms active
time at that source line. These intervals overlap and cannot be added as
wall-time shares. FLOP counts alone do not identify individual M/N/K tile
sizes, and this source attribution does not identify individual Linear modules.
Dropped DMA notifications prohibit exact traffic/bandwidth claims. Reproducible
analysis is in scratch/model_linear_spill_analysis.py and its source/hash-bound
run-logs/fs-w180-linear-spill-analysis-r1.json result.

Source190 tests 64/128KiB fixed allocator budgets first. Source192 separately
uses the library's existing automatic SBUF placement branch. The copied
allocator keeps its virtual budget, stack/heap accounting, lifetime checks and
underflow checks. Its only behavioral edit makes both OOM checks remain active
in automatic mode. The copied CTE entry changes only the allocator class and
passes use_auto_alloc=True. Original tiling, PSUM banking, token-row sharding,
activation saves and complete BF16 backward remain unchanged. Inverse AST
proofs are in run-logs/fs-w192-compiler-sbuf-ast-proof-r1.json.

The four-step finite plan will gate the full component, actual cached/uncached
Linear integration, native model control and candidate model timing, in order.
All gate failures stop the sequence. Same seed42, batch131072, pack32, four
ranks/LNC2, optlevel1, five warmups and thirty measured steps. Component speed
is diagnostic; this hypothesis concerns the surrounding model. A complete-step
win must survive replication and existing numerical gates before a quality run.
Automatic SBUF with the retained PSUM allocation still needs device validation
on the installed NKI0.5 toolchain; no compatibility is assumed from older docs.

## Review of the supplementary web research

Tavily pro request c1633403-6c4e-43b4-bf9c-43e2add85fb5 completed. Its raw
synthesis is preserved in run-logs/fs-w192-sbuf-research-raw.json. It supplies
leads, but contains errors and nonprimary references; those claims are rejected.

- The official [allocator reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/library/kernel-utils/allocator.html)
  documents use_auto_alloc as compiler address assignment. It documents scope
  management and explicit bounds. This agrees with the pinned source. It does
  not prove that an NKI allocation changes neighboring Linear placement.
- The raw report's claim that PSUM-to-SBUF copies are unsupported is false for
  the operations used here. The official [tensor_copy API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html)
  supports these buffers through Vector/Scalar engines, and our validated
  kernels already use this transfer. GpSimd cannot access PSUM.
- The official [Explorer troubleshooting page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/troubleshooting.html)
  documents missing DGE notifications and their effect on DMA metrics. It also
  warns that enabling them can time out on large binaries. Keep the existing
  warnings rather than silently treating the counters as complete traffic.
- The raw report treats third-party compiler reverse-engineering notes as
  authoritative and misnumbers some references. We do not use those claims.
  Observing -O2, --optlevel=1 and an internal walrus_driver --optlevel 2 at
  different stages does not establish which requested optimization was applied.
  No installed compiler patch or flag sweep is justified by that observation.
- Old ncc.sbuf.alloc documentation predates the installed NKI0.5 address API.
  It cannot establish the compatibility of this mixed SBUF/PSUM experiment.

The research supports the allocation experiment, but supplies no established
causal explanation of the source180 regression. Use measured counterfactuals,
not a synthesized compiler story, to decide whether to continue this path.

## Conditional follow-up: isolate how many replacements cause the regression

If source192 also loses, the next causal test is placement/count of the library
replacement: zero, first block only, last block only, and all six. Use the same
source, host, seed, batch and native HxT custom-autograd control in every block
that is not replaced. Leaving those blocks on ordinary MLP.forward would change
the saved-activation/backward contract and confound this comparison. Preserve
Parameter/cache identities, native evaluation and optimizer grouping. Gate the
partial-install adapter separately before any model timing.

A nearly fixed slowdown after a single replacement would support a graph-wide
compiler/layout effect; a cost that scales with replaced layers would support
a local boundary or per-layer cost. First-versus-last placement can further
separate a downstream head/layout interaction from a uniform cost. These are
falsifiable predictions, not explanations established by the current traces.
The shared train.py:2853 attribution cannot identify the responsible module.
If profiling a resulting difference, bind runtime executions to NEFF identities
and compare instruction work as well as elapsed time. Do not infer causality
from incomplete DMA counters or relax existing numerical gates.

Source193 also prepares a same-host three-arm forward capture (native BF16,
prepared-input ROW FP8, and fused-input ROW FP8). Its function bodies are
extracted unchanged from the source191 transform. All input/master-weight
preparation stays within each compiled call, and captured outputs must equal
unprofiled outputs. The full source191 gate is an explicit prerequisite.
This profile can identify remaining preparation/compute costs; it cannot
justify training quality or a full-model speed claim.
