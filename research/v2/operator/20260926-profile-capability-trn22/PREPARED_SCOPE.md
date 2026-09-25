# Lane P: installed capability and bounded handoff, not a runnable job

Human ALL IN directive requests one steady-state whole-step profile of current
Recursive6ef7 and isolated8x1024 after the architecture decision, by Sep26
06:00 IST. Operator performed only CLI help and installed-source reads.

## Exact profiles to bind after Implementer review

- Current best base: source6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3
  from a4aa4351789115a2d0aa947d9ca00ff689f9bd0b, dense6x1024, Recursive ON,
  matrixLR.015, seed42, four ranks/LNC2, batch131072, context1024.
- Architecture: original93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5,
  eight layers1024 with partial attention0,2,4,6,7 and U-Net, no Recursive.
  If the sole initial-cache rerun becomes the chosen architecture, bind exact
  3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c
  and its explicit flag; never silently profile another source or call it93ae.
- Prepared unique labels: fs-20260926-profile-recursive6ef7-trn21-r1 and
  fs-20260926-profile-architecture-trn22-r1. Neither is staged or READY.

## Installed interface findings

neuron-profile exists but all help invocations return exit1 and say the tool
was removed in favor of neuron-explorer. The latter is installed2.32.0.498;
its help invocations return0. No installation or SDK change is needed.

`neuron-explorer capture` executes a specified NEFF; this alone is not the
complete model forward/backward/optimizer/collective/data step. `inspect`
wraps a whole workload and has output-directory/event-count options, but its
help has no selector for exactly one steady-state training step.

The installed torch_neuronx.profiling API already provides NeuronConfig,
DEVICE/RUNTIME modes, max_events_per_nc, capture_enabled_for_nc,
profile_output_dir, precache_node_info and NeuronProfiler.export_trace for
use with torch.profiler.profile and a schedule. Source and hashes are archived
here. This is a capability finding, not proof the requested run works.

Existing research/scratch/prof_driver.py demonstrates this API, four-rank
execution with rank0-only capture and an unprofiled timing contrast. Its old
synthetic reconstructed training loop does NOT establish equivalence to the
current exact6ef7/93ae main, OwnedOptimizer, clip-after-reduce, cache refresh,
or schedule. Its ablation monkeypatches must not be enabled for Lane P.
Reuse reviewed capture/export logic, not unexamined legacy model construction.

## Required bounded diagnostic packet

An Implementer must supply exact-source instrumentation/binding and an
independent reviewer must verify normal model/gradient/optimizer semantics,
all four ranks, one bounded capture after graph compilation is complete,
rank-safe trace output, full startup/timeout accounting, fixed memory/disk
headroom, and no second launcher. A few unprofiled steady steps give the
throughput denominator; profiling overhead is not a speed win.

The packet must identify how it attributes forward, backward, owned optimizer
and Newton-Schulz, collectives, embedding backward and data. Overlapped phase
times cannot simply be summed. Unknown/unattributed intervals stay unknown.
Record actual matmul/SDPA dtypes separately from the FP32 residual/master
weights and intentional optimizer precision; do not alter precision to make
the audit pass. An MFU estimate must name its FLOP definition and verified
hardware peak denominator, and use clean timing, not the traced duration.

No profiler capture, workload launch, candidate code edit, SDK change,
runtime option change, AWS action or extra quality run was performed here.
This preparation does not create a runnable profile queue entry.
