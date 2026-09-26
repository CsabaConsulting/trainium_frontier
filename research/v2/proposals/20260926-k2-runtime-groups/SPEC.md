# Prospective K2 compile-size fallback — off-host preparation only

Supervisor selection, 26 September2026,10:15UTC. FINAL SPRINT/K2 priority;
no change to running source662c/native201, no native launch or retry release.

## Evidence and hypothesis

The current K2 compiler remains alive and CPU-active after900 startup seconds.
Archived `operator/20260926-k2-native201-handoff/trn21-live-compiler-progress.json`
shows sg00 with1026 blocks/1,895,470 instructions/694,828 allocations at
extend_shared_lifetimes. It does NOT attribute all of these to K2, or prove
failure. Exact662c `_k2_build_kernel` has64 statically unrolled row-group bodies.
The separate414c table kernel uses a real runtime group loop with grouped HBM
reshape/select, and has passed the actual compiler and full-shape simulator.

Hypothesis: preserving the row operations but making the64-group outer loop
dynamic can reduce generated code and full-model cold compilation without
changing the result. Whole-step throughput can improve, remain flat or worsen;
unknown until measured. Do not claim a speedup or shortened startup from a
smaller local NEFF alone. This is a source-geometry compile optimization inside
K2, not reopening closed architectures or changing the official clock.

## Exact permitted delta from frozen662c

New derivative only: `research/v2/lanes/k2-runtime-groups/train.py`.
Keep fixed V8192,T32768,C1024,P128,F512,G64,LNC2 and every FP32 operation.
Change `for group in range(groups)` to `nl.dynamic_range(groups)`.
Remove Python `first=group*partition`. Use the tested grouped reshape/select
accesses for counts, starts, group_maximums and output, with the ORIGINAL
K2 geometry and512-channel owner halves. No other function/global/metadata/
registration/forward/optimizer/schedule/data/eval change. Preserve order of
groups and within-row encounter order, full dense writes and zero untouched
rows. No parallel atomics, dtype change, alternative gather or unrolling.

Deliver full diff, frozen source SHA, actual reduced CPU/OFF receipts against
accepted parent and exact numerical reference, one real production backward
compile and available official simulator on existing full-shape fixtures.
Use already accepted unchanged framework-forward evidence; no new local
framework-forward test or second native smoke is introduced. One existing
narrow composition/review covers the actual delta, not repeated review cycles.

One laptop heavy process at a time; wait for the stack implementer's explicit
release before any Torch/NKI/compiler/simulator work. Static preparation is
allowed in parallel. Use existing resource-limited isolated compiler tooling;
no host/package/AWS changes. Preserve failed/older candidates and all receipts.

## Native scheduling is NOT authorized here

Complete current201-step test and use its actual startup/rate/loss result.
No cancellation/restart merely because startup is slow. This derivative stays
disabled until root selects it from actual evidence and completes its exact
proposal/command/control binding. Normal original-lock, source/resource checks,
official1800 accounting and unchanged selection rules stay. It must not
silently replace an already-released full or table packet. Current K2 full,
table preparation and trn22 confirmation continue.
