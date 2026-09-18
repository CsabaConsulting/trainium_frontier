# Reopen optimizer ownership as a compute question

Status at 21:04 UTC: exact FP32 communication feasibility passed on all four
ranks, worst-rank median6.403158ms including packing and copyback. A scratch
ownership implementation is ready for CPU/Gloo and native numerical gates.
It has not yet produced an optimizer or whole-model timing result. 2026-09-18.

The older assessment in `trn2-optimization-research.md:295` and
`experiment-rollout-plan.md:295` rejected ZeRO-1 because optimizer memory fits.
That does not answer whether avoiding four copies of optimizer computation is
profitable. The current same-host native-MLP/pack8 control measures 55.61ms of
optimizer time in a 558.63ms step. Gradient averaging already occurs before
every optimizer step; all four ranks then compute the same updates.

The four compiled Muon groups have shapes (3,8,32), (24,1024,1024),
(6,1024,4096), and (6,4096,1024). The current runtime capture attributes roughly
0.23/12.54/9.76/8.98ms to their compiled math windows. These profile windows are
for sizing only, not a promise of savings. AdamW adds five large tables and
small groups. The new functional NKI AdamW could complement ownership, but its
integration gates must pass independently first.

The [primary ZeRO description](https://www.deepspeed.ai/tutorials/zero/)
explicitly describes each process updating only its assigned optimizer
partition. Its [configuration documentation](https://www.deepspeed.ai/docs/config-json/)
describes all-gather or broadcasts to collect updated parameters afterwards.
This suggests a minimal native-stack experiment; it does not require adopting
DeepSpeed or neuronx_distributed on these provisioned hosts.

Candidate algorithm, retaining FP32 weights and the existing gradient reduction:

1. Assign complete parameter matrices to deterministic owners, balancing both
   Muon shape cost and table size. Never split a Muon matrix's norm or polynomial.
2. Each owner updates its subset using the existing math and runtime scalars.
   Per-matrix normalization and second moments remain local to that matrix.
3. Pack updated FP32 parameters into persistent owner buffers, gather once,
   and scatter into the original Parameter objects on every rank.
4. Refresh the existing BF16 caches only after that collective completes.

A theoretical balanced optimizer cost near14ms leaves about42ms for additional
packing/communication before breaking even. This is only an optimistic bound.
Approximately470MB of FP32 parameters must be reconstructed per rank; actual
collective support, padding, copies, matrix-group utilization and host dispatch
can erase the gain. No FP16/FP8 transport approximation belongs in this first test.

Gates before a full model: tiny native all-gather/broadcast support under LNC2;
deterministic assignment covering every parameter once; CPU/Gloo equivalence;
then real four-rank repeated optimizer updates against the current replicated
control, including skipped gradients, state ownership, schedules and exact
cross-rank parameter agreement. Compare per-parameter states against the
replicated reference on their owners. Evaluation checkpoints remain full model
weights; a future optimizer-resume feature would need all owners' states.

Measure the whole optimizer including staging, padding and collectives, then
the whole training step with the established post-model checks. Require at
least3% total-step reduction before a30-minute quality run. Retain the same
loader, global batch, seed, clipping order, schedules and public scorer. If
collective overhead consumes the available margin, record that compute result
and close this path for the measured configuration.

Wave60 communication evidence: `run-logs/fs-w60-optimizer-weight-exchange-r1`.
The actual 117,453,580-parameter grouping is covered exactly once; owner
element counts are 27,265,796 / 30,414,851 / 30,412,805 / 29,360,128.
Padding yields486,639,616bytes reconstructed per rank. All weights use FP32
transport, with exact rank/parameter/element patterns checked before and after
timing. Gather alone costs about4.69ms; full pack/gather/copyback about6.40ms.

The scratch adapter keeps complete public optimizer groups so scheduling,
gradient reduction, clipping and selective zeroing retain all parameters.
Persistent owner views dispatch the existing AdamW/Muon methods; Muon's None
gradient handling stays zero-filled rows with continued momentum/decay.
Unsupported kinds and per-depth metadata fail explicitly. Each owner retains
its own group state. Optimizer serialization uses an explicit rank/world/plan
schema, removes only reproducible stacked scratch, and rejects a wrong-owner
checkpoint. The existing model checkpoint remains a complete set of weights.
