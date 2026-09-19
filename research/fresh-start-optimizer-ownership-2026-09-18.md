# Reopen optimizer ownership as a compute question

Current02:21 UTC,2026-09-19: mode2 reciprocal AdamW plus ownership passes its
actual-model composition gate (`fs-w98-owned-reciprocal-adamw-integration-gate-r1`).
All24 parameter/state/rank checks match replicated identical NKI math exactly.
All13 expected table updates occur once across owners; cache, skip/schedule
and checkpoint continuation pass. Rank-local checkpoints record/reject a
different AdamW arithmetic policy. Native checkpoints without that field stay
compatible with native math. Independent FP64 authority remains wave66.
Whole-step control/candidate comparison is active; no new quality gain claimed.


Earlier status at 21:59 UTC, 2026-09-18: the padded ownership implementation passes
all CPU/Gloo and native gates with **exact weights and owned optimizer states**.
The native complete optimizer, including FP32 packing, exchange and copyback,
costs **25.35–25.38 ms versus 52.29–52.84 ms** on the four ranks. The matched
full-model phase also passes: **526.782934 versus554.873134ms, a5.06245% gain**.
The full30-minute quality run is active; BPB improvement is not yet established.

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

## Singleton Muon root cause and strict correction

The unpadded wave61 native gate failed only the two tall Muon matrices whose
owners had a singleton batch. Other owners held two such matrices and matched
the replicated reference exactly; communication and cross-rank byte equality
passed. The batch-one numerical path changed those matrices' second moments
and updates. We have localized this to batch geometry, not yet a particular
lowered instruction. No tolerance was relaxed.

Wave62 appends an independent zero matrix to a singleton owner group, retaining
the batch-two path. Muon operates independently per matrix; the dummy never
enters the model or parameter exchange. Its parameter, gradient, momentum and
second-moment rows must remain zero. Rank-local optimizer checkpoints explicitly
record this padding policy, while model checkpoints remain standard full weights.

`fs-w62-optimizer-ownership-padded-neuron-gate-r1` passes 28 equivalence records
(seven checkpoints on four ranks), including scheduled scalars, None and zero
gradients, skipped AdamW, actual BF16 Linear cache use, checkpoint restore and
wrong-owner rejection. Exactness still holds after the additional timing updates.
CPU/Gloo and native postflights are clean. Native fallback execution was disabled.
The all-rank post-timing weight digest is
`43420a0537483b49f99b6e864c2d31342e863b8f7aa54585492dc31fd158bb0e`.

The 3×20-update interleaved benchmark includes all exchange costs. It is a
synthetic-gradient optimizer test, not a full training step.

The matched trn21 full-step runs are
`fs-w62-native-mlp-pack8-owned-control-phase-r1` and
`fs-w62-native-mlp-pack8-owned-phase-r1`, both sourcec18305e,30timed/5warmups,
same dense6×1024/native-MLP/T2048/pack8/GA2/seed42 geometry. Their whole-step
medians are554.8731335 and526.7829345ms; optimizer54.9657165 and26.0838685ms;
microstep windows471.3633375 and472.8148310ms. Ownership dispatches35times.
Both pass all four post-model finite/causal/batch/repetition checks with clean
postflight. Last loss5.145325183868408 and all six attention-weight norms are
identical; these metrics are not a checksum of every model weight.

The phase uses clip-before-reduce, like its matched control. The quality run
`fs-w65-dense1024-native-mlp-pack8-owned-1800-r1` uses the best wave50 model's
clip-after-reduce configuration, unchanged131072targets/update, seed42 and
protected timing/scorer. Started21:52:43UTC on trn21, source5dcbf6a. It contains
no new arithmetic approximation or NKI AdamW combination. That combination
still needs separate interaction gates. The earlier eight-step wave44 phase
is retained as history and was not used as the matched promotion control.
