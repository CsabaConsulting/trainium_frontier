# Independent bounded review of timer ports and gathered-row patch

Reviewer: root, not the source Implementer of these patches. No host launch,
native timing or BPB is certified by this receipt.

## L2 timer port: PASS narrow source/CPU

Commit1e21c3d797eed764f108c41546b270065ad5650c; candidate
93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5.
Inspected complete four-line diff from reviewed architecture52e234e9.
Only two executable edits: clear recent durations at existing fifth boundary,
and empty-safe max default0. Independent unittest discovery ran12tests in
1.539s, allPASS. Exact inverse restores parent bytes and entire AST; timer
AST matches reviewed a4aa4351. Startup excess, official900cap,30margin,
1800budget and15checkpoint reserve are unchanged. All model/architecture
math and evaluator code remain the independently reviewed52e2 implementation.
Original architecture review:20260925-l2-unet8x1024-independent-review.md.
Exact8x1024four-rank G2 and native results remain pending.

## L1 timer port: PASS narrow source/CPU

Commit50a901374ae6f76b35b8556d32fd72b1ece28fc6; candidate
921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858.
Inspected complete four-line diff from reviewed FP8dea47910. Independent
unittest discovery in the nested startup-predictor-reset folder ran12tests
in1.043s, allPASS. The first discovery at its parent found0tests and is not
counted as evidence. Exact inverse bytes/AST and unchanged charging verified.
No L2 architecture is mixed into L1; original builder worktree remains untouched.
Original FP8 source/reduced CPU review:c15a0ac6. Exact6x1024G2, native kernels
and at least3% whole-step speedup remain mandatory before its full quality run.

## Gather boundary: PASS source/bounded CPU only; deprioritized

Commitcd43fd831a6a3e21a52d37b9bc88c8ede1869b7e; source
9bf9568d1b6fe861e4fca0826229e4644527fcd4a7296be982449a18617b85aa.
Inspected the full57-line production diff, lane synthetic adapter and tests.
The original advanced-index lookup is split from the same projection; eager
caller supplies a fresh tuple with autograd intact. Optimizer, reduction and
checkpoint ownership remain on the original model. Both training and compile
paths use it; defaultOFF retains old calls. No detach, sparse gradients,
cached rows, CPU hash or custom backward is introduced. Ordinary eval forward
retains mathematically identical lookup/projection of saved weights.

Independently reran check_boundary.py against exact6ef7 with existing CPU
Torch2.13: exit0. Parent/OFF/ON logits, loss, gradients, updates and RMS state
match through3steps and2microbatches, including collisions and nonzero tables.
CPU graph probe remained1forward/1backward over3calls, with no full-table
inputs/intermediates and nonzero raw-row gradients. Checkpoint, deterministic
batch decomposition, prefix causality and compatibility tests passed.
The lane-local tuple-slicing G2 adapter was inspected: it applies the identical
causal-prefix slice to idx/targets and row inputs, without detaching. It leaves
canonical preflight/prepare untouched and cannot establish production-shape
Neuron behavior. Existing reduced4rank receipts are Implementer evidence;
exact-size independent G2/native compilation remains outstanding.

No native speedup, resharding removal or cold eligibility is established.
Per newest POST_A1 directive this branch is parked below schedule, architecture
and FP8; do not launch a new table campaign from this source review.

Sources remain on their separate committed branches, not silently substituted
for the running6ef7 candidate or scored116188 submission vehicle.
