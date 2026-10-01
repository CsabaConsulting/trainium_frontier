# CS21 MTPW75 — R171/R172, exact T3 base

Candidate: `e056552274ac7ed3ec977589cda03428984ec373d1dc724e42f2698841645f82`. Parent: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894` at `research/v2/proposals/20260930-r168-flags-t3-opt3-laneb-f1/train.py`.

Exactly one source line changes: the leading coefficient in `mtp_auxiliary_weight` is `0.5` → `0.75`. Inverting this line restores all parent bytes. The clamped linear schedule, divisor `2.0 / 3.0`, startup-adjusted elapsed/1800 progress, fixed two-element runtime tensor, target shifts/masks and loss normalization are unchanged. Initial auxiliary coefficient is .75 rather than .5, and both reach zero at 2/3 progress. The normalized second-target share is consequently 3/7 at progress zero, rather than 1/3. This intentionally changes the training objective; no training-numerics equivalence is claimed.

Shared model forward, architecture, optimizer, loader, evaluator, load_for_eval, kernels, launch flags and all other source bytes are exact T3. In particular both T3 main and import compiler defaults remain optlevel=3; no unmeasured optlevel=1 repair is inserted. Exact source/own-checkpoint LNC1 and unset evaluation diagnostics required by R172 cannot be inherited from the optlevel=1 F1 proofs.

R171 requests this magnitude arm; R172 supersedes its initial LaneB base and queue: build on exact T3, FULL only after its two actual receipts, adopt only at <= actual T3 mean - .0005, and obtain a second candidate receipt before packaging. T3 receipts/mean/absolute threshold are NULL until authenticated. If the first T3 receipt fails its gate, this T3-based candidate stays parked; a LaneB-F1 rebase is a separate source/proof, not an automatic fallback. No throughput gate or separate timing run is introduced.

Root alone may release a unique seed42 full-chip four-rank LNC2 FULL with the original 1800s budget, cold<900s, fresh public20M/context1024 and original 25GiB entry/22GiB runtime floor, only after actual author and independent instance-CPU source checks and original machine/authority/queue gates. No launch, numerical, native, FULL, scorer or upload acceptance exists in this preparation. No laptop model/tensor/schedule execution. Author proof is bounded instance-only stdlib AST/scalar testing; the independent reviewer authors a separate oracle.

Unchanged command: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`. This command does not itself override NEURON_CC_FLAGS; the exact T3 native/full envelope policy must be retained.
