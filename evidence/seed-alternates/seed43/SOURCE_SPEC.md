# R200 — T3 seed 43 (disabled)

Exact parent: `research/v2/proposals/20260930-r168-flags-t3-opt3-laneb-f1/train.py` / `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`.
Exact candidate: `train.py` / `10ec1169ae6eaf69f7b728c34758bb5e28e3fb8df78a795309f73c6a1a9b9362`.

Only `main`'s explicit `init_runtime(device_type, seed=42)` call changes to seed 43.
One changed source line and one AST Constant; inverse substitution recovers every
parent byte. Runtime's default seed42 and the generic module docstring remain
untouched. No optimizer, model, kernel, loss, evaluator/targets=None, loader,
tokenizer, schedule, cache, flag, or launch-command changes. T3's training AND
import O3 defaults are unchanged. This intentionally changes the RNG stream, not
the mathematical model. Native reproducibility and scores are NOT predicted.

R200 explicitly requests seeds43/44 of T3 after higher-priority ready work.
The old R170 seed43 derives from LaneB-F1 O1 (02c758), not T3; its sourceCPU,
native and FULL evidence do not transfer. Its wrappers are historical recipes only.
No new seed file is a second exact-file receipt of seed42 or of the other seed.

## Proof and release obligations

Local static proof is byte/AST metadata only. Paired author CPU worker in
`../20261001-r200-t3-seeds/author_cpu.py` runs ONLY on an admitted instance CPU:
exact-byte inverse, single explicit seed, unchanged runtime, actual extracted main
prefix through CPU init_runtime, separate Generator comparison, same-seed replay,
and distinct seed42/paired-seed streams. It imports CPU Torch only after EC2,
single-core/nice19/autoload-disable and resource checks; no model, SDK, data or
whole candidate import. Fresh-session independent source review and its own actual
instance-CPU evidence are required later; this author cannot independently accept.

Every actual author/independent/FULL/own-checkpoint result is NULL until retrieved.
Root alone selects and binds current host/predecessor/authority. New hosts require
their own provisioned-data/environment and actual T3 matched60 sanity first.
Host is unassigned here; no trn21/22 mapping or launcher is changed.

FULL recipe remains original unique-root, full-chip LNC2/world4, six fresh cold
caches, original lifetime flock/Guardian/process/device guards, entry25GiB,
runtime22GiB, cold<900s, charged1800s through exit with15s reserve, stable own
checkpoint, fresh-process public20M (20,971,520 tokens), context1024. Existing
guard details are not replaced by this summary. Environment must not inject
NEURON_CC_FLAGS: T3 source owns O3 defaults. Environment seed receipt must be 43,
matching the explicit source seed (not an override of it).

Command unchanged: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`.

For a package: use THIS seed's valid FULL checkpoint and observed result for its
LNC1/unset public2M diagnostics, never seed42's checkpoint or target. R200's
best-of-three intent is not a forecast or automatic adoption. Record real scores;
retain current human selection/audit/upload requirements. No FULL or upload
authorization is supplied by this proposal.
