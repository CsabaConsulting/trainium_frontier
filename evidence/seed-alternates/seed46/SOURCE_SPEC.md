# R202 — T3 seed 46 (disabled)

Exact parent: `research/v2/proposals/20260930-r168-flags-t3-opt3-laneb-f1/train.py` / `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`.
Exact candidate: `train.py` / `54926ae7ca2c44867c94982b1cd471d031a1a5a1bc7daa8983f4b7e92768c415`.

Only `main`'s explicit `init_runtime(device_type, seed=42)` call changes to seed 46.
One changed source line and one AST Constant; inverse substitution recovers every
parent byte. Runtime's default seed42 and the generic module docstring remain
untouched. Model, optimizer, kernel, loss, evaluator/targets=None, loader,
tokenizer, schedule, cache, flag and command bytes are unchanged. Both T3 O3
defaults remain source-owned. The RNG stream changes; native reproducibility,
final validation score and an improvement are not predicted.

R202 item6 explicitly requests further T3 seeds45+ under the seed43/44 rules.
This is a new seed file, not another exact-file receipt of seed42,43,44 or its
paired seed. Prior CPU/native/FULL results are lineage only and cannot accept it.
R202 priority is CE3/constant-arm qualifying repeats, CS25, CS24 and other ready
FULL work before seeds; Root binds the first free validated host only after the
actual current higher-priority gates are known. Last FULL launch06:00/freeze07:00.

## Proof and release obligations

Local proof is byte/AST metadata only. Author CPU checks run only on an admitted
instance: exact-byte inverse, single explicit seed, unchanged runtime, extracted
main prefix through CPU init_runtime, separate Generator comparison, same-seed
replay and distinct seed42/paired-seed streams. Import CPU Torch only after EC2,
single-core/nice19/autoload-disable/resource checks. Never import a whole
candidate, model, SDK, data or tokenizer. Independent review receives only frozen
source/base, diff, SPEC and declared gates; it freezes its own oracle before
seeing author output, using its own fresh instance-CPU check. Author evidence
cannot independently accept this source. Author implementation/results are not
part of the independent review input bundle.

Every actual author/independent/FULL/own-checkpoint result stays NULL until
retrieved. This completed source proposal is disabled; host is unassigned, no
host is reserved and no launch packet is emitted. Root alone binds actual
authority, predecessor, host validation and original admission evidence.

FULL recipe is unchanged: unique root/label, full-chip LNC2/world4, six fresh
cold caches, lifetime flock/Guardian/process/device/STOP checks, entry25GiB,
runtime22GiB, cold<900s, charged1800s through exit with15s reserve, stable own
checkpoint, fresh-process public20M (20,971,520 tokens), context1024. This summary
does not replace any original guard. No caller NEURON_CC_FLAGS injection; source
owns O3. Environment seed receipt must be 46, matching its explicit source
seed. A new host needs its own passed T3 matched60 sanity and data/env parity.

Command unchanged: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`.

R202 requires THIS seed's valid FULL checkpoint for its own LNC1/unset public2M
diagnostics after FULL, then its audit copy. Do not reuse another seed's checkpoint
or target. Record actual scores and receipt counts for the freeze ranking; equal
scores retain seed42 under the earlier seed rules. No automatic adoption or upload
is authorized. A crash, NaN, causality failure, cold/budget failure, dirty run tree
or missing checkpoint fails its gate. Poojith alone uploads after human audit.
