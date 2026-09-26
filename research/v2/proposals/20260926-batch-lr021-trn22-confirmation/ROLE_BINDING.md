# Existing-code trn22 batch/LR cross-host confirmation

Researcher recommendation, 26 September 2026 09:28 UTC. Local evidence only;
no source edit, new review/test gate, host action, staging, release or launch.
The Supervisor reports the diagnosed K2 retry failed with unbound `tokens`
and is parked. Do not retry K2 or dispatch dependent P2 from this note.

## Reuse the already-completed experiment

Select the existing unstarted `batch-lr14-trn22` alternative in
`research/v2/proposals/20260926-sprint-p0-clean-batch/EXPERIMENT_PROPOSAL.json`.
Proposal SHA256:
`785bfe4ea42bd8c82cacfeabf874de39bb7f253bc5bf1b23e2f7b112e6db8aef`.
That proposal already declares both hosts and both LR arms. No new experiment
or duplicate row is needed; this note binds the comparison and current evidence.

- Host/label: trn22 / `fs-20260926-sprint-clean-batch-lr14-trn22-r1`.
- Scientific source: `research/v2/proposals/20260926-sprint-p0-clean-batch/variants/batch-lr14-trn22/package/train.py`.
- Source SHA256: `8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`.
- Source commit: `7117d57b92e19e8437158164dbce94495389d7b6`.
- Actual outcome record: `arm_id=e1e2-batch262144-lr021`, `kind=batch_arm`,
  `run_purpose=quality`, base `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`.
- Prepared files: `research/v2/operator/20260926-final-sprint-p0/batch-lr14-trn22-successor-prepared.json`.
  SHA256: `bb5d0a7977783b04568cb9bd4b54971002cd1121487a925a7136298e10d8bfdc`.

Exact existing launch command (do not append K2, tables or schedule flags):

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --total-batch-size 262144 --matrix-lr 0.021
```

Existing `package/launch-command.txt` SHA256, including its file newline:
`c74a4cc45690d395827ae95fe12bb21f628adb4a83ab2cf0c15f7445bdb15b5a`.
Eight layers, width1024, seed42, four ranks/LNC2, pack32, context1024,
batch262144 through accumulation2. Full1800 charged seconds; no short-step cap.

## Evidence and causal comparison

Existing source/CPU and disabled-binding review:
`research/v2/verification/20260926-p0-sprint-fresh-independent/REVIEW.md`,
SHA256 `c501228eb9a7f6b073005b88fe1d3041ddf90e61c569c1055b3aa723e5a7f7a9`.
It covered exact8c6d source, default identity and the intended .015/.021 arms;
its bounded CPU review did not establish native performance or host admission.
Existing actual full runs now supply native evidence; do not repeat a review.

The exact source/argv already completed on trn21 as
`fs-20260926-sprint-clean-batch-lr14-trn21-r1`: fresh20M BPB
0.9869854734569898, 1921 steps, charged through exit1788.753648138023s,
observed startup798.4203850750346s. Evidence:
`research/v2/analysis/20260926-clean-batch-lr021-results/result.json`
(SHA256 `b08608d848f661e82b7701686eb7179fc902de9c699a5668feaa39224b16bf49`).
Its selector is INSUFFICIENT pending the genuine trn21 e1e2 control; this is
a cross-host replication attempt, not a confirmed winner or adopted base.

Primary trn22 comparator is the actual valid full control
`fs-20260926-sprint-e1e2-control-trn22-r1`, exact e1e2 source/base:
fresh20M BPB0.9879041000701606, 3640 steps, charged1789.2061858640518s.
Evidence: `research/v2/analysis/20260926-e1e2-control-trn22/result.json`
(SHA256 `9d2bdb0dee58f86e78929d90500697a686168ab1fe7991cefaa22003a126062b`),
raw `research/v2/operator/20260926-final-sprint-successors/trn22-e1e2-control-completed/results-bundle.json`
(SHA256 `f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7`).

This is the already-prespecified JOINT batch262144/matrixLR.021 arm against
default batch131072/LR.015, not a claim isolating batch or LR individually.
The genuine same-source trn22 batch262144/LR.015 result0.987994054399811
may supply a secondary LR-only contrast, never substitute for the declared
e1e2 primary control or a second invented control row. Keep one actual record
under the existing arm ID so unchanged decide.py can combine actual host deltas.

## Unchanged admission, budget and result rules

Only the existing sole Operator can perform fresh admission and launch after
Supervisor selection. Existing files remain disabled; this note changes no
queue/state/ledger or launch authorization. No local ledger/analysis record was
found for the proposed trn22 label; Operator must confirm it remains unused.
Preserve the original held lifetime `device0.lock` through training, attached
fresh20M and owned-child quiescence. Fresh ps AND neuron-ls, fuser, STOP,
clean exact source/argv/runtime/proposal and measured resource checks remain.

Disk: runtime floor11274289152 B plus prospective3221225472 B, hence initial
minimum14495514624 B. Retain64GiB (68719476736 B) host MemAvailable admission;
increase reservation if actual projected needs require. No lowered floors,
protected/unscored deletion or cleanup/AWS authority. Reuse only existing
guardian mechanics; lifetime guardian hash is
`5df6897deeaea3435a963f79b912ce49f320d3a65624f60a89762249c67ecb6e`.

The prepared same-host warm research cache owner is the completed
`fs-20260926-sprint-clean-batch-lr1-trn22-r1`. Record exact paths/runtime and
observed cache evidence; no hit or cold-start claim is implied. Its inherited
first-source-timing-pending field is historical preparation state: bind actual
completed8c6d evidence mechanically, without inventing another native gate.
Keep all current failures, controls, checkpoints and caches protected.

Official1800, startup allowance capped900 with all excess charged; training
outer2730/eval outer1800/TERM grace60 do not enlarge that budget. Fresh unchanged
public20971520-token/context1024/batch1 evaluation must use the actual checkpoint.
No predictor/prepare/evaluator/data edits. Warm startup is not cold submission proof.

Use unchanged selector thresholds and actual same-host controls: one-host
nomination delta<=-0.0006, discard delta>=+0.0005, ambiguous low-prior batch
treatment unchanged; confirmation needs both deltas negative, mean<=-0.0010,
disagreement<=0.0015. Preserve big-win HOLD and all correctness/budget gates.
Step count is diagnostic (batch doubles tokens/step), not an adoption gate.
Do not adopt before the missing trn21 control and actual trn22 result arrive.
Submission remains human-only after required cold replay/fresh20M/Claude audit.
