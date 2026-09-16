# RESUME HERE — sub-1.0 campaign state (2026-09-16)

## Why no remote execution happened

The authoring sandbox has no route to the EC2 boxes: DNS fails, direct TCP is
"Network is unreachable", and the HTTP proxy refuses CONNECT (verified with
socat/nc to :22 and :443). AWS APIs DO work from here (sts + ec2
describe-instances succeed), but neither box has an SSM agent registered, so
there is no AWS-native shell channel either. No instance was stopped, started,
or modified. Both boxes are `running` trn2.3xlarge (trn21 = i-0329acb9a6244986b,
trn22 = i-0673e54d9d66e6a1e, ap-southeast-4).

## What IS done and committed (this repo, unpushed)

- `train.py`: GQA (`--n-kv-heads`, default 0 = dense) + byte-weighted loss
  (`--byte-loss-weight`, default off), both flag-gated; toggle snapshots +
  shape-critical entries updated.
- `train.py` HELP FIX (critical): bare `%` in three help strings made EVERY
  invocation crash at parser construction (verified traceback). Fixed to `%%`.
  Any box checkout older than this commit cannot run anything — rsync first.
- `research/scratch/byte_loss_gqa_equiv.py`: 14 CPU checks green.
- `research/average_checkpoints.py` (+ selftest green), `research/reconcile_eval.py`
  (dry-run green; needs Trainium+data).
- Plans: `audit-`, `plan-` (seed hunt DROPPED, seed-42 discipline), `ideas-`,
  `execution-runbook-sub1-2026-09-16.md` (exact remote commands).
- Remote facts baked into the runbook (from logs/code, not guesses): venv
  `/home/ubuntu/workspace/native_venv`, launch shape
  `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4`,
  `research/run_experiment.py` wrapper + experiment-log.jsonl (356 rows).

## Next step (needs a networked machine)

Step 0 of `research/execution-runbook-sub1-2026-09-16.md`: stock-take both
boxes, rsync this repo state over, verify `--help` exits 0, then P0
(reconcile_eval on the record checkpoint — no training cost).

## Open verifications (all need Trainium)

GQA: NKI rope kernel on repeated+contiguous K/V; byte-loss: NKI softcap+CE
"none"-reroute path; everything else per the runbook gates.
