# trn22 c5c A0 r10 Operator final retrieval

This is a raw Operator result and postflight receipt, **not** an independent
verification, `decide.py` verdict, research-base adoption, or submission
clearance. The distinct postlaunch reviewer must clear the result before it is
used as a control. No cleanup, checkpoint deletion, AWS lifecycle action, or
leaderboard submission occurred.

## Identity and training

- Unique label: `fs-20260925-v2-c5c-a0-control-trn22-r10`.
- Host: `trn22`; warm-cache manual break-glass path under
  `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260925-v2-c5c-a0-control-trn22-r10-manual-source-r1`.
- Exact copied `train.py` SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  Isolated `prepare.py` SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Exact launch: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015`.
- Driver started `2026-09-25T13:19:56Z`; training completed
  `2026-09-25T14:03:23Z`. Training rehearsal returned zero with 3,633 recorded
  step boundaries, 476.2M training tokens, 814.0847056440543 s startup
  allowance, 1790.2541294809198 s charged time within the 1800 s budget.
- Checkpoint preserved at the sibling work path
  `.../fs-20260925-v2-c5c-a0-control-trn22-r10-manual-work/train/out/final.pt`;
  469,834,849 bytes; SHA-256
  `9d225af6a3afcb2d0a15464b5514f897cd3cb29c1722a8491940a3f04a7180d6`.

## Fresh-process public evaluation

- Eval started `2026-09-25T14:03:23Z`; `driver.log` recorded
  `EVALUATED_2026-09-25T14:16:06Z`. The separate `eval/rehearsal.json` says
  `mode=eval`, `returncode=0`, wall time 760.3882084949873 s.
- Fresh-process command in `eval.log`: `python prepare.py eval-public
  --train-py train.py --checkpoint <above final.pt> --device neuron
  --batch-size 1 --seq-len 1024 --eval-tokens 20971520
  --timeout-seconds 1200`. It reported no timeout, 747.1132776270388 s
  evaluation, 20,971,520 tokens, 83,462,186 bytes, 57,360,648.0 total
  nats, and **public val_bpb 0.991513958330512**. Arithmetic check:
  `57360648.0 / (83462186 * ln(2)) = 0.991513958330512`.
- `eval.log` ends in `SUBMISSION_REHEARSAL` with return code zero. The driver
  marker alone does not establish that its detached shell checked status;
  the independent eval receipt and log establish successful eval separately.

## Postflight and retrieval

At `2026-09-25T14:16:30Z`, read-only `ps` showed no torchrun/train/eval
process; `fuser /dev/neuron0` found no holder; the pre-existing host
`device0.lock` could be acquired non-blockingly; and `/` had
56,619,425,792 bytes free. The checkpoint still matched the SHA above. No
new run had started during this snapshot. The raw final logs and receipts in
this directory were copied directly from the host and their SHA-256 values
match host-side readings:

| Local artifact | SHA-256 |
| --- | --- |
| `driver.log` | `7e46419bf2107f4264c277825a27687c2dc267f363d26a856fd87d923c923c47` |
| `train.log` | `5dfe38ee815be4ffe48cf85e351b7ee22cfcfdb6837a0e5a235aebd53fa3a741` |
| `eval.log` | `316698b691feb9cd3f6f010343220217f89c4e3ef04a9818bad9a342d8c67349` |
| `rehearsal.json` (train) | `f5b833832e2284f9fb6cd972fb2f1a84a50a6096aa02691cd79b15919b81a4a2` |
| `steps.jsonl` | `e8de53bea4e14620326d52aceed5f083b4b977e70ed558af739fb06b24d56416` |
| `eval-rehearsal.json` | `901eba35b81d7236fde2fcaa22da9f98a147a8f2704329a021593b0ffc689883` |

The earlier launch receipt, exact preflight, attempt and live four-rank
snapshot remain in the parent Operator receipt directory. The final score
has not been appended to the combined ledger by this Operator.
