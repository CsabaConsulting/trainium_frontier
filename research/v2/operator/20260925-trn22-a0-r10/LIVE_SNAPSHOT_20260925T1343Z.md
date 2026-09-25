# trn22 A0 r10 post-launch live snapshot

Observer: distinct Operator, read-only SSH only. This is launch/identity evidence, **not** an independent review or scored result.

At 2026-09-25 13:41:43 UTC, the host showed one `flock`-guarded driver (PID 2998248), one `torchrun --standalone --nproc_per_node=4` (PID 2998266), and four `train.py` workers (PIDs 2998283–2998286). The exact inner argv was:

```text
torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

The staged package and isolated train-sandbox `train.py` both hashed to `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`, matching the assignment. The log's `ISOLATED_SUBMISSION` line independently reports the same train SHA, pinned `prepare.py` SHA `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`, and identical launch command. The root filesystem had 57,110,691,840 free bytes. At 13:43:00 UTC, `fuser /dev/neuron0` named only that torchrun and its four workers; the log had reached finite step 1150, loss 3.1632, 0.49 s/step, epoch 1. The driver remained in its training phase; no checkpoint or public BPB existed yet.

Raw log captured during this snapshot: `train.progress-1341.log`, SHA-256 `1fb6903490e37ee5b2dfba0de8025874614bf3097920a25e39d839925dbaf57d`. Its `ISOLATED_SUBMISSION` and `REHEARSAL_COMMAND` entries preserve exact inner provenance. The launch receipt's `preflight.json` and `attempt.json` are unchanged. Evaluation and result use remain HOLD until training completes, fresh-process public-20M/context-1024 evaluation is retrieved, and a different reviewer examines the receipts.
