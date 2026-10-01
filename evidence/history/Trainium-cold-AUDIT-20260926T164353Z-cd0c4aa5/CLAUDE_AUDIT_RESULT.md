# Claude compliance audit: final batch/LR candidate (cd0c4aa5)

Date: 26 Sep 2026, ~22:40 IST. Auditor: Claude (source diff + evidence review).

## Verdict: SOURCE-AUDIT ACCEPT

No blocking issues. Not a guarantee of organiser acceptance.

## Inputs verified

| Item | SHA-256 | Match |
|---|---|---|
| train.py (candidate) | cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d | yes |
| accepted_parent.py | e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43 | yes, identical to the file Claude audited this morning, scored 0.9878 official (sub-86d307b4) |
| README.md | 4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316 | yes |

## Complete delta from the accepted parent (verified by diff)

```
-TOTAL_BATCH_SIZE = 131072
+TOTAL_BATCH_SIZE = 262144
-MATRIX_LR = 0.015
+MATRIX_LR = 0.021
```

Nothing else changed. Consequences, checked in the source:
- grad_accum_steps = 262144 / (1 x 32 x 1024 x 4) = 2. The loss is divided by
  grad_accum_steps before backward, so gradients are averaged correctly.
- MATRIX_LR feeds every Muon group and also the local-conv AdamW group (unchanged
  routing, same as the research runs that produced 0.98699 / 0.98677).
- Budget accounting, stop rule, record_step_boundary, load_for_eval (weights_only=True,
  config from checkpoint), NKI kernels and import-time behaviour are byte-identical to the
  audited parent. All findings in CLAUDE_AUDIT_RESULT.md for e1e2f686 carry over.

## Runtime evidence (cold-evidence.json)

- Cold startup 846.3 s (empty cache proven), startup excess 0 s.
- Charged through exit 1789.6 s, over_budget False, 1919 steps.
- Fresh public20M / context 1024: 0.986890. Same-host research comparator: 0.986985.
- Launch command SHA b75f844f... is byte-identical to the accepted parent's.

## Upload

Upload train.py only, with the exact launch command below. Expected official score about
0.9863 (the last file's official score was 0.0006 below its public score).

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```
