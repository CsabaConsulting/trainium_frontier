# Combined norm+logits+MTP — Claude source/compliance audit request

Audit the EXACT attached train.py. Poojith alone uploads; this package is not
an upload, a compliance acceptance or a guarantee of organizer acceptance.

Candidate SHA-256: 70773ef4f79edb6f5f1f6295cabb6961dc5c8657b780b361b6fd1a201685d80c
Source-audited BF16 parent SHA-256: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b
The parent was uploaded per Poojith; its official score is still pending.

## Changes and intended semantics

BF16 residual is inherited. This file removes norm-input FP32 round trips,
feeds BF16 logits to the existing softcapped CE with FP32 internal math, and
adds the k1/k2 auxiliary training objective with runtime tensor weights.
The one-graph evidence is CPU-captured; native graph count is unverified
unless separately established in the final run evidence. MTP defaults ON.
MTP leaves targets=None inference unchanged relative to9bdb norm+logits.
Relative to the attached BF16 parent, norm-input precision also changes
inference. There is no separate eval-only model, adapter or architecture
switch. Review these claims against the complete source and attached diff.
The k2 loss is training-only supervision, not extra inference computation.
CPU reference/causality/checkpoint receipts are evidence, not a Claude audit.

## Unchanged submission launch

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Upload only train.py after source-audit ACCEPT for this exact SHA. Do not
append --mtp, --batch-ramp, timing flags or research overrides. Source bytes
are the measured bytes, with no post-run cleanup or header editing.

## Authenticated measurement

- Public20M/context1024: 0.981441493395, seed42, trn21.
- 1992 steps; actual cold startup 833.244306s.
- Harness startup excess 0.000000s;
  source-conservative estimate 0.000000s.
- Charged through process exit 1789.237305s; over_budget=False.
- 9bdb same-host two-run mean 0.983153702498;
  incremental delta -0.001712209102; repeat spread 0.000326698084.
- Fresh-process public evaluation and six absent-then-empty cache receipts
  are authenticated in cold-evidence.json. One7077 result, not a7077 mean.
- No official7077 score or claimed official calibration. Official best known0.9866.

## Audit focus

Check same trained model in load_for_eval, weights_only loading, targets=None
identity, auxiliary-label causality and boundary handling, zero-weight MTP
behavior, BF16-logit/FP32-softcap numerics and gradients, legal cold budget,
full-chip launch, no evaluator/tokenizer/data/private-validation access changes.
Review inherited NKI/autograd/import-time code too; prior acceptance is not
a guarantee. Source reviewer is independent of implementation but was not a
fresh isolated session; see its explicit scope in the evidence. Claude is
the required external compliance auditor. No quota availability is assumed.

Freeze Sep29 18:00IST. This export did not modify any host job or old package.
