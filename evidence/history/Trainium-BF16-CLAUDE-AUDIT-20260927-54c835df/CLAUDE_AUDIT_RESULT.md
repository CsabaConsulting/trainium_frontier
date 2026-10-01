# Claude compliance audit: BF16 residual candidate (54c835df)

Date: 27 Sep 2026, ~00:30 IST. Auditor: Claude (source diff + evidence review).

## Verdict: SOURCE-AUDIT ACCEPT

No blocking issues. Not a guarantee of organiser acceptance.

## Inputs verified (sha256 recomputed on this machine)

| Item | SHA-256 | Match |
|---|---|---|
| train.py (candidate) | 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b | yes |
| accepted_parent.py | cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d | yes, the file Claude audited on 26 Sep; official 0.9866 (sub-35db7fa3) |

## Complete delta from the parent (verified with diff)

```
773c773
<         x = self.resid_lambda * x + self.x0_lambda * x0
>         x = self.resid_lambda.to(x.dtype) * x + self.x0_lambda.to(x.dtype) * x0
```

Nothing else changed. Checked in the source:
- x0 = norm(wte(idx).to(COMPUTE_DTYPE)) is bf16 (norm returns bf16), so the stream now
  stays bf16. Before, the shape-(1,) fp32 scalars promoted it to fp32.
- COMPUTE_DTYPE is bf16 on neuron in both the training and eval setup paths.
- Training and load_for_eval run the same Block.forward, so there is no train/eval
  implementation difference. Master weights and scalar params stay fp32.
- Budget accounting, stop rule, record_step_boundary, load_for_eval, NKI kernels and
  import-time behaviour are byte-identical to the audited parent. All earlier findings
  carry over.

## Runtime evidence (cold-evidence.json)

- Startup 828.9 s (six empty private SDK cache dirs receipted), excess 0.
- Charged through exit 1788.9 s, within budget, 1,988 steps, seed 42.
- Fresh-process public20M / context 1024: 0.984334 (same-host parent 0.986890, delta -0.002557).
- Launch command SHA b75f844f... identical to the parent's.
- One run, not a replicated mean.

## Low-severity notes

- The file header still describes an fp32 residual stream. Stale comment, harmless,
  left unedited to keep the measured bytes.
- The evidence harness SHA differs from the cd0c4aa5 package (local tooling change);
  it does not affect the uploaded file.

## Upload

Upload train.py only, with the unchanged launch command:

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Expected official about 0.9838-0.9841 (last two offsets: -0.0006 and -0.0003).
