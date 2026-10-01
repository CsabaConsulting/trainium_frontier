# Claude compliance audit: norm+logits+MTP stack (70773ef4)

Date: 27 Sep 2026, ~04:40 IST. Auditor: Claude (full diff + evidence review).

## Verdict: SOURCE-AUDIT ACCEPT

No blocking issues. Not a guarantee of organiser acceptance.

## Inputs (sha256 recomputed)

- train.py: 70773ef4f79edb6f5f1f6295cabb6961dc5c8657b780b361b6fd1a201685d80c
- audited_parent.py: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b (Claude-accepted BF16 file)

## Complete delta (verified with diff), and what I checked

1. norm(): drops x.float() before F.rms_norm. Same function in train and eval.
2. softcap CE: NKI path now accepts bf16 logits; eager fallback upcasts with logits.float().
3. GPT.forward: logits stay bf16 into the loss; the targets=None (eval) path returns
   15*tanh(logits.float()/15), the same formula as before, fp32 output.
4. MTP (training objective only): second CE on targets shifted by one (next-next token),
   last position per row masked, weight 0.5 decaying to 0 by 2/3 of the run.
   - Uses only the training labels already in the batch; no validation data, no extra head,
     no parameters.
   - Eval via targets=None is untouched. If an evaluator ever called forward with targets,
     the default weights [1, 0] give (1*first + 0*second)/1 = first, i.e. plain next-token CE.
   - GPTConfig.mtp=True is saved in the checkpoint config; load_for_eval rebuilds the same
     GPT class with strict weights_only loading, unchanged from the parent.
5. --mtp flag defaults True, so the launch command is byte-identical to the parent's.

Budget accounting, stop rule, record_step_boundary, load_for_eval and NKI kernels are
otherwise unchanged from the audited parent.

## Evidence (cold-evidence.json)

- Cold startup 833.2 s (empty private caches receipted), excess 0.
- Charged through exit 1789.2 s, over_budget False, 1,992 steps, seed 42.
- Fresh public20M / context 1024: 0.981441 (9bdb two-run mean 0.983154; BF16 0.984334).
- One run; repeat spread on the parent stack was ~0.0003.

## Upload

train.py only, unchanged command:

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Expected official roughly 0.9808-0.9814 (last offsets -0.0003 to -0.0006, plus ~0.0003 run noise).
