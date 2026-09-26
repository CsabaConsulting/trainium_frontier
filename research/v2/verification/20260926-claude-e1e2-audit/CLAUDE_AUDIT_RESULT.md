# Claude compliance audit: clock36 clean candidate

Date: 26 Sep 2026, ~12:00 IST. Auditor: Claude (independent source review + CPU reference tests).

## Verdict: SOURCE-AUDIT ACCEPT

No blocking source defects found. This is a source and CPU-math audit. It does not guarantee
organiser or reviewer acceptance, and it does not replace the runtime evidence Codex already holds
(the cold 1800 s replay and the fresh public20M of 0.988389 on this exact file).

## Inputs verified

| Item | SHA-256 | Match |
|---|---|---|
| train.py | e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43 | yes |
| competition README.md | 4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316 | yes |
| launch-command.txt | b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd | yes |

## Scope of change vs the scored vehicle (116188, official 0.9901)

71 changed lines, all readable:
- DEPTH 6 -> 8; `ATTENTION_LAYERS = (0, 2, 4, 6, 7)`; layers 1, 3, 5 are MLP-only (no attention, no conv).
- Four zero-initialised FP32 U-Net skip gates (encoder i feeds layer n-1-i), trained as AdamW scalars.
  Attribution comment to the Trainium of Thought write-up is present.
- Value embedding restricted to attention layers (only layer 7 in practice).
- `refresh_weight_cache` also at step 0 (same values as `weight.to(bf16)`; avoids a second graph).
- `startup_adjusted_schedule_elapsed`: LR/WD progress during the first 5 steps no longer counts
  compile time. Stop clock is unchanged.
- `recent_step_times.clear()` after boundary 5: fixes the known stop-predictor bug (compile-heavy
  startup steps no longer inflate the predicted next-step time).
Everything else, including the import line, env defaults, NKI kernels, optimizer, loader and
`load_for_eval`, is identical to the vehicle that already passed review and scored.

## Findings by audit item

1. **Train and eval implement the same function.** Same `GPT.forward` in both. The only train/eval
   routing difference is the MLP (`_FusedMLP` custom autograd in training, plain Linear in eval);
   it is the same math (test T3: identical logits; T4: identical fwd and grads). Returned eval logits
   are `15*tanh(z/15)`, matching the training loss. The bf16 weight cache is not saved and not used in
   eval; eval casts the saved FP32 weights to bf16, the same values as the cache. Rotary, conv and
   rope+norm kernels are per-row, so logits do not depend on batch content (T2). The training-only
   compiler auto-cast (`__main__` only) affects matmuls that are already bf16 in the forward.
2. **Checkpoint loading is ordinary.** `torch.load(..., weights_only=True)`, config from the
   checkpoint, `strict=True`, eval mode, `requires_grad_(False)`, no hooks (T5). No external weights,
   no validation-derived state, no reconstruction.
3. **Causality and kernel correctness.** SDPA `is_causal=True`, two-tap conv uses t-1 and t only,
   value embeds and gates per token (T1: zero change to earlier logits). The NKI kernel maths were
   re-derived in plain torch and match eager autograd: conv fwd/dx/dw (T6), rope+norm fwd/bwd (T7),
   softcap-CE fwd/bwd (T8, training loss only). Each custom autograd wrapper has a clear training
   purpose. CPU cannot run NKI itself, so native equivalence rests on the Neuron runs.
4. **Environment, filesystem, network.** Import-time `os.environ.setdefault` for NEURON_CC_FLAGS
   (`--optlevel=1`) and NEURON_COMPILE_CACHE_DIR also applies when the scorer imports the file.
   No network, no subprocess, no monkeypatching, no reads of validation data (the in-script public
   eval is disabled by `--no-eval-public`). Writes only `out/final.pt` and the compile cache.
5. **Accounting.** `record_step_boundary(step)` every step. After step 5 the clock restarts minus
   the charged excess (startup + 30 s margin - 900 s cap), so a slower official startup costs training
   time but cannot push the run over budget. Stop rule reserves the max of the last 20 step times
   plus 15 s for the checkpoint. Measured cold replay: startup 859.5 s, charge 1789.8 s,
   over_budget False, 3,628 steps.
6. **Launch command matches the file.** All flags are parsed and cross-checked (`parser.error` on
   mismatch). Shape is valid (LNC 2, 4 ranks = full chip). World size 4 is asserted.

## Low-severity items (no change recommended before this upload)

These are all inherited unchanged from the accepted 116188 vehicle, so changing them now would
cost a re-validation cycle for no scoring benefit. Clean them in the next version if convenient.
- Import-time env defaults (item 4). Moving lines 24-25 under `if __name__ == '__main__':` would stop
  them affecting the scorer process.
- Unused imports: `_document_batches`, `_row_groups_for_split`, `get_token_bytes`, `Tokenizer`,
  `random`. The private split helper and `get_token_bytes` are words a reviewer looks for.
- Vestigial constant checks (lines 1368-1377 mention weight EMA and batch ramp, which do not exist).
- Docstring says value embeddings on alternating layers; in this build it is layer 7 only.
- Module defaults (SEQ_LEN 2048, PACK_FACTOR 4, N_EMBD 0) differ from the launch flags. Eval is
  unaffected because the checkpoint stores the config, but the file must be launched with the exact
  command below. A bare launch would train a different model.

## Native consistency check (pending on Codex's side)

Not required for this upload, in my judgement. The official scorer path (fresh process, unchanged
evaluator, context 1024, batch 1, on Neuron) already ran end to end on this exact file and
checkpoint: 0.988389. Batch-independence holds by construction and on CPU. Keep the native B1/B2
test as a nice-to-have, not a gate.

## Exact command (unchanged from the scored vehicle)

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

## Rule note for the submitter

The competition README in this folder states: **5 submissions per team per week in Phase 1, best
score counts, no retries** (a failed run still uses a slot). Earlier notes said 15/week. Check the
remaining quota on the My Submissions page before uploading.

## Evidence

- `CLAUDE_audit_tests.py`: the CPU reference tests (imports the candidate as a module with a stub prepare).
- `CLAUDE_audit_prepare_stub.py` (copy it to prepare.py beside the tests to run them): the stub (constants and no-op functions only; no data, no scoring).
- `CLAUDE_audit_output.txt`: raw output. Summary: 9/9 PASS.
