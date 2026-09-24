# IMPL-1: let n-gram table arms run under 35c9's optimizer ownership

| Field | Value |
|---|---|
| Status | **Drafted and CPU-verified. Needs an independent Verifier receipt plus a Neuron smoke before any scored run** |
| Base bytes | `submissions/dense6x1024_public0992851/train.py`, SHA-256 `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| Result bytes | SHA-256 **`9e5dce344674b8217d8749ca3e2ce8c1961a73a9c10c1c6759f645a62ed66783`** |
| Patch | `impl1.patch` (+8 lines, 2 hunks, no deletions) |
| Apply | `patch submissions/dense6x1024_public0992851/train.py < workflow/proposals/IMPL-1/impl1.patch`, or copy it to a new research path; then check the SHA-256 above |
| Unblocks | A1 (`--ngram-ve-layers=-1`), A2 (`1,-1`), A3/A4, and any other `--ngram-ve … --ngram-ve-opt rmsprop` arm |

## Problem (reproduced, not inferred)

When 35c9 runs as `__main__`, `_install_submission_training()` wraps `GPT.setup_optimizer` so that every
optimizer passes through `OwnedOptimizer`. Its `make_plan()` raises
`ValueError('ownership probe is restricted to AdamW and Muon')` for the `kind='rmsprop'` group that the
table flags create. It would also reject the BF16 table params. The G2 harness reproduces this on the exact 35c9 bytes:
`workflow/evidence/g2-a1m8-35c9.json` and `g2-a2m8-35c9.json`, FAIL in about 10 s. The full-size A1/A2 argv fails
identically: `g2-a1-35c9.json`, `g2-a2-35c9.json`.

A second trap: `OwnedOptimizer.step()` sends **every non-AdamW owned group to `_muon_step`**. Relaxing the
`make_plan` check alone would therefore silently apply Muon to the tables. Both hunks are required.

## Change (exact)

1. `make_plan`: `if group['kind'] == 'rmsprop': continue`. Table groups are left out of the ownership plan, so the
   plan for every other group is unchanged.
2. `OwnedOptimizer.step`: for `rmsprop` groups, call `opt._rmsprop_step(source)` when `update_adamw` is true (the same
   gate `MuonAdamW.step` uses), then `continue`. The tables stay replicated. Their gradients were already all-reduced
   (AVG) by `clip_and_sync_gradients`, with separate n-gram clipping, so every rank applies the identical update. This
   is how the historical table runs synchronized before ownership existed.

Unchanged: model, forward, loss, data, schedules, clipping, checkpoint format and `load_for_eval`.

## CPU acceptance evidence (gloo, 4 ranks, the file's real `main()` loop via G2)

| Test | Result | Evidence |
|---|---|---|
| (a) `--no-ngram-ve`: ownership plan identical to 35c9 | plan SHA-256 `b89d6222…` equal | `lockstep3-35c9-vs-impl1.json` |
| (a) `--no-ngram-ve`: 3 real steps bit-identical to 35c9 | all ranks' losses and all 62 parameter hashes identical | `lockstep3-35c9-vs-impl1.json` |
| Comparator sensitivity | `--matrix-lr 0.0175` against base: step-0 loss equal, then diverges; 62/62 hashes differ | `lockstep3-35c9-vs-35c9-mlr0175.json` |
| (b) A1 and A2 layouts run | PASS; all table tensors changed; 0 cross-rank mismatches; checkpoint strict-loads through `load_for_eval` with deterministic logits | `g2-a1m8-impl1.json`, `g2-a2m8-impl1.json` |
| (b) All ranks' tokens reach the update | about 2,020 rows touched per table = distinct n-grams in 4 ranks × 512 tokens (a rank-local update would touch about 512) | `impl1-one-step-rmsprop-check.txt` |
| (c) Exactly one table step per optimizer step | `_rmsprop_step` called 2× in 2 steps on each rank; max \|table\| after step 1 = LR exactly (0.25976562 BF16), no entry above LR | `g2-a2m8-impl1-trace.json`, `impl1-one-step-rmsprop-check.txt` |
| (c) Update equals an independent float64 recomputation | max error 0.68% of LR (step 1) and ≤2.4% (step 2), i.e. BF16 storage/arithmetic rounding; untouched rows unchanged | `g2-a2m8-impl1-trace.json` |

**Scope limits.**
- CPU runs used `--ngram-ve-table-mult 8`. The row count is a runtime size, so the code path is identical. The
  full-size (m64) arms need more than this sandbox's 15 GB of RAM at 4 CPU ranks.
- BF16 device numerics, NKI kernels, Neuron compile time and step time are **not** covered. They belong to the
  Neuron gates below.

## Remaining gates before a scored A1/A2 run

1. **Verifier receipt.** Independent review of the patch. Re-run G2 on the result bytes with the exact A1 and A2
   argv (the Verifier's machine may have enough RAM for m64); confirm `g2-a1-35c9.json` still FAILs on the exact
   35c9 bytes.
2. **Neuron smoke, one host, about 20 min.** Exact A2 argv plus `--num-steps 50 --no-eval-public`. The log must show
   finite losses, `SUBMISSION_OPTIMIZER_INTEGRATION` with the same `elements_by_owner` as the control, and step
   time. Record the startup (compile) time: warm cache for research; a cold measurement is required only before any
   submission candidate.
3. **Control equivalence on Neuron.** The next same-host control runs on the IMPL-1 bytes with `--no-ngram-ve`.
   Its first logged losses must match the 35c9 control's to printed precision. Its full-run 20M BPB must be within
   ±0.0008 of the 35c9 control.

## Research lead found while testing (not part of IMPL-1)

At step 1 the median touched table-gradient element is **about 6e-10**, and RMSProp uses `eps = 1e-10`
(`g2-a2m8-impl1-trace.json`, 2,048 tokens per step). The loss is a mean over tokens, so a real 131,072-token step
should give per-row gradients roughly 64× smaller for rarely hit rows, about 1e-11. That would make the table
update ε-dominated: roughly `lr·g/ε`, about 10% of the intended step for rare n-grams early in training. This is a
hypothesis, not a measurement at full batch. **Cheapest falsifier:** log the median/percentiles of |table grad|
over touched rows at steps 1, 100 and 1,000 in the Neuron smoke. If they are ≲1e-10, queue **A-eps** (table
RMSProp `eps` 1e-10 → 1e-15, a one-constant change behind a flag) on the adopted table base.
