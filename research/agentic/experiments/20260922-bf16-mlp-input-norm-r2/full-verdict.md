# Verdict: CONFOUNDED — 20260922-bf16-mlp-input-norm-r2 (full)

- Hypothesis: In the pinned control, every transformer Block feeds its MLP an FP32 norm(x) output, so the two largest matmuls per block (MLP up/down projections) and their backward run in FP32, while the same per-block pre-attention SDPA boundary already proved that a single BF16 activation boundary buys throughput without a quality loss (--bf16-sdpa-input: 2233 steps / 1.002028 vs control 2076 / 1.005967 on trn22). The complete --bf16-norm-output policy buys far more throughput (2716 steps) but regresses quality (1.016279 vs 1.005734 on trn21) because it also casts the final norm feeding the LM head and every attention/QK norm and pushes far past the ~2195-step data wall. Casting ONLY the MLP-input norm output to BF16 isolates the cheapest-to-quality, most-FLOP-heavy boundary: it should recover a large fraction of the step gain (predeclared +8..15% steps) while keeping the FP32 residual stream, attention norms, final norm and LM head unchanged, giving a lower fixed-budget public_val_bpb than the same-host control. This is the r2 resubmission of the verifier-PASSED 20260922-bf16-mlp-input-norm-only proposal whose exact patch was returned NEEDS_WORK solely because the parallel-block path bypassed the toggle; r2 specifies that path explicitly. No run of this change exists in the ledger.
- Single causal change: One numerical-format boundary change confined to train.py: a new module-level toggle USE_BF16_MLP_NORM_OUTPUT (default True; env NEURON_COMPETITION_R1_BF16_MLP_NORM_OUTPUT='0' disables; CLI --bf16-mlp-norm-output/--no-bf16-mlp-norm-output; appended to ARCH_TOGGLE_NAMES so checkpoints persist it and load_for_eval restores it) casts exactly ONE tensor per transformer Block to torch.bfloat16 (the same cast target the existing --bf16-norm-output policy uses): the tensor passed as the MLP module's input argument in Block.forward, immediately after it is produced by norm(). The cast is guarded by a dtype check so that under --bf16-norm-output (where norm() already returns BF16) behaviour is byte-for-byte unchanged. It is applied consistently in all three Block.forward variants: (a) default serial path (parallel_block=False, mlp_sandwich=False; the path the candidate exercises): mlp_in = norm(x) -> cast -> self.mlp(mlp_in); (b) mlp_sandwich path: the same at its MLP-input norm, post-MLP sandwich norm untouched; (c) parallel_block path: the shared h = norm(x) is NOT modified in place (attention keeps consuming FP32 h); a separate h_mlp = cast(h) is created and passed as the MLP input argument (e.g. self.mlp(h_mlp, fc=fc)); when parallel-block coalescing != 'none' precomputes fc inside a fused qkv+fc projection on shared h, that fused projection is left untouched (explicitly documented, one startup log line) and only the direct MLP input argument is cast. Acknowledged downstream propagation: the MLP up-projection/activation/down-projection consume BF16 activations via the already-exercised BF16-input Linear weight-cache path, and the MLP output is added to the FP32 residual with the existing promotion semantics (exactly the path the --bf16-norm-output full run exercised end-to-end). NOT changed: norm(q_in)/norm(k_in)/norm(v_in), the norm feeding local_conv, QK norms/rope_norm, the initial embedding norm in GPT.forward, the final norm before lm_head, lm_head, Linear.forward, the BF16 weight-cache logic, MLP internals, softcap/CE, ngram-ve, optimizer, schedule, batch, data, eval, causality, launch shape or environment. num_params_M must remain 592.9 and every existing config oneliner must be identical to the control except for one added field. Launch smoke and full with the exact control command/environment and no additional flags. Full-run classification vs same-host pinned control (trn21 2071 / 1.005734; trn22 2076 / 1.005967), in order: (1) HALT-INERT if static inspection finds the MLP-input norm output is already BF16 under default flags (no patch, no run); (2) CONFOUNDED-SHORT if full-run steps < 90% of same-host control (<1864 trn21, <1869 trn22); (3) FALSIFIED-INERT if steps within +/-2% of control (the MLP-input FP32 boundary is not a measurable cost); (4) SUCCESS if steps >= +3% of control and public_val_bpb < same-host control by >= 0.002; (5) FALSIFIED-QUALITY if steps >= +3% and public_val_bpb >= control (BF16 at the MLP input costs more quality than the extra steps buy); (6) INCONCLUSIVE otherwise (+2..3% steps, or >= +3% with a gain smaller than 0.002). Verdicts use only fixed-budget public_val_bpb; the Analyst records as attribution notes (not verdict inputs) the fraction of steps beyond the ~2195-step data wall and the smoothed training-loss value at the control's final step count.
- Flags: ``
- Commit: `b8fab55a4a6a21098cad7200a0001000aa9d76f3`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-mlp-input-norm-r2-full-r2`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- steps differ by more than 10% (2346.0 vs 2071.0)

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.020187 | 1.005734 |
| num_steps | 2346.0 | 2071.0 |
| steps/s | 1.3144329896907216 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

The full-tier verdict is CONFOUNDED.
The BF16 MLP norm-output candidate recorded 1.020187 validation bpb versus 1.005734 for the control, a worsening of 0.014453 bpb.
The confounder is unequal training steps: 2,346 versus 2,071, approximately 13.3% more for the candidate and above the 10% threshold.
Both trained for 1,784.8 seconds, but candidate throughput was 1.314 steps/s versus 1.160 steps/s, yielding 307.5M versus 271.5M training tokens.
Parameter count (592.9M), seed, hardware configuration, and time budget matched; both passed causality checks and stayed within budget.
The observed bpb difference therefore does not isolate the numerical-format change from the unequal training exposure.
