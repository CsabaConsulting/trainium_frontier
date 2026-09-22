# Verdict: CONFOUNDED — 20260922-bf16-mlp-input-norm-r3 (full)

- Hypothesis: In the pinned default configuration the transformer Block feeds an FP32 norm(x) output into the MLP, so the MLP up/down projections run as FP32 matmuls. The ledger shows this class of boundary is the dominant per-step cost: the global --bf16-norm-output policy (all norm outputs BF16) reached 2716 steps vs 2071 (+31%) but regressed public_val_bpb to 1.016279, while the single isolated --bf16-sdpa-input boundary reached 2233 vs 2076 (+7.6%) and improved BPB to 1.002028 (KEEP). BF16 inputs to hidden-layer matmuls are standard mixed-precision practice; the plausible quality culprit in the global policy is the final norm -> lm_head/softcap-CE path (and/or other non-matmul norm sites), not the MLP. Casting exactly one tensor per Block -- the MLP-input norm output -- to BF16 should recover a large share of the step-rate gain (predicted +8..18% steps) while preserving per-step learning, lowering fixed-budget public_val_bpb. This is the narrowest untested precision boundary; the previous attempt (20260922-bf16-mlp-input-norm-only) PASSED proposal review and failed only implementation review because the parallel_block code path bypassed the toggle, which this revision specifies explicitly.
- Single causal change: One numerical-format boundary change confined to train.py: a new module-level toggle USE_BF16_MLP_NORM_OUTPUT (default True; env NEURON_COMPETITION_R1_BF16_MLP_NORM_OUTPUT='0' disables; CLI --bf16-mlp-norm-output/--no-bf16-mlp-norm-output; appended to ARCH_TOGGLE_NAMES so checkpoints persist it and load_for_eval restores it) casts exactly ONE tensor per transformer Block to torch.bfloat16 (the identical cast target the existing --bf16-norm-output policy uses): the norm output that is the direct input of the MLP module. In the default serial path (parallel_block=False, mlp_sandwich=False, the path exercised by the candidate) this is the result of the norm(x) call passed to self.mlp(...). The cast is guarded by a dtype check so that under --bf16-norm-output (norm already returns BF16) behaviour is byte-for-byte unchanged. For consistency of the toggle's meaning across Block.forward variants: (i) the mlp_sandwich path casts its own MLP-input norm output the same way (post-MLP sandwich norm untouched); (ii) the parallel_block path with non-coalesced projections keeps the shared h = norm(x) untouched for attention and passes a separately cast copy of h to self.mlp only; (iii) the parallel_block path with coalesced projections (where the MLP consumes a projection computed jointly with qkv from the shared h) is declared inapplicable: no cast is applied and one explicit startup line is printed. Acknowledged downstream propagation: MLP up-projection/activation/down-projection run on BF16 activations via the already-exercised BF16-input Linear path (Linear.forward selects the weight cache matching input dtype), and the MLP output is added to the FP32 residual with the existing promotion semantics -- the same consumer path the --bf16-norm-output full run exercised end-to-end. NOT changed: norm(q_in)/norm(k_in)/norm(v_in), the norm feeding local_conv, QK norms/rope_norm, the GPT.forward initial embedding norm, the final norm before lm_head, lm_head, logits/softcap/CE, ngram-ve, optimizer, schedule, batch, data, eval, causality, launch shape or environment. num_params_M must remain 592.9 and every existing config oneliner must be identical to the control except for one added field. Launch smoke and full with the exact control command/environment and no additional flags. Full-run classification vs same-host pinned control (trn21 2071 / 1.005734; trn22 2076 / 1.005967), evaluated in order: (1) HALT-INERT if static inspection finds the MLP-input norm output is already BF16 under default flags (no patch, no run); (2) CONFOUNDED-SHORT if full-run steps < 90% of same-host control (<1864 trn21, <1869 trn22); (3) FALSIFIED-INERT if steps within +/-2% of control (the MLP-input FP32 boundary is not a measurable cost); (4) SUCCESS if steps >= +3% of control and public_val_bpb < same-host control by >= 0.002; (5) FALSIFIED-QUALITY if steps >= +3% and public_val_bpb >= control (BF16 at the MLP input costs more quality than the extra steps buy); (6) INCONCLUSIVE otherwise (+2..3% steps, or >= +3% with a gain smaller than 0.002). Verdicts use only fixed-budget public_val_bpb; the Analyst records as attribution notes only (not verdict inputs) the fraction of steps beyond the ~2195-step data wall documented in train.py and the smoothed training-loss log value at the control's final step count.
- Flags: ``
- Commit: `b2a5bf5ba7e6f3c071a4fa941f95b7723534237b`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-mlp-input-norm-r3-full-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- steps differ by more than 10% (2346.0 vs 2071.0)

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.019731 | 1.005734 |
| num_steps | 2346.0 | 2071.0 |
| steps/s | 1.3148013226475368 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

The full-tier verdict is CONFOUNDED.
The candidate’s validation BPB was 1.019731 versus 1.005734 for control, a worsening of 0.013997 (lower is better).
The confounder is unequal training steps: 2,346 versus 2,071, about 13.3% more for the candidate, exceeding the 10% threshold.
Training time was nearly identical, but the candidate ran at 1.315 versus 1.160 steps/second and processed 307.5M versus 271.5M tokens.
Both runs used 592.9M parameters, seed 42, and the same listed execution configuration; both passed causality checks and stayed within budget.
The observed BPB difference therefore cannot be cleanly attributed to the candidate’s BF16 MLP normalization-output change independently of training exposure.
