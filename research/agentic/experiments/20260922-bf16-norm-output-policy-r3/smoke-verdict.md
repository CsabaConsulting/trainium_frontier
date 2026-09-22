# Verdict: KEEP — 20260922-bf16-norm-output-policy-r3 (smoke)

- Hypothesis: Under the pinned environment (NEURON_CC_FLAGS='--optlevel=1', no compiler matmul autocast) the control executes every consumer of norm() output in FP32: the qkv/out/MLP projections, the embedding-normalization consumers, the local-convolution input, and the final LM-head input (train.py norm() 1446-1462; Block.forward; GPT.forward; Linear.forward selects weight/cache by input dtype). The only isolated precision arm run so far, --bf16-sdpa-input, converted a single boundary to BF16 and recovered +7.6% steps (2233 vs 2076) with better public BPB (1.002028 vs 1.005967). The existing --bf16-norm-output policy converts the much larger share of matmul FLOPs and activation traffic to BF16 (and, as a documented side effect, also delivers BF16 Q/K/V to SDPA because CausalSelfAttention aligns V to Q's dtype and rope_norm preserves input dtype). I hypothesize the complete norm-output precision policy yields a larger fixed-budget step gain (>= +8%) than the SDPA arm and that BF16 rounding of normalized activations at this 55.7M-dense/537M-table scale costs less quality than the extra steps recover, giving lower public BPB than control.
- Single causal change: Pass --bf16-norm-output only. This is the complete existing norm-output precision policy of train.py (not a projection-only cast): all norm() outputs become BF16, so qkv/out/MLP projections, embedding-normalization consumers, local-conv input, the final LM-head input, and (via V-to-Q dtype alignment and dtype-preserving rope_norm) SDPA inputs execute in BF16; FP32 master weights, optimizer state, loss code, evaluator, and every other option stay at 9a4caa7b defaults. No --bf16-sdpa-input, no --ngram-rms-fp32, no --nki-*/--compile-* flag, no optimizer/schedule/batch/architecture/ngram-ve/eval change. Launch shape and environment identical to the pinned controls (NEURON_LOGICAL_NC_CONFIG=2, torchrun --standalone --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42, NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1). This arm is NOT an independent boundary from the SDPA KEEP: its scope overlaps and supersedes the SDPA cast; a later stacking run with --bf16-sdpa-input is expected to be near-inert and must be proposed separately with that interaction hypothesis. Pre-full checks from the smoke: resolved precision oneliner shows bf16_norm_output=True and bf16_sdpa_input=False; num_params_M 592.9; ngram_ve oneliner identical to control ('True tables=4 rows=1048576 dim=128 index_gather=True trigram=True opt=rmsprop flat_lr=True bf16=True rms_fp32_state_math=False nki_rms=False'); architecture oneliners identical to control; causality_passed True. Comparison: same-host pinned control on public_val_bpb (trn21 2071 / 1.005734; trn22 2076 / 1.005967). ONE authoritative outcome classification, evaluated in order: (1) HALT-SEMANTIC-REVIEW if any pre-full check fails - do not launch the full run; (2) CONFOUNDED-SHORT if full-run steps < 90% of same-host control (<1864 trn21, <1869 trn22); (3) FALSIFIED-INERT if steps within +/-1% of control (norm-output precision is not a measurable cost); (4) EXCESS-GAIN-REVIEW if steps > +10% of control (>2278 trn21, >2283 trn22): record throughput and BPB; before any quality claim compare the candidate's logged validation trajectory to the control's at matched step counts; if matched-step trajectories are unavailable or noncomparable, quality attribution is recorded unresolved; SUCCESS may be declared only if public_val_bpb < control by >= 0.001 and no matched-step deficit > 0.010 at the control's final step; (5) SUCCESS if steps >= +2% and public_val_bpb < same-host control by >= 0.001; (6) FALSIFIED-QUALITY if steps >= +2% and public_val_bpb >= control (BF16 rounding of normalized activations costs at least as much as the recovered steps); (7) INCONCLUSIVE otherwise, including any gain in [0, 0.001) or steps in (+1%, +2%).
- Flags: `--bf16-norm-output`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-bf16-norm-output-policy-r3-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.13367 | 1.165855 |
| num_steps | 438.0 | 334.0 |
| steps/s | 1.539002108222066 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; the candidate passes the gates and is eligible for a full run.
The candidate enables --bf16-norm-output, applying the existing precision policy to all norm outputs, not just projection inputs.
Validation BPB improves from 1.165855 to 1.133670, a delta of -0.032185 versus the allowed maximum of +0.003.
Step rate rises from 1.1748 to 1.5390 steps/s, about 31.0%, exceeding the 1.1983 steps/s floor.
The candidate completes 438 steps versus 334 for control, clearing the 342-step minimum.
Both runs pass causality checks with zero maximum absolute difference, exit successfully, and remain within budget.
Parameters, seed, source hashes, host, compiler flags, and distributed configuration match.
The main confounder for interpreting BPB is unequal training exposure: the candidate processes 57.4M tokens versus 43.8M in nearly identical training time.
Thus, the BPB gain supports the time-budgeted result but does not isolate a precision-policy benefit at equal steps; this remains a single-seed smoke comparison.
