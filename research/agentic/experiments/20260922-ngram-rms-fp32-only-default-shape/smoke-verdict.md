# Verdict: DISCARD — 20260922-ngram-rms-fp32-only-default-shape (smoke)

- Hypothesis: The default 9a4caa7b model carries 4 hashed n-gram tables of 1,048,576 rows x 128 dims stored in BF16 and trained with a flat-LR RMSProp whose second-moment state math also runs in BF16 (control ledger rows record `ngram_ve: ... opt=rmsprop flat_lr=True bf16=True rms_fp32_state_math=False nki_rms=False`). With beta2 near 1, the per-step increment (1-beta2)*g^2 is frequently smaller than 1/256 of the running state and is lost to BF16 rounding, so the RMSProp denominator for the sparsely-touched rows is biased and the effective per-row step size is erratic. Doing the RMSProp state math in FP32 (--ngram-rms-fp32) removes that quantization bias for a compute cost proportional only to the gathered rows (index_gather=True), so loss per step should improve at essentially unchanged throughput. Supporting paired evidence: fs-w3-w896-direct-control-r1 (trn21, 2026-09-18 09:18, seed 42, 2188 steps, 0.987568) vs fs-w4-w896-nki-rms-fp32-r1 (trn21, same day 09:58, same seed/shape/flags plus FP32 n-gram RMSProp, 2204 steps, 0.986947): -0.00062 BPB with +0.7% steps. Caveats acknowledged: that pair also enabled the fused NKI kernel (--nki-ngram-rms) and spans two commits (00f82d8a -> a62df273, the commit that added the feature), so it is suggestive, not isolating; it was run at width 896 under BF16 matmult autocast, not the default shape under CC='--optlevel=1'. No default-shape run with --ngram-rms-fp32 alone appears in the supplied ledger, DECISION_LOG.md, or CURRENT_STATE.md, and this direction is not among the parked ones.
- Single causal change: Pass --ngram-rms-fp32 only: RMSProp state math for the hashed n-gram value-embedding tables is performed in FP32 instead of BF16. Nothing else changes: no --nki-ngram-rms kernel switch, no table dtype change (--ngram-ve-bf16 stays at its default True), no precision/kernel/optimizer/schedule/shape flag. Launch is otherwise identical to the pinned controls: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 5000 --max-train-seconds 1800 --seed 42, environment NEURON_CC_FLAGS='--optlevel=1' exactly (no autocast flags), NEURON_COMPETITION_R1_SEED=42, evaluated with the unchanged public evaluator settings. The matched comparison is the same-host pinned control: trn21 -> al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1 (2071 steps, 1.005734); trn22 -> al-20260922-control-trn22-lnc2x4-9a4caa7b-full-r1 (2076 steps, 1.005967).
- Flags: `--ngram-rms-fp32`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-ngram-rms-fp32-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke step rate 1.1092 < floor 1.1540 steps/s

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.173336 | 1.165604 |
| num_steps | 316.0 | 335.0 |
| steps/s | 1.1091611091611093 | 1.1775043936731107 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the smoke tier: candidate throughput was 1.1092 steps/s, below the required 1.1540 steps/s floor.
The candidate enabled FP32 RMSProp state math for hashed n-gram value embeddings; the control used BF16 state math.
Throughput fell about 5.8% from the control’s 1.1775 steps/s, yielding 316 versus 335 steps in nearly equal training time.
Validation BPB was worse: 1.173336 versus 1.165604, a delta of +0.007732.
The BPB comparison is confounded by unequal training progress: the slower candidate processed 41.4M versus 43.9M tokens, so this does not isolate a precision effect on quality.
Both runs matched on clean source hashes, commit, seed, host, compiler settings, and parameter count of 592.9M.
Both completed successfully and passed causality checks; neither exceeded budget or evaluation timeout.
The throughput-floor failure determines the discard decision, regardless of the ledger rows’ “keep” status.
