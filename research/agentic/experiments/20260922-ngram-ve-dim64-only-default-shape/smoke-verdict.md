# Verdict: KEEP — 20260922-ngram-ve-dim64-only-default-shape (smoke)

- Hypothesis: At the pinned defaults the four hashed n-gram value-embedding tables (rows=1048576, dim=128, 4 tables = ~537M of the 592.9M parameters; core model is ~55.7M per the --no-ngram-ve smoke row) dominate per-step parameter traffic (gather, RMSprop update, and gradient reduction across the 4 ranks). The --no-ngram-ve smoke shows that removing them entirely raises the 300 s step count from 334-335 to 410 (~+22%) but costs ~+0.008 BPB at smoke horizon, so the tables carry real information. Halving the table width to dim=64 should recover a substantial fraction of that step-rate headroom (predicted +6% to +12% steps in 1800 s) while retaining most of the n-gram information (address space and number of tables unchanged; only the per-slot vector width halves). Because the only 30-minute win at this commit (--bf16-sdpa-input: 2233 vs 2076 steps, 1.002028 vs 1.005967) came from a throughput lever with unchanged capacity, a throughput lever with a mild capacity reduction is predicted to net a lower public_val_bpb than control.
- Single causal change: Pass --ngram-ve-dim 64 only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1). No precision flag (no --bf16-sdpa-input, --bf16-norm-output, --ngram-rms-fp32), no kernel flag (no --nki-*), no --compile-* change, no other ngram-ve option (--ngram-ve stays on; tables, rows/table-mult, trigram, opt=rmsprop, flat_lr, bf16, clip-exclude, layers, lr, beta2 all untouched), no optimizer/schedule/batch/architecture/evaluation change. The only behavioral delta is the per-slot width of the hashed n-gram value-embedding tables: 128 -> 64. Pre-full checks: the candidate's resolved ngram_ve oneliner must read 'True tables=4 rows=1048576 dim=64 ...' (controls read dim=128) and reported num_params_M must fall to roughly 55.7 + 4*1048576*64/1e6 ~ 324M (+/- a few M for any dim-dependent projection). If the oneliner still shows dim=128 or the run fails to construct the model, classify NOT-EXERCISED and do not spend the full run. Comparison: same-host pinned control on public_val_bpb (trn21 -> 2071 steps / 1.005734; trn22 -> 2076 steps / 1.005967). Related, distinguished evidence: al-20260922-no-ngram-ve-only-default-shape-smoke-r1 (trn21, 410 steps, 1.173413) tests total removal; this proposal tests the intermediate width and is predicted to land between control and the no-ngram-ve arm on both steps and BPB. Outcome classes, mutually exclusive, evaluated in order: (1) CONFOUNDED-SHORT if candidate steps < 90% of same-host control steps (experiment-protocol.md section 5; record as throughput regression only); (2) FALSIFIED-INERT if steps within +/-1% of control and |delta BPB| < 0.0005 (width change did not alter throughput or quality); (3) SUCCESS if steps >= control and public_val_bpb <= same-host control - 0.001; (4) FALSIFIED-CAPACITY if steps >= +3% of control but public_val_bpb >= same-host control (throughput gained, but halved n-gram width lost more than it bought); (5) INCONCLUSIVE otherwise (small step gain and/or |delta BPB| < 0.001).
- Flags: `--ngram-ve-dim 64`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-ngram-ve-dim64-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.163309 | 1.165604 |
| num_steps | 355.0 | 335.0 |
| steps/s | 1.2491203377902886 | 1.1775043936731107 |
| params (M) | 324.3 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; correctness gates passed, and the candidate is eligible for a full run.
Reducing ngram embedding dimension from 128 to 64 lowers parameters from 592.9M to 324.3M.
Candidate validation BPB is 1.163309 versus 1.165604, an improvement of 0.002295.
Step rate rises from 1.1775 to 1.2491 steps/s, about 6.1%, exceeding the required 3% gain.
The candidate completes 355 steps, above the 345-step minimum, versus 335 for control.
Both runs pass causality checks, exit successfully, and remain within budget and evaluation timeout.
Commit, source hashes, seed, host, compiler flags, and launch shape match; neither source tree is dirty.
Confounder: the time-limited candidate trains on more steps and tokens, so the BPB gain does not isolate embedding dimension at equal training exposure.
This single smoke comparison supports a full run, not a confirmed full-budget quality improvement.
