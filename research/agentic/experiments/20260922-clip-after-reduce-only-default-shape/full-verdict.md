# Verdict: DISCARD — 20260922-clip-after-reduce-only-default-shape (full)

- Hypothesis: At the pinned 9a4caa7b defaults the 4-rank data-parallel step clips each rank's locally accumulated gradient by its OWN norm before all_reduce (clip-before-reduce, the mode the fs-w15 ledger row calls 'baseline clip-before-reduce'). Because each local gradient carries roughly 4x the sampling variance of the 4-rank mean, E||g_r||^2 = ||g_mean||^2 + variance terms > ||g_mean||^2, so (a) clipping engages more often and more aggressively than the configured threshold intends and (b) ranks apply different scale factors, so the averaged update is a variance-weighted mean rather than the true batch gradient. Clipping the averaged gradient after all_reduce (--clip-after-reduce) restores single-device clipping semantics with one scale factor on the true global gradient. The fresh-start ledger contains one near-paired full-budget comparison on the dense 6x1024 model: fs-w12 (no flag) 0.98171 BPB @2977 steps vs fs-w16 (only --clip-after-reduce added per its description) 0.981174 BPB @3005 steps, after which every later fs run carried the flag as 'winning clip-after-reduce'. This mode has never been tested in isolation on the 592.9M ngram-VE default shape at 9a4caa7b, and it is throughput-neutral by construction (the same single norm computation, only ordered after the reduction), so it is a clean isolated optimizer-side test.
- Single causal change: Pass --clip-after-reduce only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --standalone --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1). No precision, kernel, compile, architecture, batch, schedule, or evaluation switch is passed; --clip-foreach and --grad-bucket are left at their resolved defaults. The only behavioral delta is the ORDER of gradient-norm clipping relative to the all_reduce: clip the averaged global gradient once after reduction instead of clipping each rank's local gradient before reduction. Whether the default at 9a4caa7b is clip-before-reduce is a hypothesis inferred from the fs-w15/fs-w16 descriptions and from every later fs run still passing the flag explicitly; it MUST be confirmed statically before launch (see abort conditions). Matched comparison: same-host pinned control (trn21 -> 2071 steps / 1.005734; trn22 -> 2076 steps / 1.005967) on public_val_bpb from the controller's shared scorer (private shard not accessible, not claimed). Outcome classes, mutually exclusive, evaluated in order: (1) CONFOUNDED if candidate steps are more than 10% BELOW the same-host control (experiment-protocol.md section 5 quality rule); (2) THROUGHPUT-ANOMALY (separately declared conservative rule) if candidate steps are more than 10% ABOVE the same-host control — the mechanism predicts no rate change, so record as an unexplained result and do not credit BPB; (3) INCONCLUSIVE if steps are between 2% and 10% below control (throughput-neutrality prediction failed, quality not attributable); (4) FALSIFIED-INERT if steps are within [-2%, +10%] and |public_val_bpb - control| <= 0.0003 (below the observed cross-box same-seed spread of 0.00023 — numerically indistinguishable, consistent with the mode already being active or clipping never engaging); (5) SUCCESS if steps are within [-2%, +10%] and public_val_bpb <= control - 0.001; (6) FALSIFIED-QUALITY otherwise (steps within [-2%, +10%] and public_val_bpb > control - 0.001 with |delta| > 0.0003). Attribution note: a SUCCESS establishes a fixed-budget public-BPB improvement of the flag at this shape, not the specific variance mechanism; matched-step trajectory logs (loss at equal step indices, and the fraction of steps where the clip coefficient < 1 if logged) are the evidence for the mechanism claim and are reported separately.
- Flags: `--clip-after-reduce`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-clip-after-reduce-only-default-shape-full-r2`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1`
- Budget: num_steps=5000 max_train_seconds=1800
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- public_val_bpb delta +0.00024 did not meet threshold -0.00200

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.005974 | 1.005734 |
| num_steps | 2074.0 | 2071.0 |
| steps/s | 1.162100072841374 | 1.1603541012998655 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the full tier: --clip-after-reduce produced public_val_bpb 1.005974 versus 1.005734 for control.
The candidate-minus-control delta was +0.000240 (worse), failing the required threshold of -0.002000.
Both runs used the same clean commit and source hashes, seed, host, launch shape, and 592.9M parameters.
Candidate throughput was slightly higher: 1.16210 versus 1.16035 steps/s, completing 2074 versus 2071 steps.
The minor confounder is unequal training exposure under the time budget: the candidate completed three additional steps, so this is not an exactly step-matched comparison.
Both runs passed causality checks, exited successfully, and stayed within the charged budget; these checks do not alter the rejection.
