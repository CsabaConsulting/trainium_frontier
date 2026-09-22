# Verdict: KEEP — 20260922-clip-after-reduce-only-default-shape (smoke)

- Hypothesis: At the pinned 9a4caa7b defaults the 4-rank data-parallel step clips each rank's locally accumulated gradient by its OWN norm before all_reduce (clip-before-reduce, the mode the fs-w15 ledger row calls 'baseline clip-before-reduce'). Because each local gradient carries roughly 4x the sampling variance of the 4-rank mean, E||g_r||^2 = ||g_mean||^2 + variance terms > ||g_mean||^2, so (a) clipping engages more often and more aggressively than the configured threshold intends and (b) ranks apply different scale factors, so the averaged update is a variance-weighted mean rather than the true batch gradient. Clipping the averaged gradient after all_reduce (--clip-after-reduce) restores single-device clipping semantics with one scale factor on the true global gradient. The fresh-start ledger contains one near-paired full-budget comparison on the dense 6x1024 model: fs-w12 (no flag) 0.98171 BPB @2977 steps vs fs-w16 (only --clip-after-reduce added per its description) 0.981174 BPB @3005 steps, after which every later fs run carried the flag as 'winning clip-after-reduce'. This mode has never been tested in isolation on the 592.9M ngram-VE default shape at 9a4caa7b, and it is throughput-neutral by construction (the same single norm computation, only ordered after the reduction), so it is a clean isolated optimizer-side test.
- Single causal change: Pass --clip-after-reduce only. Every other option stays at the 9a4caa7b defaults used by the two pinned controls: identical launch shape (NEURON_LOGICAL_NC_CONFIG=2, torchrun --standalone --nproc_per_node=4, --num-steps 5000, --max-train-seconds 1800, --seed 42), identical environment (NEURON_CC_FLAGS exactly '--optlevel=1', NEURON_COMPETITION_R1_SEED=42, PYTHONUNBUFFERED=1). No precision, kernel, compile, architecture, batch, schedule, or evaluation switch is passed; --clip-foreach and --grad-bucket are left at their resolved defaults. The only behavioral delta is the ORDER of gradient-norm clipping relative to the all_reduce: clip the averaged global gradient once after reduction instead of clipping each rank's local gradient before reduction. Whether the default at 9a4caa7b is clip-before-reduce is a hypothesis inferred from the fs-w15/fs-w16 descriptions and from every later fs run still passing the flag explicitly; it MUST be confirmed statically before launch (see abort conditions). Matched comparison: same-host pinned control (trn21 -> 2071 steps / 1.005734; trn22 -> 2076 steps / 1.005967) on public_val_bpb from the controller's shared scorer (private shard not accessible, not claimed). Outcome classes, mutually exclusive, evaluated in order: (1) CONFOUNDED if candidate steps are more than 10% BELOW the same-host control (experiment-protocol.md section 5 quality rule); (2) THROUGHPUT-ANOMALY (separately declared conservative rule) if candidate steps are more than 10% ABOVE the same-host control — the mechanism predicts no rate change, so record as an unexplained result and do not credit BPB; (3) INCONCLUSIVE if steps are between 2% and 10% below control (throughput-neutrality prediction failed, quality not attributable); (4) FALSIFIED-INERT if steps are within [-2%, +10%] and |public_val_bpb - control| <= 0.0003 (below the observed cross-box same-seed spread of 0.00023 — numerically indistinguishable, consistent with the mode already being active or clipping never engaging); (5) SUCCESS if steps are within [-2%, +10%] and public_val_bpb <= control - 0.001; (6) FALSIFIED-QUALITY otherwise (steps within [-2%, +10%] and public_val_bpb > control - 0.001 with |delta| > 0.0003). Attribution note: a SUCCESS establishes a fixed-budget public-BPB improvement of the flag at this shape, not the specific variance mechanism; matched-step trajectory logs (loss at equal step indices, and the fraction of steps where the clip coefficient < 1 if logged) are the evidence for the mechanism claim and are reported separately.
- Flags: `--clip-after-reduce`
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-clip-after-reduce-only-default-shape-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- correctness gates passed; step rate within tolerance; eligible for full run

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.165428 | 1.165855 |
| num_steps | 335.0 | 334.0 |
| steps/s | 1.1775043936731107 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

KEEP at smoke tier; correctness gates passed and the candidate is eligible for a full run.
The candidate adds only --clip-after-reduce; source hashes, clean commit, seed, host, launch shape, and parameter count match the control.
Validation BPB is 1.165428 versus 1.165855, a decrease of 0.000427, satisfying the maximum allowed delta of +0.003.
The candidate completed 335 steps versus 334, exceeding the 325-step minimum.
Step rate is 1.17750 versus 1.17482 steps/s, about 0.23% faster and above the supplied 1.15132 floor.
Both runs compiled, exited successfully, passed causality checks, and stayed within budget and evaluation timeout.
The extra candidate training step is a confounder for the small BPB improvement; separate run times also leave room for runtime variability.
This smoke result supports full-run testing, not a demonstrated quality or throughput gain.
