# Verdict: KEEP — control-trn22-lnc2x4-9a4caa7b (smoke)

- Hypothesis: matched control at authoritative commit
- Single causal change: none
- Flags: ``
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2`; control label: `None`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- control run correct; usable as matched control

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | None | 1.165604 |
| num_steps | None | 335 |
| steps/s | None | None |
| params (M) | None | None |

These are development measurements, not leaderboard scores.
