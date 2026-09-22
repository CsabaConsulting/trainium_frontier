# Verdict: KEEP — control-trn21-lnc2x4-9a4caa7b (smoke)

- Hypothesis: matched control at authoritative commit
- Single causal change: none
- Flags: ``
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`; control label: `None`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- control run correct; usable as matched control

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | None | 1.165855 |
| num_steps | None | 334 |
| steps/s | None | None |
| params (M) | None | None |

These are development measurements, not leaderboard scores.
