# Verdict: FAILED — control-trn22-lnc2x4-9a4caa7b (smoke)

- Hypothesis: matched control at authoritative commit
- Single causal change: none
- Flags: ``
- Commit: `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`
- Box: trn22 (`ip-172-31-32-253`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r1`; control label: `None`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- runner exit code 1
- ledger status 'crash'
- causality_passed is not True
- public_val_bpb missing or non-finite
- final.pt missing
- num_steps missing

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | None | None |
| num_steps | None | None |
| steps/s | None | None |
| params (M) | None | None |

These are development measurements, not leaderboard scores.
