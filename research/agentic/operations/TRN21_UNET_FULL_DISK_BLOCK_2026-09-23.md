# Trn21 U-Net full-promotion disk block — 2026-09-23

## Outcome before the block

Proposal `20260923-unet-skip-zero-init-r2` passed independent proposal and implementation
verification. Its smoke run used candidate commit `7e500778265cf2c55b39ab00666b73072970f089`,
label `al-20260923-unet-skip-zero-init-r2-smoke-r1`, box `trn21`, and seed 42.

The run completed with return code 0, 330 steps, public development BPB `1.167766`, exact
causality (`max_abs=0`), an explained clean post-run worktree, and a SHA-verified checkpoint.
The independent Analyst returned `KEEP`, making it eligible for a full run. Evidence was committed
and pushed in `798abdb4`; the 1,297,958,776-byte smoke checkpoint was then hash-verified and pruned
in `cfbb4fb1`.

## Full-run gate

No full run started. Two fresh inventories measured `7.26 GiB` free on idle `trn21`, below the
documented `7.40 GiB` launch gate. After the second identical failure, the controller opened its
breaker at `2026-09-23T03:11:24Z`. The durable inventory is
`experiments/20260923-unet-skip-zero-init-r2/full-inventory.json`.

## Read-only storage attribution

After the breaker opened, a read-only size inventory measured:

- root filesystem free: `7,791,947,776` bytes;
- `/home/ubuntu/workspace/autoloop_runs`: `3,893,264,384` bytes across 19 isolated controller run
  directories, each approximately 205 MB after checkpoint pruning;
- `/home/ubuntu/autoloop/bundles`: `255,463,424` bytes;
- `/home/ubuntu/.cache`: `10,260,152,320` bytes, including the data/compiler and package caches.

No file or directory was deleted. The standing authorization permits hash-verified pruning of
controller-created scratch checkpoints after evidence is pushed, but it does not currently permit
removing run worktrees, source, logs, bundles, or caches. Resuming the full run therefore requires
one of the following human-coordinated actions:

1. explicitly authorize selective removal of archived controller-created run worktrees after each
   candidate's durable evidence and checkpoint-prune receipt are verified; or
2. have the account owner free or add storage without touching the active data/compiler caches.

The breaker must remain open until a fresh inventory passes the configured gate. Lowering the gate
again or deleting unknown/teammate data is not an accepted recovery.
