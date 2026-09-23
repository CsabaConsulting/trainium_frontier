# trn21 archived-worktree cleanup — 2026-09-23 r6

Purpose: provide a stable margin above the mandatory 7.4 GiB launch gate after staging the
verified attention-scale full checkout consumed the prior narrow margin. The controller opened a
breaker at 7.3 GiB and did not launch.

Human authority: the user explicitly approved archived-run cleanup in the active task.

## Deleted targets

Only these three literal closed-run worktrees were removed:

1. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260923-demon-beta1-r2-default-shape--al-20260923-demon-beta1-r2-default-shape-full-r3`
2. `/home/ubuntu/workspace/autoloop_runs/7e500778--20260923-unet-skip-zero-init-r2--al-20260923-unet-skip-zero-init-r2-smoke-r1`
3. `/home/ubuntu/workspace/autoloop_runs/7e500778--20260923-unet-skip-zero-init-r2--al-20260923-unet-skip-zero-init-r2-full-r3`

## Pre-delete gates

- Host `ip-172-31-34-85` was idle; no training, phase timer, or reconciliation process was present.
- `/dev/neuron0` had no holder.
- Every target resolved directly under `/home/ubuntu/workspace/autoloop_runs`.
- The Demon checkout was at `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`; both U-Net
  checkouts were at `7e500778265cf2c55b39ab00666b73072970f089`.
- No target contained `final.pt`, `*.pt`, or `*.ckpt`; all preserved checkpoints had already been
  hash-verified and pruned by the controller.
- For all three runs, the remote per-run launch JSON, run log, and experiment ledger hashes
  matched the pushed artifacts byte-for-byte.
- Durable verdicts and checkpoint-prune receipts already existed: Demon full DISCARD
  `74fc7adf` / prune `5dcef2a5`; U-Net smoke KEEP `798abdb4` / prune `cfbb4fb1`;
  U-Net full DISCARD `e856a210` / prune `02b2c2b9`.

## Result

- Reported directory sizes: 190,085,697; 190,081,878; and 190,102,447 bytes.
- Available bytes before: 7,838,167,040.
- Available bytes after: 8,452,919,296.
- Filesystem-reported gain: 614,752,256 bytes.
- All three literal targets were confirmed absent after deletion.

The removed checkouts are reconstructable from their pinned Git commits. All experiment evidence
and verdicts remain in the pushed repository. No checkpoint, active process, staged attention-scale
full checkout, or AWS instance lifecycle state was modified.
