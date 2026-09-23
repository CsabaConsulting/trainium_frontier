# trn21 archived-worktree cleanup — 2026-09-23 r4

Purpose: recover enough disk for the already-verified
`20260923-attn-scale-012-default-constant-code` smoke run after its pre-launch safety gate
stopped at 7.17 GiB free (minimum 7.4 GiB).

Human authority: the user explicitly approved cleanup in the active submission/research task.

## Deleted targets

Only these two literal archived autoloop worktrees were removed:

1. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-batch-ramp-only-default-shape--al-20260922-batch-ramp-only-default-shape-smoke-r1`
2. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-batch-ramp-only-default-shape--al-20260922-batch-ramp-only-default-shape-full-r1`

## Pre-delete gates

- Host `ip-172-31-34-85` was idle; no `train.py`, `phase_timer`, or `reconcile` process was present.
- `/dev/neuron0` had no holder.
- Both targets resolved directly under `/home/ubuntu/workspace/autoloop_runs` and their
  basenames matched an exact allow-list.
- Both worktrees were at authoritative commit
  `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`.
- Neither worktree contained a `final.pt`, `*.pt`, or `*.ckpt` file.
- The remote per-run launch JSON, run log, and experiment ledger hashes matched the pushed
  artifacts under
  `research/agentic/experiments/20260922-batch-ramp-only-default-shape/artifacts/{smoke,full}`.
- Durable verdict commits already existed: smoke KEEP `1f224e52`, full CONFOUNDED
  `75abc598`, and post-run evidence enforcement `691faccb`.

## Result

- Reported directory sizes: 190,065,881 and 190,089,843 bytes.
- Available bytes before: 7,702,913,024.
- Available bytes after: 8,112,713,728.
- Filesystem-reported gain: 409,800,704 bytes.
- Both literal targets were confirmed absent after deletion.

The removed worktrees are reconstructable from the pinned Git commit. Their experiment evidence
and verdicts remain in the pushed repository. No checkpoint was deleted, no active or staged
attention-scale worktree was touched, and no instance lifecycle action was performed.
