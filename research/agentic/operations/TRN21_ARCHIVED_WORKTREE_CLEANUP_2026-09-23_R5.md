# trn21 archived-worktree cleanup — 2026-09-23 r5

Purpose: restore the mandatory 7.4 GiB pre-launch margin for the verified
`20260923-attn-scale-012-default-constant-code` full run. After the smoke artifacts were
archived and its checkpoint was hash-verified and pruned, the controller measured only 7.3 GiB
free and correctly did not launch the full tier.

Human authority: the user explicitly approved archived-run cleanup in the active task.

## Deleted target

Only this literal abandoned archived worktree was removed:

`/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-clip-after-reduce-only-default-shape--al-20260922-clip-after-reduce-only-default-shape-full-r1`

## Pre-delete gates

- Host `ip-172-31-34-85` was idle; no training, phase timer, or reconciliation process was present.
- `/dev/neuron0` had no holder.
- The target resolved directly under `/home/ubuntu/workspace/autoloop_runs` and its basename
  matched the exact allow-list entry.
- The checkout was at authoritative commit
  `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`.
- `git status --porcelain` was empty.
- No `final.pt`, `*.pt`, or `*.ckpt` file existed anywhere in the target.
- This was the unused clean `full-r1` checkout. The later completed full result is separately
  durable as `full-r2`, with its DISCARD verdict and checkpoint-prune evidence already pushed
  under `research/agentic/experiments/20260922-clip-after-reduce-only-default-shape`.

## Result

- Reported directory size: 189,996,091 bytes.
- Available bytes before: 7,838,343,168.
- Available bytes after: 8,043,139,072.
- Filesystem-reported gain: 204,795,904 bytes.
- The literal target was confirmed absent after deletion.

The removed clean checkout is reconstructable from the pinned Git commit. No experiment evidence,
checkpoint, active process, current attention-scale checkout, or AWS instance lifecycle state was
modified.
