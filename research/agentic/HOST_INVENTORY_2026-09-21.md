# Read-only Trn2 inventory — 2026-09-21

Captured at approximately `2026-09-21T16:42Z`. No process was killed, no file was changed remotely,
no code was synchronized, and no training or diagnostic was launched.

## Shared status

- SSH authentication succeeds with the provided team key.
- Both machines report `trn2.3xlarge`, one device, four NeuronCores, 96 GB device memory, and LNC2.
- `neuron-ls` shows no active process.
- `/dev/neuron0` has no holder.
- No `torchrun`, experiment runner, phase timer, Neuron compiler, or parallel compiler process was
  found by the bounded process query.
- Each host has approximately 128 GB `MemAvailable`.
- Root filesystems are 93% used, leaving approximately 15–16 GB. This currently clears the 8 GiB
  protocol minimum but leaves little margin for new profiles or checkpoints.

Idle hardware is not permission to launch. Every future run still needs a fresh preflight, unique
label, completed proposal, clean immutable candidate, host lock, and human approval.

## trn21

- Expected source195 path does not exist; this is normal because later work used different immutable
  wave worktrees.
- Relevant checked paths and commits:
  - `trainium_freshstart_20260918_wave196`: `d8461533`, 6 dirty/untracked entries
  - `trainium_freshstart_20260918_wave198`: `9a793768`, 6 dirty/untracked entries
  - `trainium_freshstart_20260918_wave199`: `5d1ff50e`, 18 dirty/untracked entries
- The original `trainium_frontier` checkout has 405 dirty/untracked entries and must not be used or
  cleaned by the agent loop.
- Many immutable historical wave directories exist. Do not remove them merely to gain disk space;
  follow the repository's preservation/retirement procedure.

## trn22

- Source197/source195 checkout exists at `trainium_freshstart_20260918_wave195`.
- HEAD is `282935ca9b1e30f6c21c397143770b03034363f9`, matching the source197 evidence.
- The checkout contains numerous untracked experiment logs and maintenance receipts. They are
  evidence, not debris. Preserve and reconcile them before declaring this tree clean.

## Local key handling

- `trn2.pem` was received with mode `664`; it was changed locally to `600`.
- `/trn2.pem` is explicitly ignored by Git.
- The key must never be printed, copied into agent context, committed, or supplied to a Bedrock
  model. A per-user SSH public key remains preferable when the teammate can provision one.

## Next operational gate

1. Preserve/reconcile remote untracked receipts without overwriting canonical local evidence.
2. Verify available disk/inodes and select a new clean immutable worktree for an approved proposal.
3. Repeat the complete experiment-protocol preflight immediately before any run.
