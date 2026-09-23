# trn22 submission-validation cleanup — 2026-09-23 r1

Purpose: satisfy the immutable r3 Neuron discriminator's mandatory 8 GiB pre-launch disk gate.
After deploying the finalized checkout, trn22 had 8,572,989,440 bytes free, which was 16,945,152
bytes below the gate. The request failed closed and did not launch.

Human authority: the user previously gave explicit permission to clean trn22 while preserving
progress.

## Deleted target

Only this literal archived validation checkout was removed:

`/home/ubuntu/trainium_freshstart_submission_validation_20260923_finalgates`

## Pre-delete gates

- Host `ip-172-31-32-253` was idle; no training, phase timer, reconciliation, or discriminator
  process was present.
- `/dev/neuron0` had no holder.
- The target resolved directly under `/home/ubuntu` and matched the exact allow-list basename.
- It was at commit `5856c32bcd4023a5e6cf2561e5627713a3169d7c`.
- It contained no `final.pt`, `*.pt`, or `*.ckpt` file.
- Its preflight, launch, job, process, postflight, and log artifacts for
  `fs-20260923-dense6x1024-lean-review-fresh-neuron-contract-r3` matched the durable pushed files
  byte-for-byte. The evidence is retained in the submission-validation repository, including
  commit `5856c32b` and later readiness commits.

## Result

- Reported directory size: 139,083,649 bytes.
- Available bytes before: 8,572,977,152.
- Available bytes after: 8,727,019,520.
- Filesystem-reported gain: 154,042,368 bytes.
- The literal target was confirmed absent after deletion.

The checkout is reconstructable from Git and all run evidence remains durable. No checkpoint,
current r3 discriminator checkout, process, dataset, cache, or AWS instance lifecycle state was
modified.
