# trn21 archived-worktree cleanup — 2026-09-23 r7

Purpose: restore a safe margin above the mandatory 7.4 GiB launch gate after the closed
double-hash smoke checkout reduced free space below the gate.

Human authority: the active task explicitly approved archived-run cleanup while preserving all
progress; r3-r6 recorded the same exact-target protocol.

## Deleted targets

Only these literal controller-created, closed-run checkouts were removed:

1. `/home/ubuntu/workspace/autoloop_runs/05422cf0--20260923-ngram-ve-double-hash-r2--al-20260923-ngram-ve-double-hash-r2-smoke-r1`
2. `/home/ubuntu/workspace/autoloop_runs/7c744964--20260923-attn-scale-012-default-constant-code--al-20260923-attn-scale-012-default-constant-code-smoke-r1`
3. `/home/ubuntu/workspace/autoloop_runs/7c744964--20260923-attn-scale-012-default-constant-code--al-20260923-attn-scale-012-default-constant-code-full-r2`

## Pre-delete gates

- Host `ip-172-31-34-85` was idle; no training/runner process or Neuron holder was present.
- Each literal target resolved directly under `/home/ubuntu/workspace/autoloop_runs`.
- HEADs matched pinned candidate commits `05422cf0503821f9031e349613eb38db1af8f5c4`
  and `7c7449648547692e0877cc26b6b4a70c29cbf251`.
- No target contained `final.pt`, `*.pt`, or `*.ckpt`; controller checkpoint hash/prune receipts
  were already pushed (`b2cea2e7`, `161ccded`, and `1cdad219`).
- Remote run-log, launch-receipt, and ledger SHA-256 values matched the retrieved artifacts for
  all three runs. Durable verdict commits were already pushed (`2ed3c130`, `4af69edb`, and
  `e88d4bad`).

## Result

- Reported target sizes: 190,219,986; 190,168,710; and 190,189,522 bytes.
- Available bytes before: 7,756,324,864.
- Available bytes after: 8,371,343,360.
- All three literal targets were confirmed absent.

The removed checkouts are reconstructable from their pinned Git commits. All logs, manifests,
metrics, verdicts, and checkpoint hashes remain in the pushed repository. No checkpoint, data,
cache, active process, or AWS instance lifecycle state was modified.
