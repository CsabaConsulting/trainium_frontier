# Trn2 autonomous-loop disk block — 2026-09-23 R1

Purpose: preserve the read-only evidence for the fail-closed controller stop after independent
verification of `20260923-local-conv-pre-attn-serial-only-r3`.

## Controller state

- Controller branch/commit: `agent/control-plane-20260921` at
  `1493ff2b080a5eb3b2a45fd72742906e43906736`, synchronized with origin before this receipt.
- Candidate commit: `79b0cecb2f655441f4e0898f6140308eea6072f0`.
- Candidate state: proposal and implementation verified; no smoke or full run launched.
- Breaker: `trn21`, `free disk 7.3 GiB below 7.4 GiB`, recorded at
  `2026-09-23T16:32:26Z` after the second failed inventory.
- Service: exited fail-closed with status 3; systemd `Restart=on-failure`, `RestartSec=300`.

## Live read-only inventory

Captured approximately `2026-09-23T16:34Z` by hostname, filesystem availability, process listing,
and bounded `du`. Neither host had a `torchrun`, `train.py`, or `run_experiment.py` process.

| Controller box | Hostname | Available bytes | Controller run directories | Controller scratch |
|---|---|---:|---:|---:|
| `trn21` | `ip-172-31-34-85` | 7,835,312,128 | 2,254,217,216 | 51,818,496 |
| `trn22` | `ip-172-31-32-253` | 8,628,547,584 | 4,371,390,464 | 153,665,536 |

The shared competition data cache used 551,325,696 bytes on each host. The large `/tmp` trees were
observed but not attributed or modified because they can contain Neuron compiler/runtime state and
other agents' work.

Two narrow `trn21` recovery candidates were then inspected read-only. They contain no `*.pt` or
`*.ckpt`; their candidate HEADs are exact, and the dirty ledger/run-log state is already retrieved.
Remote launch-receipt and run-log hashes exactly match the durable local copies.

| Closed run directory | Bytes | HEAD | Retrieved launch/log SHA-256 |
|---|---:|---|---|
| `b8fab55a--20260922-bf16-mlp-input-norm-r2--al-20260922-bf16-mlp-input-norm-r2-full-r2` | 204,996,608 | `b8fab55a4a6a21098cad7200a0001000aa9d76f3` | `fa521732…` / `e5339fb5…` |
| `fff4b9ba--20260922-bf16-norm-output-except-mlp-input--al-20260922-bf16-norm-output-except-mlp-input-full-r1` | 204,976,128 | `fff4b9ba808e18f8012537cb80b445889f2a9608` | `d6f7bb7a…` / `1bd980ec…` |

Their full metrics, manifests, verdicts, retrieved artifacts, and checkpoint-prune receipts are
already present under the corresponding `research/agentic/experiments/` directories. Removing both
would nominally recover about 410 MB, but this monitor did not delete them because they are another
agent's isolated worktrees.

`trn22` is above the current 7.4 GiB preflight floor but remains intentionally absent from
`autoloop.json`. Its recovery receipt requires a separate hash-preserving cleanup/re-enable
decision. The persisted `trn21` breaker stops the daemon before another box can be selected.

## Safety boundary and next action

No remote path, cache, checkpoint, process, AWS resource, controller configuration, or breaker was
changed. Existing isolated run directories are evidence-bearing/reconstructable worktrees and were
not cleaned under this monitoring action.

Resuming hardware research requires a human-coordinated recovery that preserves retrieved logs and
checkpoint receipts, restores the configured disk margin, decides whether to re-enable `trn22`, and
then clears the persisted breaker. Lowering the disk gate is not an acceptable substitute.
