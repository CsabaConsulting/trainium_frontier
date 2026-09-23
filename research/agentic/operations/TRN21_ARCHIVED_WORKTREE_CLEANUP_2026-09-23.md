# Trn21 archived-worktree cleanup receipt — 2026-09-23

## Authority and scope

The human operator explicitly approved the narrow cleanup proposed in commit `a90879a8` on
2026-09-23. The local autonomous-loop service was stopped before remote action. A fresh read-only
check proved `trn21` idle with no Neuron device holder, all three exact worktrees present, every
corresponding `final.pt` already absent, and the six durable verdict/prune commits available.

## Exact removal

At approximately `2026-09-23T05:10Z`, only these three controller-created archived worktrees were
deleted from `trn21`:

1. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-weight-ema-fullbudget-r3--al-20260922-weight-ema-fullbudget-r3-smoke-r1`
2. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260923-engram-only-default-shape--al-20260923-engram-only-default-shape-smoke-r1`
3. `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-bf16-norm-output-policy-r3--al-20260922-bf16-norm-output-policy-r3-smoke-r1`

Each occupied exactly `204,886,016` bytes before deletion. Their durable evidence remains in the
repository:

| Proposal | Verdict evidence | Checkpoint-prune receipt |
|---|---|---|
| `20260922-weight-ema-fullbudget-r3` smoke | `2f8b4ee7` | `bef2fb7a` |
| `20260923-engram-only-default-shape` smoke | `d1fa2ab0` | `070e5698` |
| `20260922-bf16-norm-output-policy-r3` smoke | `bce18ba6` | `cb9cf875` |

## Verification

- All three exact paths are absent after deletion.
- `/home/ubuntu/workspace/autoloop_runs` decreased from `3,893,264,384` to `3,278,606,336`
  bytes, exactly `614,658,048` bytes.
- Root free space increased from `7,790,657,536` to `8,405,307,392` bytes (about 7.83 GiB), above
  the configured 7.40 GiB gate.
- No cache, dataset, launcher, bundle, active candidate checkout, teammate path, AWS resource, or
  other machine state was removed or changed.

The controller breaker may be cleared only after this receipt is committed and pushed. The next
launch must still pass a fresh full preflight and the second inventory inside `op.launch`.
