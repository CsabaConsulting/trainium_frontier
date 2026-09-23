# Trn21 archived-worktree cleanup receipt — 2026-09-23 R2

## Authority and scope

The human operator explicitly approved this exact cleanup on 2026-09-23. The local
`trainium-autoloop.service` was stopped before remote action. A fresh read-only check at
`2026-09-23T11:19:04Z` proved that `trn21` had no training-related process or Neuron device
holder, the exact archived checkout existed at the authoritative commit, and its checkpoint was
already absent.

Before removal, the three retrieved run artifacts still matched their durable recorded hashes:

| Artifact | SHA-256 |
|---|---|
| run log | `56d29df4540c258c7d1fffecc0f86bbd55c4fd571679b8642b7000c0fdb7fcf4` |
| launch receipt | `c1423968998cc4be5979126013ef951a7688d8f2587a11cc38b6e88b00d4ee30` |
| experiment ledger | `9b655f0331398a7ac261bab80a07298bbb98326e8c584185f0c04d819d4a1d9e` |

The smoke result, verdict, retrieved artifacts, and checkpoint-prune receipt are durably present
on pushed branch `agent/control-plane-20260921`. The result/verdict commit is `d11ec75e`; the
checkpoint-prune commit is `8e2e5f7e`; both are ancestors of pushed commit `69779622`.

## Exact removal

At approximately `2026-09-23T11:20Z`, only this controller-created archived checkout was deleted
from `trn21`:

`/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-clip-after-reduce-only-default-shape--al-20260922-clip-after-reduce-only-default-shape-smoke-r1`

The resolved path exactly matched that literal target. Its apparent byte size before removal was
`190,065,009` bytes. No cache, dataset, launcher, bundle, active candidate checkout, teammate
path, AWS resource, or other machine state was removed or changed.

## Verification

- The exact path is absent after deletion.
- Root available bytes increased from `7,776,006,144` to `7,980,867,584`, a filesystem-accounted
  gain of `204,861,440` bytes.
- A fresh protocol inventory at `2026-09-23T11:21:05Z` reports `7.43 GiB` free, no training
  processes, no Neuron holder, the data cache present, the pinned Python present, and no findings.
- The active Demon-beta1 staged checkout was not touched.

The controller breaker may be cleared only after this receipt is committed and pushed. The next
launch must still pass a fresh inventory and the independent pre-launch inventory inside the
remote operator.
