# Trn21 archived-worktree cleanup receipt — 2026-09-23 R3

## Authority and scope

The human operator explicitly approved the verified `trn22` validation sequence and the
previously specified three-checkout `trn21` cleanup on 2026-09-23. The local
`trainium-autoloop.service` was inactive before remote action. A fresh protocol inventory at
`2026-09-23T12:27:19Z` proved that `trn21` had no training-related process or Neuron device
holder. Each exact archived checkout resolved to its literal target, existed at authoritative
commit `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`, and had no remaining `final.pt` checkpoint.

Before removal, every remote artifact below matched its pushed local copy byte-for-byte:

| Archived run | Run log SHA-256 | Launch receipt SHA-256 | Ledger SHA-256 |
|---|---|---|---|
| Demon-beta1 smoke R1 | `0d38bb7c77afcda6b26d52472c78295567b11121de74a05c72d8017fae6ec88b` | `7c7a054a2a1da54ed8d3d7b054ab5d5c0eb868591ddd7e01f1234f7975abf532` | `9409a07ce2924ac26a70e9cff95a082d8adfa860310d8d98a640649e1402da64` |
| Clip-after-reduce full R2 | `84ecccc2c82310742231064a6112b8643756d21428da59c42f4fc23b8c66eeef` | `76e87ab278ddf0684854299273554db683f025ea8efe5ab6a6d6fd1f6e60e66a` | `f543637c0c88d7af769536f752bed30f4eb471854b758601590bd4ee40946d4a` |
| No-ngram-VE smoke R1 | `ea2c2593666457995ca21df8c65b2bcef15c3f47a65c535853503bee1235fa46` | `fda87b91a1526429b2e4f06a896329cc1fdb8c381689b7cb8d7a999ca2887458` | `ad6d369f902cde0bc5e00f81c9d445550eb4608469c4ff21c9d0e8ad13f987b1` |

The corresponding result/verdict and checkpoint-prune commits are durable ancestors of this
branch: Demon `330793dd` / `35488ee5`, clip-after-reduce `aa07480b` / `1d543362`, and
no-ngram-VE `f2ac57b8` / `192e99a6`.

## Exact removal

At approximately `2026-09-23T12:27Z`, only these controller-created archived checkouts were
deleted from `trn21`:

- `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260923-demon-beta1-r2-default-shape--al-20260923-demon-beta1-r2-default-shape-smoke-r1` (`190,064,890` apparent bytes)
- `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-clip-after-reduce-only-default-shape--al-20260922-clip-after-reduce-only-default-shape-full-r2` (`190,085,819` apparent bytes)
- `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-no-ngram-ve-only-default-shape--al-20260922-no-ngram-ve-only-default-shape-smoke-r1` (`190,065,508` apparent bytes)

No cache, dataset, launcher, bundle, active candidate checkout, teammate path, AWS resource, or
other machine state was removed or changed.

## Verification

- All three exact paths are absent after deletion.
- Root available bytes increased from `7,764,992,000` to `8,379,666,432`, a
  filesystem-accounted gain of `614,674,432` bytes.
- A fresh protocol inventory at `2026-09-23T12:28:09Z` reports `7.80 GiB` free, no training
  processes, no Neuron holder, the data cache present, the pinned Python present, and no findings.

The controller breaker may be cleared only after this receipt is committed and pushed. The next
launch must still pass a fresh inventory and the remote operator's independent pre-launch checks.
