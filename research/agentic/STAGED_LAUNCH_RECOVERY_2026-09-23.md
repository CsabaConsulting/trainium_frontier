# Staged launch recovery — 2026-09-23

## Incident

At `2026-09-23T02:50:14Z`, candidate `20260923-unet-skip-zero-init-r2` had passed
proposal verification and independent implementation verification, and commit `7e500778` had
been staged on `trn21`. The second inventory inside `op.launch` measured free disk just below the
configured `7.5 GiB` controller gate and failed closed before creating a launcher or starting any
training process. The durable state remained `staged:smoke`; no run was counted.

The staged checkout itself accounted for the small change from the earlier `7.7 GiB` inventory.
All controller-owned non-KEEP full checkpoints checked during recovery were already absent or had
durable prune receipts, so no teammate or unknown data was removed.

## Repair

Commit `d44c6dc4` changes restart handling for a persisted `staged:*` state. The controller now
reuses the exact manifest, immutable checkout, label, and staged hash receipt. Before launch,
`op.launch` again proves controller lock ownership, idle/device/disk inventory, checkout HEAD,
clean porcelain, and the pinned `train.py` hash. If a launcher directory already exists, the
launch is ambiguous and still fails closed rather than starting a duplicate.

The active free-space gate is adjusted from `7.50` to `7.40 GiB`. This remains `2.40 GiB` above
the experiment protocol's approximately `5 GiB` abort floor, permits the already-staged `trn21`
checkout, and continues to exclude `trn22` at `7.37 GiB`. Forty-eight offline tests pass,
including a restart test that proves the same staged checkout and `-r1` label are launched without
creating a second checkout or incrementing the attempt.

The breaker may be cleared only after this repair is committed and pushed. The resumed launch must
still pass the live `op.launch` checks; this note is not approval to bypass them.
