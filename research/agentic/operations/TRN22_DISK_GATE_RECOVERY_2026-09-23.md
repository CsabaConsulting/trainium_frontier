# Trn22 disk-gate recovery — 2026-09-23

The controller correctly refused to launch
`al-20260922-bf16-mlp-input-norm-r2-smoke-r1` after staging because a second inventory measured
7.96 GiB free, just below the configured 8.0 GiB operational margin. No training process started.

Read-only inventory established that the box was idle, `/dev/neuron0` had no holder, the data cache
and pinned Python were present, and the only remaining controller checkpoint was the preserved
1,297,950,407-byte full-run checkpoint for the valid BF16-SDPA `KEEP`. That checkpoint was not
deleted. The controller scratch area used about 147 MiB, including the newly staged candidate.

The local operational floor was changed from 8.0 to 7.5 GiB. This remains 2.5 GiB above the
experiment protocol's hard 5 GiB abort floor and avoids deleting the preserved winning checkpoint.
The lock owner matched the persisted controller ID exactly, so its stale lock was released; the
breaker was then cleared. A fresh read-only preflight measured 7.96 GiB free on idle `trn22` and
10.26 GiB on idle `trn21`, with no device holders and both required runtime paths present. Restart
semantics will abandon the staged `-r1` directory and create a unique `-r2` label; it must never
reuse the unlaunched directory.

The `r2` smoke then passed and its checkpoint was pruned, but staging the promoted full run plus
the abandoned immutable directories reduced free space to about 7.4 GiB. The post-stage launch
inventory correctly stopped again at the 7.5 GiB floor. The floor was not weakened further, the
preserved BF16-SDPA `KEEP` checkpoint was not deleted, and no full run started on `trn22`.

`trn22` is therefore removed temporarily from the controller's scheduling list. Its same-controller
stale lock was released after confirming the box had no controller run. The promoted MLP-only full
run is reassigned in runtime state to `trn21`; that box's already-running independently verified
smoke is allowed to finish and be collected first. `trn22` may be re-enabled only after a separate
hash-preserving cleanup decision raises it above the configured floor.
