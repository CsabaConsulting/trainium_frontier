# Independent verifier receipt: trn22 disk-breaker state transition

Reviewed 2026-09-24 22:37 UTC. Scope: commit `9169177f` against parent
`7aceaeef`; this is a **state-transition review only**, not host preflight,
G2, native verification, a run, or submission approval.

## Verdict

**PASS — narrowly scoped breaker re-arm; dispatch remains HOLD.** The human
acknowledged clearing the named trn22 disk safety stop only after fresh checks.
The independent safety review passed before the state change. The commit
clears only `breakers.trn22` for
`DISK_FLOOR_BREACH_DURING_A0_COMPILE_PRE_STEP`, retaining that reason, the
consumed `r1` label, the clearance time, and the evidence links in
`last_cleared_incident` and `latest_breaker_rearm`.

## Evidence checked

- Human acknowledgement:
  `research/v2/authorizations/20260925-trn22-disk-breaker-human-ack.md`,
  SHA-256 `5dcb2f64860001dacd3d70dfbaf530a745da49e232230fb1a0a387323db034d0`.
- Independent safety review:
  `research/agentic/verification/20260925-trn22-disk-breaker-rearm-review.md`,
  SHA-256 `05dc7ff165117b7f32cc79008d647410c09c73014d543ac523c5e7da1a91b087`.
- Post-resize read-only audit:
  `research/agentic/operations/TRN2_POST_RESIZE_READ_ONLY_AUDIT_2026-09-25_0356.json`,
  SHA-256 `1aacf82457debedcc657751c2de0f20dada67769834cde9ed12de662f8cafdd5`.
  The fresh safety review additionally measured 58,454,974,464 root-free bytes
  at 22:31:36 UTC, above the provisional 10-GiB prelaunch requirement and
  6.5-GiB operational floor, with no train/eval, Neuron-device, or host-lock
  holder observed at that instant.
- Archived terminal incident:
  `research/v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json`,
  SHA-256 `65b88b3ba08dc5a896da0d59e1cc065821ebcd95d4b209e71c958fad2caa1719`.
  It still records a disk safety stop before step 0, no checkpoint/eval/BPB,
  and `host_clean=false` for the old postflight.
- `git diff 7aceaeef 9169177f` changes only `research/v2/state.json`,
  `research/v2/STATUS.md`, and `research/agentic/CURRENT_STATE.md`.
  The incident, ledger, G1 receipt, and post-resize audit are byte-unchanged.
  `git diff --check` and `jq empty research/v2/state.json` pass.

## Exact state boundary

- `breakers.global_correctness` and `breakers.trn21` are unchanged and clear.
  `hosts.trn21` is unchanged.
- `hosts.trn22.launch_allowed=false`, top-level
  `launch_authorized=false`, and both `ready_job_counts=0` remain unchanged.
  Only the trn22 host status, timestamp, and fresh-free-byte observation are
  updated alongside its named disk-breaker state.
- The sole pending human decision, the trn22 disk-breaker acknowledgement, is
  moved to resolved with its narrow scope. No additional human decision is
  silently treated as authorized. The two status documents explicitly say no
  training/evaluator was launched and no new BPB exists; their older sections
  are marked historical.

Before any new-label A0 `r2` dispatch, an independent final request must bind
the completed proposal, clean pinned source and `prepare.py` identity, exact
cache/environment, current host/disk/device/lock preflight, exact-effective-
environment four-rank G2 and numerical comparison, a live external optimizer-
boundary observer, and a separate one-attempt Supervisor authorization.
Recheck free space immediately before and during the run. This receipt does
**not** make either host launch-ready or convert the failed `r1` into a result.
