# Independent safety review: trn22 disk-breaker re-arm

Reviewed 2026-09-24 22:31–22:33 UTC by an Analyst independent of the
controller, retry-package Implementer and host Operator. Scope is **only** the
persisted workflow-v2 trn22 breaker for
`DISK_FLOOR_BREACH_DURING_A0_COMPILE_PRE_STEP`. This is not a run, G2, native,
submission, or general host-preflight approval. No breaker or host state was
changed during this review.

## Bound evidence and authority

- The human's conditional authorization is
  [`20260925-trn22-disk-breaker-human-ack.md`](../../v2/authorizations/20260925-trn22-disk-breaker-human-ack.md),
  SHA-256 `5dcb2f64860001dacd3d70dfbaf530a745da49e232230fb1a0a387323db034d0`.
  It permits re-arming this disk breaker only after an independent fresh
  post-resize safety review; it does not authorize a launch.
- The archived post-resize read-only host audit is
  [`TRN2_POST_RESIZE_READ_ONLY_AUDIT_2026-09-25_0356.json`](../operations/TRN2_POST_RESIZE_READ_ONLY_AUDIT_2026-09-25_0356.json),
  SHA-256 `1aacf82457debedcc657751c2de0f20dada67769834cde9ed12de662f8cafdd5`.
- The consumed A0 `r1` [terminal incident](../../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json)
  has SHA-256 `65b88b3ba08dc5a896da0d59e1cc065821ebcd95d4b209e71c958fad2caa1719`;
  its [independent safety review](20260925-a0-trn22-disk-safety-review.md)
  has SHA-256 `58e24cd7a3cf5e4b511574abeb8fc679b9a8850b9a194589eb4760333e4a473b`.
  `r1` stopped below the 6.5-GiB floor before step 0, with no checkpoint,
  evaluation, or BPB. The old postflight remains historically
  `host_clean=false` solely because disk space was below the floor; it must
  never be rewritten as a success.
- The new-label A0 `r2` [gate contract](../../submission-validation/20260925-v2-c5c-a0-control-trn22-r2/GATE_CONTRACT.json)
  has SHA-256 `fcf0194c6210beabfa3f2a5cfcd77e5cd00881afbd720bc1ea23cf3a75bbbf27`.
  Its [independent static review](20260925-v2-c5c-a0-trn22-r2-static-review.md)
  is PASS for package consistency but explicitly HOLD for dispatch. The
  provisional pre-dispatch disk requirement is 10,737,418,240 bytes; the
  immutable operational floor is 6,979,321,856 bytes.
- At reviewed commit `3d041a45fc27165e39cc74723fc78d0bc2f6b1eb`,
  `research/v2/state.json` SHA-256 was
  `d0503859c35f3c50087513c6ab36cf33191335ca040b6ee3ac37cf0be5c621ef`:
  trn22 breaker `tripped=true`, the named disk reason, human reset required,
  global correctness breaker clear, `hosts.trn22.launch_allowed=false`, and
  `launch_authorized=false`. The local STOP sentinel was absent.

## Fresh read-only trn22 observations

Using only the pinned SSH endpoint/key **by path**, BatchMode, IdentitiesOnly,
and strict host-key checking, no credentials were read or printed. Remote UTC
time was `2026-09-24T22:31:36Z`; hostname was `ip-172-31-32-253` and
read-only Neuron inventory identified instance `i-0673e54d9d66e6a1e`.
The root NVMe serial normalizes to the expected EBS volume
`vol-0d64a9cafbb4912e8`.

| Observation | Fresh result |
|---|---:|
| Root disk `/dev/nvme0n1` | 268,435,456,000 bytes (250 GiB) |
| Root partition `/dev/nvme0n1p1` | 267,360,648,704 bytes, ext4 mounted at `/` |
| Root ext4 total | 258,906,374,144 bytes |
| Root free, `df -B1 /` | **58,454,974,464 bytes** |
| Margin above A0 `r2` 10-GiB pre-dispatch gate | 47,717,556,224 bytes |
| Margin above 6.5-GiB operational floor | 51,475,652,608 bytes |
| Matching train/torchrun/eval/guard/timer/reconcile PIDs | 0 (`pgrep` rc 1) |
| `/dev/neuron0` holders | 0 (`sudo -n fuser` rc 1, empty output) |
| Shared host-lock holders | 0 (`sudo -n fuser` rc 1, empty output) |
| Neuron inventory | one device, LNC 2, zero `neuron_processes` |

`findmnt` independently reported `/dev/nvme0n1p1` ext4 and 58,454,970,368
available bytes during its later sample. The exact remote `r1` postflight
still hashes to the archived
`72518b1ca9522626d35b73ce81d77780b3d1d15d6e5a1838ef52ef0878471ade`.
No new top-level `trainium_freshstart_20260925*trn22*` directory was modified
since the 22:26:14 UTC audit. A read-only kernel-journal screen since that audit
found zero lines matching Neuron error/failure/reset, OOM-kill or I/O error.
These negative screens are point-in-time evidence, not a substitute for a
later run's full preflight.

## Decision and exact transition boundary

**PASS to re-arm the named trn22 disk breaker only.** The human explicitly
authorized that narrow action, the independent resize audit and this fresh
check show that the specific disk-shortage condition is resolved with large
headroom, and no current train/eval/device/lock holder was observed. There is
no new incident in the bounded checks above. The historical `r1` failure and
its non-clean postflight remain unchanged and its label remains consumed.

The controller may make a separately recorded, atomic workflow-v2 state
transition from `breakers.trn22.tripped=true` to `false`, binding this review,
the human-acknowledgement receipt, the prior incident and the fresh disk
measurement. Retain the cleared reason and timestamp as history. Remove only
the now-resolved breaker item from `pending_human_decisions`; keep
`hosts.trn22.launch_allowed=false` and top-level `launch_authorized=false`.
Do not reset the legacy autoloop breaker or modify the remote historical
postflight.

Before any trn22 dispatch, the separate A0 `r2` gates still require a clean
pinned stage, exact source/prepare/cache identity, current host/disk/lock
preflight, exact-effective-environment four-rank G2 and numerical comparison,
live external boundary observer, completed proposal, independent Verifier
review, and a new hash-bound one-attempt Supervisor authorization. Recheck
free space immediately before dispatch and monitor it throughout compile and
training; any new conflict or below-floor sample is a fresh HOLD/FAIL. This
review neither starts a job nor makes a BPB or submission claim.
