# Track A 35c9 controls and A1 runbook

This Implementer package does not authorize dispatch. An independent Verifier must first identify
the exact finalization HEAD and approve each requested arm. No command here performs SSH unless the
Operator deliberately omits `--dry-run` after that approval.

The two controls use separate labels and hosts and may run only under the repository host lock.
Each runs full 1800-second seed-42 LNC2/world4 training, then a fresh 2M evaluation, and retains its
checkpoint. A1 uses trn21 only and its adapter fails until `a1-trn21-authorization.json` binds an
archived successful trn21 control by commit plus exact completion/manifest/result/metrics digests.

Verifier-safe local materialization shape:

```text
python research/submission-validation/20260924-v2-track-a-35c9-controls-a1/launch_adapter.py \
  --arm ARM --concrete-request /tmp/UNIQUE.request.json --dry-run \
  --dry-run-origin-head EXACT_FINALIZATION_HEAD
```

The Operator's eventual non-dry invocation must use the canonical path
`/tmp/<arm-label>.request.json`; the adapter then calls `research/fresh_start_launch.py`. The 6.5 GiB
launch floor and hard 5 GiB abort, host lock, pre/postflight, source fingerprints, O_EXCL label and
request, immutable artifacts, finite checkpoint scan, and fresh2M count checks all fail closed.

After each run, retrieve and hash the dispatch/job/launch/preflight/process/postflight/driver files
and the entire `out/<label>` evidence directory. Do not delete a checkpoint or cache. A1 improvement
of at least 0.001 only nominates an other-host full20M replicate; it is not KEEP or adoption.
