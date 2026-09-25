# trn22 r10 direct-stage timing feasibility — DISABLED / READ-ONLY HOLD

This Implementer-only diagnostic is bound to the corrected disabled [trn22 e65→r10 direct source-stage package](../trn22-c5c-a0-r10-incremental-stage-r1-hold/RUNBOOK.md) at commit `020bfa7ebd048f947d5207cbedaf028a29a4b159`. Its target training-source HEAD is `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. The predecessor stage package `3d26e963` had a mistyped inherited-review hash and remains historical HOLD. This timing diagnostic changes neither stage code nor its one-shot label, target, bundle or marker.

No SSH/host contact, local timing result, marker, transfer, checkout, cleanup, native training, evaluation, AWS action or submission occurred. `GATE_CONTRACT.HOLD.json` has `dispatch_authorized=false` and all remote-write/training permissions false. The absent distinct enabled timing request, independent final timing review and Supervisor one-attempt assignment are required before the collector can contact a host. Running `operator_timing_readonly.py` from these checked-in bytes refuses locally before SSH or a result file. Source-stage dispatch is a separate authority and remains HOLD.

## Exact source pins

| Item | SHA-256 |
| --- | --- |
| `remote_timing_readonly.py` | `a9bc253680403883dacc601b35cb1935c81672c512fc9a04cff9ce12e7944ed4` |
| `operator_timing_readonly.py` | `4d35b139e2d1cd44e96afb25458b10415de94dd211791ed4172e2893f95e21ab` |
| `GATE_CONTRACT.HOLD.json` | `bc5f5efc1fa86606c0dbdc03203ceb0dc4f3c0c197fa5bf696e0dd9ffbac98c2` |
| `test_timing_readonly.py` | `cf5ba6b15128820ce60ecdff1e4f6232e155e980314ef47aa6c3c8d116016800` |
| exact corrected-stage remote source | `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f` |
| exact corrected-stage operator source | `7c04b0288de04a514b3da0a6a9c01f525551f2cfc92f69d4ba63f8941aa2d0b5` |
| exact corrected-stage disabled request | `625b4a53c0f67fba0a7b4ef5a6705f64f3935eb8f5841ef5e2212f6c8e820b6e` |

The host/source pin is trn22 `ip-172-31-32-253`, UID/GID 1000, ext4 device 66305, accepted linked e65 checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2` at `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, and the exact source `train.py`/`prepare.py` hashes. The timed label is `fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1`; its target and bundle paths match the corrected-stage code. The existing private-cache lock is opened read-only and acquired nonblocking. This diagnostic checks source identity and activity but is intentionally narrower than the stage's full ACL/NSS/mount/protected-artifact preflight. Earlier host snapshots are not fresh stage preflights.

The older r9-r4 timing source `6876f084` supplied the bounded collector pattern; no r9 timing result or static review grants r10 authority. Every r10 pin, label, target, schema and output name has a distinct binding here.

## Exact clock-only model of the r10 final boundary

Each of twenty trials holds the existing cooperative lock and follows the corrected stage's final read-only order:

1. Full STOP/process/fuser/Neuron/process check; two broad label scans. Measure the double-scan duration against the inclusive one-second bound.
2. Only if timely: another full activity check, root byte/inode floor check and direct STOP/bundle/target absence check; sample label/activity/space ages before the would-open.
3. Take a monotonic **clock-only** would-open sample and recheck all three ages. No `O_CREAT`, `O_EXCL`, payload or target creation occurs.
4. Only if that model passes: third broad label scan, third full activity check, second root byte/inode floor check and direct absence check; sample post-would-open ages at a clock-only would-payload point.

A timing miss stops that trial where the stage would refuse; unrun checks are not fabricated. Persistent STOP, process/Neuron activity, label appearance, low disk, helper failure or uncertainty aborts. Helper timeouts mirror the stage: fuser/neuron-ls 20 s, broad find 45 s. The outer timing phase is bounded to 120 s. Each complete JSONL start/trial/end record is flushed before the next trial. The one-attempt local collector retains raw stdout and stderr bytes/base64, hashes, exact redacted argv and timeout status, even if only a completed prefix is retrieved. It attempts a distinct full read-only post-probe after timing starts. Missing/failed post-probe, timeout, incomplete/aborted records, identity drift, malformed age arithmetic or any other uncertainty is HOLD, not a timing PASS. A local STOP can prevent even that post-probe connection; no completed post-probe is inferred.

This measures **feasibility**, not write safety. Adjacent clock samples do not measure real `O_EXCL` syscall/scheduler delay. The read-only post-would-open direct check requires the bundle absent, whereas the real stage would retain a provisional empty bundle. It cannot prove fsync, payload timing, reconstruction/readback, same-UID/root race freedom or atomicity of broad scans. A subsecond twenty-trial result would only show these read-only helpers fitted the modeled windows on those observations. Independent static and raw-result review, separate enabled stage request, fresh preflight and Supervisor assignment remain mandatory; this package does not close any native/BPB gate.

Offline checks, with no host contact:

```text
python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-timing-hold -p 'test_*.py' -q
python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-timing-hold -p 'test_*.py' -q
python3 -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-timing-hold/operator_timing_readonly.py
python3 -O -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-timing-hold/operator_timing_readonly.py
```

The Implementer tests are local synthetic checks only. Both entrypoints must refuse before SSH or an output receipt; a distinct independent Verifier must review the final package before any later enabled timing request.
