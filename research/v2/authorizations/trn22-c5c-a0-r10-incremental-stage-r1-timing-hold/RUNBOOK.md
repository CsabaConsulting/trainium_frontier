# trn22 r10 direct-stage timing feasibility — DISABLED / READ-ONLY HOLD

This r4 disabled correction closes the independently observed postflight
local-gate bypass in r3: **every** SSH phase, including the best-effort
post-timing probe, now checks local Desktop/Documents/package STOP and active
controller breaker state before opening a new connection. If STOP appears
mid-run, the collector preserves its prefix and terminal HOLD receipt but
does not contact the host for postflight. A synthetic late-STOP regression
checks that neither host-key identity nor SSH subprocess is reached. No host
contact was made to implement or test this correction.

This Implementer-only diagnostic is bound to the corrected disabled [trn22 e65→r10 direct source-stage package](../trn22-c5c-a0-r10-incremental-stage-r1-hold/RUNBOOK.md) at commit `b3ec7406f0e7a3df6848698876e2106305dd283f`. Its target training-source HEAD is `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. The predecessor stage package `3d26e963` had a mistyped inherited-review hash; its r2 successor `020bfa7e` read a potentially stale Documents controller state. Both remain historical HOLD. This branch includes a byte-identical cherry-pick of the separately pushed r3 correction so its pins resolve locally. The timing-specific change carries the r3 Desktop state pin and additionally checks the former Documents STOP paths. It changes neither the remote timing sequence nor the source stage's one-shot label, target, bundle or marker.

No SSH/host contact, local timing result, marker, transfer, checkout, cleanup, native training, evaluation, AWS action or submission occurred. `GATE_CONTRACT.HOLD.json` has `dispatch_authorized=false` and all remote-write/training permissions false. The absent distinct enabled timing request, independent final timing review and Supervisor one-attempt assignment are required before the collector can contact a host. Running `operator_timing_readonly.py` from these checked-in bytes refuses locally before SSH or a result file. Source-stage dispatch is a separate authority and remains HOLD.

## Exact source pins

| Item | SHA-256 |
| --- | --- |
| `remote_timing_readonly.py` | `c43fbe1224de291c31a797079ed9708db4ca7fa0d7fe785b3ff67274c835f3ca` |
| `operator_timing_readonly.py` | `90660ab0892ecb091f9dc852cea46f4d59f76a3365c46451381d9c78d6ba4ef8` |
| `GATE_CONTRACT.HOLD.json` | `09086904ae8ab1cba54e4c1fb35476a90c5654d7a8692a61eb167cd8bddd6bfc` |
| `test_timing_readonly.py` | `59b27538b48e009acfeccd21f9858ec137191af2f6150281440f33372e17f6ce` |
| exact corrected-stage remote source | `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f` |
| exact corrected-stage operator source | `a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda` |
| exact corrected-stage disabled request | `48e3ad5d0bb1d077dbf8304e3ddfe7e3263892e522dc277f32812eea5c8e111a` |

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
