# trn22 r4 source-stage timing feasibility — DISABLED / READ-ONLY HOLD

This is a separate Implementer-only timing diagnostic for the exact disabled [trn22 r4 e65→r9 source stage](../trn22-c5c-a0-r9-incremental-stage-r4-hold/RUNBOOK.md), commit `19fbfdc88763b6b5762e219339dcd156e7977a30`. It neither enables that stage nor supplies native-training authority. No SSH/EC2/AWS contact, local result, host marker, transfer, checkout, cleanup, training, evaluation, breaker reset or submission occurred. The checked-in contract has `dispatch_authorized=false`; the separately named enabled timing request, independent final timing review, Supervisor one-attempt assignment and raw result are absent. Running `operator_timing_readonly.py` now refuses locally before SSH or a result file.

## Exact immutable pins

| File or source | SHA-256 |
| --- | --- |
| `remote_timing_readonly.py` | `b454aa5524cd1df572ea5437a4588239a7b01eee12498474a99df958695160e2` |
| `operator_timing_readonly.py` | `3d3bc7b99088314ddeea1e0b9ea5e4cc6fe782c60abcc2eec735f92423d268f1` |
| `GATE_CONTRACT.HOLD.json` | `68c0b32b469c6a16062a9587eeb2dc567415f7d27158f557a1257aebc17e01ad` |
| `test_timing_readonly.py` | `e5560f808e10f1eb7e82005bdcb7a6ff5576b253408a04f6f8c3927c223267ec` |
| exact r4 stage remote | `871fd00aa61d0db2df8597949c8cad43c4c0a9a94ffda218a44851d60091c9b7` |
| exact r4 stage operator | `64d0ed49261cd231b3193f315d114d43e39a94f3b3dea21833a37dd15c0c2d7e` |
| exact r4 disabled request | `8cd3b2debb0c309ab72faef78b057e344fc0525b1269332d3d46b13337797f56` |

The host/source pin remains trn22 `ip-172-31-32-253`, UID/GID 1000, ext4 device 66305, accepted linked e65 checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2`, HEAD `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, exact `train.py`/`prepare.py` hashes, observed checkout inode/mode `2432738/0775`, linked `.git` inode/mode `2432740/0664`, and existing `.cache/trainium_freshstart/device0.lock` identity. The r4 target, bundle and attempt label remain the same exact absent stage-r1 names; this diagnostic never opens them. The earlier 07:33 read-only snapshot is point-in-time planning evidence, not a fresh stage preflight. A timing probe is intentionally narrower than r4's full ACL/NSS/mount/protected-artifact/source-stage preflight.

The older r2 timing diagnostic `384eb822c12c05cd994e98e81d2fe4f632cb6118` received [independent STATIC PASS only](../../verification/20260925-trn22-r9-final-continuity-timing-r4-independent-static-PASS.md) at `9aa9a124` (receipt SHA-256 `4b2e4e86240433b922f5394b8dd6d692e79c7241c52ac2e5dd255abcd1f8971d`). It measured only the r2 sequence and did not authorize a host attempt or establish r4 feasibility. This package reuses its strict local authority/host-key gates, read-only transport, existing nonblocking cooperative flock, per-trial flushed JSONL, raw timeout-prefix retention and separate pre/post probes. It binds the **r4** source hashes and changes the timed sequence; the r2 result cannot be substituted.

## Read-only r4 timing model

Each trial, under the existing flock, performs the r4 writer's read-only order:

1. Full STOP/process/`fuser`/Neuron/process activity check; two broad attempt-label scans; measure their combined duration against the inherited inclusive one-second limit.
2. If still timely, another full activity check, root byte/inode floor check, direct STOP/bundle/target absence check, then sample the pre-open label/activity/space ages.
3. If still timely, take a **monotonic clock-only `would_open` sample** and recheck those ages. There is **no** `os.open(O_EXCL)` or file creation.
4. Only for a passing would-open model, run the third broad label scan, third full activity check, second root floor check and direct absence check, then sample post-would-open ages at a clock-only `would_payload` point.

A timing miss stops that trial where r4 would refuse; it does not fabricate unrun third checks. Persistent STOP, activity, label, low disk or helper uncertainty aborts the read-only trial. The diagnostic uses the r4 helper timeouts (20-second `fuser` and `neuron-ls`, 45-second broad `find`); the 120-second outer timing timeout bounds the twenty-trial request and may leave a strict completed JSONL prefix. Each complete record is flushed before the next trial. The local collector preserves raw stdout/stderr in its exclusive one-attempt receipt, attempts a distinct post-probe after timing starts, and treats timeout, aborted/incomplete records, probe drift, bad age arithmetic or failed post-probe as HOLD. A local STOP may prevent even the attempted post-probe from contacting the host; no completed post-probe may be inferred without its actual receipt.

This is a **feasibility measurement**, not a safety or transfer proof. The two adjacent clock samples cannot measure the real `O_EXCL` syscall or its scheduler delay; the read-only post-would-open direct check requires the bundle to stay absent whereas the real writer would have a provisional empty bundle. It cannot test r4's fsync, retained-partial behavior, actual payload delay, Git reconstruction, cross-session readback, malicious same-UID/root races or the non-atomic broad-scan gap. A twenty-trial subsecond result would only show that these read-only helpers fitted the modeled windows in those observations. It cannot close a source-stage HOLD or permit dispatch. Independent review must judge both this static source and any eventual raw timing result.

The Implementer ran **18/18** synthetic tests under normal Python and **18/18** under `python3 -O`. Tests cover exact three-activity/three-scan/two-space order, inclusive 1-second edge and misses at the double scan/pre-open/would-open/post-would-open boundaries, late helper aborts, no target-open primitives, exact pin validation, strict JSONL/age parsing, timeout prefix preservation, distinct post-probe attempt and absent authority before SSH/output. These are offline regressions, not independent verification or live timing.

R4→r9 timing is **not the current scored A0 critical path** while trn21's planned checkout is r10: experiment-protocol §1 requires the same Git HEAD locally and on both hosts. The collector pattern can be adapted to a **separate exact r10 trn22 stage profile** after that profile's review, but r4 pins, label, target and any r4 raw result cannot be reused as r10 authority or evidence.

Offline commands:

`python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r4-timing-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r4-timing-hold -p 'test_*.py' -q`
