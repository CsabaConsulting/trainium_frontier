# Disabled K2 reporting-successor native binding

This is a separate immutable successor to 8555da97, not an amendment or launch.
The original packet and the original 077481d0 source REFUSE 90db28e0 remain
unchanged. Source 18b9365a changes only failed-result JSON reporting/cleanup;
this binding does not edit that source, kernel, reference, tests or thresholds.
Independent source, genuinely fresh successor review, binding review and a
separate sole Operator assignment are still required. No host is reserved.

## Exact identity and mechanical delta

Extract source files from 18b9365aabb63fa85e1720fdefc11177ca8e7853,
research/v2/lanes/k2-resident-consumer/:

- check_native.py:89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850
- consumer_nki.py:57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843
- reference.py:14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f

Only source/checker provenance, the unique label/root, checker/rank-entry/driver
pins and derived manifest identities differ from 8555. The driver and rank-entry
files are otherwise byte-equivalent after those constant substitutions.
validate_receipts.py is byte-identical. The kernel/reference, V256/C128/T256,
20case/dtype pairs,150 comparisons/rank, zero tolerance, timing sample counts,
10.5GiB disk floor,600+60s watchdog, compiler flags, shared idle guard/dispatch
and original lock are unchanged. Candidate e4d93 remains REFUSE.

Proposed label:fs-20260926-k2-resident-consumer-reporting-trn22-r2.
Proposed trn22 root:
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-k2-resident-consumer-reporting-trn22-r2-packet

## Future Operator route only after all gates

Use the original manual route described in
../20260926-k2-resident-native-binding/README.md, with ONLY this successor's
exact NATIVE_BINDING.disabled.json paths/pins/argv. Never dispatch the original
8555 packet with the repaired checker. Never launch bound-command.txt directly.
Preserve active jobs and stopped old queues; verify STOP/breaker/host/disk,
unique absent root and every source/helper hash. No staging happened here.

Stage the three frozen source files, this driver's rank entry/validator, and
the byte-identical common idle_guard.py/dispatch_driver.py. The standard
dispatcher invokes the same exclusive persistent
/home/ubuntu/.cache/trainium_freshstart/device0.lock. Do not replace its inode.
One synchronous four-rank child inherits the original FD for its whole lifetime;
fresh ps AND neuron-ls checks inside the lock fail closed on busy/unknown/error.

As in8555, caches/metrics/receipts must be absent. The driver creates six empty
job-private SDK cache roots and records device/inode/emptiness before the child;
compiler URL equals compiler DIR. No data/shared-cache edits or prewarm.
The rank entry records only whitelisted environment/argv/hash metadata before
Torch/NKI import and invokes the exact checker bytes with runpy; no model/kernel
reconstruction. Actual SDK use of these variables needs separate evidence.

A 600s timeout plus 60s TERM grace bounds this tiny diagnostic, not a 1800s quality
run. No prepare/evaluator/real data/training/production model is used.
Exit0/four files alone do not pass: all four wrappers, all 20 cases and150exact
finite comparisons per rank plus prescribed component samples must complete.
A repaired max_abs:null/max_abs_nonfinite record still refuses; descriptive
JSON-safe failure reporting never relaxes byte exactness.

Retain raw lock/preflight/cache/launch/log/child/terminal records, four rank
environment/wrapper/checker receipts, all-rank graph_stats and compiler logs.
Missing runtime evidence remains unknown; zero compile_us is not a cache hit.
Component timings exclude routing/upload and never establish accepted_speedup.
Timeout/failure may leave descendants: Operator must prove worker/compiler/
device quiescence before any later job. No retry, eval, full run, next job or
adoption is automatic. Architecture/submission/CT priorities remain unchanged.

## Local checks and preserved limitations

85 stdlib checks plus 3 mechanical test groups pass normal/-O. The existing fake
driver/receipt suite is reused, with label/root normalization and two reporting-
failure negatives. Effects are mocked; no actual driver, tensor/native import
or host call occurs. Additional tests prove every original 8555 file unchanged,
constant-only driver/entry deltas, unchanged validator and exact Git-blob pins.
Original preparation failed lookups remain in8555's archived packet; none are
relabelled as new evidence. Native compilation/rounding/model routing are unproven.

22:15 UTC night-mode decision: prepare this reversible disabled successor to the
reviewed reporting repair. Undo: decline the packet. No remote, source, queue,
data, evaluator or submission state changes exist to undo. Supervisor owns the
canonical decision log and all later assignment.
