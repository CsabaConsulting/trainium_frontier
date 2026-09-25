# Exact 93ae architecture CPU gates — independent PASS

Reviewer: root, distinct from source Implementer and trn22 Operator.
Archives3f87928a/f9270d77, canonical46d7939d/1274a0ea. Exact source
93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5.

Independently parsed all4G2 reports and16raw rank files. Each probe completed
4ranks x3finite optimizer updates, exit0, no timeout or cross-rank mismatch,
all parameters updated, deterministic strict checkpoint reload. Every raw
rank equals its aggregate entry. ON is actual8x1024,113,256,724elements and
65parameter tensors; OFF/references are117,453,580elements and62tensors.
CPU forward is intentionally synthetic2x256 and uses the unchanged declared
G2 shims; fake-clock/native-kernel bypass is not timing or native proof.

Default-OFF has exact12losses, all62parameter digests and ownership plan
identity to BOTH c5c and timer-fixed6ef7/Recursive-OFF. ON science argv uses
--unet-8x1024 --attention-layers0,2,4,6,7 with separate argument values,
seed42, no tables, LR0.015,1800seconds,pack32/context1024. Neither Recursive
nor FP8 is silently enabled. Source/evaluator pins match the reviewed packet.

Recomputed G2 hashes:

- ON:c2fee697d02322198d61951363861fb31201bd7b68d4b91f159dbd44001392b0
- OFF:1d102c7d5fd7b09c871b5e022abf2a7f623bbb20e5fcb16206f23849b1a26cf6
- c5c:988d3a1eb639efdf1cd9670d3e31c07176453618ed108b47b0bfe9d35e7f19b2
-6ef7:7ccf42ecc822be74fabbae69ce03341be3f6498804d2437562130f53b981f725

## Supervisor assignment

After fresh normal preflight, distinct trn22 Operator may launch the already
prepared exact93ae manual full run
`fs-20260925-l2-unet8x1024-93ae-trn22-full-r1`, then its fresh public20M eval.
Use the reviewed11e5f64a binding/driverda58b422 and common device lock with old
queue paused; preserve six fallback jobs and serial hand-back. No extra
human ceremony or launcher is required. Existing source/timer/correctness
review remains in force; native failure cannot be overridden by CPU PASS.

Judge final BPB against the declared same-hostc5c control under the new
general fixed1800budget rule. Steps are diagnostic, with<60% or stalls/spikes
investigated, never a step-window veto. Record startup estimate and driver
timings separately. This initial research directory is warm, NOT true cold
proof; genuine cold accounting/audit remain mandatory before submission.
Current schedule adoption does not alter this frozen isolated contrast.
