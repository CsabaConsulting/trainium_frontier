# Repaired-source role and binding note

Prepared locally by Researcher, 26 September 2026. These are conditional
superseding proposals, not host assignments, release packets or launch authority.
All three `dispatch.launch_authorized` values are false; neither host is selected.

## Exact source lineage

- K2: frozen `63925519` plus the two-line int32/int64 input guard/error repair
  is `a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`,
  source commit `2ad051aa365e60817859be2969e1c559a91b974e`.
- P2: frozen `b4728d17` plus the identical guard port
  is `0127d02dcaae0e6fdb70db693697caf02233db31d926dddd4649a1d496e6b543`,
  source/evidence commit `697c0d05a55ae6c037ad475ebc179e5d561fc16a`.
- Bind original K2 review `2ddb87ac`, original P2 review `c41e16d6`,
  and the one tiny-repair fresh review `b689fb6e` at their original scopes.
  Root/Analyst's `lanes/k2-last-layer-tables-index-repair/COMPOSITION.md`
  records exact composition; it is not a second independent source review.
- Both dtype CPU receipts are pinned in the proposals. OFF3 used K2 disabled
  in both parent/P2 with SGD, not production K2-ON/table-OFF. ON2 used interpreted
  lookup with actual RMS. Cross-shape diagnostic remains failed at
  `7.62939453125e-6`; native execution is not proved.
  Reference notation uses actual PyTorch argument order: rows, FP32 master;
  FP32 parameters/gradients/RMS and BF16 gathered activations are unchanged.

## Prospective roles, frozen before results

1. Existing a318 native retry remains
   `fs-20260926-k2-indexrepair-on-parentcontrol-trn22-r1`; no duplicate retry
   or reset of the failed639 attempt. Actual preselected whole-step rate
   must qualify at >=1.03x. These proposals do not schedule that retry.
2. `20260926-indexrepair-k2-qualified-base-control` prospectively assigns the
   first qualified a318 full to one actual `kind=control/base=a318` record,
   under `fs-20260926-indexrepair-k2-control-<host>-r1`. One selected host,
   one actual run/label/row, including failure; the role never depends on BPB.
   It replaces the unstarted old throughput/control role, not an extra full run.
   Raw delta to genuine e1e2 may be reported separately, without fake K2 adoption.
3. `20260926-indexrepair-k2-fp32-last-layer-tables-timing` is repaired0127
   native45, label `fs-20260926-indexrepair-k2-fp32-tables-timing-<host>-r1`.
4. `20260926-indexrepair-k2-fp32-last-layer-tables-full` removes only
   `--num-steps 45`, label
   `fs-20260926-indexrepair-k2-fp32-tables-full-<host>-r1`.
   Require successful P2 native evidence and an actual valid same-host a318
   full control before dispatch. No old639/e1e2/timing/other-host row substitutes.

## Unchanged conditions

P2 does not need 3% over K2. Its quality decision remains fixed1800 charged
seconds and fresh public20M/context1024 with unchanged selector thresholds.
Startup above900 is charged; submission still needs genuine cold full replay,
fresh20M, Claude audit and human action. No schedule/batch/LR stacking here.

Preserve all original guardian/lock, idle, runtime, accounting and artifact
guards. Reuse machinery only after mechanically rebinding source/proposal/
label/argv/cache ownership; old639/b472 prepared payloads are not these jobs.
P2 floors remain initial15032385536 B, runtime11274289152 B, RAM68719476736 B,
including the additional536870912 B FP32 checkpoint. Measure actual prospective
needs and increase reservations if needed; no lowered floor, cleanup or AWS action.

Old proposals remain intact. Researcher changed only these three new proposals
and this note, with no source/selector/shared-state/ledger edits, new reviews,
tests, host calls or launch assignments. Root/Analyst handles evidence composition;
only the existing sole Operator may execute a separately admitted job.
