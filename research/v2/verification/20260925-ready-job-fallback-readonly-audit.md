# Read-only audit: no dispatchable dense or S0 fallback

Observed at **2026-09-25T02:58:41Z** against canonical controller commit
`9d298f3c311c164551ecaa24694303b33f0c8527`. This is a planning audit
of committed receipts and state only. I did not SSH to either host, inspect
secrets, run G2, train, evaluate, edit source, reset a breaker, or authorize a
launch. Host observations in the cited receipts are point-in-time, not a fresh
Operator preflight.

## Decision

**No existing dense or S0 warm job is fully checked or legally dispatchable.**
`research/v2/state.json` (updated `2026-09-25T02:48:32Z`) records
`launch_authorized=false`, `ready_job_counts={"trn21":0,"trn22":0}`, and
`launch_allowed=false` on both hosts. Both host-specific G1 calibrations PASS;
the trn22 c5c CPU G2 numeric result is independently accepted for unchanged
training bytes/argv. Recent read-only observations show roughly 54 GiB free
per host and cleared workflow breakers. These are necessary conditions, not
native one-run authority. No c5c A0 control, S0 warm replay, or new BPB has
been scored.

| Candidate | What is proved | Decisive missing gate |
| --- | --- | --- |
| `fs-20260925-v2-c5c-a0-control-trn21-r1` | Proposal ratification, static package review, trn21 G1, exact persistent-ext4 capsule/tool transfer, checkout implementation review. The repaired **disabled** checkout draft passed offline mechanics in `research/v2/verification/20260925-trn21-stage-r2-checkout-repair-static-review.md`. | No enabled checkout request/final independent review/one-shot checkout assignment or host checkout/result review. Therefore no exact-host four-rank CPU G2 result, fresh native preflight/monitor/observer, or scored-run authorization. `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/GATE_CONTRACT.json` explicitly has `cpu_g2_dispatch_authorized=false`; `research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/GATE_CONTRACT.json` has `dispatch_authorized=false` and no final run request. |
| `fs-20260925-v2-c5c-a0-control-trn22-r1` | One prior native attempt was made. | The archived `research/v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json` records disk-floor stop before step 1, no checkpoint/BPB, and a consumed label. It cannot be relaunched. |
| `fs-20260925-v2-c5c-a0-control-trn22-r2` | Static package and earlier exact-environment CPU G2 passed; disk was later expanded and breaker rearmed. | `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.request.pending.json` still has `dispatch_authorized=false` and pins the older `e65a3097` guard/launcher bytes. The reviewed fail-closed r4 safety source is `446b6576`; its changed launcher/guard cannot be substituted under r2's request. The r3 interlock failed independent safety review. Earlier preflight is stale, not fresh native authority. |
| `fs-20260925-v2-c5c-a0-control-trn22-r4` | Frozen r4 source safety, disabled-package consistency, capsule static review, planning-only host feasibility, and c5c CPU G2 **numeric** carry-forward passed. | `research/v2/disk-monitor/r4-hold/GATE_CONTRACT.json` has `dispatch_authorized=false`, no source transfer or checkout; `research/v2/authorizations/trn22-c5c-a0-r4-source-feasibility/INDEPENDENT_PLANNING_VERIFIER_RECEIPT.md` explicitly refuses transfer authority. A separately reviewed exact transfer request, fresh preflight, checkout/result review, live monitor/observer and one-shot native assignment remain absent. |
| `fs-20260925-s0-warm-native-trn22-r1` | Bare S0 source and CPU checks, plus static review on `experiment/20260925-s0-warm-native-trn22-r1`. The old disk-size objection no longer applies after expansion. | That branch's `research/submission-validation/20260925-s0-warm-native-trn22-r1/request-template.json` has `dispatch_authorized=false`; `research/agentic/verification/20260925-s0-warm-native-package-review.md` is static PASS / dispatch HOLD. Exact live source/cache identity, current safety interlock, fresh preflight and a separately reviewed one-shot request are missing. A warm S0 check would not itself be a same-source c5c A0 control or a cold submission replay. |

Historical `35c9` controls used different source bytes and public-2M screening;
they cannot be silently relabeled as current `c5c410af...` public-20M A0
controls. A1/trn21 and A2/trn22 static packages also require valid same-host
c5c A0 controls, their own exact G2/native table gates, and separate unique
run authorizations. Queue entries in `research/v2/queue.json` are proposals,
not launch permissions.

## Shortest permissible route to a first scored c5c control

On trn21, finish and independently review an **enabled** exact checkout
request; issue a one-attempt checkout-only Supervisor assignment; have a
distinct Operator repeat full fresh host safety checks, perform one checkout,
and archive an independently reviewed result. Then run the stage-r2 exact
four-rank CPU G2 under its own reviewed assignment, independently compare its
numeric result, and perform fresh source/cache/checkpoint/disk/quiescence,
monitor and boundary-observer checks. Only a new exact one-shot native
authorization permits the full 1,800-second A0 run; a separate fresh-process
public-20M/context-1024 evaluation and retrieved log establish its score.
The trn22 r4 source-stage path may finish first, but it must meet its own
reviewed transfer/checkout and live native gates. Do not lower a gate or use
old approvals as substitutes.
