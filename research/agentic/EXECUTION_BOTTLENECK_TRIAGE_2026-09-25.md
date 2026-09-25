# Execution bottleneck triage — 25 September 2026

This is a Supervisor priority note, not a new queue, proposal, gate waiver, or
launch authority. `research/v2/state.json`, the approved proposal, and the
independent receipts remain authoritative.

## Measured problem

- The workflow-v2 ledger still has only the trn22 A0 attempt that stopped
  before step 1 on the earlier disk floor. It has no valid scored control or
  table result. `research/v2/state.json` still has `launch_authorized: false`.
- The 07:33 UTC dual-host read-only snapshot showed both expanded disks with
  about 54 GiB free and no matched job. That is point-in-time evidence, not
  current host preflight or a claim of present utilization.
- The accepted e65 science checkout lacks the reviewed native monitor and
  containment source, so a safe A0 control needs a new source checkout.
  Independent reviews of successive transfer wrappers found real STOP,
  activity, freshness, cleanup-authority and finally a mistyped SHA pin.
- A2's CPU G2 package also names tools absent from the pinned e65 Git tree;
  host presence is unknown. Its one-shot marker must not be consumed until
  a separate read-only exact-path proof or reviewed tool stage resolves this.
- Both disabled r10 source-stage operators were found to read their local
  breaker state from the older `Documents/ChatGPT/Trainium` worktree instead
  of the active `Desktop/Trainium` controller. At 10:05 UTC their state-file
  SHA-256 values differed (`9e859fdf…` versus `07543c62…`). A tripped active
  breaker could therefore be missed. Final staging remains HOLD until both
  operators bind the current authority and independent regressions prove a
  stale Documents state cannot override an active Desktop STOP/breaker.

## Shortest approved critical path

1. The exact trn22 e65→r10 receipt pin was corrected and received a
   source-only static PASS. Repair the newly found controller-root pin in
   both r10 stage operators, add fail-closed stale-state regressions, and
   independently review the exact repaired bytes. Do not stage r9 on trn22
   for scored work: it would break the same-HEAD pre-run rule.
2. In parallel, review and run strictly read-only live timing/feasibility
   probes for the exact r10 stage on each host. A safe refusal is not a
   usable stage result; record the timing before consuming a one-shot label.
3. For each host, prepare the exact enabled stage request, independent final
   review, Supervisor assignment and fresh Operator preflight. Stage r10,
   retrieve readback/postflight, and obtain independent result review.
4. Prepare the exact native A0 control packages on the staged source with
   G2 carry-forward, monitor/containment, disk projection and one-run-per-
   host checks. Run and archive one valid full control per host before any
   quality arm is compared.
5. Resolve A2's tool paths in parallel. Then exact-size CPU G2, native table
   smoke, A1/A2 full runs, fresh public-20M/context-1024 eval, and same-host
   comparisons. Old-stack table gains are priors, not a dense result.

Until a valid control exists, do not spend host or review capacity on the
old 640 stack, new architecture flags, or speculative kernel work. A checked
S0 cold replay is the fallback if it becomes launchable before a table arm.
Keep the existing `queue.json`, ledger and `decide.py`; do not create a
second scheduler. Automate exact-pin/hash validation in the corrected stage
package rather than relying on another manual transcription.

No one should promise an improved BPB or a clock-time ETA before a first
valid run actually starts. A full training arm still needs 1,800 s plus
fresh evaluation and retrieved evidence after all launch gates pass.
