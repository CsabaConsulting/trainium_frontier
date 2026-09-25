# General fixed-budget selector — independent review PASS

Authority: Poojith's direct general amendment, preserved in
`workflow/FIXED_BUDGET_SELECTION_2026-09-25.md`.
Implementer commitsc2e3323d/de75ef1d; canonical4578abae/c7094808.
Reviewer: root, distinct from Implementer. Exact selector SHA-256:
`fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`.
Rules version: `v2-2026-09-25-general-fixed-budget`.

Read the complete production/test/template diff. All step-window rejection
branches and exemption requirements are removed for every arm. Step counts,
below60%-of-control alarms and recorded spikes/stalls are diagnostic only.
The BPB branches do not consult them. Missing measured startup and unproven
cold evidence remain visible, not silently zero or claimed cache hits.

Explicit, mutually consistent1,800-second budget declarations are required.
Existing authenticated environment-only archival controls remain accepted;
no source-default guess or ledger rewrite is used. Short diagnostics/smokes
cannot be selected as quality runs. `over_budget=false` remains necessary;
contradictory explicit charged-time evidence above1,800, negative/nonfinite
or malformed charged values is rejected. Missing/null charge evidence remains
unknown subject to the existing result-review boundary; this is not certification
of missing accounting. Native capped startup estimates are separately labeled.

Every numeric nomination/discard/adoption/disagreement/drift/big-win threshold,
same-host control rule, high-prior class and two-host replication requirement
is unchanged. Cold-start proof is still required before submission and is
explicitly not implied by a warm research adoption.

Independent checks completed:

-16/16 regression tests PASS under normal Python and16/16 under `-O`.
-270 independent old-versus-new BPB cases across6arm kinds,9first-host
 deltas and5second-host configurations: old nominal-step verdicts equal
 new nominal-step verdicts; changing candidate step ratios to10/3600 and
 7200/3600 never changes the new BPB verdict.
-Canonical ledger regression: A1 DISCARD, A2 INSUFFICIENT preserving both
 invalid statuses, Recursive single-host NOMINATE, drift alarms empty.
-No training source, evaluator, data, decision thresholds or ledger contents
 changed by the selector commits. Whitespace check passes.

The Implementer separately reports the existing local fake-runner integration
passed C1/C2,F6,E1,B1,X1 and left no tagged processes. That report is supporting
evidence, not falsely described as an independently repeated host test.

This PASS releases the reviewed selector under the new human rule. It grants
no native launch, submission, cold-cache certification or adoption by itself.
