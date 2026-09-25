# Independent review: explicit table budget exemption

Reviewer: `/root/host_queue_implementer` (did not author this decision change).
Reviewed commit: `b286bda2dc3b90f6f9169bfa33f3425071b40d00`.
Human authority: `workflow/RESUME_AND_TABLE_CORRECTION_2026-09-25.md`, item 1.

Verdict: **PASS for the requested narrow change**.

The only scientific gate changed is the step-ratio window. Each latest host
record must itself have `kind == "table_arm"`, `throughput_exempt is True`,
and `training_budget_seconds == 1800` to gain the new exemption. String truth,
other arm kinds, absent/false declarations and 120-second budgets do not pass.
The existing throughput-arm exemption remains. The normal validity filter,
same-host control selection, table high-prior behavior, delta thresholds,
two-host agreement, big-win hold and control drift logic remain unchanged.

Independent execution of all six supplied regression tests passed. Their
cases cover the explicit per-host declaration; wrong kind, flag type and
budget; prior throughput behavior; invalid budget/correctness/20M/dirty
records; high-prior ambiguous results and big-win HOLD; and two-host adoption
with one host's exemption removed.

This receipt reviews decision-code scope only. It does not certify any A1/A2
run as valid or completed. The Operator/Analyst must still prove actual
1800-second budget compliance and complete fresh public-20M results before
declaring a run evaluated or supplying these fields to the decision engine.
