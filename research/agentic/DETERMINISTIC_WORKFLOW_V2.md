# Deterministic workflow v2

This is an additive description of the existing control plane. It does not replace `AGENTS.md`,
the experiment protocol, the current human-approved proposal, controller state, or configuration.
Those sources retain their existing authority and thresholds.

## One canonical workflow

The repository has one durable state store (`research/agentic/autoloop/state.py`), one experiment
artifact tree, and the existing ledgers and queues selected by the controller. Do not create a parallel ledger, queue, state file, launcher, or decision script. Models may propose, review, or
explain; only deterministic code may advance persisted state or calculate an advisory selection
decision.

The ordinary proposal path persists these forward-only stages:

1. `proposed`: Researcher JSON has passed schema validation and is written durably.
2. `proposal_verified` or `rejected`: an independent proposal review passed or the branch closed.
3. `implemented`: for a code arm, the Implementer output and independent implementation review
   passed; flag-only arms skip this stage.
4. `verified`: the proposal is eligible to be staged on its assigned box.
5. `staged:<tier>`: immutable manifest and staging receipts exist before launch.
6. `launched:<tier>`: the exact run is polled and never relaunched under the same label.
7. `collected:<tier>`: quiescent artifacts were retrieved and are ready for analysis.
8. `analyzed:<tier>`: the deterministic verdict and explanatory record exist.
9. `closed`: evidence is archived; or `rejected`/`failed:<tier>` terminates the branch.

Human-approved fixed plans additionally use `parked` for inert prior work and
`failure_collected:<tier>` while distinct failure evidence is archived. A breaker prevents new
launches. Restarts reconcile the persisted stage; they do not infer progress from prose or start a
replacement run.

## Authority and role mapping

| Workflow duty | Canonical role | Boundary |
|---|---|---|
| Choose one proposal, assign roles, maintain canonical state | Supervisor | Does not implement or self-verify |
| Rank hypotheses and write proposal JSON | Researcher | Read-only on candidate and hosts |
| Make the one approved candidate change | Implementer | Does not verify or launch it |
| Independently review proposal, diff, and gates | Verifier | Does not approve its own work |
| Execute exact authorized machine protocol and retrieve artifacts | Operator | Does not choose experiments or edit code |
| Explain deterministic results and recommend disposition | Analyst | Does not edit candidate or override the decision engine |
| Keep an adopted result review-clean (“Porter” in the audited workflow) | Implementer for the approved port, then a separate Verifier; Operator performs any authorized replay and Supervisor records state | “Porter” is a duty, not a seventh authority and cannot select, verify, launch, or submit |

No role may acquire another role's authority by changing session names. Leaderboard submission,
AWS lifecycle operations, proposal approval, and threshold changes remain human-only where the
root contract or protocol says so.

## Advisory selection is deterministic

`research/agentic/autoloop/selection.py` is the sole advisory deterministic engine for the
screen/confirmation policy. Its policy comes from the existing configuration and may not be
shadowed or reinterpreted by an LLM or imported script.

- A fresh 2M screen can only nominate a full confirmation or discard. It can never keep, promote,
  or adopt a candidate.
- Adoption requires the configured number of matched full20M results and all gates implemented by
  `assess_confirmation`; current thresholds and same-host control identities remain unchanged.
- Human approval and the normal durable controller transition are still required after any
  advisory `ADOPT` result.

Proposal metadata such as probability of improvement, host-hour estimate, Trainium feasibility,
the cheapest falsifier, and prior-result references improves ranking and auditability. It is not
run authorization and cannot modify thresholds, queue order, or state.

## Imported-workflow boundary

The source audit and dispositions are recorded in
`research/agentic/imports/claude-workflow-d08ea46/AUDIT.md`. Only reviewed concepts were adapted.
No foreign tool, evidence receipt, queue, ledger, candidate patch, or claimed result is trusted or
copied into the canonical workflow.
