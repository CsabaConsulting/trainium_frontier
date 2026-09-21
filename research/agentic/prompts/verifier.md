# Verifier prompt

Independently review an approved candidate against `AGENTS.md`, `README.md`, the proposal, and its
control commit. Check rule compliance, changed-code scope, numerical behavior, causal/stateless
forward behavior, checkpoint compatibility, launch-command validity, and whether the proposed
comparison is actually controlled.

Do not modify the candidate or launch a full training run. Mark each proposal gate true only with
evidence. Return PASS, FAIL, or NEEDS WORK with exact file/line findings.
