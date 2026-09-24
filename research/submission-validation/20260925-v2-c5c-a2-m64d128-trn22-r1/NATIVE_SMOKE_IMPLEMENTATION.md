# Implementer handoff: A2 native-30 smoke gate

This additive package consists only of
`NATIVE_SMOKE_GATE_CONTRACT.json` and `NATIVE_SMOKE_RUNBOOK.md`.
It does not alter the independently reviewed A2 proposal/contract or any
runtime/model/evaluator bytes. The JSON's literal command is the original
static-PASS A2 `source_command_argv_prefix + a2_argv + argv_extension`;
local ordered-array self-check returned `true`. The c5c model, `prepare.py`,
queue, original contract and independent review hashes are pinned in the
new contract.

The package is **HOLD**, not a host request. A different Verifier must check
it. Exact-size four-rank G2/RMS rank trace, host staging, breaker/STOP and
disk/HBM checks, the immutable `env -i`/STEP_LOG/observer request, one-shot
Supervisor authorization, native run, strict synthetic checkpoint check and
independent native verdict are all pending. The existing source does not log
native per-rank table/moment hashes, so a 30-step smoke cannot be declared a
rank-equality PASS without a separately reviewed diagnostic. The older
queue's 50-step IMPL-1 criterion also remains unresolved by this 30-step
package; the Supervisor/Verifier must reconcile it before a scored A2 run.

No SSH, CPU G2, native smoke, training, evaluation, AWS action, protected
deletion, breaker reset, adoption or leaderboard submission occurred here.
