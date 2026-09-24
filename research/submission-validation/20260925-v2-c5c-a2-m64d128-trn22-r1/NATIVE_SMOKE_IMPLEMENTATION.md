# Implementer handoff: repaired A2 native-30 smoke gate

This additive repair responds to the independent STATIC FAIL receipt at
`research/agentic/verification/20260925-v2-c5c-a2-native30-static-review.md`
(`a802f15a…`). The original package did not prove the STEP_LOG parent
existed and was writable, matching a real zero-step trn22 failure. The
repair adds a positive same-UID parent/probe preflight and a smoke-only
`NATIVE_SMOKE_BOUNDARY_OBSERVER.py` adapter with local-only tests. The
adapter is needed because the reviewed A0 observer CLI requires its
boundary file beside guard receipts inside the checkout, whereas this gate
requires a trusted STEP_LOG parent outside the clean source checkout. It
hash-checks and delegates to the unchanged A0 observer core; it is not a
training or evaluator change.

It does not alter the independently reviewed A2 proposal/contract or any
runtime/model/evaluator bytes. The JSON's literal command is the original
static-PASS A2 `source_command_argv_prefix + a2_argv + argv_extension`;
local ordered-array self-check returned `true` again. The c5c model,
`prepare.py`, queue, original contract and independent review hashes are
pinned in the repaired contract. No host path or value has been guessed:
the exact scratch root, UID, parent device/inode/owner/mode, write-probe
hash, guard and adapter hashes, and immutable request must all be bound at
stage, then independently verified before dispatch.

Local static self-checks on this repair: the exact ordered smoke argv still
equals the untouched A2 contract's prefix + A2 argv + extension; ten
local-only adapter tests pass, including missing parent/probe, symlinked,
inside-checkout, unwritable or identity-changed parent, and pre-existing
STEP_LOG failures. The guard-receipt parent must also be present and keep
its stage identity. `git diff --check` passes. The adapter and regression
test SHA-256 values are pinned in the repaired JSON contract; a Verifier
must recompute them, review the adapter's delegation to the unchanged core,
and independently confirm the complete request binding.

The package is **HOLD**, not a host request. A different Verifier must check
these repaired bytes. Exact-size four-rank G2/RMS rank trace, host staging,
breaker/STOP and disk/HBM checks, the immutable `env -i`/STEP_LOG/observer
request, one-shot Supervisor authorization, native run, strict synthetic
checkpoint check and independent native verdict are all pending. A valid
same-host c5c A0 control is additionally required before any full scored
A2 comparison. The existing source does not log native per-rank table/moment
hashes, so a 30-step smoke cannot be declared a rank-equality PASS without a
separately reviewed diagnostic. The older queue's 50-step IMPL-1 criterion
is explicitly **not** satisfied: a separately proposed, new-label,
independently reviewed at-least-50-step exact-source rank-hash diagnostic
must pass, or a human must durably supersede that older criterion, before
full A2 promotion. Simply extending this 30-step job to 50 steps would
not prove rank equality and would alter its reviewed argv.

No SSH, CPU G2, native smoke, training, evaluation, AWS action, protected
deletion, breaker reset, adoption or leaderboard submission occurred here.
