# FP8 VJP diagnostic — independent binding review and conditional assignment

Root reviewed the diagnostic implementation/independent receipt53bc070f and
the separately prepared Operator binding413b48f0+d729a3be. Exact source921271
and the prior native failure remain unchanged. Root did not implement the
checker or this manual driver.

The existing manual route retains exact source/prepare/checker hashes,
new exclusive run paths, owned between-job pause, STOP/disk/device checks,
300-second timeout plus60-second grace, four ranks and serial execution.
It must be launched under the common device lock; no separate launcher may run.

A tensor-free AST test evaluated the actual two receipt predicates against
45 cases: valid diagnostic observations (including numeric RMS1.0), wrong
rank/world/label/source/checker/verdict, every per-case ERROR, and missing cases.
All behaved as required. Exit0 with caught ERROR is incomplete; numeric mismatch
with OBSERVED is information, never component acceptance. This test does not
claim a native pass or validate unknown future code.

- Source SHA: 921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858
- Checker SHA: d8187ad2f3e68d204203a6252e16062e52521e8b93f119c7f0a9fbe82e4367fd
- Binding SHA: abd2c37456b5ea6faf6c73d9b6285ca2027f7e17ca4fe6918e73ec7ac086a2ad
- Driver SHA: b1774f17528bbd5d124ec490e6232062d1bdadc800df34394ac2a6c476ffc5d9

## Exact conditional assignment

The distinct Operator trn22_queue_operator is assigned ONE diagnostic on trn21:
`fs-20260925-l1-921271-vjpdiag-r1`, with the absolute argv/environment in
`research/v2/operator/20260925-trn21-l1-vjpdiag-r1/DIAGNOSTIC_BINDING.json`.

Prepare only the existing owned between-job pause and immutable files while
the active scalar train/eval runs. Do not interrupt it. Launch only after its
training AND fresh20M eval are terminal, old queue is exited, device is clear,
source pins match, no STOP/breaker and at least10.5GiB free, with the common
device lock acquired. Current global/host breaker flags must be checked by
Operator; this assignment never resets one.

Archive raw logs, four diagnostic receipts, exit, quiescence and real errors,
including an incomplete/timeout outcome. Do not automatically retry or start
FP8 timing/full training. After complete quiescence, promptly restore the
preserved fallback queue unless a separately checked priority job is assigned.
No cleanup, AWS action, submission or decision-threshold change is authorized.

The immutable binding's preparation-only `native_authorized:false` is
superseded ONLY by this exact conditional assignment and
`EXPERIMENT_PROPOSAL.completed.json`; preserve its hash rather than pretending
the historical template was itself launch authority. The original failed
component remains HOLD irrespective of the diagnostic outcome.
