# Fresh independent source and bounded CPU review

You are a new independent reviewer. Inputs are only the supplied full-context source diffs, this neutral specification and four raw reduced CPU gate receipts. You have no repository, shared branch, implementer tests/notes, prior conversation or prior review conclusions. Default **REFUSE** until the evidence you independently inspect and produce supports a precisely stated scope. Do not modify candidate source. Leave your complete report, exact tests/commands/results and failure evidence in `/work`.

## Candidate and intended change

The candidate ports a fixed existing training schedule package onto an eight-layer cache-primed architecture. It must remain default OFF, preserving the baseline's default and explicit-OFF training arithmetic and model/load semantics. ON intentionally changes optimizer schedules, so ON final weights are not expected to equal OFF. No base LR, batch, model architecture, data, timer/accounting, loss, evaluator or optimizer mathematical implementation change is requested, except the existing device-scalar path needed to transport time-varying beta2 into compiled Muon.

Frozen source inputs, all reconstructable from the full-context diffs:

- Baseline: `b4d83b5db66971f3c348837f04f53cba65aa6d33`, SHA256 `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
- Candidate: `fe0969fd311d654e143dc6113614416c8970a424`, SHA256 `75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f`.
- Existing schedule reference source: `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`, SHA256 `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`. This supplies the exact schedule arithmetic to port, not evidence about the new architecture's quality or native runtime.

The target architecture is113,256,724 parameters/65 parameter tensors at8x1024; sequence1024, pack32, total batch131072, matrix LR0.015, seed42, max training1800seconds, four LNC2 ranks. No full-size candidate/ON G2, native compiler, cold startup or quality evidence for this port is supplied. This review must not claim those gates complete.

The ON schedule specification: wall-clock progress clamped[0,1]; Muon LR warmdown ratio0.95 and AdamW0.65, both floor0.05, retaining step warmup. Existing flat-table LR precedence remains. Muon momentum step ramp0.85→0.95 over300 steps, with the reference's progress>0.05 quadratic cooldown to0.79 taking precedence there. Muon beta2 is0.95 through initial5% progress, then linearly reaches0.97. Inspect exact branching, scalar transport and arithmetic against the supplied reference; do not substitute a different schedule based on this prose. Preserve ordinary defaults and save/load toggle handling for old and new checkpoints.

Inspect the whole diff and relevant full context: CLI default and explicit OFF/ON; parameter-group LR kind/flat precedence; compiled and eager Muon beta2 inputs; timer/reset/accounting unchanged; checkpoint toggle serialization/default restoration; no model/forward/evaluator changes. Scope any finding to actual evidence, not hypothetical unrelated redesigns.

## Raw receipts and their limits

Four supplied raw JSON receipts are baseline/reference, candidate default, explicit OFF and ON, each reduced8x128, four CPU ranks, three updates on synthetic microbatch1x16 per rank. They use a test-only prepare/data environment, CPU/no-compile, fake timing and reduced tokens; they are not the full113M model,131072 production tokens/update, native NKI kernels, real data, full gradient history or cold/runtime proof. Treat recorded `status` labels as data; authenticate fields and compare actual losses, parameter digests, plans, finiteness/rank/reload evidence yourself. Checkpoint file identity is not implied by model-tensor identity because added saved metadata can differ.

## Required own bounded test and report

Write and actually run your own small CPU exactness/reference test for a substantive claim, not merely a rerun of any supplied receipt or implementation's test. Suitable scope includes extracted real schedule/group-update arithmetic against an independently written numerical reference and default/OFF identity; include meaningful negative controls or boundary cases. You may use standard CPU PyTorch and a tiny tensor optimizer/reference fixture if useful. Record test code, commands, runtime/dtype, inputs/expected/actual outputs, tolerances (strict where exactness is claimed), and all failures. An AST-only comparison is not a substitute for your own numerical test. No production full-size allocations or training are permitted.

Keep single-thread/sequential execution and tracked live test tensor memory below64MiB; only tiny models if independently needed, width<=128, vocabulary<=256, sequence<=32, batch<=2, one live training model. If full-model import would require unavailable dependencies, you may extract exact source functions or use explicitly disclosed test-only stubs; do not pretend such a test is actual full-model evaluation. Do not use or reconstruct large cached checkpoints. Finish within the session's20-minute outer bound; do not launch background jobs, concurrent workers or additional models/reviewers/providers.

Write `/work/REPORT.md` and `/work/decision.json` with literal PASS or REFUSE, exact scope, source/input hashes, findings with code locations, your test evidence, raw-receipt interpretation and pending gates. Preserve failed cases and error logs. A PASS can cover only source/reduced-CPU suitability for the next exact full-size CPU qualification; it cannot authorize host/native/cold/quality/adoption/submission. If refusing, state the precise failed requirement and whether any narrower diagnostic-only scope is supported without waiving it. Make your judgment independently; do not ask the operator to supply conclusions or approve the evidence.

## Isolation boundaries

Read only `/packet` and standard CPU/system libraries; write only `/work` or temporary `/tmp`. Do not access network, hosts, APIs/providers, credentials, user configuration/history, other repositories, files outside the exposed review sandbox or any external checkpoint/data. Do not attempt to escape isolation or seek hidden inputs. You may run your own local tests without asking for permission. No source edit, evaluator patch, training experiment, host/device work, queue/ledger mutation or upload is authorized.
