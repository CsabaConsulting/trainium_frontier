# Pending exact36 trn21 replica result audit

This is a local archive-only Analyst tool. No result is accepted yet. It reuses the immutable reviewed `closed_result` helper from `ae725297`, SHA256 `1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770`; its original main/selector is never called. The analogous control auditor7c05b8e9 supplied the small binding/pending pattern, not a new pipeline.

Scope is exact label `fs-20260926-arch36a8-startup-clock-trn21-r2-full`, packetf4fe5d8e, assignment18727a55, source7419927c/SHA36a862. Five immutable launch/progress observations authenticate all19 known training/compiler identities, including walrus. The train command includes startup clock/architecture/initial cache, noRecursive, and fixed1800. Original lock is66305:596823.

I previously prepared the replication packet; this is explicitly an **outcome/archive audit**, not independent source or packet approval. Source gates remain separate. The tool does not contact a host, run a model/evaluator, write source/ledger/state, query or change the selector, adopt a source, or authorize submission.

## Before completion

```bash
python3 -I -B research/v2/analysis/20260926-clock36-trn21-final-audit/audit.py
```

This authenticates immutable preparation/actual-progress/comparator evidence and prints `ONLY_PENDING_REAL_RAW`, nullBPB, null evaluationPIDs and null selector verdict. Final evaluation IDs are unknown at preparation. They are not guessed from the next PID or supplied only as an unchecked postflight list.

## After actual immutable archives arrive

```bash
python3 -I -B research/v2/analysis/20260926-clock36-trn21-final-audit/audit.py \
  --raw-commit FINAL_40_HEX_COMMIT --bundle-sha FINAL_64_HEX_SHA256 \
  --eval-observation EVAL_OBSERVATION_40_HEX_COMMIT \
    research/v2/operator/20260926-trn21-startup-clock-replication/progress/ACTUAL_EVAL_OBSERVATION.json \
    EVAL_OBSERVATION_64_HEX_SHA256
```

Placeholders are intentionally invalid. The final commit must contain the existing Operator format under this run's `completed/`: `results-bundle.json`, `explicit-process-postflight.json`, `post-eval-checkpoint-hash.json`. The separately pinned live-evaluation observation uses the existing read-only progress archive format (root/files with exact hashes and sizes, UTC, job_lines and device/lock observation). It must show the exact bound timeout, rehearsal and evaluator command chain, and the actual evaluator PID on neuron-ls. If actual evidence has a different schema, inspect and make a narrowly reviewed adapter; do not synthesize IDs or receipts.

The audit requires all known19 IDs plus every identity in the supplied live-eval observation to appear absent in the final explicit PID census, and checks final declared PID rows plus unfiltered PID/PPID/PGID/SID lineage, compiler names including walrus, device idle and original lock. This is evidence from supplied actual observations, not continuous process surveillance. Any additional compiler/eval IDs observed in subsequent archives must also be preserved in the final census; do not discard observations to pass.

Other required evidence: all inner/outer hashes and sizes; frozen binding/manifest/source/prepare/harness; exact argv/environment; observed training/log prefixes; consecutive finite step boundaries; all printed losses finite; actual uncapped fifth and capped allowance; process-through-exit charged time<=1800; source exit0/over_budgetFalse; retained checkpoint and unchanged post-evaluation hash under original lock; fresh20971520-token/context1024/B1 evaluation, recomputed nats/bytes BPB and timeouts; all4 graph CSVs; fresh guards before train and eval; complete terminal/outer exits and chronological cleanup. Retained research caches never become cold-full proof or cache-hit evidence.

Output is an in-memory **candidate row only**, arm `L2-architecture-startup-clock-36a862`, kind `architecture_startup_clock_arm`, base6ef7, never kindcontrol. It reports the same-host genuine control0.9899528600562257 and exact delta when real evidence passes. Immutable4037a818 also pins the trn22 genuine control0.9912010195941572 and exact36 counterpart0.9883821509811542. Historical schedule arms retain their original roles. The unchanged selector can be queried separately by Supervisor; this tool does not append or adopt.

## Readiness tests and limits

Run `test_audit.py` normally and with `-O`; all checks use explicit exceptions, not assertions. An existing **trn22** closed36 result is only a parser fixture. A labelled synthetic three-process live-eval fixture checks identity extraction and failure handling; its90001–90003 IDs never enter the real auditor. Mutants cover hashes, duplicate/nonfiniteJSON, official budget, checkpoint, missing fourth graph, eval totals/tokens, foreign binding, extraRecursive, missing/broken eval lineage/device/lock, omitted observed compiler/evaluator and survivingSID/walrus. These are readiness tests, not a completed trn21 result or source review.
