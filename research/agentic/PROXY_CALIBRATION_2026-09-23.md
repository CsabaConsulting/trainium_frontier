# Smoke-to-full proxy calibration

Generated from durable proposal and verdict artifacts with
`python research/agentic/proxy_calibration.py`. Negative BPB delta is better. Historical
mechanism labels are inferred and marked as such. Only full verdicts `KEEP` or `DISCARD` count as
comparable in correlation statistics; `CONFOUNDED` pairs remain visible but do not calibrate a
gate.

| Class | Pairs | Comparable full | Clean sign agreement | Clean Spearman rho | Evidence |
|---|---:|---:|---:|---:|---|
| all | 14 | 8 | 75.0% | 0.690 | smoke_gate_supported |
| throughput | 5 | 1 | 100.0% | — | insufficient_evidence |
| per_step_quality | 4 | 4 | 75.0% | 0.600 | insufficient_evidence |
| schedule | 2 | 2 | 100.0% | 1.000 | insufficient_evidence |
| capacity | 2 | 1 | 0.0% | — | insufficient_evidence |
| data_regime | 1 | 0 | — | — | insufficient_evidence |

## Decision

The pooled result is descriptive only. No individual mechanism class has the six comparable
pairs required to change routing automatically. Keep the current smoke as a correctness and
throughput gate, but do not use early BPB alone to reject schedule, capacity, or data-regime ideas.
Continue recording explicit mechanism classes so this report becomes decision-grade.

## Paired experiments

- `20260922-batch-ramp-only-default-shape` — data_regime (inferred); smoke -0.002403, full +0.006456; full verdict `CONFOUNDED`.
- `20260922-bf16-mlp-input-norm-r2` — throughput (inferred); smoke -0.015366, full +0.014453; full verdict `CONFOUNDED`.
- `20260922-bf16-mlp-input-norm-r3` — throughput (inferred); smoke -0.015396, full +0.013997; full verdict `CONFOUNDED`.
- `20260922-bf16-norm-output-except-mlp-input` — throughput (inferred); smoke -0.014421, full +0.014575; full verdict `CONFOUNDED`.
- `20260922-bf16-norm-output-policy-r3` — throughput (inferred); smoke -0.032185, full +0.010545; full verdict `CONFOUNDED`.
- `20260922-bf16-sdpa-input-only-default-shape` — throughput (inferred); smoke -0.009270, full -0.003939; full verdict `KEEP`.
- `20260922-clip-after-reduce-only-default-shape` — per_step_quality (inferred); smoke -0.000427, full +0.000240; full verdict `DISCARD`.
- `20260922-ngram-ve-dim64-only-default-shape` — capacity (inferred); smoke -0.002295, full +0.001116; full verdict `DISCARD`.
- `20260922-no-ngram-ve-only-default-shape` — capacity (inferred); smoke +0.007558, full +0.020397; full verdict `CONFOUNDED`.
- `20260922-no-qk-shift-fixedbudget-r3` — per_step_quality (inferred); smoke -0.000122, full -0.000201; full verdict `DISCARD`.
- `20260923-attn-scale-012-default-constant-code` — per_step_quality (inferred); smoke +0.001546, full +0.004030; full verdict `DISCARD`.
- `20260923-demon-beta1-r2-default-shape` — schedule (inferred); smoke +0.000492, full +0.000190; full verdict `DISCARD`.
- `20260923-ngram-ve-cooldown-lr-flag-only-default-shape` — schedule (inferred); smoke +0.002788, full +0.002097; full verdict `DISCARD`.
- `20260923-unet-skip-zero-init-r2` — per_step_quality (inferred); smoke +0.001911, full +0.002657; full verdict `DISCARD`.
