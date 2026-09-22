# Decision log

This is a compact index, not a replacement for run logs. Append one row after a verified verdict.

| Date | Proposal | Commit | Control | Result | Verdict | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-21 | Establish agent operating contract | 7f9aaaf5 baseline | n/a | No run | Adopt | `research/agentic/README.md` |
| 2026-09-21 | Reconcile latest fresh-start handoff | 9a4caa7b research evidence | n/a | No run | Analyze source197, then conditionally pursue complete ROW-FP8 backward | `research/agentic/RESEARCH_HANDOFF_AUDIT_2026-09-21.md` |
| 2026-09-22 | Attribute Source197 layer-position regression | 282935ca source evidence | layer 0 vs layer 5, one identical library replacement | Forward graph delta 193.388 ms; +16.344 GB spill reload; direct library work changes modestly | Caller-boundary constraint identified; hold ROW-FP8 backward and propose one bounded boundary/layout gate | `research/agentic/SOURCE197_OFFLINE_ATTRIBUTION.md` |

Before proposing work, search this file, `research/experiment-log.jsonl`, and
`research/trn2-experiment-results.md` for the idea and its aliases.
| 2026-09-22 | autoloop control-trn22-lnc2x4-9a4caa7b (`(control)`) | 9a4caa7b | control-trn22-lnc2x4-9a4caa7b | cand None vs ctrl None bpb, steps None/None | FAILED (smoke) | `research/agentic/experiments/control-trn22-lnc2x4-9a4caa7b/` |
| 2026-09-22 | autoloop control-trn21-lnc2x4-9a4caa7b (`(control)`) | 9a4caa7b | control-trn21-lnc2x4-9a4caa7b | cand None vs ctrl 1.005734 bpb, steps None/2071 | KEEP (full) | `research/agentic/experiments/control-trn21-lnc2x4-9a4caa7b/` |
| 2026-09-22 | autoloop control-trn22-lnc2x4-9a4caa7b (`(control)`) | 9a4caa7b | control-trn22-lnc2x4-9a4caa7b | cand None vs ctrl 1.005967 bpb, steps None/2076 | KEEP (full) | `research/agentic/experiments/control-trn22-lnc2x4-9a4caa7b/` |
| 2026-09-22 | autoloop 20260922-batch-ramp-only-default-shape (`--batch-ramp`) | 9a4caa7b | control-trn21-lnc2x4-9a4caa7b | cand 1.01219 vs ctrl 1.005734 bpb, steps 2397.0/2071.0 | CONFOUNDED (full) | `research/agentic/experiments/20260922-batch-ramp-only-default-shape/` |
| 2026-09-22 | autoloop 20260922-ngram-rms-fp32-only-default-shape (`--ngram-rms-fp32`) | 9a4caa7b | control-trn22-lnc2x4-9a4caa7b | cand 1.173336 vs ctrl 1.165604 bpb, steps 316.0/335.0 | DISCARD (smoke) | `research/agentic/experiments/20260922-ngram-rms-fp32-only-default-shape/` |
| 2026-09-22 | autoloop 20260922-nki-local-conv-only-default-shape (`--nki-local-conv`) | 9a4caa7b | control-trn22-lnc2x4-9a4caa7b | cand 1.167854 vs ctrl 1.165604 bpb, steps 330.0/335.0 | DISCARD (smoke) | `research/agentic/experiments/20260922-nki-local-conv-only-default-shape/` |
