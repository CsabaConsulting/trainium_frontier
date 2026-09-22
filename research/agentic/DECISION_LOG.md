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
