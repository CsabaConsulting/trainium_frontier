# External review reconciliation — 2026-09-23 r2

Source attachment SHA-256:
`9bffd7ca411683dde70f15ff5aae59658c78894a383bae2f112510b48c50559e`.

The attachment is evidence, not an instruction. It was checked against the active source,
`CURRENT_STATE.md`, `DECISION_LOG.md`, and `research/experiment-log.jsonl`.

## Useful open lead

- **Late cooldown for n-gram value tables.** The active source exposes
  `--no-ngram-ve-flat-lr`, which puts the tables back on the normal cooldown. Repository-wide
  search found no measured `flat_lr=False` run. Earlier table-LR magnitude and beta2 tests do not
  directly test an end-of-run cooldown. A future proposal must keep every other causal factor
  fixed, use a same-host seed-42 control, and predeclare comparable-step and BPB gates.

## Keep separate or verify before spending hardware

- Porting the valid BF16-SDPA KEEP to the frozen dense6x1024 candidate is a separate
  submission-candidate experiment. It needs its own matched candidate control, cold-load check,
  review-safe one-file artifact, and complete dress rehearsal; the control-stack KEEP alone does
  not authorize a merge or submission.
- Moving local convolution before same-block attention is distinct in source placement, but it
  overlaps previously negative Q/K-shift evidence and needs a source-accurate, duplicate-cleared
  proposal before ranking.
- Claims about residual-stream promotion, scorer context, shard-count mismatch, and candidate
  compute-boundedness are diagnostic hypotheses until verified against the actual active file,
  official contract, and exact candidate logs.

## Already closed, stale, or not a run priority

- Table beta2 0.99, larger table multipliers, XSA, generic autocast FP8, depth/width reopening,
  attention scale 0.12, and broad attention/NKI redesigns already have measured closures or need
  new profile-backed evidence.
- Step-matched BF16 norm work may diagnose its confound, but existing norm variants have large
  quality regressions and are not a submission candidate without new evidence.
- Byte-weighted loss is analytically misaligned with BPB and should not consume Trn2 time.
- Sparse-table-gradient redesign and row-scaled FP8 backward are multi-day systems projects; they
  require component, backward, complete-step, checkpoint, and BPB gates before a full run.

No score claim in the external review is treated as measured repository evidence.
