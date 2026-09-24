# Offline packager checks

This file records packager checks only. It is not an independent verification receipt and does not approve submission.

## Passed in the metadata-repair worktree

- `python -m py_compile submissions/insurance_116188_review_20260924/train.py` — PASS.
- `sha256sum -c SHA256SUMS` from the package directory — PASS after resealing every package artifact except `SHA256SUMS` itself, as documented in that file.
- AST/static audit — PASS: 85,557 bytes; exact candidate SHA; exactly one `load_for_eval`; sole `torch.load` has `map_location='cpu'` and `weights_only=True`; no hook registrations; `record_step_boundary` call exists; prior `_SubmissionEvalGPT`, `MethodType`, and `_submission_*` adapter markers are absent.
- Exact causal-diff audit against `28b89bfc:submissions/dense6x1024_direct_review/train.py` — PASS: the only changed line is `MATRIX_LR = 0.02` to `MATRIX_LR = 0.015`.
- Bare launch-command audit — PASS: exact manifest equality; no embedded newline, pipe, redirect, chaining, or background operator. The exact 246-byte file has **one trailing newline**.
- Evidence-manifest audit — PASS: all 13 referenced reviewer, CPU/load, finite-checkpoint, rehearsal, metric, result, and operator receipts exist and match their declared SHA-256. Both runs' result, runtime-job, and postflight JSON files are additionally bound directly by exact path, size, and hash and were independently re-read and verified.
- `confirmation-r2/test_static.py` — PASS (`LR0.015 confirmation r2 static/Decimal gates: PASS`).
- Official `prepare.py check` in an offline dependency-complete Python 3.12 environment (`torch`, `numpy`, `pyarrow`, `requests`, `tiktoken`) — PASS (`check passed`).
- Direct module import in that environment — PASS.
- Synthetic CPU `load_for_eval` checkpoint round trip — PASS: strict load, exact logits, eval/frozen model, and zero module hooks.

## Preserved limitation

- The first-arm legacy `test_static.py` could not rerun in this checkout because its review-clean base package path is absent at audit commit `83b433e3`. The equivalent one-line diff was independently recomputed from the pinned Git object and passed above.
- Real historical checkpoints were not copied into this package or reloaded during metadata repair. Their preserved weights-only finite/load results remain hash-bound in `evidence/canonical-independent-verifier.json` and both run receipt sets.

The package status is `PENDING_INDEPENDENT_REVERIFICATION`; these packager results are not a self-issued verifier receipt or submission approval.
