# Offline packager checks

This file records packager checks only. It is not an independent verification receipt and does not approve submission.

## Passed in this packaging worktree

- `python -m py_compile submissions/insurance_116188_review_20260924/train.py` — PASS.
- `sha256sum -c SHA256SUMS` from the package directory — PASS for all four packaged byte artifacts.
- AST/static audit — PASS: 85,557 bytes; exact candidate SHA; exactly one `load_for_eval`; sole `torch.load` has `map_location='cpu'` and `weights_only=True`; no hook registrations; `record_step_boundary` call exists; prior `_SubmissionEvalGPT`, `MethodType`, and `_submission_*` adapter markers are absent.
- Exact causal-diff audit against `28b89bfc:submissions/dense6x1024_direct_review/train.py` — PASS: the only changed line is `MATRIX_LR = 0.02` to `MATRIX_LR = 0.015`.
- Bare launch-command audit — PASS: exact manifest equality and no newline, pipe, redirect, chaining, or background operator.
- Evidence-manifest audit — PASS: all 13 referenced reviewer, CPU/load, finite-checkpoint, rehearsal, metric, result, and operator receipts exist and match their declared SHA-256.
- `confirmation-r2/test_static.py` — PASS (`LR0.015 confirmation r2 static/Decimal gates: PASS`).

## Environment-limited checks

- `python prepare.py check ...` did not execute because this controller lacks `pyarrow` (`ModuleNotFoundError`). This is an environment limitation, not a package PASS.
- Direct module import did not execute for the same missing `pyarrow` dependency. The controller also lacks `torch`.
- The first-arm legacy `test_static.py` could not rerun in this checkout because its review-clean base package path is absent at audit commit `83b433e3`. The equivalent one-line diff was independently recomputed from the pinned Git object and passed above.
- Tensor CPU and real-checkpoint `load_for_eval` checks were not newly rerun. Their preserved results and limitation are hash-bound in `evidence/canonical-independent-verifier.json` and the two run receipt sets listed by `EVIDENCE_MANIFEST.json`.

These limitations remain inputs to the required independent final package audit. They are not silently converted into passes.
