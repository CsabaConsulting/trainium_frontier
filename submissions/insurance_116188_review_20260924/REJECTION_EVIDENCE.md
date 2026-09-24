# Prior rejection evidence

The repository contains two authoritative portal-record artifacts, but both visible messages are truncated. The full reviewer explanation is therefore **unknown** and is not reconstructed here.

From commit `58e341d4` (`submissions/dense6x1024_lean/rejection.md`), submission `sub-a7629c90`:

> REJECTED_BY_REVIEW: The submission is a sophisticated GPT training script with custom NKI kernels, Muon optimizer, and several training adapters. The most significant concern is the `_SubmissionEvalGPT` class, which over

From commit `68bacfd7` (`research/submission-validation/20260923-dense6x1024-direct-review-safe/rejection-sub-7be49b6b.json`), submission `sub-7be49b6b`, submitted `2026-09-23T13:43:00+05:30`, candidate SHA-256 `9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344`:

> REJECTED_BY_REVIEW: The submission is a dense GPT with custom NKI kernels, Muon optimizer, and several training-only adapters. The code is largely honest, but several constructs require closer human inspection: (1) `_sub

Neither record identifies the staged `116188cf...` bytes as the uploaded candidate. The exact missing evidence is the portal's untruncated reviewer text plus a portal receipt binding submission ID, uploaded filename/hash, timestamp, and status to these exact staged bytes.
