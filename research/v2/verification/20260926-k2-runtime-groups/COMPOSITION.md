# Supervisor source binding — 26 September 2026, 10:25 UTC

The single fresh independent review is accepted for its stated narrow scope.
Its four files were imported unchanged from Documents checkout commit
`0f16b089c6f9260bb2cb677aea23cb1ec2372a2f` as canonical `d6e385693`.
The reviewer independently executed exhaustive address and ordered-FP32
reference checks. No second numerical review or native smoke is added.

The Supervisor separately executed full-file hashing and top-level AST
comparison in the canonical Desktop checkout. Results:

- Parent SHA256: `662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.
- Candidate SHA256: `f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
- Complete diff SHA256: `4ed25ae91e40bb07ed6fbeb5eb4a80ecc69c994463de9a8803ecd82ce8bc4ade`.
- Only `_k2_build_kernel` changes; all 113 other top-level AST nodes match.

This supplies the full-source identity binding deliberately outside the
reviewer's diff-only context. The real local production backend and both
official simulator receipts remain the evidence for runtime-register legality;
the reference alone does not prove native behavior. Nonfinite diagnostics
remain labelled failures, not finite-training evidence.

Actual whole-model cold startup, steady throughput, loss trajectory and net
tokens are still unknown. Source correctness does not release a full run or
tables. A completed prospective proposal and the sole Operator's normal
original-lock/source/idle/resource checks bind any one native201 measurement.
That native measurement is the native test, not a prerequisite to another
smoke. Continuation requires the human's >=3% cold-start-adjusted net tokens.

The current 662c test remains protected. Its observed SDK compile-cache lock
timeouts are distinct from the original device lifetime lock and are not
permission to reset either. The new source never overwrites its files or
the existing 414c table and 662c full-run packets.
