# K2 cast-boundary evidence, not a gate waiver

2026-09-26 03:13 UTC. Supervisor/Analyst analysis while both full native jobs
run; no model source, prior test, evaluator, host or selector was changed.
Command: `/home/poojith-devan/venv/bin/python -B research/v2/analysis/20260926-k2-cast-boundary/audit.py`.
Exit0;24 tiny CPU autograd cases. No full-size model or NKI was instantiated.

## New evidence

Exact adopted36a862 source at7419927c immediately casts both ordinary token
and value-embedding outputs to COMPUTE_DTYPE; compute_dtype_for(neuron) returns
BF16. AST inspection authenticates these two expressions. Ngram/table paths
are outside this finding and disabled in the adopted configuration.

An independently written tiny CPU analogue uses FP32 embedding weights,
BF16 output, hooks before/after the cast, and an explicit ordered FP32 scalar
recurrence. It confirms the backward cast widens BF16 tangent values into
FP32 exactly. The FP32 minimum2^-149 becomes zero at the BF16 boundary. Normal
FP32 operands00800001 +80800000 would cancel to00000001 without the cast,
but round to00800000 +80800000 and cancel to zero after the BF16 boundary.
This is why merely excluding tiny *inputs* would have been insufficient.

Crucially, representable BF16 subnormals are not dismissed:00810000 +80800000
cancels to00010000 (2^-133), preserved by the BF16 path and FP32 accumulation.
Both signs, duplicate IDs and untouched rows were checked, with exact bits.
The FP32-output contrast preserves00000001, demonstrating the two contracts
are not equivalent. All raw input/tangent/intermediate/final bits are saved.

Authenticated native rank receipts from the original registered-kernel test
also distinguish these cases. On every rank:

- BF16-subnormal fixture:32768 nonzero inputs;1408 nonzero expected/actual
  output gradients; all five routes exact, including ordinary compiled.
- FP32-minimum fixture sent as BF16: zero input, zero output. Its PASS is a
  cast-to-zero observation, not proof of nonzero subnormal transport.
- FP32-minimum fixture sent as FP32: candidate routes exact with1408 nonzeros;
  ordinary compiled returns zero and the original aggregate FAIL is retained.

Thus this evidence does NOT support a universal native flush-to-zero claim.
It narrows a useful future question: can a production BF16-output-only
specialization be qualified against the actual reachable tangent domain,
while the unrestricted FP32 replacement stays disabled?

## Limits and next action

Static production casts plus tiny CPU hooks are not a full-model native
trace, all-input proof, metadata-generation implementation or timing test.
The diagnostic consumer is still only256x128, not a production-size patch.
All old unrestricted FP32/ordinary-control failures remain failures. No
existing acceptance criterion was weakened and no native launch is released.

Next is a separately completed BF16-only proposal and fresh independent
review of source reachability, cancellation/rounding, exceptional inputs and
guarded fallback. A reviewer must supply its own test, not repeat this note.
Only if that contract is justified should a new production-sized integration
be implemented and tested. Preserve the exact FP32 oracle and raw failures;
do not replace expected values with observed zeros or silently drop cases.

This is analysis by the Supervisor, NOT a fresh independent candidate approval.
The already accepted clean submission source is unrelated and unchanged.
