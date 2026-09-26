# PASS_FOR_ONE_NATIVE_RETRY

Independent bounded CPU review of the two-line int32-or-int64 guard repair. The supplied diff changes only the accepted dtype predicate and its error string. No cast, metadata algorithm, gather, backward, or OFF-path alteration appears in this diff. Default REFUSE was held until the supplied author receipt arrived and was inspected.

## Executed independent claim

Command: `/home/poojith-devan/venv/bin/python -B /tmp/trainium-k2-index-review-u6xevN/independent_check.py`

Exit 0, torch 2.13.0+cpu. My script reconstructs only the new metadata function from the supplied diff context. Independently written ordinary Python row lists supply stable positions, counts, starts, and group maxima. Four small deterministic cases cover mixed duplicates, all-zero IDs, all-8191 IDs, and descending IDs, using vocabulary 8192/group 128 and both int32 and int64.

Actual outputs: all four cases PASS; 32 exact metadata/reference comparisons PASS; 14 invalid dtype/dimension rejections PASS. For each case and dtype, ordinary embedding gather exactly matches explicit reference row lookup, and sequential FP32 accumulation driven by returned positions exactly matches a direct token-order reference. Tangents and weights are BF16; accumulated outputs are finite FP32. Mutating original IDs after preparation leaves every returned tensor and snapshot gather unchanged. int32/int64 metadata values and routing match exactly, with snapshot dtype preserved. These are exact tensor value comparisons, not a signed-zero bit-pattern assertion.

Full captured output is in `independent_check.receipt.txt`; reference code is in `independent_check.py`. No source worktree, implementer notes, SSH, native execution, or launches were accessed.

## Attached gate receipt

Read AUTHOR_GATE_RECEIPT.json in full. Its candidate source hash matches the lane specification. It records 12 EXACT_FINITE operation fixtures per dtype, OFF three-step identity, ON three-step CPU-interpreter identity, and 21 input-contract rejection checks. Four nonfinite fixtures per dtype remain explicitly FAIL_NONFINITE_DIAGNOSTIC; they are not training passes. Range rejection is attributed to unchanged F.embedding, not the metadata guard. I independently tested neither those author fixtures nor native compilation. Null metadata/identity/lifetime summary fields are not treated as additional evidence.

## Limits and disposition

PASS_FOR_ONE_NATIVE_RETRY applies only to this tiny guard repair and these exact supplied bytes. It does not prove native int32 argsort/searchsorted/gather lowering, kernel correctness on hardware, performance, whole-step execution, checkpoint production, or scientific improvement. There is no new range-check claim: valid IDs only were used by my reference cases. The previous zero-step native failure and source-file identities are supplied evidence, not independently retrieved here. Root must bind the reviewed diff to the final full source bytes and apply all separate authorization, machine, and run gates. This review itself is not launch authority or permission for repeated retries.

## SHA256 bindings

- candidate.diff: `a2735f824e6207217533c2a098e8992cd2176043626e956147651976ca8701e5`
- LANE_SPEC.md: `aa3535669b4f69779ee0a2acaa3912195bd3400a9a4fc61d6c2261560205cfce`
- AUTHOR_GATE_RECEIPT.json: `930b80e38684aafee6805cdf57bb90a2d8ced0815ad8a14e0219ffe64fdc7c46`
- independent_check.py: `296548d08ee1e26e8fd8bbbc940138525294561f73087f8046585c716c5ee6c5`
- independent_check.receipt.txt: `e57f9e57ed09d6680e01ce154ef77d59c700642527084efdd89ab6f6cb963a32`
- Candidate full source, supplied binding only: `a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`
