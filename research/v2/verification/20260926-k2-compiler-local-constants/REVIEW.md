# Independent fresh Verifier review

Verdict: **PASS_FOR_NEW_SOURCE_NATIVE_CHECK**

Candidate SHA-256: `662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.

This narrow verdict applies to the six geometry constants moving from the outer factory to `resident_consumer`. It grants no native success, training launch, adoption, or submission approval.

## Independence and input identity

Only `candidate.diff`, `SPEC.md`, `cpu-result.json`, `parser-result.json`, and `MANIFEST.json` in this review directory were read as task evidence. No implementer source, branch, notes, repository status, host, queue, or ledger was inspected or changed. Review outputs are confined to this directory. No subagents were used.

The supplied manifest hash independently matched `06f77b50cd45fb8f31716f1dd118353c4520f99a3beb32fa610d4500df530fd3`. Every listed file's byte length and SHA-256 matched. The full candidate file is not an allowed input: its full-file SHA is attested by the consistent packet manifest and CPU/parser receipts, not independently recomputed from full source. The diff provides the complete relevant factory and a single hunk; all changes shown are within that factory.

## Independently executed exactness evidence

Command executed successfully twice:

```
/home/poojith-devan/venv/bin/python -B /home/poojith-devan/k2-local-constants-review-TaQUwd/independent_check.py
```

Return code 0; stderr empty. The second execution's captured stdout was compared byte-for-byte against `independent_check.stdout` and matched.

The reviewer-authored standard-library CPU script reconstructs both complete factory ASTs from the diff. It proves that moving the two assignment nodes back makes the factory AST identical; all subsequent kernel statements, decorators, and the return selecting two programs are unchanged. A separate textual comparison covers the entire supplied hunk and rules out other changes in that context. Exactly two tuple assignments and their existing comment moved, with one blank line removed.

Independent executable scope probes retain the original assignment nodes at their respective original scopes and replace native operations with a tuple return. Both return exactly `(8192, 32768, 1024, 128, 512, 64)`. Compilation of the unmodified factory ASTs independently proves that the inner Python code object changes from exactly six geometry free variables to none. Full production row/column enumeration confirms that 64 groups of 128 cover the 8192 rows and two programs of 512 cover the 1024 channels exactly once. This executes the central binding/value exactness claim; it is expressly not numerical execution of NKI operations.

## Supporting receipts assessed

Both int32 and int64 CPU receipts contain 12 finite fixtures with matching actual/expected FP32 bits. The independent script verifies their bit equality and finite exponent patterns. Four nonfinite diagnostics per dtype remain excluded. OFF3 and ON3 status, three losses for each, three metadata calls, ordinary evaluation, and 21 rejection cases are internally consistent. OFF and ON losses are not asserted equal to each other; the supplied receipt describes each comparison against its corresponding base execution. Their full numerical test programs and model parameter/gradient arrays are not supplied, so those executions were not independently repeated.

The real-parser receipt consistently identifies this candidate, exit 0, empty stderr, exact unmodified factory AST, empty closure, NKI `0.5.0+28631259367.ga768afa6`, Trn2/LNC2, and all production BF16/int32 metadata shapes. It records verified MLIR, 1,062,963 bytes, SHA-256 `404acb968b16f1bb922b57b8ee2422eab3aedb7de57a54a59512597889dbc76b`. No backend or device execution is claimed. The parser was not rerun in this review and its MLIR bytes are outside the supplied packet. The closure diagnosis is consistent with the independently established scope change and the supplied successful parser receipt.

## Remaining uncertainties and authority

Backend lowering, allocation/liveness, executable generation, framework capture, native numerical/DMA behavior, cold startup, whole-step timing, checkpoint/budget behavior, and validation BPB remain unproved. CPU bit agreement and verified parser MLIR do not resolve those uncertainties. Host wheel binary equivalence is also not independently authenticated by this packet. This review creates no additional review layers or prospective gates and changes none of the existing run/proposal/operator requirements. No performance win is claimed.

## Durable review evidence

- `independent_check.py`: SHA-256 `51311034cc6bd4b8bd5fc77509824a5ffe6affda0c76d073f9245a7e94a484bb`
- `independent_check.stdout`: SHA-256 `b297a05edddeb957096043728d6295bd9a32e94f7b26a932d3f978cc89373043`
- Execution: return code 0, empty stderr; saved stdout independently byte-matched on repeat execution.

No failing claim was found within this explicitly bounded constants-relocation review.
