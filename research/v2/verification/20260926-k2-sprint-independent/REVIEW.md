# Independent review: PASS for the short native test only

Reviewed 2026-09-26, within the assigned 30-minute limit. Decision: **PASS** at
the SOURCE/CPU/OFF3 gate for a short native whole-step and cold-start test of
the exact frozen candidate below. I found no concrete incorrect semantics,
unsafe source behavior, or invalid CPU/OFF result that warrants refusing that
test. This is not native correctness, speed, full-run, adoption, launch-authority,
or submission approval. Existing proposal, machine, and Operator gates remain
outside this review.

## Scope and evidence

I read only the seven frozen inputs in this directory. I did not read repository
state/history, other reviews, implementer correspondence, or external sources;
I did not contact the implementer, access a host, or launch anything native.
The supplied files were left unchanged. The actual unified diff agrees with
SOURCE.diff, including its hunk line counts and content.

The new `reference_test.py` uses a plain-Torch reference that visits tokens in
logical flattened order and performs one FP32 in-place row addition per BF16
tangent. This reference uses neither metadata nor sorting nor a grouped
reduction. It independently checks the actual extracted metadata and registered
autograd helpers. To exercise the actual inlined consumer source on CPU, it
uses the supplied instruction interpreter with only reduced geometry. That
interpreter remains an assumption about native instructions, not an SDK
simulation or native proof.

Executed successfully, exit 0:

```
cd /home/poojith-devan/k2-final-sprint-review-JsMWCX
/home/poojith-devan/venv/bin/python -B review/reference_test.py
```

`reference-results.json` preserves the observed result. Torch was 2.13.0+cpu,
one CPU thread, seed 42 for comparison initialization, denormal flushing disabled.

- All 25 finite operation fixtures matched both the new sequential reference
  and ordinary Torch embedding/cast backward by bytes. Coverage includes
  interleaved duplicates, all-one-row skew, empty rows/groups, noncontiguous IDs
  and tangents, both BF16 subnormal signs, cancellation, signed zeros, and
  the `2^24, 1, -2^24` and `1, 2^-24, -1` order witnesses. The first fixture's
  repeated row occurs at logical tokens 0, 4, 8. Interpreter write masks verify
  each owned output element is written once and the other owner's half is not.
- Two tables and two simultaneously live forward microsteps matched both
  references after the common loader storage was overwritten before any
  backward. Metadata is per-forward, and the forward snapshot and backward
  routing stay associated. No mutable global metadata cache is present.
- Direct invocation of the actual registered backward callback rejects FP32
  tangents before dispatch: `K2 refuses a non-BF16 output tangent`. Ordinary
  autograd may convert a user-supplied tangent to the BF16 output dtype before
  this callback; the claim here concerns the callback's actual input.
- A fresh import of each full source passed three default-OFF model steps:
  losses, every gradient, and all state values matched by bytes. Three ON steps
  using the interpreted consumer also matched. Geometry was eight layers,
  width64, context8, batch2, with SGD .01. One metadata build occurred per ON
  forward. Parameter identities stayed fixed.
- With K2 globally enabled, eval with gradients, eval without gradients, and
  training-mode no-grad all matched ordinary embedding and never called K2
  helpers. OFF helpers were also set to raise if accidentally reached.

The full-source CPU imports use reviewer-local, unused `prepare` symbol and
`pyarrow.parquet` placeholders. Any attempted data/evaluator function call
would raise. No data, evaluator, tokenizer, or scoring code was changed or run.
This is a model-level OFF3 check with a test optimizer, not production optimizer
or full-training-loop qualification.

The author's receipt matches the exact base/candidate hashes and contains 12
finite raw-bit matches plus four explicitly failing nonfinite diagnostic
fixtures. I do not count those four as passes. The supplied author's standalone
test entry point also refers to sibling `k2.py`/`consumer_nki.py` files for its
final hash receipt and to an external test-only prepare import; those inputs
are not in this frozen bundle. I did not execute that entry point or access
those files. The new test above runs from the frozen bundle without them and
independently reproduces the required CPU/OFF evidence.

## Source assessment and native risks

The patch changes the explicitly enabled training/grad route for WTE and VE7,
plus adds the opt-in flag and geometry checks. The original ordinary branches,
parameter construction, optimizer, evaluation loading, and budget machinery
are preserved. Stable ordering and left/right boundaries give chronological
positions within each valid vocabulary row. Integer positions stay below the
declared exactness bound; the consumer iterates occurrences sequentially in
FP32 and masks inactive lanes with explicit initialization. The fixed two
channel owners have disjoint stores.

The following remain unresolved by design and should be measured by the
already proposed short native whole-step/cold-start test:

- Installed backend lowering of stable argsort/searchsorted and integer
  metadata reductions; registered-autograd capture and consumer invocation.
- Native indirect DMA addressing, 512-channel tiles, out-of-bounds skip,
  runtime loop limits, buffer liveness/allocation, and compiler scheduling.
- Actual BF16-to-FP32 widening, subnormal preservation, FP32 rounding/order,
  and whole-step numerical behavior on the device.
- End-to-end cost of the producer and both consumers, strong-skew padding,
  compilation/startup cost, and interactions with the ordinary optimizer,
  collectives, and weight-cache refresh.

These are not established defects. No native speed or numerical result is
claimed. A native compile/numerical failure must be recorded as a failed gate;
the code supplies no silent fallback. A short-test PASS would still not by
itself establish a full-run speed win or final validation improvement.

## Exact SHA-256 identities

| Frozen input | SHA-256 |
| --- | --- |
| base.py | e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43 |
| train.py | 63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5 |
| SOURCE.diff | b24a13c3a639b3946150ac3c6b14187880b36e611fdab4c83763c0f2c20e37aa |
| SPEC.md | 441bff0d111bf69af99915144979f56f3d12bd44acc8165e9ed7f42bdcd11448 |
| cpu-result.json | f2ccec824749764e8460502b9c8377a1bc7a3eb30e86c47f99d2cca0f1280b1c |
| cpu_instruction_model.py | 98ec82913b314bfcb2fea519540c5806109609f80422210999955bd9eb8e9024 |
| test_k2.py | d5bbd981c7b0992515350e9047088d79f3daa50fdf39a0a28c1c3cfb2e147916 |

Reviewer test SHA-256:
`afd4bcb40476ac369669c1b2ed4dac89c32ddfb6d9e358c63109fd9e39e7b079`.

No refusal witness was found within this bounded source and CPU scope.
