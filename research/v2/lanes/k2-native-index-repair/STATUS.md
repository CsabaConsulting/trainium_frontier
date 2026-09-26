# K2 native integer-index repair — Implementer delivery

2026-09-26 14:40 IST. Source/CPU evidence only; no independent approval,
native qualification, timing gain, or host action is claimed.

Candidate SHA-256:
`a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`.
Frozen parent `k2-final-sprint/train.py` remains
`63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5`.
`SOURCE.diff` is the entire train.py change: accept only int32 or int64
two-dimensional IDs, and update the error text. There is no input conversion.

## Exact cause and dtype flow

The Operator reports the original native attempt failed on all four ranks
before step 0 at the int64-only guard; it did not log the runtime tensor dtype.
The unchanged actual `prepare.py` establishes the loader contract: lines
479 and 481 allocate CPU/device inputs as int32; line 514 explicitly copies
token rows to int32; lines 516 and 518 copy/yield those inputs. Candidate
`fetch()` at lines 1571–1575 only reshapes them. Forward line 1056 passes
the resulting IDs directly to the K2 metadata producer. Thus the normal
int32 loader input is rejected, independently reproduced by the CPU test
against the exact frozen helper. No speculative backend conversion is needed
to explain this failure. Observed log failure and source-derived dtype remain
distinct evidence.

The repaired guard preserves input dtype through clone, stable argsort,
searchsorted and the ordinary embedding gather. Both legal integer dtypes
produce identical routing metadata. Bounded positions/counts/starts remain
int32 as before. No hash, ordering, gradient precision, NKI instruction,
optimizer, budget, evaluation, loader, or flag-off execution change exists.
Invalid row rejection remains with the unchanged `F.embedding` operation.

## Executed CPU evidence

Command (exit 0):

```
cd /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-native-index-repair
/home/poojith-devan/venv/bin/python -B test_k2.py --base /home/poojith-devan/Desktop/Trainium/workflow/candidates/confirmed-clock36-116188/train.py
```

`cpu-result.json` contains exact source/test/base hashes and raw gradient bits.
The original test and instruction interpreter were reused; the test now runs
every suite with both int32 and int64 inputs and adds explicit contract checks.
For **each dtype**:

- 12 finite operation fixtures match ordinary embedding output and FP32
  gradient bytes, including duplicates, collision ordering, cancellation,
  subnormals, all-one-row skew and untouched rows.
- Four NaN/Inf/overflow fixtures remain explicitly
  `FAIL_NONFINITE_DIAGNOSTIC`; these are never finite-training passes.
- Full production 32x1024 metadata checks match the chronological permutation
  and counts for balanced, all-one-row and random IDs.
- Reused source storage and two tables/two microsteps preserve exact gradients.
- Three OFF steps and three ON steps versus audited e1e2 are bit-identical for
  loss, every gradient and every updated parameter; ordinary eval matches.
  Geometry is 8 layers, width64, context8, batch2, test SGD 0.01.

Additional checks establish int32/int64 routing identity, preserved snapshot
dtype, float32/float64/BF16/bool/int16/uint8 rejection, scalar/1D/3D rejection,
invalid geometry and >=2^24 token rejection, -1/vocab row rejection for both
integer dtypes, and no narrowing of int64 ID 2^32 into a valid row.

The actual inlined metadata and registered-autograd helpers execute in the
tests. The actual NKI body is CPU-interpreted with reduced geometry. Existing
test-only prepare and unused pyarrow placeholders are disclosed import shims.
This is not native compilation, full-size optimizer, or full-loop evidence.
Native stable sort/searchsorted support, capture, consumer semantics and whole
step timing remain unestablished for this repaired source.

Next owner: root's one fresh independent reviewer, then sole Operator only
after the exact repaired bytes and applicable proposal/preflight gates.
