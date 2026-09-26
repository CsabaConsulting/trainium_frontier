# P2 dependent integer-index port — conditional CPU delivery

2026-09-26. Implementer only. No new review, approval, host action, native
qualification, launch, timing gain or BPB claim.

Source SHA-256:
`0127d02dcaae0e6fdb70db693697caf02233db31d926dddd4649a1d496e6b543`.
Frozen P2 source remains
`b4728d17d082b44018290d2f465b5956f99ecbebe61da8b81508365c7dc53190`.
The full source delta is the two lines in `SOURCE.diff`: the shared token
metadata guard accepts int32 or int64 two-dimensional inputs and names both
in its error text. This is identical to the repaired K2 parent's guard.

The actual unchanged loader emits int32 tokens; the inherited int64-only
guard would reject them. No casting or other production change is added.
WTE/unigram metadata now accepts the loader dtype. The table hash function
still performs its original signed-int32 arithmetic and returns int64 row
IDs. Table capacity/layout, hashes, FP32 master weights/gradients/RMS state,
BF16 gathered activations, ordering, clipping and optimizer are unchanged.

## Executed existing P2 suite

Both commands exited 0 from this directory:

```
/home/poojith-devan/venv/bin/python -B test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-native-index-repair/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int32
/home/poojith-devan/venv/bin/python -B test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-native-index-repair/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int64
```

`cpu-result-int32.json` and `cpu-result-int64.json` contain source/test
hashes and results. The supplied P2 tests/interpreter were reused, with an
explicit input-dtype option (default int32) and seed42. Both dtypes establish:

- Original K2 helper ASTs exactly equal repaired K2 parent a3181d0d.
- Production hashes at lengths1/2/4 and boundary IDs match the independent
  signed-int32 oracle.
- Six operation fixtures/two accumulation microsteps match FP32-master
  embedding/cast gradients by bytes: historical BF16 collision witness,
  large/small FP32 ordering witnesses, both subnormal signs and cancellation.
- FP32 RMS arithmetic matches three direct updates; four mocked ranks retain
  replicated table routing/one update, and separate clipping/FP32 AVG routing.
- Three tables-OFF steps versus repaired K2 parent and two tables-ON steps
  versus ordinary FP32-master/BF16-gather behavior match losses, every gradient
  and updated parameter by bytes. Actual optimizer/RMS runs in the ON check.
- Initialization, same-shape causality/batch independence, enabled checkpoint
  and old-parent checkpoint roundtrips pass.

The existing cross-shape prefix diagnostic remains
7.62939453125e-6 for both dtypes; this is retained, not reclassified as exact.
Tests use eight layers/width128/context8/batch2 and table capacity32x128.
The actual NKI body is CPU-interpreted with reduced geometry; mocked ownership
collectives, existing test-only prepare, unused parquet import and CPU BF16
load policy are disclosed test shims. No production data/evaluator changed.

Native stable sort/searchsorted, runtime-group DMA, numerical semantics,
full-size memory, compilation and whole-step timing remain untested.
No P2 native/full launch is authorized by this port; conditional use awaits
K2 retry qualification and the applicable P2 gates/sole Operator assignment.
