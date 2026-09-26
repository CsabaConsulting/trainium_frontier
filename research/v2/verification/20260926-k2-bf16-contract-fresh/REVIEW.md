# Independent CPU specification review

Verdict: ACCEPT mathematical narrowing as a prospective research contract; HOLD implementation/native/launch qualification. This is not approval of any patch. The unrestricted FP32 failure remains FAILED, and its evidence must remain visible. No candidate implementation or implementer notes were inspected.

Inspected only the requested SPEC.md and EXPERIMENT_PROPOSAL.json, exact train.py body at 7419927c7417b5666a984ec9c2b849feadabaaa9, and adoption JSON at cited research snapshot e018498e4bdd91ce5c93a184c3e7602a42228311. Independently verified source SHA256 36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb. No source imports, full models, SDK, host, network, dataset, evaluator access, or repository edits.

Frozen source evidence: compute_dtype_for returns BF16 for Neuron at lines 539-544; main sets COMPUTE_DTYPE at 3898. Ordinary embeddings constructed at 1948/1950 use default floating dtype, and no set_default_dtype call exists in the source. Main moves model by device without a dtype conversion at 3920. WTE/VE lookup outputs are immediately cast at 2282/2294. has_ve selects parity matching layer 7 and requires has_attention; adopted depth8 and attention0,2,4,6,7 leave only VE7, plus WTE. The source's parser enables partial attention under unet_8x1024 at 3666-3668 and installs the architecture arguments at 3675/3803. Pack32/context1024 reshape is at 3969. These establish source intent, not a compiled runtime boundary or actual native tangent dtype.

Independent reference: review.py uses integer multiples of 2^-149, exact integer addition, and explicit binary32 round-to-nearest-even; no FP64 accumulation or candidate imports. Fourteen one-element-tensor fixtures compare every recurrence intermediate against CPU FP32. All 65,280 finite BF16 encodings were independently decoded and checked against the 2^-133 lattice. Normal and optimized Python runs succeeded with byte-identical receipts.

Important receipts:
- BF16 0001/8001 widen to FP32 00010000/80010000.
- 00810000 + 80800000 gives 00010000; the negative counterpart gives 80010000. Reachable FP32 subnormals therefore remain required.
- [2^24,1,-2^24] yields 00000000 in token order, versus 3f800000 under reassociation. [1,2^-24,-1] yields 00000000 versus 33800000. Domain narrowing cannot excuse tile partial sums.
- Positive-zero initialization plus negative-zero tangent yields positive zero. The tangent's negative-zero raw bit survives widening; preserving zero signs means matching operation semantics, not copying tangent signs to output. Negative-zero plus negative-zero was separately checked in the scalar oracle.
- Two maximum finite BF16 inputs overflow to positive infinity; their negative counterparts overflow to negative infinity. A subsequent opposite finite input does not undo infinity. All such results remain failed finite-training gates.
- Infinity cancellation produced CPU ffc00000 versus the reference canonical diagnostic 7fc00000; NaN input 7fc1 canonicalized during the CPU cast. These raw differences are recorded, not equality passes. The first script attempt rejected that NaN cast difference; the final script limits exact widening assertions to finite values/infinities and records NaN bits. No finite mismatch was relaxed.

Spec acceptance conditions and remaining blockers:
1. A BF16-output registration must encompass the actual gather/cast boundary. Wrapping a generic FP32 embedding and casting arbitrary tangents down is not this contract. Native trace/replay must establish actual use and no compiler removal, before numerical qualification.
2. Ordinary native accumulation must itself match the mandated ordered recurrence throughout the reachable domain. CPU scalar proof does not prove native ordinary parity or absence of FTZ.
3. Preserve per-invocation recurrence and existing microstep gradient accumulation boundaries. Wording about order across microsteps must not be interpreted as concatenating all token contributions into one persistent accumulator: that changes existing FP32 rounding boundaries. A future patch must make this explicit.
4. Metadata must snapshot the actual per-call device IDs, remain immutable until both WTE/VE7 backwards finish, and be invalidated on loader reuse/replay. The frozen loader reuses device buffers; retaining a mutable alias or caching by tensor identity is insufficient. The spec states immutable metadata but an implementation must prove it.
5. No NaN equality gate is necessary or acceptable for training. Diagnostic policy can compare exceptional classification/sign where defined and report payload/sign raw differences, while every nonfinite output remains FAIL_NONFINITE. If a later packet asks for bit-identical NaN payloads, it needs an explicit separate policy and review.
6. Default-OFF behavior, option guards, IDs/metadata validation, real-shape producer and consumer, native code legality/liveness/skew performance, and useful complete-component timing remain unreviewed. Fresh diff/spec/receipt review is required for any future patch. No launch is authorized by this report.

Commands:
`/home/poojith-devan/venv/bin/python -B /home/poojith-devan/k2-bf16-independent-review-QGZRIJ/review.py`
`/home/poojith-devan/venv/bin/python -O -B /home/poojith-devan/k2-bf16-independent-review-QGZRIJ/review.py`

Environment: Python 3.14.4, torch 2.13.0+cpu. Largest tensor: one element.
Script SHA256: e4c0044541565aa527e6648673c5d807897a7e1128b2b1a41406b55871f25d76.
Both receipt SHA256: c96f9a79700bba7e8f32accf4681dd50caf6d61b2e9b28e93242739e19b4e5f5.
