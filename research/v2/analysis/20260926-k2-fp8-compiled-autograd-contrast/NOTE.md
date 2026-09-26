# K2 versus FP8: compiled-autograd contrast, not a common-cause diagnosis

Analyst-only, 26 September 2026. Both original failures remain failed/HOLD. No candidate edit, dtype restriction, tolerance change, tensor test, host access or retry. I previously implemented these candidates; this note is source/evidence analysis, not their independent verification.

## What is shared, and what is not

The shared observed risk is **an enclosing `torch.compile(..., backend='neuron', fullgraph=True, dynamic=False)` custom-autograd path producing present, finite but all-zero gradients when eager controls produce nonzero gradients**. Both archives used Torch 2.11 native Neuron, four ranks/LNC2. This supports investigating the differentiation/input-consumption integration before altering working arithmetic. It does **not** establish the same SDK defect, kernel defect or repair.

| Boundary | FP8 exact `921271d7` | K2 symmetric exact `2c8295a4` |
| --- | --- | --- |
| Custom forward | NKI weight preparation and projection, via `torch_neuronx.wrap_nki`; returns BF16 projection | Ordinary `F.embedding` followed by requested output cast; no NKI forward |
| Saved context | Two NKI-produced BF16 effective operands | Four supplied INT32 routing metadata tensors |
| Custom backward | Ordinary PyTorch BF16 matmuls, cast dW to FP32; no NKI backward kernel | Direct `nki.jit` symmetric `[2]` consumer on incoming gradient; FP32 result |
| Observed failure | BF16 upstream: both compiled dX/dW zero; bound and enclosing paths both fail | FP32-output/upstream wrapper: compiled dW zero on `unique`; BF16 wrapper has nonzero exact gradients |
| Working controls | Eager apply and manual native BF16 VJP within original 1% RMS | Direct/repeated consumer, dense CPU oracle and eager custom backward byte-exact on the failed FP32 fixture |

Exact source locations: FP8 `50a90137:workflow/proposals/research-v2/train.py:1458–1473,1575–1619,1650–1655`; K2 `research/v2/lanes/k2-symmetric-consumer/check_native.py:93–109,118–161` and `consumer_nki.py:16–75`. The NKI call's location and wrapper API are different. FP8's exposed saved tensors were nonzero; their consumed compiled input mapping remained unknown. K2 did not inspect consumed tangent/metadata slots or compiled backward IR. Neither raw receipt identifies the failing lowering stage.

## Raw evidence and the zero-fixture trap

K2 archive `7f0fbbde`, `research/v2/operator/20260926-trn21-k2-symmetric-native/completed/results-bundle.json`, SHA `16dc0f7d84eed33eaaa821cb6afd21d00fc5b6d60b4ad1b9f93e55f3e07b9c7b`: all 31 files rehashed. Every rank completed 13/20 cases then failed `unique/fp32`, exclusively at the compiled backward comparison: actual nonzero count 0, expected 32,709, maximum error 8.0. Direct and eager FP32 gradients exactly match reference SHA `dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671`. The BF16 `unique` compiled gradient matches that same **FP32 output** SHA, and other BF16 duplicate/nonzero cases pass. Thus this is not a general inability to return FP32 parameter gradients. Both dtype fixtures' ordinary values are exactly representable multiples of 1/32; the failed unique comparison is not BF16 quantization drift.

The first two FP32 cases (`cross_tile_large`, `cross_tile_unit`) have **expected all-zero outputs**. Their PASS cannot establish that FP32 compiled backward ever consumed its nonzero upstream correctly. `unique` is the first FP32 case exposing a nonzero expected result. A frozen-zero tangent/metadata hypothesis is possible but unproved; BF16 also started on zero-output cases and subsequently worked. CPU weights differ across the dtype loops because RNG continues; they are irrelevant to this mathematical embedding VJP, but this was not a fully reset matched experiment. Remaining seven FP32 cases were never executed; no full K2 native gate passed.

FP8 archive `c39844eb`, four `diagnostic-rank*.json` receipts under `research/v2/operator/20260925-trn21-l1-vjpdiag-r1/completed/receipts/diagnostic/`, independently rehashed against Git and the retrieval manifest: all compiled variants have zero dX/dW and relative RMS 1.0; eager/manual have 65,536 nonzeros each and RMS errors approximately .00409/.00298. Original checker failure `c1ac162c` is preserved. Replaying the existing independent archive auditor passed its 110 stdlib checks; this was not a new native test.

## Production embedding dtype: name the boundary

For the ordinary WTE/VE route in exact `6ef7` and `36a8`, the logical source contract is FP32 table/output **then an external BF16 cast**. `nn.Embedding` uses ordinary default FP32 parameters, and `model.to(device)` changes device only; these files contain no global `set_default_dtype`/model BF16 cast. `compute_dtype_for(neuron)` returns BF16. Source locations:

- `a4aa4351` / `6ef7d090`: lines 520–525, 1912–1914, 2230/2239, 3830, 3843–3850.
- `7419927c` / `36a86274`: lines 539–544, 1948–1950, 2282/2294, 3898, 3913–3920.
- The training installer changes MLP/CE/optimizer behavior, not these embedding calls (`36a8:4930–4962`).

Under the normal fresh-process FP32 default, gradient at the **ordinary embedding output / dense embedding-backward boundary** is semantically FP32: autograd reverses the external BF16 cast before embedding backward. Downstream BF16 activations do not make that boundary BF16. This is the relevant contract for replacing only dense embedding backward, so dropping FP32 support is unjustified.

A different future custom function that encloses **both** embedding and its BF16 cast, as the K2 BF16 checker does, receives a BF16 tangent at that custom-function boundary and returns FP32 dW. That is a different integration boundary, not the existing production source, and is not implemented/adopted by this consumer. Source semantics alone do not observe the installed backend's actual lowered tangent slot, hidden conversions or physical storage dtype; those remain unknown. Nor does the tiny BF16 checker prove that a full production graph would preserve its behavior.

## At most two discriminating next checks — proposals only

1. **Source-faithful tiny CPU boundary check**, reusing the Verifier's independently reconstructed wrapper work: compare the actual external-cast embedding expression with the fused embedding+cast diagnostic boundary, for identical FP32 masters/IDs and zero then two distinct nonzero tangents, BF16 and FP32. Record logical output/tangent/dW dtypes and bytes under eager and `aot_eager`, retaining repeated calls and metadata changes. This can expose a Python/custom-backward contract error or show why the two boundary dtypes differ. It cannot validate Neuron lowering. The Verifier already reports generic-wrapper CPU success; do not duplicate it or present it as native proof. Only the source-faithful boundary contrast is incremental.

2. **One separately reviewed tiny native localization matrix**, unchanged kernel/strict oracle at V256/C128/T256: retain the exact failing compiled custom wrapper, contrast a compiled standalone consumer and an ordinary compiled embedding control, both dtypes, starting with `unique` nonzero rather than the cancellation fixtures; replay with a changed nonzero tangent and changed valid routing metadata. Preserve existing direct/eager controls and all original failures. Capture actual backward graph/input metadata if the supported capture route permits, otherwise label those slots unknown. If standalone compiled consumer fails while eager succeeds, failure is not specific to custom-autograd linkage; if standalone and ordinary controls work but the custom path fails, linkage/capture is more narrowly implicated. Starting nonzero tests the zero-first specialization hypothesis. None proves equivalence to the FP8 cause or completes the original 20-case gate; no retry or implementation is authorized here.

`extract_evidence.py` performs 58 explicit stdlib hash/scope checks in normal and optimized Python; outputs are byte-identical in `EVIDENCE.json`. Source hashes, wrapper/support pins and all-rank observations are retained. No performance or quality inference follows from this note.
