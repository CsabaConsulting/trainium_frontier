# Independent gate-fixed VE reuse / MLP3 review

Recorded 2026-09-27 09:08:41 UTC. **Both PASS_SOURCE_CPU_ONLY**, no source/CPU blocker. Parent and required fresh causal control: `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`.

| Arm | Exact new source SHA256 |
|---|---|
| VE reuse 4/6/7 | `7daaecc292c3ac0ca11fcbec90e242b9346abf2571fd99338a0e6f2a25d3b261` |
| MLP ratio 3 | `64eb5e17f897db38ba823fd47ae25c6f0f5eb0ec4d3b4e09625e5afa6da7b60c` |

Independent byte reversal proves each file is exactly its original science delta composed with 500f. Removing the gate fix recovers the archived science source; removing science recovers 500f. All matrix gates are AdamW (LR 0.021, betas 0.9/0.95, epsilon 1e-10, WD 0). Scalar U-Net gates retain LR 0.2. Historical c239 differences combine gate routing and architecture and are not single-arm causal comparisons.

## Executed evidence

One bounded CPU process, first invocation exit 0, body time 1.2214 seconds, Torch `2.13.0+cpu`. The verifier's new independent script is `research/v2/research_cycles/20260927-r12-priority-gatefixed-reference-verification.py`, SHA `76657a5d5004c2bbddbba50ee0714205a2380574d3b82270e47a62758bed984f`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r12-priority-gatefixed-reference-verification.py
```

Actual `build_config` constructs production meta shapes (`8 layers, width1024, heads8, vocab8192, context1024`) and tiny real CPU models (`width128, heads1, vocab128, sequence7`). This avoids accidentally using `GPTConfig`'s unchanged default MLP ratio 4 for the MLP3 arm. The script extracts hash-bound real definitions without importing the training runtime or `prepare.py`. Author tests were not replayed or counted as independent evidence.

**VE reuse:** 67 tensors / 113,257,236 elements, exactly two additional `[8,32]` gates, one table `7`. Gates at 4/6/7 are independent, zero-initialized and AdamW-routed; their initial `2*sigmoid` multiplier is 1. Hooks verify exactly one embedding lookup and the **same BF16 tensor object** at all three layers in both training and targets-none eval, with no detach. Its actual embedding-weight gradient equals an independently computed scatter-add of the shared BF16 tensor's gradient, including repeated token indices. All 67 parameter gradients are finite and nonzero in the synthetic connectivity stress case. A separately written per-use-lookup forward oracle matches bitwise. Prefix causality, length-one finite output, complete optimizer/four-owner coverage and strict checkpoint/eval parity pass.

**MLP3:** only `MLP_RATIO=4→3` differs from 500f. Actual production meta shapes are `[3072,1024]`/`[1024,3072]` in all eight layers; 65 tensors / 96,479,508 elements. The existing gate7 stays in AdamW. All 65 tiny-model gradients are finite and nonzero; eight explicit eager MLP equations match bitwise. BF16 CE inputs, prefix causality, length-one output, complete optimizer/four-owner coverage and strict checkpoint/eval parity pass.

For both arms the connectivity stress deliberately makes MLP down-projections and U-Net skip gates nonzero; it is not a training step. Actual `load_for_eval` executes in memory with `weights_only=True`, strict load, eval/no-grad mode and the stored model configuration. Its CPU FP32 dtype policy is compared against an independent same-dtype model, including train/eval parity; this is **not** a cross-FP32/BF16 equality claim.

## Scope and caveats

Shared BF16 accumulation can round differently from three separate lookups; only forward equivalence and the correct shared-gradient scatter are asserted. Adding VE gates changes initialization RNG consumption, so common-weight identity is not claimed. No automatic port to six layers is implied by fixed indices 4/6/7. MLP3's constructor/config/shape path is verified; no native or fused-MLP qualification is claimed by the tiny eager check.

The MLP3 spec's copied reference to added VE4/6 gates was flagged and corrected by its author without a source change; corrected spec SHA `6ace3844682a6d2775716dcf8439260fa52983ea5f70b6d91034d27803184942`. The companion JSON binds all final specs, author receipt hashes and proposal stubs.

No host, native compilation, real collective, dataset/evaluator or full-model optimizer update was used. CPU synthetic losses are logic checks, not BPB. Local Torch differs from the previously retrieved host Torch; speed/quality remain unknown. No candidate or operational file was edited by the verifier. Normal completed proposal/Operator/artifact gates remain; this receipt adds no extra native experiment and is not launch/adoption authority.
