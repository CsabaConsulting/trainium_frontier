# Independent gate-fixed C1 implementation verification

Recorded 2026-09-27 08:46:41 UTC. **PASS_SOURCE_CPU_ONLY**, no source/CPU blocker, for source `6312b5426bc4c5b8cf791d6b763ad7c77b9e1587fbce92d69354089bbd41e03e` against the reviewed gate-fix parent **500f** (`500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`).

This receipt supersedes no native evidence: it verifies the corrected composition and gate compliance for packet preparation, not a host/command, native qualification, launch authorization, adoption or a quality gain. No extra native experiment is required by this receipt. Normal approved proposal/Operator checks still apply.

**Causal control is a fresh 500f run.** The old c239-based C1 draft is archival science context only. A historical-c239 difference combines gate routing, MuonEq and requested timing instrumentation; it must not be attributed solely to C1 or MuonEq.

## Exact composition and gate compliance

- Independently removed the exact five gate-routing lines from the new source and recovered old C1 source `051b06e4dc037bff7486f43912f0f2e696c9dc71a7b41ada98e8e03265aa0f4b` byte-for-byte. Independently removed only C1's helper/two call sites, canonical-storage guard and sampled timing block from the new source and recovered **500f byte-for-byte**. Old/new science `EDITS.json` hashes also agree (`9c1445c6bab10488dc5b0e21d4290564672649fde40d97002c459a5e4b99f814`), but the byte proofs do not depend on trusting that manifest.
- Actual production-parameter meta configuration `L=8,D=1024,H=8,V=8192,T=1024`: **65 unique tensors, 113,256,724 elements**, complete exactly-once optimizer coverage. Every group policy and the actual four-owner assignment/packing plan are identical to 500f.
- Every matrix gate is in AdamW. The sole current matrix gate, `transformer.h.7.attn.ve_gate.weight` `[8,32]`, has LR 0.021, betas `(0.9,0.95)`, epsilon `1e-10`, WD 0. The four scalar U-Net gates retain AdamW LR 0.2. All actual Muon groups have `transposed=False`; unsupported transposed storage raises before an update.
- Stored MLP matrices remain canonical `[out,in]`: `c_fc=[4096,1024]`, `c_proj=[1024,4096]`. Row equilibration is across the last dimension, not columns or an intermediate backward transpose. Model/eval/checkpoint implementations, AdamW implementation, schedules, PE coefficients, NorMuon statistics, data and existing clock/stop logic are unchanged by the exact diff. The only timing change is the expressly requested sampled owned-step instrumentation.

## Independent CPU execution

Script `research/v2/research_cycles/20260927-r13-c1-gatefixed-reference-verification.py`, SHA `13548fbcc1cdb360e71b1fc80eb4b8e68a64ffb6c90e9cd136a0e686b0759cde`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r13-c1-gatefixed-reference-verification.py
```

First invocation exited 0, reported body time 0.7408 seconds, local Torch `2.13.0+cpu`. It executes hash-bound extracted definitions without importing the training runtime or `prepare.py`. Author tests were not executed or counted as independent proof.

- **72 row-oracle cases:** scalar FP64 norm reference with explicit output-dtype scaling/multiplication, covering square/tall/wide, stacked matrices, zeros, zero rows, tiny rows, unequal row magnitude, FP32 and BF16. BF16 outputs match exactly in these cases. FP32 maximum absolute error is `1.52587890625e-05` on a large-magnitude case, within the tested `rtol=2e-6, atol=2e-7` bound. All outputs remain finite and all-zero input remains exactly zero. A nonsymmetric example proves rows, not columns, are equalized.
- **27 tiny optimizer group updates:** square/tall/wide two-parameter groups, three steps, FP32-master/FP32-compute, FP32-master/BF16-compute and BF16-master/BF16-compute. This exercises both actual FP32-master pure-math path and eager fallback. The independent comparison is **500f inherited math with the independent scalar row oracle inserted at the specified point**, not unmodified c239 or unmodified 500f. Maximum parameter error `1.4901161193847656e-07`; tolerance `rtol=2e-3, atol=2e-5`; all updates finite. Transposed-storage refusal passes.
- **202 actual-timer mock cases:** steps 0–100 for ranks 0 and 3. Every call executes exactly one owned optimizer step. Only one-indexed steps 50 and 100 execute `sync → start clock → step → sync → end clock → flushed log`, and the scope label explicitly includes all-gather. This is an event-order check, not a native timing measurement.

### Numerical and timing caveats

The target is Frobenius norm divided by square-root row count (the RMS of row L2 norms), with row denominator clamped at `1e-6` and scale cast before multiplication. Exact equal row norms/Frobenius preservation are **not universal invariants**: zero/clamped tiny rows and BF16 rounding matter. For `[[3,4],[0,0]]`, the zero row stays zero and the resulting/input Frobenius ratio is `0.7071067094802856`. This matches the specified helper; it is not a blocker or a claim of equivalence to the original optimizer.

The timing sample includes local owned AdamW/Muon work plus all-gather, not pure Muon compute. Synchronization/logging overhead is charged under the unchanged budget and may affect time-based schedules. Sparse samples do not establish all-step overhead or a speed gain.

## Primary-source scope and remaining gates

Independently reopened [nanochat's primary optimizer source](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py) on this review: its pre-orthogonalization row-scaling formula supports the conceptual helper. That source separately applies Muon+ post-orthogonalization renormalization; this candidate does **not** import that distinct operation. The live master source is provenance, not a pinned SDK guarantee or local performance evidence.

Corrected spec SHA `13f2f7e0d7bd2839a88015b62cbb2a6ea18df9332507fdebe3712d69b2237d0a` was reread: its intro now distinguishes matrix gates from scalar gates, as requested by the verifier. Author CPU receipt SHA `ee9ff2b681bb6bd37ea35d9962b5dbbb575ff18f3d31f7dd8331a398e9df28cb`; proposal stub SHA `a4eac94d56bcd7ef195c0769694ce28a4e4fd4d7ca23e37cc564723ae5c0ba12`. Source SHA was rechecked after testing.

No native compilation, real collective, full-model optimizer update, dataset, host or evaluator was exercised. Model/eval preservation is established by the exact source union; the author's model/causality/load tests are not mislabeled as this verifier's execution. Local CPU Torch differs from previously retrieved host Torch `2.11.0+cpu`. Native behavior and local BPB/throughput remain unknown. No candidate, host, queue, ledger, state or launcher was edited by the verifier.
