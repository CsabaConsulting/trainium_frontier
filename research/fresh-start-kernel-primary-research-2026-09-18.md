# Primary-source kernel research audit — 2026-09-18

The external Tavily Pro report is complete. It adds useful source pointers, but
its implementation recommendations require correction for the installed native
PyTorch/NKI0.5 stack. It does not establish a documented fix for our gather
failure or evidence about any leaderboard team's implementation. The original
JSON response, including citations and request identity, is preserved under
`out/fresh-start-research/fs-w31-primary-kernel-research-r2.response.json`; the
tracked audit receipt records its hash. This document is the reviewed synthesis.

## Findings that survive source and device checks

`nc_n_gather` uses GpSIMD on SBUF, with uint32 indices and matching data/output
dtypes. Its documented lowering groups512 **elements**, not bytes. The public
contract describes larger outputs through multiple groups;512 is not a documented
maximum. Our explicit512-output tiling fixes the tested full computation, while
larger calls have allocation/instrumentation-dependent failures. This is evidence
about this installed stack, not proof of a hardware limit or a named SDK bug.
[Public API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html).

Legacy FP8 double-row matmul is explicitly supported on NeuronCore-v3. It packs
the contracting axis as `[128,2,...]`; NC-v3 has a512-wide moving/output limit,
FP32 accumulation, and legacy E4M3/E5M2 formats. OCP E4M3FN and wider limits must
not be transplanted from NC-v4. The report's absence of a complete production
training example does not imply absence of the hardware feature. Our existing
installed-stack FP8 probes already compile it and quantify accuracy/layout costs.
[Matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html).

Contiguous transfers and explicit on-chip transposes are useful directions,
confirmed here by measurements rather than generic bandwidth advice. Our fused
MLP's initial785ms/then554ms forward regressions became1.69ms after fixing DMA
layout/cast combinations. That still misses native1.35ms. Its actual-shape
profile records4,096 DMA_DIRECT2D instructions versus512 native, and~1.34ms
GpSIMD activity per physical core versus~0.181ms. The revised investigation is
about transfer count and saved-activation layout, with backward included.
[DMA guidance](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/nki-dma-overview.html).

## Recommendations rejected or qualified

- The report recommends old `nki.compiler.sbuf.mod_alloc` examples and even
  `nl.load_transpose2d`. Those are not assumed portable to this installation.
  The kernels use the inspected installed `nl.ndarray(..., address=...)` and
  explicit ISA APIs. The installed allocator passes byte addresses; that was
  checked in its actual source.
- It conflates hardware DGE restrictions with indirect DMA generally. Software
  DGE indirect gathers are already tested successfully on these hosts. Hardware
  DGE and software DGE must be distinguished; blanket avoidance is unjustified.
- It suggests MXFP8 library kernels without establishing an NC-v3/NKI0.5 training
  path. That is not an actionable replacement for legacy FP8 here. No SDK upgrade
  or hardware-generation assumption follows from the report.
- It proposes cross-partition out-of-range indices as an experiment. Our failure
  already occurs with valid in-range indices and correct shapes, so invalid-input
  stress would not discriminate the active hypotheses. Do not run it just to
  populate a generic checklist.
- A simulator gather-transpose bugfix is not evidence that our native hardware
  failure has the same cause. Neither the report nor the checked public gather
  API identifies that connection.
- General source claims that SBUF gather is faster than indirect DMA do not settle
  this workload. The tested scan, including layout, gather, metadata and output,
  is still slower than native. Compare complete implementations empirically.

## Experiments justified by the combined evidence

1. Keep512-output gathers as the currently validated workaround. Continue using
   uninstrumented compiled checks because debug stores themselves change failures.
2. For the MLP, consume the saved HxT activation through backward, instead of
   converting it immediately back to TxH. Check all three gradients and full
   forward/backward cost. Any gain must survive the whole-model phase gate.
3. Revisit the correct indirect-DMA scan implementation after moving dtype casts
   into SBUF and replacing its strided layout transfers. This is a different
   transfer strategy from the repaired GpSIMD scan, not another seed trial.
4. A fused FP8 training kernel remains a longer-term candidate after transfer
   overhead is controlled. Quantization/scaling and backward must be charged,
   overflow measured, and full-model finite/causal/quality gates passed. Existing
   component negatives only reject their tested layouts; they do not rule out
   all FP8 pipelines.

For run identities, failures, exact timings and health receipts, use the
[measured-results ledger](fresh-start-results-2026-09-18.md). The broader model,
optimizer, corpus-exposure and scoring analysis remains in the
[deep research report](fresh-start-deep-research-2026-09-18.md).


## Device follow-up, 15:19 UTC

The two primary-source-informed layout directions now have actual integration
results. DMA/scan embedding is 1.42% faster in a nearby whole-step comparison,
short of promotion. HxT MLP backward is faster in isolation but slower in the
whole model. This reinforces the need to test the surrounding graph and upstream
gradient consumers. New fused-transfer and input-gradient-layout controls are
justified; a 30-minute quality win has not been established for either component.
