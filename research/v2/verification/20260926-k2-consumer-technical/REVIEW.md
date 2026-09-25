# K2 supplied-metadata consumer: bounded technical audit

Candidate: `077481d0476cebfd79ddc751f8f63e61c1fbf82f`. Verdict: **PASS_STDLIB_ARITHMETIC_ONLY; no concrete blocking source/checker defect found in this bounded review.** This reviewer has historical context and is not the separately required fresh reviewer. Native legality/exactness, launch, production integration, routing, speedup and adoption remain unaccepted.

Exact support hashes:

- `consumer_nki.py`: `57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843`;
- `reference.py`: `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`;
- `check_native.py`: `17e2d2ad417407c12327e1446aed857f273d445946839357102e93dea46d4435`.

The commit changes only the new lane packet. No production `train.py`, optimizer, evaluator, data, batch/LR default or launch path changes. Factory and checker defaults refuse before NKI/Torch import. The enabled checker requires exact kernel/reference pins and four-rank/LNC2 environment; it is a workload, **not** the missing sole-Operator guarded host driver.

## Own executed evidence

`review_reference.py` is an independent stdlib test, not a rerun of the implementer's test. Both normal and `-O` runs completed successfully, each with **5,211 explicit checks, 20 cases, zero byte differences**. Normal/optimized result SHA256 is identical: `82eb402430838cb795a222e04df80e9a81302b566f5d75ca95ec75a7ba5f2e06`. The actual candidate `reference.consumer` ran against independently generated stable metadata and a token-major `array('f')` dense oracle with one FP32 store per occurrence. No Torch, NKI, tensor jobs, host calls or model-sized allocation occurred.

Cases cover both BF16-rounded and FP32 inputs: two cross-tile cancellation cases, all occurrences in each separate row group, reversed unique IDs, boundary rows 0/127/128/255, random duplicates, signed zeros, tiny subnormals, and logical empty input. Exact per-row visits, all 256 output stores, inactive-row initialization and active occurrence counts were checked. Four malformed metadata variants were rejected. OFF checker behavior was exercised under normal Python and `-O`.

Retained negative controls distinguish sequential accumulation from grouped sums: `(2**24, 1, -2**24)` across tokens 0/128/129 gives **0 versus 1**; `(1, 2**-24, -1)` gives **0 versus 5.960464477539063e-08**. This fixes neither the earlier rejected production K2 patch nor any native arithmetic; it only tests the proposed ordered recurrence. These test results were not supplied to the fresh reviewer.

Reproduce from this review worktree:

```text
python3 -B research/v2/verification/20260926-k2-consumer-technical/review_reference.py /home/poojith-devan/Desktop/Trainium-k2-resident-consumer-20260926/research/v2/lanes/k2-resident-consumer
python3 -O -B research/v2/verification/20260926-k2-consumer-technical/review_reference.py /home/poojith-devan/Desktop/Trainium-k2-resident-consumer-20260926/research/v2/lanes/k2-resident-consumer
```

## Static native/checker assessment

The consumer keeps one FP32 accumulator per row/channel, loads only the next occurrence in each active row, adds once per occurrence, selects the old accumulator for inactive lanes, and stores each dense row once. Inactive addresses are initialized to the out-of-bounds sentinel before both gathers, avoiding accidental stale-address reuse. Supplied stable metadata is validated by the fixture before transfer. This says nothing about production metadata cost or generation, which is absent.

The documented vector-indirect DMA form uses an HBM source, `(P,1)` SBUF indices and `indirect_dim=0`; software DGE supports gather with out-of-range transfers skipped. The chosen source access patterns match that documented structure. [AWS 2.31 DMA API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html). Register-fed dynamic loops are documented, but allocation/control-flow lowering depends on frontend/version. [AWS 2.32 dynamic-loop documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/deep-dives/nki-dynamic-loops.html).

These documents are contextual API evidence, **not authentication of installed NKI 0.5**. Operator confirmed that no already-archived installed API inventory covers these primitives. No new host inventory was requested/performed. Dynamic-loop lowering, signed int indices, SW-DGE sentinel semantics, BF16-to-FP32 DMA conversion, subnormal/rounding behavior, selection/broadcast and actual SBUF allocation remain precisely what the future native falsifier must test. Logical named-tile sizes are not compiler peak/liveness proof.

The checker compares direct and repeated results, eager and compiled enclosing autograd, and a dense CPU gradient oracle with zero tolerance. A missing gradient is failure. It records per-rank partial failure and does not call completion a production/source/whole-step acceptance. A prospective binding must still require all prescribed cases/checks on all four ranks, nonzero failure exits or missing receipts fail closed, full process/compiler/device quiescence and the common lifetime lock with both idle checks. None is created by this review.

Minor interpretation detail: optional consumer timing is inside the dtype loop. Thus BF16 timing can exist before later FP32 checks fail. The final diagnostic must remain REFUSE in that event; those partial timings cannot be described as a successfully qualified consumer or whole-step speedup. No threshold or code change is requested for this nonblocking reporting distinction.

## Fresh gate in progress

A separate new packet-only Astra session started `2026-09-25T20:58:35.867000+00:00`, thread `01a0da5c-e47c-76c2-997d-ab14ed900de1`, requested model `gpt-6-astra`, input manifest `5e0dde0832d3031b56ea4143a705808e1bd62326674799837b3444e1af440561`. It receives only three complete new-file diffs, a neutral lane specification and two raw recorded stdlib receipts. No implementer test, notes, branch, previous review, this test or conclusions are mounted. This is launch provenance only; its verdict/own test remain pending until separately archived. Native remains unexecuted.
