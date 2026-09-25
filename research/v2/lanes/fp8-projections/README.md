# L1: local Codex FP8 projections

This is an implementation candidate, not a native PASS or performance result.
The only production edit is `workflow/proposals/research-v2/train.py`.
The source starts at canonical commit `2e676a09dd26e1d6c8d6e17f12be2e36953a0e73`,
whose research file is exactly
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
The final candidate is
`dea47910547572c704d8d5cc0e3dd058282fc8d862c913939bfc5e40830713b4`.
No hardware, credentials, remote model service, evaluator, dataset, timing hook,
or submission was touched. Newest human authorization moved L1 to local Codex
after the Bedrock policy refusal; no provider retry was made.

## Numerical contract

`--fp8-proj` defaults OFF. It marks only each block's MLP `c_fc`/`c_proj` and
attention Q/K/V/O projections (including the existing fused-QKV alternative).
Embeddings, language-model head, and small gates retain the base path. Every
parameter remains FP32 and keeps its name and shape. The numerical flag is
saved/restored with the existing checkpoint toggle mechanism; an old checkpoint
without it restores OFF. The trained and frozen forward share the same code.

Forward casts current activations and masters to BF16, computes independent
per-token/per-output-channel `max(abs(row))/240` scales with a `1e-6` floor,
quantizes into legacy E4M3, accumulates FP8 products in FP32, applies the weight
then token scales in FP32, and rounds the result to BF16. Native FP8 operands
use `nisa.nc_matmul(..., perf_mode=double_row)` with the documented
`[128,2,M]` / `[128,2,N]` contraction layout. They are never converted back to
BF16 before this forward GEMM. Weights are prepared from the current master
inside each projection forward, then held in SBUF across its token tiles.
No persistent quantization state or delayed statistics can become stale.

Backward uses BF16 reconstructions of the **actual quantized operands**:
`dX = dY @ W_effective`, `dW = dY.T @ X_effective`. This is an explicit STE
through the scales and rounding. The token scales stay inside dW's token
reduction. Both products have BF16 operands; dW is returned as FP32 for master
gradient accumulation. MLP ReLU-squared keeps its existing BF16 activation
derivative. No loss scaling is used; incompatible scaling flags are rejected.
FP8 ON bypasses the old fused-BF16 MLP adapter so it cannot silently bypass
either projection. OFF still installs the original adapter.

On CPU only, the explicit reference uses `float8_e4m3fn` after clipping to
±240, where its finite encoding agrees with legacy E4M3, and a FP32 reference
GEMM. This validates algebra, not Neuron execution. CUDA is rejected. Native
K must be divisible by 256, N by 128, K <=8192, and the SBUF bound must fit.
Token rows are zero-padded to 256 when needed and trimmed afterwards. Their
scale is independent of companions and padding. Transposed/coalesced projection
layouts are rejected rather than silently taking a partial path.

## Support and attribution

AWS documents Trn2 FP8 double-throughput Tensor Engine execution with FP32
accumulation and a 256-element contraction in its
[Trainium2 NKI architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html#double-fp8-matmul-performance).
The [nc_matmul reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html)
distinguishes legacy E4M3 on NeuronCore-v3 from OCP E4M3FN support on v4.
Documentation alone does not prove the installed competition SDK compiles
this particular program.

The quantizer and byte-safe transpose follow this repository's
`research/scratch/nki_library_row_weight_prepare.py` and
`research/scratch/fp8_mlp_token_rows_nki.py` at
`origin/research/fresh-start-20260918`. The two-byte PSUM stride derives from
AWS NKI Library commit `92d11f63`,
`core/mlp/mlp_cte/basic/mlp_cte_basic_transpose.py`, Apache-2.0,
Copyright Amazon.com, Inc. or its affiliates; see
[license](https://www.apache.org/licenses/LICENSE-2.0).
The source includes an attribution comment. No vendor dependency is required
at runtime: the two new kernels are inline in the production file.

Repository evidence rules out treating direct aten FP8 matmul or `_scaled_mm`
as an efficient path: historical direct FP8 matmul returned NaNs and the
correct `_scaled_mm` measured about 46 seconds/call. Those are historical
environment observations, not an API guarantee. This candidate uses neither.
Commit `71b8b9c2` was a **CPU VJP scale-axis contract only**. Source199's
approximately 9.4% single-rank / 19.1% contended gains measured a saved fused
MLP **forward component**, not this projection kernel, backward, or a full step.

## Local evidence and remaining gates

The laptop tests are deliberately bounded: four ranks, two layers x256,
one x16-token effective microbatch, three optimizer steps, seed 42,
OMP/MKL/OpenBLAS threads all 1, and no tables. Runtime is
`/home/poojith-devan/venv/bin/python` (Torch 2.13.0+cpu).
The test-only PyArrow import stub rejects all data access; G2's existing
synthetic prepare/loader and BF16/clock shims are recorded in each receipt.
Neither stub is part of production.

- `g2-base-c5c-small.json`: unchanged exact-c5c base.
- `g2-off-r3-small.json`, `g2-on-r3-small.json`: final candidate, OFF/ON.
- `off-identity.json`: three-step loss, all parameter byte hashes, and
  ownership-plan comparison against exact c5c.
- `cpu-checks-final.json`: zero/tiny/outlier inputs, explicit BF16 VJP,
  row-scale placement, projection inventory, three finite SGD steps,
  causality, changed-input behavior, batch independence, checkpoint restore,
  and missing-toggle legacy behavior.
- `static-checks.json`: syntax/diff checks, unchanged entire training-loop,
  data-loader, schedule, clip/sync, checkpoint and evaluator-hook ASTs, protected
  prepare hash, and synthetic benchmark-decision tests.

Earlier `g2-*-small.json` files with hashes `dfef42...` or `5bc09...` cover
earlier revisions and are retained as history only. `g2-base-small.json`
accidentally read the canonical file after it changed to `f9b6a9...`; it is
**not** the c5c comparison control. The corrected local c5c control above is
used. The first custom CPU test had a test-only checkpoint dtype mismatch
(BF16 training versus the CPU loader's FP32 default); the test now applies
the same explicit BF16 shim as G2. No production change was needed for it.

These are Implementer self-checks, not an independent Verifier receipt.
Exact 6x1024 four-rank G2, Neuron compile, native numerical checks, full-model
causality/checkpoint checks, and whole-step performance remain outstanding.
The laptop has neither NKI nor torch-neuronx installed; no native simulator
result is claimed. An independent Verifier must review the diff and evidence
before any Operator machine use. `check_native.py` is a supplied, **unrun**
four-rank component check, including compiler/autograd capture and actual
packed bytes, saved operands, gradients, zero/tiny/outlier rows and padding.

Expected benefit is limited to replacing the six projection forward GEMMs
per block with FP8 arithmetic. No BPB or speedup magnitude is promised. Risks
include quantization error, tiny-row flooring, BF16 gradient rounding,
conversion/state-write overhead, SBUF allocation/spills, full-model compiler
boundaries, loss of the old fused MLP backward schedule, and cold compilation.
The extra BF16 effective operands needed for backward can increase live HBM.
At the current pack32 configuration there is one microbatch per rank per step;
larger accumulation counts repeat weight preparation and may erase the gain.

## Operator whole-step benchmark

This package has **no launcher**. After independent review and normal exact-
source, proposal, unique-label, idle, disk, STOP, one-job-per-host and full-chip
preflight, the Operator can run these separate diagnostics. Preserve and
retrieve every log, terminal status and checkpoint. Failed gates are failures.

Native component command, under the host's checked containment machinery:

```sh
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 research/v2/lanes/fp8-projections/check_native.py --train-py workflow/proposals/research-v2/train.py --out-dir <unique-native-receipts>
```

For timing, use four new labels in chronological **OFF, ON, ON, OFF** order
on one host, same day, same exact candidate bytes, same warm cache/compiler
flags and approved launch environment. Every run starts from seed 42.
Production argv (ON adds only `--fp8-proj`; output and step-log paths are unique):

```sh
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 workflow/proposals/research-v2/train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 --num-steps 100 --max-train-seconds 1800 --out-dir <unique-output>
```

Before each launch start the external observer below and set the unchanged
organizer `NEURON_COMPETITION_R1_STEP_LOG` to that exact fresh absent step-log
path in the launch environment. Both receipt and step-log parents must exist.

```sh
python research/v2/lanes/fp8-projections/whole_step_benchmark.py observe --train-py workflow/proposals/research-v2/train.py --step-log <unique-steps> --out <unique-timing.json> --label <unique-label> --arm off
```

Use `--arm on` for ON. The observer measures the 80 complete step intervals
after 20 warmup steps without patching the train loop or timing/scoring code.
The intervals include backward, synchronization, optimizer, preparation and
cache refresh, control reductions, and subsequent data acquisition. It rejects
coalesced/missing/duplicate boundaries. Compile time and checkpoint completion
are separate recorded gates, not part of steady-state timing.

```sh
python research/v2/lanes/fp8-projections/whole_step_benchmark.py compare off1.json on1.json on2.json off2.json
```

Timing passes only if **both** matched pairs improve whole-step time by
>=3%, conservatively accounting for observer polling. The script exits nonzero
on failure. A PASS is conditional on the Operator/Verifier separately checking
the exact argv/environment, finite losses/gradients, all four clean exits,
checkpoint, and native/full-model correctness receipts. It is not a BPB win
and does not authorize a 30-minute quality run by itself. If the timing gate
fails, retain the failure and profile before any further kernel revision.
