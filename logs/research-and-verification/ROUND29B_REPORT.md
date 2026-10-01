# Round29b — installed stack, own-kernel DMA and build gates

2026-09-28. Read-only host inventory at 02:40:16 / 02:40:18 IST;
source-only transfer audit. No model execution on the laptop, no device
capture, SDK change, running-job change or fused-backward implementation.

## 1. Actual versions and frontend

Both trn21 and trn22, exact active native venv:

| Component | Actual version |
| --- | --- |
| neuronx-cc | `2.26.6360.0+6f180f47` |
| NKI | `0.5.0+28631259367.ga768afa6` |
| torch-neuronx | `2.11.3.0.0+unknown.dev` |
| torch | `2.11.0+cpu` |
| system aws-neuronx-tools | `2.32.28.0-526c2b7f6` |
| system runtime / collectives | `2.34.10.0-ac18d186d` / `2.34.10.0-74eaafac6` |

Evidence: [trn21 receipt](operator/20260928-round29b-readonly-sdk/trn21-receipt.json),
[trn22 receipt](operator/20260928-round29b-readonly-sdk/trn22-receipt.json).
The probe imported only Python distribution metadata, not torch or NKI;
`neuronx-cc --version` and package inventory had no device calls.

The relevant **training compiler/frontend is from the SDK 2.31 generation**:
AWS maps NKI 0.5 to SDK 2.31 and NKI 0.6 to SDK 2.32. This is a mixed
component installation, not proof that every installed package is SDK 2.31.
The newer system tools/Explorer cannot satisfy a training-compiler SDK>=2.32
gate. **Gate NOT MET; no upgrade.**
[AWS NKI release mapping](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/release-notes/components/nki.html#neuron-kernel-interface-nki-0-5-0-neuron-2-31-0-release),
[AWS compiler release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/release-notes/components/compiler.html).

CLEAN `881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749`
uses **standalone `import nki`, `nki.language`, `nki.isa`**, lines175–177,
not the legacy `neuronxcc.nki` frontend. RoPE+norm uses jit calls inside
an autograd Function; CE and convolution use `torch_neuronx.wrap_nki`
inside their Functions. These are observations, not a new registration choice.

## 2. Own-kernel transfer audit

Detailed source inventory: [ROUND29B_KERNEL_DMA_AUDIT.md](ROUND29B_KERNEL_DMA_AUDIT.md).
All35 explicit transfer sites are listed. Main findings: RoPE activation
loads/stores2 KiB/partition, cos/sin1 KiB logical but128B unique; CE logits
16 KiB BF16 per partition, scalar metadata4B; conv activations2 KiB,
dW partial stores4B. MLP has no explicit NKI transfers. The most concrete
transfer-only proposal batches eight conv partial stores into one4 KiB
row, retaining each existing reduction; its extra on-chip repartitioning
still needs equality/legality and actual timing. No speed claim or launch.
It separates explicit HBM/SBUF transfer sites from SBUF/PSUM computation,
source bytes from destination widening/broadcasting, and logical tiles from
physical DMA packets. MLP forward in CLEAN and gate-fix is **plain PyTorch**;
its misleading `_fused_mlp_kernel` name does not establish an NKI kernel.

AWS recommends at least 2 KiB per active partition and shows near-saturation
around 4 KiB; this is an efficiency target, not a correctness requirement or
a universal promise that batching helps. Strides, broadcast duplicates,
tails and buffer pressure still matter.
[AWS DMA guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/get-started/about/nki-dma-overview.html).

No batching patch is implemented or declared math-identical by this report.
The requested route remains CPU equality of outputs/gradients, then a
same-host 60-step timing (>=1.5% tok/s), then a separately reviewed full cold
run. Reordering a reduction or removing a BF16 rounding boundary is NOT
automatically an exact batching change.

## 3. MLP-backward profile gate

Existing Round29 summary `63ada7e8` and full-coverage query `3b296982`
are sufficient to establish the missing attribution; another join cannot
recover names that were not captured. No repeat decode selected.

| Requested MLP-backward measurement | Actual answer |
| --- | --- |
| `average_dma_size` | **Not available for MLP backward.** No identified region. The installed schema instead lists `dma_transfer_average_bytes` for Summary. |
| Small DMA transfers band | **Not available for MLP backward.** A global small packet cannot be assigned to MLP by guessing from size. |
| Transposes >10% of FLOPs | **Unknown.** No region attribution; missing HLO/compiler metrics. 600 recorded DMA-transpose flags are not a FLOP ratio. |
| Separate elementwise pass between backward matmuls | **Not established by trace.** Source has `dpre = dhidden * (relu * 2)` and `relu.square()`, but compiler fusion/materialization cannot be inferred from Python statements. |

All251,542 Instruction rows lack framework/HLO names and framework IDs.
Warnings report dropped DMA events and missing dynamic metadata. The recorded
aggregate transfers cover only10.80% of filtered packet bytes. Therefore
session-wide transfer statistics are neither an MLP-region result nor a
production-step result. See [Round29 evidence](ROUND29_REPORT.md).

## 4. Decision, integration amendment and deadline

**NO BUILD: profile condition unproven AND SDK>=2.32 condition not met.**
Do not upgrade the organizer-matched hosts to bypass this decision.
The earlier10–12h plan is historical conditional design, not launch authority.

If legitimately reopened on satisfying evidence and environment, Round29b
replaces the old Function-style proposal with separate `mlp_fwd` / `mlp_bwd`
custom ops, fake implementations, `register_autograd`, and `opcheck` for both.
Gradient numerics must also pass through the normal real call path;
`opcheck` alone is not mathematical gradient verification. No baremetal
result is accepted as substitute native-call evidence. The claim that
baremetal universally returns zeros in2.32 was not independently established
from the release notes; no such test was run or relied upon.

AWS's [blockwise MoE backward source](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/bwd/blockwise_mm_backward.py)
does expose `skip_gate_proj`, selectable accumulation dtype and token/expert
mapping inputs. Setting one expert and unit affinities is a design starting
point, **not proof of an exact dense ReLU-squared replacement**. Its
[dropless implementation](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/bwd/bwmm_bwd_dropless.py)
and MXFP8 loop reference remain references only; no code copied or deployed.

Any reopened attempt still has the<=12h timebox and must be **timed on
device by Monday28 September16:00 IST (10:30 UTC)** or be dropped.
Safe CLEAN cold run stays first. No running job changed by this audit.
