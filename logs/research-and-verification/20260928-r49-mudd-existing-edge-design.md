# R49 MUDD existing-edge gates: bounded research design

Date: 2026-09-28  
Status: advisory-only research; not selected, not a proposal, and not launch authority.

## Verdict and scope

One defensible cheaper MUDD-scope hypothesis is token-dependent gating of the **existing skip edges in blocks 6 and 7**. It is a scientific variant, **not a same-math optimization** of the native-tested MUDD-lite source `958fe7439d795914da80bac4fbedbddad99bce771faf4fb75c4ac1dae79acdd6`.

This report does not claim a >=0.005 public-BPB improvement, a measured <=2% overhead, or quality preservation. Root/supervisor determines whether it is selected under R48/R49. No candidate code, proposal, queue, launch command, evaluator, optimizer recipe, or host state was changed.

## Exact design hypothesis

Keep two independent tokenwise `Linear(1024, 2, bias=True)` gates, one for each final block, with weights and biases initialized to zero. Compute the two coefficients from that block's local residual **before** its existing U-Net addition:

```text
(d_u, d_0) = W_i x + b_i
z = x + (u_i + d_u) * s_i
y = r_i * z + (a_i + d_0) * x0
```

Continue the existing block unchanged from `y`.

Definitions:

- `u_i`: existing U-Net scalar for block i.
- `s_6 = h1`, `s_7 = h0`: outputs of blocks 1 and 0, respectively, already consumed by those U-Net edges.
- `x0`: existing normalized input embedding state.
- `r_i` and `a_i`: the block's existing residual and x0 scalars.
- `d_u` and `d_0`: per-token scalar deltas, broadcast across the hidden dimension.

Thus block 6 dynamically mixes the two earlier sources `{x0, h1}`; block 7 uses `{x0, h0}`. Gates are independent: no parameter tying, shared coefficients, or shared gate-query input. Preserve existing scalars and optimizer settings; new gates use the prior MUDD gate parameter-group policy. New gate parameters total 4,100, versus the tested arm's 6,150.

All operations remain in the common causal train/evaluation forward path. No detached sources, future-token access, cross-batch mixing, or train/evaluation branch difference is proposed. Zero deltas algebraically reproduce the baseline, but floating-point parity and the compiled graph still require independent verification.

### Explicit change from the tested arm

The tested arm used three stored outputs `{h0, h2, h4}`, independent Linear(1024,3) gates, and separate additive mixtures before blocks 6/7 **after** their U-Net additions. This design changes both source selection and gate-query location, removes access to h2/h4, and routes dynamic control through existing products. It is not numerically or scientifically identical to the tested arm.

## Primary-source evidence

### Original MUDDFormer paper

The paper reports dynamic dense connections improving perplexity beyond static connections and identifies baseline-equivalent initialization as critical. Its sparse 2x2 pattern improves relative training throughput from 89.8% to 97.8%, but loses 0.18 perplexity. Sparsity can reduce cost without preserving full quality automatically. These are different model sizes, training regimes, and hardware; they are not measurements of this eight-layer Trainium design.

Primary source: [MUDDFormer, sections 3 and 3.6](https://arxiv.org/html/2502.12170v2).

### modded-nanogpt PR259

The author reports source-subset sweeps: adding h3/h5 cost time without reliable loss benefit, while smaller selected source sets retained the measured gain. Its v3 replaces static x0/residual coefficients with token-dependent MUDD outputs, initialized to match original coefficients. This supports sparse source selection and existing-edge modulation, not the exact Trainium source choices.

The author also flags consolidation onto one gate-query snapshot as unvalidated because it removes later-block information. Therefore neither tied coefficients nor a shared query input is recommended here.

Primary source: [PR259, source-selection and v3 discussions](https://github.com/KellerJordan/modded-nanogpt/pull/259).

### modded-nanogpt PR315 and implementation

PR315 explicitly introduces MUDD-generated x0-injection and skip-path gates. At matched 1,335 steps, its combined change reports loss 3.28477 -> 3.27714. The author attributes about 0.0045 nats/token to DC and about 0.0031 jointly to MUDD-gating/cleanup. This is bundled evidence, not an isolated x0/skip-gating ablation.

The current implementation multiplies an existing skip coefficient by a token-dependent gate and consumes dynamic x0 coefficients in existing residual arithmetic. Absorbing dynamic control into existing edges is therefore an implemented technique.

Primary sources: [PR315](https://github.com/KellerJordan/modded-nanogpt/pull/315), [current upstream implementation](https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py).

The implementation URL is a mutable master reference, inspected during this bounded research on 2026-09-28; the PR discussions provide the historical evidence.

## Native cost threshold: measurement versus inference

Local evidence: [R49 native overhead wall](/home/poojith-devan/Desktop/Trainium/research/v2/walls/round49-mudd-native-overhead.md).

Measured max-rank completed-step means over steps 20-59:

| Arm | Seconds/step |
| --- | ---: |
| Matched control | 0.44988762827706524 |
| Tested MUDD-lite | 0.46612296674866227 |

The tested arm costs +3.608754%, an incremental **16.235338 ms/step**. A +2% ceiling permits 0.45888538084260654 s/step. Recovering **7.237586 ms/step**, or **44.58% of the measured increment**, is necessary.

Structural inference for the proposed variant:

- The tested arm adds six full-width source products and their additions across the final two blocks.
- The proposed variant adds no separate full-width source mixture: its four selected-source products already exist in baseline.
- It removes the extra h2/h4 source-retention requirement; x0/h0/h1 are already existing-path sources.
- Two small gate projections remain, with output width reduced from three to two. Gate-query activations still need retention for backward.
- Coefficient additions operate on per-token scalar tensors, not new full-width source tensors.

This is a substantially stronger cost rationale than simply dropping three sources to two. Pure proportional scaling of the measured increment would predict approximately +2.41%, still above the ceiling. That proportional estimate is only a screening calculation, not a hardware prediction.

No wall-clock bound follows from operation counts. Per-token coefficient gradients, broadcast lowering, retained gate inputs, gradient fan-in, and compiler fusion can still dominate. Native measurement would remain necessary if independently selected.

## Counterevidence and eligibility limits

1. The narrower two-edge design has no isolated primary-source quality ablation and no local quality result. The failed native arm itself has no completed public-BPB quality comparison to preserve.
2. PR315's measured MUDD-gate/cleanup contribution is bundled. Its later rebase reports needing architecture changes to recover the gain and notes gating instability. These caution against assuming straightforward transfer. [PR315 rebase notes](https://github.com/KellerJordan/modded-nanogpt/pull/315)
3. Original MUDD and the upstream implementation use richer gate networks and/or additional paths. The requested bare linear gates on exactly these two local sources are an inference.
4. Removing h2/h4 and using only embedding/U-Net states may lose useful intermediate-depth information. Existing U-Net connectivity establishes cheap availability, not optimal source quality.
5. There is **no supported >=0.005 public-BPB forecast**. This is not a qualifying new bold arm under that threshold.
6. This report supplies one MUDD-scope cost-reduction hypothesis only. It authorizes no implementation, queue insertion, native test, training run, or host action.

## Work boundary

Read-only local/source research completed; the only writes are this report and its identical live-state copy. No candidate edits, host contact, device work, new research, code execution on the model, or launches.

