# Round13 optimizer audit — exact c239

Source SHA: `c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0`. CPU/meta checks only; no hosts or data touched.

Confirmed routing mismatch: `transformer.h.7.attn.ve_gate.weight` [8,32] is in Muon (LR .021, initial WD .2). The user requires gate matrices in AdamW. Prepare an isolated dedicated AdamW group using the existing repository gate policy (LR MATRIX_LR, betas .9/.95, eps1e-10, WD0), not scalar LR .2. No BPB benefit is assumed; independent review and one full run required.

All 65 parameters appear once, with exactly one owner. All master parameters and actual initialized Adam moments are FP32. Actual MLP storage is canonical [out,in]: up [4096,1024], down [1024,4096]. The .T expressions in backward are intermediate layouts, not transposed storage. The aspect-aware NorMuon reduction matches [nanochat's current source](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py); [PR168](https://github.com/KellerJordan/modded-nanogpt/pull/168) motivates the audit but does not prove our MLP has that bug. Square/up reduce input dim(-1); wide down reduces output dim(-2). The gate reduces dim(-2), but will leave Muon in its isolated routing arm. No separate matrix-orientation bug was found.

CPU verification: 15 AdamW zero-gradient decay cases and 12 Muon decay cases passed. AdamW factor is `1 - scheduled_lr * group_wd`, once; Muon factor per entry is `1 - shape_scaled_lr * .2 * (1-progress) * sign_agreement_mask`, once. Ownership executes only each owner's update then copies/all-gathers; dummy padding is not a second real parameter update. 12 equal-token accumulation cases compare two half-seeded micro losses to concatenated token mean: FP32 max relative-L2 9.09352806388597e-8; BF16 0.002537461929023266. BF16 is not bit-identical due to accumulation/matmul rounding. No factor-of-two objective bug. Native/device numerical parity is not established by this CPU test.

Canonical-transpose covariance max absolute error 2.384185791015625e-7; 3 independent MLP-autograd cases max absolute error 0.00002288818359375 within scaled tolerance. Full receipt: CPU_RECEIPT.json; reproducible checker: check_cpu.py (existing local Torch environment, no Neuron imports).

## Every parameter

| Name | Shape | Optimizer group | Initial LR | WD | NorMuon reduced axis / averaging |
|---|---|---|---:|---:|---|
| transformer.wte.weight | 8192×1024 | 0: adamw | 0.2598076211353316 | 0.001 | — |
| lm_head.weight | 8192×1024 | 1: adamw | 0.005196152422706633 | 0.01 | — |
| transformer.h.0.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.0.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.1.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.1.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.2.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.2.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.3.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.3.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.4.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.4.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.5.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.5.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.6.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.6.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.7.resid_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| transformer.h.7.x0_lambda | 1 | 2: adamw | 0.2 | 0 | — |
| unet_skip_gates.0 | 1 | 2: adamw | 0.2 | 0 | — |
| unet_skip_gates.1 | 1 | 2: adamw | 0.2 | 0 | — |
| unet_skip_gates.2 | 1 | 2: adamw | 0.2 | 0 | — |
| unet_skip_gates.3 | 1 | 2: adamw | 0.2 | 0 | — |
| value_embeds.7.weight | 8192×1024 | 3: adamw | 0.2598076211353316 | 0 | — |
| transformer.h.0.local_conv.weight | 1024×1×2 | 4: adamw | 0.021 | 0 | — |
| transformer.h.2.local_conv.weight | 1024×1×2 | 4: adamw | 0.021 | 0 | — |
| transformer.h.4.local_conv.weight | 1024×1×2 | 4: adamw | 0.021 | 0 | — |
| transformer.h.6.local_conv.weight | 1024×1×2 | 4: adamw | 0.021 | 0 | — |
| transformer.h.7.local_conv.weight | 1024×1×2 | 4: adamw | 0.021 | 0 | — |
| transformer.h.7.attn.ve_gate.weight | 8×32 | 5: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.0.attn.c_q.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.0.attn.c_k.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.0.attn.c_v.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.0.attn.c_proj.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.2.attn.c_q.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.2.attn.c_k.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.2.attn.c_v.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.2.attn.c_proj.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.4.attn.c_q.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.4.attn.c_k.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.4.attn.c_v.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.4.attn.c_proj.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.6.attn.c_q.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.6.attn.c_k.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.6.attn.c_v.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.6.attn.c_proj.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.7.attn.c_q.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.7.attn.c_k.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.7.attn.c_v.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.7.attn.c_proj.weight | 1024×1024 | 6: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.0.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.1.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.2.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.3.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.4.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.5.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.6.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.7.mlp.c_proj.weight | 1024×4096 | 7: muon | 0.021 | 0.2 | -2 / output |
| transformer.h.0.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.1.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.2.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.3.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.4.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.5.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.6.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
| transformer.h.7.mlp.c_fc.weight | 4096×1024 | 8: muon | 0.021 | 0.2 | -1 / input |
