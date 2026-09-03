# Competition GPT vs. Karpathy's Original NanoGPT

Comparison of the `GPT`/`Block`/`CausalSelfAttention`/`MLP` classes in `trainium-frontier/train.py` against `nanogpt/model.py` (classic nanoGPT, 2022).

## Verdict

The competition model is **not** classic nanoGPT. It descends from Karpathy's newer **nanochat / modded-nanogpt** lineage (same lineage as the `autoresearch` repo — see `autoresearch-comparison.md`). Architecture is substantially different from nanoGPT, not just a hardware port.

## Side-by-side

| Component | nanoGPT (original) | Trainium competition GPT |
|---|---|---|
| Normalization | `LayerNorm` (learnable weight + optional bias) | `RMSNorm` (`F.rms_norm`, no learnable params at all) |
| Position info | Learned absolute position embeddings (`wpe`), added to token emb | None — rotary embeddings only (precomputed `cos`/`sin` buffers) |
| QKV projection | Single fused `c_attn` Linear, splits into q/k/v | Separate `c_q`/`c_k`/`c_v` Linear layers |
| Attention extras | Plain SDPA, no norm on q/k | **QK-norm**: RMSNorm applied to q/k after rotary, before attention |
| Bias | `config.bias` toggle (Linears + LayerNorms) | No bias anywhere — all Linear `bias=False` |
| MLP activation | GELU, 4x expansion | **ReLU²** (`relu(x)**2`), 4x expansion, optional custom NKI kernel |
| Dropout | `config.dropout` throughout (attn, resid, MLP) | None — no dropout modules at all |
| Residual stream | Standard `x = x + attn(ln(x))`; `x = x + mlp(ln(x))` | Per-block learnable `resid_lambda` scale + **x0 skip**: `x0_lambda` mixes the *initial* embedding into every block's input (stabilizes deep nets) |
| Weight tying | `wte.weight` tied to `lm_head.weight` | **Not tied** — separate weights, very different init std (wte: 0.8, lm_head: 0.001) |
| Vocab padding | Hardcoded 50304 (comment: "GPT-2's 50257 padded to nearest 64") | Computed at runtime: `((vocab_size + 63) // 64) * 64` |
| Logits | Raw logits into cross-entropy | **Logit softcap**: `15.0 * tanh(logits / 15.0)` before loss |
| Init | `normal_(std=0.02)` everywhere, scaled init for `c_proj` per GPT-2 paper | Custom per-tensor init: uniform for attn/mlp weights scaled by `sqrt(3)/sqrt(n_embd)`, zeros for `c_proj`/`mlp.c_proj`, `resid_lambda=1.0`, `x0_lambda=0.05` |
| Optimizer | Single **AdamW**, 2D+ tensors decay / <2D don't, fused if CUDA | Hybrid **MuonAdamW**: AdamW for embeddings/lm_head/scalars (each its own LR, embedding LR scaled by `(n_embd/768)^-0.5`), **Muon** (orthogonalized momentum via polar-express Newton-Schulz coefficients) for all matrix params, grouped by shape |
| MFU / FLOPs | `estimate_mfu()` — hardcoded vs A100 312 TFLOPS bf16 peak | `estimate_flops()` — hardware-agnostic FLOPs/token count, no peak-FLOPS ratio baked in |
| Compute dtype | fp32/bf16 via autocast, config-driven | Hardcoded `COMPUTE_DTYPE = bfloat16` module constant |
| Sampling | `generate()` method (temperature, top-k, multinomial sampling) | **Absent** — no generation/sampling code, training-only script |
| Pretrained loading | `from_pretrained()` pulls GPT-2 HF checkpoints | **Absent** — trains from scratch only |
| `crop_block_size` | Present (model surgery for shorter block size) | **Absent** |

## What carried over unchanged in spirit

- Core transformer block shape: attention + MLP with residual connections, causal masking via `F.scaled_dot_product_attention`.
- Vocab padding to a multiple of 64 for efficiency — same idea, different implementation.
- Both are single-file, dependency-light, "readable" reference implementations (explicitly stated in both docstrings) meant to be modified by the reader/participant.

## Why it matters for the competition

Every non-obvious architectural choice here (RMSNorm, rotary, QK-norm, ReLU², no biases, x0 skip, logit softcap, Muon optimizer) is a **known modded-nanogpt/nanochat speedrun trick** — these are exactly the levers a participant is expected to understand, tune, or rip out per `program.md`'s simplicity criterion. Anyone mentally modeling this as "nanoGPT on Trainium" will be surprised by the optimizer (Muon, not AdamW) and the residual stream (x0 skip + learnable lambdas) in particular.
