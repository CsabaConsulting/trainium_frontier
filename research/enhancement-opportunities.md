# Enhancing the Competition GPT — What Karpathy's Autoresearch Already Proved, Applied to Trainium

Source of evidence: `autoresearch/` git history (`git log --all`) contains a **real overnight autoresearch run** — commit `fedfef3` ("add experiment results log (125 experiments, best val_bpb=0.969686)") plus the 20+ individual commits it summarizes. That's not speculation from the docs/articles — it's an actual validated ablation log (baseline 0.9979 → 0.9697 val_bpb, ~2.8% relative improvement, on presumably H100 per the `exp/H100/mar8` branch). The commits were later reverted off `master` so fresh clones start at baseline again, but the log (`git show fedfef3:results.tsv`) and the intermediate commits survive in history. This is the strongest evidence available — real gradients-on-metal results, not hypotheses.

Two lenses, as requested:
- **[QUALITY]** — lowers the val_bpb ceiling (model intelligence / sample efficiency), independent of hardware.
- **[SPEED]** — more useful optimizer steps inside a fixed 30-minute wall-clock window (`TRAIN_TIME_BUDGET_SECONDS = 30*60` in `trainium-frontier/prepare.py`), which is the actual Phase 1 scoring lever since val_bpb is measured after a *fixed time*, not a fixed step count.

Every item below states whether the trainium baseline already has it, what Karpathy's ablation found, and Trainium-specific feasibility (static-shape recompilation cost, no `torch.compile`, no flash-attn3, 24GB HBM/LNC=2, NKI kernels available).

## Device portability at a glance

Three buckets, by how much Trainium-specific work each item needs:

- **A — zero device-specific code.** Plain PyTorch/schedule math, identical on GPU or Trainium. Safe to prototype on an H100 (see "Rollout plan" above) with zero porting step.
- **B — same source line runs on both, but efficiency is backend-dependent and unverified/likely worse on Neuron.** Nothing crashes; the GPU path quietly gets a fused/optimized kernel for free (flash-attn, cuDNN backends, etc.) that Trainium's registered backend may not have. Needs profiling, and possibly a follow-up NKI kernel, to actually realize the win.
- **C — can't run as-is on Trainium at all.** No GPU code to "port" — either it's Trainium-native from the start (NKI kernels), or the GPU reference implementation is a hand-written CUDA kernel with zero Neuron equivalent.

| # | Item | Bucket | Why |
| --- | --- | --- | --- |
| 1-3 | LR cooldown/warmup, WD decay schedule | A | pure scalar schedule math |
| 4 | Value embeddings | A | plain `nn.Embedding` + a small gate `Linear` |
| 5 | Sliding-window attention | **B** | mask-building code is portable; GPU gets flash-attn's native `window_size` kwarg for free, Trainium has no such registered backend — needs a custom NKI kernel to actually realize the speedup, not just the mask |
| 6 | Muon momentum-warmup step count | A | scalar schedule |
| 7-8 | Batch size / depth-width resweep | A (values, not code) | method is portable; the *numbers* must be re-measured on real Trn2 throughput, not ported from H100 — no code changes needed either way |
| 9 | Packing-bin reshape / custom packer | A | pure dataloader-arg change + `.reshape()`, entirely inside `train.py` |
| 10 | NKI kernel work (fused RMSNorm/rotary/softcap, possible flash-attention-style kernel) | **C** | Trainium-native by definition — nothing to port *from*, written fresh against the shipped `relu²` template. (The Linear weight-cast-caching sub-fix under this item is bucket A — plain PyTorch, no kernel involved.) |
| 11 | Init scale | A | plain `torch.nn.init.*` calls |
| 12 | Mamba2/hybrid architecture swap | **C — backburnered** | root reason it's shelved: the reference CUDA kernels (`selective_scan_cuda`, `causal_conv1d_cuda`) are hand-written CUDA with zero Neuron equivalent — not a portability inconvenience, a hard wall |
| 13 | SSM-inspired transformer add-ons (local causal conv, gated MLP, selective gating) | A, with a **B** caveat on the conv | plain PyTorch ops (`conv1d`, elementwise gate, small `Linear`s); GPU's `causal_conv1d` package is a hand-tuned CUDA kernel that Trainium has no equivalent for — unlikely to matter at kernel size ~4, but if the conv ever shows up as a bottleneck under profiling, that's a bucket-B→C follow-up |

The root cause behind every bucket-C item: `program.md`/README explicitly ban `torch.compile` and any CUDA-specific library (`torch.cuda.*`). Any technique whose GPU speed comes from a hand-written CUDA/Triton kernel (flash-attn, `causal_conv1d`, `mamba_ssm`, fused optimizers) lands in bucket C here by construction — either write the NKI equivalent, or accept the plain-eager-PyTorch fallback and its efficiency cost.

## Already at parity (don't touch)

These looked like gaps from the first-pass architecture diff but on closer read of `trainium-frontier/train.py` they're already implemented, and match exactly the details Karpathy's ablation flagged as high-leverage:

- **Cautious weight-decay mask** on Muon (`mask = (g * w) >= 0` before applying WD) — trainium has this (`train.py:517-518`). Karpathy's log: removing this mask was one of the single worst regressions tested (`b19f649 remove cautious WD mask (much worse)`). Correctly kept.
- **NorMuon-style second-moment normalization** in the Muon step (`second_momentum_buffer`, `beta2=0.90`) — trainium has this too. This is *more* sophisticated than plain Newton-Schulz Muon and matches the direction Karpathy's own optimizer evolved toward.
- Untied `wte`/`lm_head` weights — correct; Karpathy's log shows tying them is catastrophic (`d70987b weight tying ... broken`, val_bpb exploded to 3.2).
- RMSNorm, rotary embeddings, QK-norm, ReLU² MLP, x0-skip + per-block `resid_lambda`, logit softcap — all present, all validated-important architecture in the nanochat lineage (see `gpt-model-comparison.md`).

## Concrete gaps — validated wins Karpathy found that trainium's port dropped or never had

### 1. No LR cooldown at all — [SPEED + QUALITY], highest priority, near-zero cost

`trainium-frontier/train.py:574-577`:
```python
def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    return 1.0
```
LR ramps up over 20 steps then sits **flat at peak for the entire remaining ~30 minutes**. There is no decay.

Karpathy's baseline *already* had a 50%-of-budget linear warmdown, and his ablation pushed it further and found it kept helping:
```python
def get_lr_multiplier(progress):  # progress = training_time / TIME_BUDGET
    if progress < WARMUP_RATIO: ...
    elif progress < 1.0 - WARMDOWN_RATIO: return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```
Log entries: `4e6697f warmdown 0.5→0.7 (keep)`, `438a26e warmdown 0.7→0.75 (keep, best run)`, `772dada FINAL_LR_FRAC 0.0→0.05 (keep — small LR floor beats decaying to exactly 0)`. Every single warmdown-related change that was tried and kept pointed the same direction: more cooldown, plus a small nonzero floor instead of a hard zero.

This is pure LR-schedule math — no shape changes, no recompilation risk, trivially portable to Trainium. **Single highest-confidence, lowest-risk change available.** Base it on `progress = elapsed_wall_time / TRAIN_TIME_BUDGET_SECONDS` (time-based, not step-based) — important on Trainium specifically, since NEFF compile overhead makes the first steps slower than steady-state, so step-count-based progress would misjudge how much wall-clock is actually left.

### 2. Warmup should probably be removed, not lengthened — [SPEED + QUALITY], free

Trainium has `WARMUP_STEPS = 20` (nonzero). Karpathy's shipped baseline uses `WARMUP_RATIO = 0.0` (none), and the ablation tried adding warmup twice, independently, and both were discarded: `187e419 add 5% warmup (discard)`, `b1f02f7 add 2% warmup — "any warmup hurts" (discard)`. In this fixed-short-budget regime, every step spent ramping up is a step not spent learning at full LR. Try `WARMUP_STEPS = 0` (or 1-2 for numerical safety on first NEFF-compiled step) as an experiment — validated-likely win, essentially free to test.

### 3. Weight decay is constant; Karpathy decays it to zero — [QUALITY], cheap

Trainium: `WEIGHT_DECAY = 0.10` used unchanged for the entire run (`train.py:664-665`, `group["weight_decay"] = WEIGHT_DECAY`, no schedule). Karpathy's baseline already decays it:
```python
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
```
Ablation `d3840ec constant WD at 0.1 — "decaying better" (discard)` explicitly confirms the decaying version wins. Same category as #1: pure scalar-schedule change, zero shape/recompile risk.

### 4. Value embeddings (VE) — dropped entirely — [QUALITY], the biggest architectural gap

Karpathy's model injects a learned per-position "value embedding" into attention's V stream at (by default, in the ablation) every layer, gated per-layer (`ve_gate`, `value_embeds` ModuleDict, `VE WD`/`VE LR` as distinct tunables). trainium's `CausalSelfAttention`/`GPT` has **zero references** to value embeddings — confirmed by grep (`value_embed`: 12 hits in autoresearch, 0 in trainium). This is real extra model capacity that's cheap in params (one embedding table per VE-enabled layer, `kv_dim`-wide) and Karpathy's ablation shows it's load-bearing: reducing VE coverage from "every layer" to "every 3rd layer" (`bff5cda`) or "last 3 layers only" (`5db6ed4`) both made things measurably *worse*, i.e. full-coverage VE is the local optimum, not a nice-to-have.
- **Trainium feasibility**: straightforward — it's just an extra `nn.Embedding` per layer plus a small gate Linear and an add into V before attention. Static shapes, no recompilation risk, negligible HBM (well within the 24GB/LNC=2 budget). This is the single highest-value *architectural* addition to port over.

### 5. Sliding-window attention pattern — absent — [SPEED, secondarily QUALITY]

Karpathy's `GPTConfig.window_pattern` (e.g. `"SSSL"`, ablation-winning `"SSSSL"` — 5 short-window layers per 1 full-context layer) makes most layers attend over a short local window (ablation-winning: 1/8 of context, 256 of 1024 tokens) instead of full context. This directly cuts attention FLOPs/memory-traffic per step → more steps fit in the fixed wall-clock budget, and the ablation found it *also* improved val_bpb (better inductive bias for short contexts), not just throughput: `8363d52 SSSSL pattern (keep)`, `7da0b67 short window 1/8 context (keep)`. Trainium's `CausalSelfAttention` has no windowing — every layer is full dense causal attention every time.
- **Trainium feasibility**: needs care. `F.scaled_dot_product_attention(..., is_causal=True)` has no built-in "local window" flag; a banded window requires passing an explicit `attn_mask` tensor. That's fine on Trainium as long as the window size is a **static, compile-time-fixed per-layer property** (it is — the pattern string is fixed at model-build time, same as `SEQ_LEN`/`DEPTH`), so it compiles once like everything else and doesn't trigger the "changing tensor shapes → recompile" penalty `program.md` warns about. Precompute one boolean/additive mask per window size (short, long) and reuse across all layers of that type — cheap.

### 6. Muon momentum warmup length — [QUALITY], trivial, low-impact

Trainium: momentum ramps 0.85→0.95 over 300 steps (`muon_momentum`, `train.py:580-582`) — this matches Karpathy's *original baseline* value. His ablation found 200 steps better (`a7aa309 muon momentum warmup 300→200 (keep)`) and reducing further to 150 was worse (`a6b6476 discard`). Small, cheap, one-line tune — low priority but free to grab alongside the others.

### 7. Batch size vs. step count — [SPEED], needs on-device re-tuning, not a direct port

Karpathy's log: `bea057b halve batch 524K→262K — more steps in 5 min (keep)`, and separately overly-small batch crashed on a shape assertion (`e05a87d`). The lesson — "trade batch size for step count until something else (variance, an assert, or memory) pushes back" — is real and generalizes, but the **specific numbers don't port**: Trainium's tokens/sec, HBM bandwidth, and NEFF recompilation behavior are entirely different from an H100, and trainium's baseline is already smaller (`DEVICE_BATCH_SIZE=1`, `TOTAL_BATCH_SIZE=65536` vs Karpathy's 262K–524K). This needs its own small sweep on real Trn2 hardware, at the model's actual `DEPTH`/`ASPECT_RATIO`, rather than copying Karpathy's ratio. Note also each batch-size change is a shape change → triggers a ~60s NEFF recompile, so this sweep is best done once, offline, before committing to a value for the 30-minute scored run.

### 8. Depth/width sweet spot — [SPEED + QUALITY], needs on-device re-tuning, not a direct port

Karpathy found depth 8-9 / dim ~512-576 near-optimal *for his 5-minute H100 budget* — bigger (depth 10-11, dim 640) measurably hurt because fewer optimizer steps completed in the fixed time (`00a7c09 depth 11 dim 640 — "too big, too few steps" (discard)`, `3cde993`/`8be9036` similar). This is a real, generalizable principle for a **fixed wall-clock objective**: model FLOPs/step trades directly against steps/budget, so there's a hardware- and budget-specific optimum, not a universal one. Trainium's 30-minute budget is 6x longer than Karpathy's 5 minutes, and NeuronCore throughput/memory-bandwidth characteristics differ from H100 — so trainium's current `DEPTH=10, ASPECT_RATIO=64` needs its own sweep rather than reusing Karpathy's absolute numbers. The *method* (bisect depth/width against measured tokens/sec until val_bpb stops improving) is exactly what's proven out, not the specific depth value.

### 9. Dataloader packing bin size — [QUALITY + SPEED], new lever, not from Karpathy's ablation

Forum ruling (devpost thread 45042): repacking train shards via a custom scheme in `train.py`, or just calling `prepare.make_dataloader` with different args, is admissible ("data sampling strategy", not the protected "data pipeline"). Verified `trainium-frontier/prepare.py:make_dataloader` and `autoresearch/prepare.py:make_dataloader` are **byte-identical algorithms** (BOS-aligned best-fit document packing, `row_capacity = seq_len + 1`) — inherited verbatim, not a trainium regression, and not something Karpathy's ablation tuned either. This is a genuinely new opportunity the 125-experiment log never touched.

The real mechanic underneath the forum's "low-effort variant": when no buffered document fits the remaining space in a row, the packer takes the **globally shortest** buffered doc and does `doc[:remaining]` — cropping and **permanently discarding** the remainder (not requeued). At `SEQ_LEN=1024` this crop-and-drop fires on close to every row fill, so any document that doesn't land on an exact fit loses its tail for good.

- **Call `make_dataloader(seq_len=8*SEQ_LEN, ...)` then `.reshape(batch*8, SEQ_LEN)`** on the returned `(inputs, targets)`. Verified by re-deriving the math: the shift-by-one target relationship is computed over the *whole* 8192-token row before reshaping, so chunk boundaries chain correctly (last position of sub-row *i*'s target lands exactly on sub-row *i+1*'s first input token) — no realignment bug.
  - **[QUALITY]**: 8x bigger packing bin → crop-and-drop events per token trained fire far less often; documents up to 8192 tokens (instead of 1024) survive intact instead of losing their tail. More real, undamaged text seen per token of training.
  - **[SPEED]**: this also transposes into effectively raising `DEVICE_BATCH_SIZE` from trainium's current `1` to 8 real rows per host-side dataloader call — cuts Python-loop/grad-accum iteration count 8x for the same total tokens/step. `DEVICE_BATCH_SIZE=1` is a lot of host↔device round-trips paying fixed per-launch dispatch overhead every micro-step; batching this up is a legitimate NeuronCore utilization win independent of the packing-quality argument.
  - **Trainium-safe**: the reshape sits outside the compiled graph — the model still only ever sees one fixed static shape (`8, SEQ_LEN`) for the whole run, so it's a single one-time NEFF compile, same recompilation cost as any other constant you'd tune, not a per-step risk.
  - Residual gap: documents longer than the new 8192-token bin still lose their tail — smaller cap than before, not zero.

- **Further, untested idea** (more invasive, higher effort, no ablation evidence either way): instead of wrap-and-reshape, actually reimplement the packer in `train.py` per the ruling's explicit allowance, and **requeue the un-taken remainder** of a cropped document back into the buffer instead of discarding it. Removes the tail-loss problem at any bin size, at the cost of tracking partial-document continuation state across `refill()` calls. Flag as speculative — worth trying, no precedent in Karpathy's log since his packer has the identical crop-and-drop behavior and he never touched it.

- **Related, separately-flagged issue, also inherited identically by both repos**: neither packer resets attention at document boundaries — `F.scaled_dot_product_attention(..., is_causal=True)` lets tokens from one packed-in document attend across into an unrelated preceding document in the same row. Karpathy's ablation doesn't address this either (not a trainium gap, a shared open question). Bigger packing bins (per the 8x trick above) mean *more* distinct documents concatenated per row, so this leakage source gets proportionally worse exactly when the packing-bin fix is applied. A static-shape-safe fix exists — build a per-step boolean/additive block-diagonal mask from BOS positions and pass it as `attn_mask` instead of `is_causal=True` (values vary per step, shape doesn't, so no recompilation risk) — but this is speculative, not validated by any real ablation run, medium effort. Worth a real A/B before trusting it, especially once larger packing bins are in place.

### 10. NKI / Neuron-level kernel opportunities — [SPEED], hardware-specific, no autoresearch precedent (Karpathy has FA3+torch.compile, so his run never needed this)

`program.md` mandates eager `F.scaled_dot_product_attention` and explicitly forbids `torch.compile` in the competition. On CUDA, Karpathy's autoresearch gets op-fusion for free (`torch.compile`) plus a hand-tuned flash-attn3 kernel. Neither exists here — confirmed via `program.md:9-13` and the fact the shipped baseline already ships one illustrative NKI kernel (`_relu_sq_fwd`/`_relu_sq_bwd` for `relu(x)**2`) specifically *because* fusion has to be hand-written on this stack. That recipe generalizes to every other elementwise hot-path op, none of which are fused today:

- **RMSNorm (`norm()`)** — the single highest-frequency target. Fires ~42x per forward at `DEPTH=10` (2 pre-sublayer norms + 2 QK-norms per block × 10, + 2 outer norms), each a separate `F.rms_norm` dispatch (reduce-mean-square → rsqrt → multiply, likely 3+ kernel launches apiece with no compiler to fuse them). High call count on small tensors = launch-overhead-dominated, exactly the profile a hand-fused NKI kernel (single-pass reduce+rsqrt+mul, tiled to the 128-partition SBUF limit, same shape as the shipped example) fixes best.
- **Rotary embedding (`apply_rotary_emb`)** — ~20x/forward (2 per block × 10). Current impl slices into halves, does 2 muls + 2 adds, then `torch.cat`s the result — the `cat` is an extra full materialization/copy that a fused kernel writing directly into both output halves avoids.
- **Logit softcap (`LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)`)** — low call count (once per forward, plus once per `evaluate_bpb` call) but applied to the **widest tensor in the whole network** (`[B, T, padded_vocab_size]` — vocab is the largest dimension anywhere in the model). Three full elementwise passes (div, tanh, mul) over that tensor every train and eval step is real HBM traffic. Best version isn't even a standalone softcap kernel — it's fusing softcap directly into the cross-entropy computation so the softcapped logits tensor is never materialized in HBM at all.
- **Attention itself (`F.scaled_dot_product_attention`)** — potentially the biggest lever of all, but **unverified**: whether Neuron's eager SDPA backend already lowers to something flash-like (tiled, online-softmax, no full `T×T` materialization) or falls back to a naive "math" path that materializes the full score matrix is not knowable from source-reading alone — it depends on what torch-neuron registers for that op. Do not assume either way; profile first (see below). If it's the naive path, a hand-written NKI flash-attention-style kernel (tiled Q/K/V into SBUF, online softmax accumulation, custom forward *and* backward) is the closest thing to recovering what FA3 gives Karpathy on Hopper — but it's also by far the highest-effort, highest-risk item here (a subtly wrong custom backward silently corrupts gradients rather than crashing). Save for last, after cheaper wins, and only if profiling actually shows attention dominating step time.
- **Muon's Newton-Schulz / polar-express iteration** (the `a*x + x@(b*xtx + c*(xtx@xtx))` loop in `_muon_step`) — matmuls here are already well-served by the PE array (nothing to fuse there), but the surrounding elementwise scale-adds could be fused too. Lower priority: this runs once per *optimizer* step, not once per microstep, so it's called `grad_accum_steps`x less often than anything in the forward path above.

**Adjacent, non-kernel Neuron-level fix, cheaper than any of the above**: `Linear.forward` does `self.weight.to(dtype=x.dtype)` — a full weight-tensor cast — on *every* forward call. Since weights only change once per optimizer step but forward runs once per microstep, this recasts every weight matrix `grad_accum_steps` times more than necessary, each a redundant HBM write. Caching the cast copy once after `optimizer.step()` and reusing it across the grad-accum inner loop is a plain PyTorch-level fix, zero kernel-writing effort, do this before touching NKI at all.

**Process, not just targets**: profile before writing any kernel — don't guess which op actually dominates step time from source alone. The `neuron-nki-profiling` / `neuron-nki-profile-analysis-agent` tooling (deployed earlier this session from `neuron-agentic-development-nativept`) exists for exactly this: capture a real execution trace, get the bottleneck broken down by engine/op, then aim kernel-writing effort at what's actually hot. Once a target is confirmed, `neuron-nki-writer-agent` authors the kernel (following the shipped `_relu_sq_fwd`/`_relu_sq_bwd` pattern) and `neuron-nki-debugger-agent` handles compilation-error iteration. Before trusting any new kernel in a scored run, validate it against the eager reference with `neuron-framework-equivalence` — a kernel that's merely shape-correct but numerically off will pass `causality_check` (trainium's existing causal-mask correctness check) while quietly moving val_bpb the wrong way.

None of these change tensor shapes — same functional signature in, same out — so none carry the "changing DEPTH/ASPECT_RATIO/DBS triggers recompilation" risk `program.md` warns about; each is a one-time compile of the new kernel, same as any other constant.

## Rollout plan — how this fits with running `autoresearch`

The autonomous `autoresearch` loop (`autoresearch/program.md`) and this hand-picked list are two different tools, sequenced rather than run in parallel from the start.

**Why sequencing matters here specifically**: our "experiment" unit is a full 30-minute Trn2 run — Phase 1's actual scoring window — not Karpathy's 5-minute GPU cycle. That's ~2 experiments/hour versus his ~12/hour, i.e. **6x more expensive per experiment**. Every experiment the loop burns rediscovering something already proven in his 125-run ablation log (`fedfef3`) is 6x more wasteful here than it was for him. That's the core argument for applying known wins by hand before ever starting the loop, not just tidiness.

**Phase A — apply by hand first, before starting the loop.** Zero/near-zero risk, correctness-obvious, already validated in Karpathy's real ablation, no reason to pay 30-min slots rediscovering them:
- LR cooldown + small floor (#1)
- remove/shrink warmup (#2)
- WD decay-to-zero schedule (#3)
- Muon momentum-warmup step count (#6)
- packing-bin wrap-and-reshape (#9)
- Linear weight-cast caching (free fix under #10)

Land these, confirm val_bpb moved the right direction with one manual run, commit. This becomes the loop's starting baseline.

**Phase B — structural additions, one at a time, engineering-verified before handoff to the loop:** value embeddings (#4), sliding-window attention (#5), NKI kernel work (#10). These need a human/engineer check the loop's blind "run → grep val_bpb → keep if lower" judgment can't provide: shape sanity, the existing `causality_check`, and for kernels specifically numerical equivalence against the eager reference (`neuron-framework-equivalence`). A subtly-wrong custom kernel or a broken VE gate can still produce a *lower* bpb number by accident and get "kept" by a loop that only ever looks at the number. Land each one deliberately, verify it, do a manual A/B run to confirm real improvement — *then* hand that checkpoint to the loop.

**Where `autoresearch` actually runs**: after each Phase A/B checkpoint, not instead of them — as the long-running "mine the local neighborhood" pass. Exactly right for: retuning VE's own LR/WD once it's added, retuning RoPE base/window ratio once windowed attention lands, and the batch-size/depth-width resweep (#7/#8) — all small-nudge, greedy keep/discard territory, same shape as Karpathy's actual log. Don't turn it loose on a baseline still missing Phase A's free wins, or it'll spend expensive 30-minute slots re-deriving "yes, cooldown helps" instead of exploring genuinely unknown Trainium-specific territory.

**In short**: Phase A (hand-applied) → Phase B items landed one at a time with verification → `autoresearch` runs continuously in the gaps after each checkpoint and beyond, spending its expensive per-experiment budget on what's actually unexplored rather than known answers.

### 11. Init scale — [QUALITY], validated, cheap, missed as its own item earlier

Karpathy's ablation independently tuned the *overall* transformer init scale down from the naive default across three separate confirmed wins: `41d50a8 reduce init scale 0.8x`, `f5979a7 init scale 0.7x (keep)`, `97dda85 init scale 0.7→0.68 (keep)`. This sits alongside the already-present good init choices in trainium's baseline (zero-init on `c_proj`/`mlp.c_proj`, muP-style width-scaled LR via `dmodel_lr_scale`). Note a negative result too — don't over-extrapolate the "principled scaling always helps" intuition: depth-dependent init scale (`3c19fba 1/sqrt(layer+1)`, discarded) and several other init variants in the log lost. Test each, don't assume.

This lever is unusually high-value *specifically because* the run is short (~500-1500 optimizer steps, not a normal-length pretrain) — there's little time for training to wash out a bad starting point, so init quality carries more weight here than it would in a long run. Cheap, zero shape/architecture risk — fold into Phase A.

**Not permitted, don't attempt**: warm-starting from an earlier checkpoint (yours or anyone's). The literal rule bans external pretrained weights outright; reusing your *own* prior submission's checkpoint is grayer in the text but breaks the stated premise directly — *"lowest val_bpb after exactly 30 minutes of training"* means each scored run is self-contained, not cumulative across submissions. There's also no submission slot for attaching a checkpoint (only `train.py` + launch command + optional NKI kernels), and scoring almost certainly runs in a clean environment with nothing else present.

**Gray-area, ask-before-relying-on-it**: deriving embedding init from token-frequency statistics computed on the *provided* training data (not gradient-based, not external) is plausible and arguably legal — "assets produced within your own training loop" is ambiguous about whether a pure data-statistics pass counts. Anything further (e.g. a small offline word2vec/SVD pass to actually *learn* embeddings before the timed run starts) crosses into what the pretraining/external-data ban exists to prevent. Don't rely on either without checking the forum first.

### 12. Architecture swap to a state-space model (Mamba2) — [QUALITY], **not recommended for Phase 1**, evidence-backed

Verified directly against the fixed harness (`prepare.py:571` `evaluate_bpb`, `:638` `causality_check`): both only ever call `model(x) -> logits` given tokens, and `causality_check` just verifies flipping future tokens doesn't change earlier-position logits. Nothing in that contract, or in the README's "what you may modify" list, is transformer-specific — a causal SSM satisfies both by construction, so this is *legal*. It just isn't good expected value for Phase 1, on every axis checked:

**FLOPs math (own derivation, from `train.py`'s own `estimate_flops()` formula)**: at the current baseline (`SEQ_LEN=1024`, `DEPTH=10`, `n_embd=640`), attention is only ~19% of total per-token FLOPs; the crossover point where attention would equal the matmul term is ~4,250 tokens for this model width — we're at ~24% of that. Cross-checked against the literature (see below) and consistent: "Mamba vs Transformers: A Technical Comparison" cites empirical Mamba advantage kicking in around 2,000-4,000 tokens; a formula-derived FLOPs-share chart in "Transformers vs Mamba vs Linear Attention" puts a d=1024 model's crossover at ~3-5k tokens. Two independent methods agree: at `SEQ_LEN=1024` we're solidly in transformer-favorable territory, and even a *perfect* SSM swap (zero SSM-side cost, which is unrealistic) caps out around a ~19% FLOPs saving — smaller than what the already-validated, far-lower-risk sliding-window attention change (#5) targets.

**The finding that actually settles it**: every quantitative "Mamba is faster" claim across 5 surveyed articles (Mamba Explained; Beyond Transformers; Mamba vs Transformers: A Technical Comparison; How Mamba Beats Transformers; Transformers vs Mamba vs Linear Attention) is an **inference-time** throughput/latency/KV-cache-memory claim — never a training-sample-efficiency claim. Direct quote, "How Mamba Beats Transformers at Long Sequences": *"your training costs remain similar to Transformers, so don't expect huge speed-ups during pre-training. The efficiency boost comes at inference."* Worse: "Mamba vs Transformers" separately notes Mamba shows **lower MFU during training** than an optimized Transformer at equivalent scale — the parallel scan is less hardware-aligned than plain matmul, even on mature CUDA kernels with years of tuning. None of the 5 articles contain a short-context (<2k), small-scale (tens of millions of params), training-steps-to-target-loss comparison — that specific data point doesn't exist in this literature; the "Mamba wins" story is consistently a long-context-serving story, not a short-context-training story.

**Repo reality check** (`awslabs/state-space-models-neuron`, cloned to `state-space-models-neuron/`): nothing is directly portable. Confirmed hard `torch_xla` dependency at the kernel level, not just training glue (`mamba2/mamba2_kernel.py:4`, `train.py:11`), and `neuronx_distributed` (NxD) baked into the `Mamba2Mixer` class itself via `RowParallelLinear`/`ColumnParallelLinear` (`mamba2/mamba2_mixer.py:26-27`) — even the misleadingly-named `mamba2_inference_no_tp.py` still imports NxD's `parallel_state`. There is no NxD-free, no-XLA, single-device code path anywhere in the repo — direct conflict with the competition's plain-eager `torch.device("neuron")` baseline. Also uses an older `torch_neuronx.nki_jit` kernel-invocation style, not the `@nki.jit` decorator the competition's example kernel uses. Porting means writing the SSD chunked-scan and grouped-conv1d kernels from scratch in NKI with **zero existing Neuron reference kernel to start from** — this repo is useful only as an algorithm reference (the SSD scan structure, the grouped-conv design, the hybrid layer-allocation idea), not as reusable code.

**Hybrid-specific findings** (also from the repo and literature): the repo's own `Mamba2Hybrid` default is ~8% attention layers + ~50% plain-MLP-only layers + remainder Mamba2 (`mamba2hybrid/configuration_mamba2hybrid.py:97-100`) — a third layer type (attention-free, SSM-free MLP blocks) not present in our baseline at all. Real-world hybrids (Jamba: ~1-in-8 attention layers; NVIDIA hybrids: "Mamba for the majority of layers, strategically placed attention") are motivated by **long-context serving cost and recall/in-context-learning quality tradeoffs** — attention is kept specifically because pure SSM layers are weaker at exact recall/needle-in-haystack/ICL tasks. Neither motivation applies to Phase 1 (short context, training-time-only objective, no serving).

**Verdict**: not recommended for Phase 1. The one place this could conceivably matter is Phase 2 — CORE has a 60-minute inference timeout and the README explicitly flags inference efficiency as a first-class Phase 2 constraint — but even there, the same literature's hybrid rationale argues for *keeping* attention layers (CORE is fundamentally an ICL/recall benchmark, exactly where pure SSM is weak), and you'd still be building genuinely novel NKI kernels from zero reference. If pursued at all, it's a Phase-2-only, hybrid-only, high-risk bet — not a Phase 1 move.

### 13. Ideas borrowed from the Mamba/SSM research, applied to the transformer path (no architecture swap) — [QUALITY], untested, good `autoresearch` candidates

Item #12 rules out swapping to Mamba2/hybrid for Phase 1. That doesn't waste the research — a few design elements from the SSM world are transferable to the existing transformer *without* adopting the recurrence, the SSD scan, or any new kernel category. None of these appear in Karpathy's 125-run ablation (he never tested them, unlike items #1-11) — treat them as real experiments to run, not assumed wins, and they're good first candidates to hand to `autoresearch` once Phase A/B land, since each is a single, bounded diff.

- **Cheap local causal conv ("short convolution" / token-shift) alongside attention** — the most concrete and most precedented. Mamba (and H3, Hyena, RWKV) all place a small depthwise causal conv1d (kernel size ~4, or RWKV's degenerate kernel-2 shift-and-mix) in front of the actual sequence-mixer, specifically to handle short-range pattern-matching cheaply (linear cost, tiny kernel) so the expensive mixer doesn't spend capacity on it. Purely additive to a transformer block — no recurrence, static-shape-safe on Trainium, directly complements the already-planned sliding-window attention (#5): local patterns get absorbed by the conv at near-zero cost, freeing windowed-attention layers to do less per-token work.
- **Gated MLP (SwiGLU/GEGLU-style) instead of plain ReLU² MLP** — Mamba's block gates its SSM output multiplicatively (`y * silu(gate)`) before the final projection; the transferable idea is the multiplicative-gate pattern, not the SSM itself. Karpathy's log tuned the MLP's *hidden ratio* (3.5x/4x/4.5x, 4x won) but never tried changing its *structure* to a gated variant. Real cost to weigh against any gain: an extra weight matrix (more params + FLOPs), tested against the model's own simplicity criterion and the fixed time budget.
- **Input-dependent (data-dependent) per-token gating of `resid_lambda`/`x0_lambda`** — Mamba's "selective" mechanism computes its recurrence parameters (Δ, B, C) from the current token rather than using fixed scalars; the baseline's `resid_lambda`/`x0_lambda` are one static learned scalar per layer. Making them a function of the hidden state (a small per-token gate) is the same principle without the SSM. Higher effort/risk than the two above (extra projection, more compute per token, deviates from the current minimal-scalar design) — lower priority, further-out idea.

## What doesn't transfer / is off the table on Trainium

- **`torch.compile`, flash-attn3 kernel** — both explicitly forbidden/unavailable on NeuronCore (`program.md`: "Do NOT use torch.compile"; no FA3-equivalent kernel exists for Neuron). This is *why* the illustrative NKI relu² kernel exists in the trainium baseline — on Trainium, op-fusion speedups have to be hand-written as NKI kernels instead of coming for free from `torch.compile`. If pursuing item #5 (windowed attention) or #4 (VE) pushes step time up, a hand-fused NKI kernel for the extra elementwise/gather ops (following the existing `_relu_sq_fwd`/`_relu_sq_bwd` template: tile to 128-partition SBUF chunks, wrap in `torch.autograd.Function`) is the correct lever to claw the throughput back — not something to reach for pre-emptively.
- Multi-query/grouped-query attention (`n_kv_head` < `n_head`) — Karpathy's own ablation found this hurt at this model scale (`b85567f multi-query n_kv_head=1 — "too few KV heads" (discard)`). Not worth porting.
- Parallel attention+MLP block structure — tried and discarded by Karpathy (`46cf5f2 "much worse"`). Not worth porting.

## Suggested priority order (risk-adjusted)

1. LR cooldown schedule (#1) — do this first, zero shape risk, strongest evidence, likely largest single val_bpb delta available.
2. Remove/shrink warmup (#2) — free, pairs naturally with #1's re-derivation of the schedule.
3. Decaying weight decay (#3) — free, same category as #1/#2.
4. Value embeddings (#4) — bigger lift (new module), but static-shape-safe and the clearest *architectural* capacity gap versus Karpathy's model.
5. Muon momentum warmup length (#6) — trivial one-line tune, grab it alongside #1-3.
6. Sliding-window attention (#5) — good speed/quality win but needs a static-mask implementation; do after the schedule fixes so you're tuning against a stronger baseline.
7. Packing bin size (#9, wrap-and-reshape variant) — cheap, no model changes, do early alongside #1-3; it's a dataloader-only change and directly increases effective device batch size, so re-derive #7/#8 (batch/depth sweep) *after* this lands, not before.
8. Batch size and depth/width re-sweep (#7, #8) — do last, on real Trn2 hardware, once the above are in, since the optimal point shifts once the schedule, VE, and packing-bin changes land.
9. Document-boundary attention masking, custom requeue-remainder packer (#9, speculative variants) — no ablation evidence either way; treat as an experiment to run and log, not an assumed win.
10. NKI kernel work (#10) — cache-the-weight-cast fix first (free, no kernel needed), then profile (don't guess), then fuse whatever the trace actually shows as hot (RMSNorm/rotary/softcap are the likely candidates by call-frequency/tensor-size reasoning above). Save a hand-written flash-attention-style kernel for last — highest effort, highest risk, only worth it if profiling shows attention itself dominating and everything cheaper is already banked.
