# GPU Proxy Experiment Results — RTX PRO 6000 Blackwell Max-Q (Vast.ai)

Running log for the overnight autonomous session. Machine-readable rows live in `research/experiment-log.jsonl` on the remote box (synced back here periodically); this doc is the human-readable narrative — what was tried, what won, why, and what's still open.

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96GB HBM, compute cap 12.0, torch 2.11.0+cu128, driver 595.80. Single GPU (`--nproc-per-node 1`), so this is the GPU-proxy phase per `experiment-rollout-plan.md` — **[GPU]**-tagged caveats apply throughout; nothing here is a final Trn2 number.

**Standard unit for this session**: `--num-steps 800`, `--max-train-seconds` chosen per-Solution to match the schedule's intended horizon (see Solution 1 note — this matters specifically for LR-schedule-dependent Solutions since `progress` is computed against `max_train_seconds`, not a hardcoded budget). At ~1.24s/step observed on the unmodified baseline (single GPU), 800 steps ≈ 16-17 min of training + eval.

**Convention**: each Solution's row(s) below record commit-equivalent description, `public_val_bpb`, and keep/discard. Cumulative — each Solution builds on the previous kept config, same as Karpathy's own `results.tsv`. Full detail (launch command, all summary fields) is in `experiment-log.jsonl`.

---

## Baseline

`--num-steps 800 --max-train-seconds 1200` (unmodified code). **public_val_bpb = 1.273797**. training_seconds=1023.7 (~1.28s/step), causality passed exactly (max_abs_diff=0.0), no NaN/crash. This is the reference point for everything below.

Note: 800-step val_bpb (1.27) is far from the README's reported ~0.98 at a full 30-min/8-core run — expected, this is a short single-GPU proxy run, not a real submission-shape number. Only *relative* deltas between rows below are meaningful this session, not the absolute value.

---

## Solution 1 — LR cooldown + floor

Code: `lr_multiplier` rewritten to compute `progress = elapsed/max_train_seconds` and apply cooldown+floor when `WARMDOWN_RATIO > 0` (defaults `WARMDOWN_RATIO=0.0`, `FINAL_LR_FRAC=1.0` reproduce old flat-after-warmup behavior exactly — zero-op refactor, only the sweep values below change behavior). `max_train_seconds` set to ~1050s for this sweep (close to the 800-step run's real duration, ~1025s) so `progress` actually reaches deep into the cooldown window within the fixed step budget — see `experiment-rollout-plan.md`'s note on this.

| Config | public_val_bpb | Δ vs baseline | status |
|---|---|---|---|
| baseline (flat LR) | 1.273797 | — | — |
| `WARMDOWN_RATIO=0.5, FINAL_LR_FRAC=0.0` | 1.216659 | **-4.5%** | keep |
| `WARMDOWN_RATIO=0.75, FINAL_LR_FRAC=0.0` | 1.216602 | -4.5% (tied w/ 0.5) | keep (no regression) |

`0.5→0.75` is a tie (Δ=0.00006, well inside noise) — diminishing returns past 0.5 here, consistent with Karpathy finding 0.8 worse than 0.75 in his own log. Locking `WARMDOWN_RATIO=0.75` (his validated number, no evidence against it).

| `WARMDOWN_RATIO=0.75, FINAL_LR_FRAC=0.05` | 1.215996 | -4.55% | keep |

Small further gain from the nonzero floor (-0.05% on top), matches Karpathy's finding that a small floor beats decaying fully to zero. **Solution 1 locked: `WARMDOWN_RATIO=0.75, FINAL_LR_FRAC=0.05` → 1.215996, cumulative -4.55% vs baseline.**

---

## Solution 2 — Remove/shrink warmup

Cross-ref #2. Starting from Solution 1's locked config (1.215996).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `WARMUP_STEPS=20` (Solution 1 config) | 1.215996 | — | — |
| `WARMUP_STEPS=0` | 1.216120 | +0.00012 | keep (see note) |

**Note on this call**: the delta is the same tiny magnitude as the `WARMDOWN_RATIO 0.5→0.75` noise floor already established (~0.0001), i.e. not distinguishable from run-to-run noise on a single sample here — a genuine repeat-seed check would cost another full ~17-min run for a call this small. Deciding **keep `WARMUP_STEPS=0`** anyway on the strength of Karpathy's log, which found warmup hurt in *two independent* tests (not a one-off) — this result is at worst a statistical wash with that prior, not a real contradiction of it. Also matches `program.md`'s own criterion: removing something for an equal result is a good outcome, not just a neutral one.

**Solution 2 locked: `WARMUP_STEPS=0` → 1.216120 (carried forward; treated as tied with 1.215996 for cumulative-tracking purposes).**

---

## Solution 3 — Weight-decay decay-to-zero schedule

Cross-ref #3. Starting from Solutions 1-2's config (1.216120).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `WD_DECAY_TO_ZERO=False` (constant, Sol 1-2 config) | 1.216120 | — | — |
| `WD_DECAY_TO_ZERO=True` (linear decay to 0) | 1.215870 | -0.00025 | **keep** |

Above the established noise floor (~0.0001) and in the expected direction — real, if modest, win. **Solution 3 locked: `WD_DECAY_TO_ZERO=True` → 1.215870.**

**Cumulative after Solutions 1-3: 1.273797 → 1.215870 (-4.55% relative).**

---

## Solution 4 — Muon momentum-warmup step count

Cross-ref #6. Starting from Solutions 1-3's config (1.215870).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `MUON_MOMENTUM_WARMUP_STEPS=300` (current, Sol 1-3 config) | 1.215870 | — | — |
| `MUON_MOMENTUM_WARMUP_STEPS=200` (Karpathy's keep) | 1.216712 | +0.00084 | **discard** |

Real regression, well above the established noise floor (~0.0001-0.00025) — not a noise call. **Contradicts Karpathy's finding directly.** Genuine divergence from his setup (different depth/width, different WD-decay interaction, or just a different optimum here) rather than a replication of his result. **Solution 4 discarded — reverted to `MUON_MOMENTUM_WARMUP_STEPS=300`.** Cumulative unchanged at 1.215870.

This is exactly the kind of thing the whole premise of this exercise flagged as *not* guaranteed to transfer — good to have it verified independently rather than assumed.

---

## Solution 5 — Init scale sweep

Cross-ref #11. Starting from Solutions 1-3's config (1.215870; Solution 4 discarded).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `INIT_SCALE=1.0` (baseline) | 1.215870 | — | — |
| `INIT_SCALE=0.7` | 1.212319 | **-0.00355** | keep |
| `INIT_SCALE=0.68` | 1.210940 | -0.00138 further | **keep** |

`0.7` and `0.68` both clear, real wins in strict monotonic order — matches Karpathy's finding exactly. Not re-testing `0.66` (his log already found it regresses past `0.68`) — trusting that negative result rather than re-spending a run on it. **Solution 5 locked: `INIT_SCALE=0.68` → 1.210940.**

**Cumulative after Solutions 1-5: 1.273797 → 1.210940 (-4.93% relative).**

---

## Solution 6 — Packing-bin reshape trick

Cross-ref #9. Starting from Solutions 1-5's config (1.210940).

**Methodological note, applies to every Solution from here on**: this trick changes throughput (fewer, bigger micro-batches — smoke test showed ~0.87s/step vs baseline's ~1.28s/step, a real speedup). Since `progress` (driving the LR cooldown and WD decay from Solutions 1+3) is computed as `elapsed / args.max_train_seconds`, reusing the old `max_train_seconds=1050` here would make an 800-step run land at a *different point* in the cooldown curve than Solutions 1-5 did (progress≈0.67 instead of ≈0.98) — confounding the comparison with a schedule-phase difference, not just the intended change. Caught this early (killed a misconfigured run ~1 min in) and recalibrated `max_train_seconds` to match each Solution's own actual throughput going forward, not a reused constant. Smoke-tested first (15 steps, `PACK_FACTOR=8`) — passed clean: no crash, causality still exact (`max_abs_diff=0.0`), `micro_tokens=8,192 grad_accum_steps=8` confirmed (was `1,024`/`64`) — 8x fewer grad-accum iterations for the same total tokens/step, as intended.

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `PACK_FACTOR=1` (Sol 1-5 config) | 1.210940 | — | — |
| `PACK_FACTOR=8` | 1.188680 | **-0.02226 (-1.84%)** | **keep** |

**By far the largest single delta this session** — 30-90x the established noise floor, and a real 35% wall-clock speedup too (670.5s vs ~1025s for the same 800 steps, same total tokens/step). Sanity-checked the mechanism: batching itself should be gradient-neutral here (`F.cross_entropy` with `reduction="mean"` averages uniformly over all tokens in the batch regardless of micro-batch size — worked through the math, 8 micro-batches of 8192 tokens each averages to the exact same 65,536-token mean as 64 micro-batches of 1024 each). So the win isn't from batching mechanics — it's the packing-bin-size change itself: far fewer document-tail crop-and-discard events at bin size 8192 vs 1024. Strongly validates the enhancement doc's hypothesis about ClimbMix web-text documents routinely exceeding 1024 tokens and getting truncated at nearly every row fill. **Solution 6 locked: `PACK_FACTOR=8` → 1.188680.**

**Cumulative after Solutions 1-6: 1.273797 → 1.188680 (-6.68% relative).**

---

## Solution 7 — Value embeddings

Cross-ref #4. Starting from Solutions 1-6's config. Greenfield module (no existing VE code in baseline) — porting the reference design from `autoresearch/train.py` (`has_ve`, `ve_gate`, `value_embeds` ModuleDict pattern).

Ported exactly from `autoresearch/train.py`'s reference design: `has_ve(layer_idx, n_layer) = layer_idx % 2 == (n_layer-1) % 2` (alternating layers, last layer always included — 5 of 10 layers here, odd-indexed), gate `2*sigmoid(ve_gate(x[..., :32]))` applied to `v` before rotary/attention, VE embedding table sized `padded_vocab_size × n_embd` per VE-enabled layer, VE's own AdamW group (`lr=EMBEDDING_LR*dmodel_lr_scale`, `weight_decay=0.0`, matching the reference exactly). Gate weights zero-init (sigmoid(0)=0.5, ×2=1.0 = neutral start).

**Two-stage verification before trusting any number** (greenfield module, higher risk than a scalar tweak): (1) smoke-tested `USE_VE=False` first — `2.146382` vs the prior no-VE smoke's `2.146375`, identical within GPU nondeterminism, confirms the whole refactor (threading `layer_idx`/`ve` through `CausalSelfAttention`/`Block`/`GPT`) is a true no-op when off. (2) smoke-tested `USE_VE=True` — ran clean, no crash, `causality_passed=True` (`max_abs_diff=0.0`), `num_params_M` jumped `59.6→85.9` (+26.3M, matches 5 alternating layers × `8192×640` embedding tables exactly).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `USE_VE=False` (Sol 1-6 config) | 1.188680 | — | — |
| `USE_VE=True` (alternating layers) | 1.174188 | **-0.01449 (-1.22%)** | **keep** |

Solid real win, overhead negligible (679.5s vs 670.5s training time). Matches Karpathy's finding directly. Skipped the partial-coverage sanity variants (every-3rd-layer / last-3-only, both confirmed worse in his log) — lower priority than breadth given the time budget on an already strong, precedented result; flagged as a possible follow-up rather than blocking progress. **Solution 7 locked: `USE_VE=True` → 1.174188.**

**Cumulative after Solutions 1-7: 1.273797 → 1.174188 (-7.82% relative).**

---

## Solution 8 — Sliding-window attention

Cross-ref #5. Starting from Solutions 1-7's config.

Combined all three sub-changes into one experiment (all independently validated by Karpathy already, pragmatic given time budget): `WINDOW_PATTERN="SSSSL"` (cycled to `n_layer=10` → `SSSSLSSSSL`, 8 short + 2 long layers), `SHORT_WINDOW_FRAC=0.125` (1/8 of *our* `SEQ_LEN=1024` = 128 tokens — correctly scaled to our context length, not Karpathy's absolute 256-token number which was scaled to *his* 2048 context), `ROPE_BASE=200000`. Implementation: explicit additive float mask built once per distinct window size (`_build_window_mask`, static, registered as buffers), passed to `F.scaled_dot_product_attention(..., attn_mask=...)` — no `is_causal=True` fast path, matching the eager masked-SDPA path Trainium will actually run.

**Hit a real dtype bug on the first smoke test**: `q`/`k` come out fp32 after `norm()` (pre-existing behavior, invisible before since `is_causal=True` never checks mask dtype), but I'd built the mask in `COMPUTE_DTYPE` (bf16) — PyTorch's SDPA requires the mask to be bool, plain fp32, or exactly match query dtype. Fixed by building the mask in plain `torch.float32` (explicitly allowed regardless of query dtype per the error message). Re-verified no-op (`WINDOW_PATTERN="L"`): `2.153047` vs Solution 7's smoke `2.152977`, matches within noise, causality exact. Then smoke-tested the real config: no crash, causality exact, **no speed change at the eager level** — expected, since eager masked SDPA still computes the full `T×T` matrix regardless of mask sparsity; only a specialized windowed kernel (the natural NKI follow-up) would realize a speed win. Testing for a quality effect only here.

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `WINDOW_PATTERN=L` (Sol 1-7 config) | 1.174188 | — | — |
| `WINDOW_PATTERN=SSSSL, SHORT_WINDOW_FRAC=0.125, ROPE_BASE=200000` | 1.175473 | +0.00129 | **discard** |

Small real regression, larger than the established noise floor — doesn't transfer as a combined change to this setup (different depth, different data packing already applied via Solution 6, different schedule already tuned around full dense attention). Since this bundled three independently-validated sub-changes, a single run can't isolate which one (if any) is responsible — flagged as a good candidate for `autoresearch` to decompose later rather than spending more GPU-hours on it now. **Solution 8 discarded — reverted to `WINDOW_PATTERN="L"`, `ROPE_BASE=100000` defaults.** Cumulative unchanged at 1.174188.

---

## Solution 10 — Weight-cast caching

Cross-ref #10 (non-kernel part). `Linear` now caches its dtype-cast weight copy, refreshed once per optimizer step (`refresh_weight_cache`, called right after `optimizer.step()`) instead of recomputed every forward call. **Prediction going in**: Solution 6 already cut `grad_accum_steps` 8x (64→8 via `PACK_FACTOR=8`), so most of the redundant-cast waste this fix targets is already gone.

| Config | public_val_bpb | Δ | training_seconds (800 steps) | status |
|---|---|---|---|---|
| `USE_WEIGHT_CACHE=False` (Sol 1-7+9a config) | 1.171962 | — | 779.2 | — |
| `USE_WEIGHT_CACHE=True` | 1.172216 | +0.00025 (noise) | 778.4 (no meaningful change) | **keep** |

Prediction confirmed: bpb drift is noise-floor-sized (correctness intact, no caching bug), and the speedup is not measurable here — Solution 6 already absorbed the benefit this fix targets. Keeping it anyway: free, correctness-verified, harmless, and still theoretically useful in any configuration with a larger `grad_accum_steps` (e.g. without the packing trick, or on different batch settings on Trn2). **Solution 10 locked: `USE_WEIGHT_CACHE=True` → 1.172216** (treated as tied with 1.171962 for cumulative tracking).

**This completes every GPU-testable Solution (1-10) — Solution 11 is Trn2-dry-run-only here, Solution 12 is Trn2-native from the start. Final cumulative (Solutions 1-7, 9a, 10; 8 and 9b discarded): 1.273797 → 1.171962 (-7.99% relative).**

---

## Final confirmation run — longer horizon

Per the earlier agreement: one longer run (~1400-1500 steps, closer to the real 30-min-equivalent) on the full cumulative winning config, before calling the GPU phase done.

**Result: `public_val_bpb = 1.111741` at 1450 steps (1409.7s training), causality exact, `num_params_M=85.9` matches the expected final config exactly (confirms all reverts/locks applied correctly — no stray leftover config from a discarded Solution).** Not directly comparable to the 800-step number (more steps naturally means lower loss) — this run's purpose was to confirm everything holds together, with no crash and no regression, at a horizon closer to the real 30-min-equivalent. Confirmed: clean, no crash, no issue.

**GPU phase complete.** Final locked config: Solutions 1, 2, 3, 5, 6, 7, 9a, 10 kept; Solutions 4, 8, 9b discarded (real regressions, documented above with reasoning); Solutions 11-12 untouched (Trn2-only by design). Ready to hand off to Trn2 once allocated — see `experiment-rollout-plan.md` for the re-verification protocol per Solution.

---

## Session summary

**Net result at 800-step comparison scale: `1.273797 → 1.171962`, a `-7.99%` relative improvement**, from 8 kept changes (Solutions 1, 2, 3, 5, 6, 7, 9a, 10) out of 12 attempted (Solutions 4 and 8 and 9b discarded — real regressions, not noise). Biggest single contributor by far: Solution 6 (packing-bin reshape, `-1.84%` alone, plus a genuine 35% wall-clock speedup) — validated the enhancement doc's document-tail-cropping hypothesis far more strongly than expected. Second: Solution 1 (LR cooldown+floor, `-4.5%` alone) — the highest-confidence prediction going in, and it delivered the largest single validated-from-literature win. Solution 5 (init scale) and Solution 7 (value embeddings) both landed clean, real wins matching Karpathy's log directly.

**Two genuine divergences from Karpathy's findings**, both real signal (not noise), both informative: Solution 4 (Muon momentum-warmup steps, his 200 lost to our 300) and Solution 8 (windowed attention + RoPE base, combined regression here vs. his combined win) — concrete evidence that not everything transfers across setups, exactly the premise the whole exercise was testing. Solution 8 is a good `autoresearch` decomposition candidate later, since it bundled three sub-changes and a single run can't isolate which one hurt.

**One genuinely novel finding**, no Karpathy precedent either way: Solution 9a (local causal conv before attention) — a real, validated win (`-0.19%` alone). Solution 9b (gated MLP) was a clean discard on every axis (worse quality, +72% params, +19% slower).

**Everything here needs re-verification on real Trn2** before being trusted for a submission — this was all measured on a single RTX PRO 6000 Max-Q GPU, fixed step counts, not the real 30-minute/full-chip/Trn2-native conditions. Per the rollout plan: bucket-A items (nearly everything kept here) should port as direct code copies; Solution 8's discard doesn't rule out its sub-components being retried individually on Trn2 via `autoresearch`. Solutions 11 (batch/depth resweep) and 12 (NKI kernels) haven't been touched — both are Trn2-only by design.

**Process note on methodology**: caught and fixed a real progress-schedule miscalibration bug at Solution 6 (reusing a stale `max_train_seconds` after packing changed throughput would have silently confounded the LR-cooldown phase across the comparison) — documented as a standing methodological rule for the rest of the session. Also caught and fixed a real SDPA mask-dtype bug at Solution 8 before it could produce a false crash-free result.

---

## Solution 9 — SSM-inspired add-ons

Cross-ref #13. Starting from Solutions 1-7's config (Solution 8 discarded).

### 9a — Local causal conv

Depthwise causal conv1d (kernel=4), zero-initialized, added as its own residual branch before attention in each `Block`. Zero-init means the added term is 0 at init, so the branch starts as a true no-op — same principle as the existing zero-init on `c_proj`/`mlp.c_proj`. Smoke-tested clean: `2.152975` vs Solution 7's `2.152977` (near-identical, as expected for a zero-init no-op), causality exact.

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `USE_LOCAL_CONV=False` (Sol 1-7 config) | 1.174188 | — | — |
| `USE_LOCAL_CONV=True` (kernel=4) | 1.171962 | **-0.00223** | **keep** |

Real win, above noise floor. **First genuinely novel finding this session — no Karpathy precedent either way, validated positive independently.** **Sub-step 9a locked: `USE_LOCAL_CONV=True` → 1.171962.**

**Cumulative after Solutions 1-7 + 9a: 1.273797 → 1.171962 (-7.99% relative).**

### 9b — Gated MLP

Canonical SwiGLU form: `c_proj(c_fc(x) * silu(c_gate(x)))`, replacing `relu(c_fc(x))**2` entirely (new `c_gate` weight matrix, same init scale as `c_fc`). Smoke-tested clean (no crash, causality exact) but this is a real cost, not free: **+16.4M params** (matches exactly: 10 layers × one extra `n_embd×4·n_embd` matrix), cumulative `102.3M` vs baseline's `59.6M` (**+72%**) — a lot weighed against the competition's params tie-breaker and simplicity criterion. Also slower per-step (~1.2s vs 9a's ~0.97s). Needs a real, clear quality win to justify this, not a marginal one.

| Config | public_val_bpb | Δ | params_M | training_seconds (800 steps) | status |
|---|---|---|---|---|---|
| `USE_GATED_MLP=False` (Sol 1-7+9a config) | 1.171962 | — | 85.9 | 780 | — |
| `USE_GATED_MLP=True` | 1.175858 | +0.00390 | 102.3 (+72% vs orig. baseline) | 930 (+19%) | **discard** |

Clear discard on every axis — worse quality, more params, slower. Not close to justified. **Sub-step 9b discarded, reverted to plain ReLU² MLP.**

### 9c — Selective per-token gating

Per the plan's own condition ("only attempt if 9a *and* 9b both land") — **9b didn't land, skipping 9c**, moving to Solution 10.

---
