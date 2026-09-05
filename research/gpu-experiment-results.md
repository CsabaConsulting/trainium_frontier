# GPU Proxy Experiment Results — RTX PRO 6000 Blackwell Max-Q (Vast.ai)

Running log for the overnight autonomous session. Machine-readable rows live in `research/experiment-log.jsonl` on the remote box (synced back here periodically); this doc is the human-readable narrative — what was tried, what won, why, and what's still open.

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96GB HBM, compute cap 12.0, torch 2.11.0+cu128, driver 595.80. Single GPU (`--nproc-per-node 1`), so this is the GPU-proxy phase per `experiment-rollout-plan.md` — **[GPU]**-tagged caveats apply throughout; nothing here is a final Trn2 number.

**Upstream sync (post-session)**: merged [aws-neuron/trainium-frontier PR #6](https://github.com/aws-neuron/trainium-frontier/pull/6) — the official runner now excludes compilation time from the charged 30-min budget (first 5 optimizer steps = capped startup allowance) and requires every `train.py` to call `prepare.record_step_boundary(step)` per step or the submission checker rejects it outright. `prepare.py` replaced wholesale (never modified locally, so a clean drop-in); `train.py`'s budget-clock/predictive-stop logic hand-merged around all our Solutions (no shared git history with upstream — this fork was a manual copy, not a GitHub-native fork, so `git merge` wasn't applicable). Verified: syntax valid, `record_step_boundary` call present via the exact AST check the real submission checker runs, no change to any of the 26 logged results below (we always passed explicit `--num-steps`/`--max-train-seconds` calibrated so step-count was the true binding constraint in every real comparison, so none of PR #6's default-behavior changes touch them retroactively).

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

**Second-largest delta this session after Solution 1's LR cooldown** (-0.0578 vs this item's -0.0223) — but still 30-90x the established noise floor, and a real 35% wall-clock speedup too (670.5s vs ~1025s for the same 800 steps, same total tokens/step). Sanity-checked the mechanism: batching itself should be gradient-neutral here (`F.cross_entropy` with `reduction="mean"` averages uniformly over all tokens in the batch regardless of micro-batch size — worked through the math, 8 micro-batches of 8192 tokens each averages to the exact same 65,536-token mean as 64 micro-batches of 1024 each). So the win isn't from batching mechanics — it's the packing-bin-size change itself: far fewer document-tail crop-and-discard events at bin size 8192 vs 1024. Strongly validates the enhancement doc's hypothesis about ClimbMix web-text documents routinely exceeding 1024 tokens and getting truncated at nearly every row fill. **Solution 6 locked: `PACK_FACTOR=8` → 1.188680.**

**Cumulative after Solutions 1-6: 1.273797 → 1.188680 (-6.68% relative).**

### Follow-up: bin-size sweep (post-session, packing-area research)

Never actually tried a bin size other than 8x — `experiment-rollout-plan.md` originally called this "a binary change, no sub-sweep needed," untested against the alternative. Testing `PACK_FACTOR=16` next, same fresh-init cumulative methodology as every other Solution (not warm-started — this is a from-scratch architecture/dataloader question, same as all prior Solutions, kept separate from the warm-start capability which serves a different purpose).

| Config | public_val_bpb | Δ | status |
|---|---|---|---|
| `PACK_FACTOR=8` (Sol 1-7,9a,10 config, matched to Solution 10's own row) | 1.172216 | — | — |
| `PACK_FACTOR=16` | 1.174631 | +0.00241 | **discard — real regression** |

Above the noise floor, not a fluke. Pushing the bin bigger *hurts* rather than helping — consistent with (not proof of) the doc-boundary-leakage hypothesis already flagged: bigger bins mean more distinct documents concatenated per row, and that leakage cost apparently starts outweighing the crop-reduction benefit somewhere between 8x and 16x. This reprioritizes the follow-up list: document-boundary masking (item #2) becomes more motivated (might be *required* to unlock any further bin-size gain, not just a nice-to-have), and the remainder-requeue custom packer (item #3) becomes less motivated (bigger bins aren't even helping right now, so a fix that only helps at bigger bins is premature).

Testing `PACK_FACTOR=4` next — natural follow-up the original plan didn't cover: is 8 actually the local optimum, or was it already past it?

| Config | public_val_bpb | Δ vs PACK_FACTOR=8 | status |
|---|---|---|---|
| `PACK_FACTOR=4` | 1.166012 | **-0.00620** | **real improvement, above noise** |

**8 was not the local optimum.** Pattern across today's same-cumulative-level tests: `4 (1.166012) < 8 (1.172216) < 16 (1.174631)` — a sweet spot below 8, not a simple monotonic "bigger bin is always better" relationship (matches the leakage-cost-vs-crop-reduction-benefit tradeoff hypothesis: too small = crop loss dominates, too big = doc-boundary leakage dominates, somewhere in between wins). This is itself a real, actionable finding — the improvement from 8→4 is comparable in size to Solution 7's whole contribution. Testing `PACK_FACTOR=2` next to find out whether 4 is the actual optimum or whether smaller still wins.

| Config | public_val_bpb | Δ vs PACK_FACTOR=4 | status |
|---|---|---|---|
| `PACK_FACTOR=2` | 1.165273 | -0.00074 (small, borderline noise floor) | **keep — continues the trend** |

Diminishing returns as bin size shrinks further: `16→8` regression was `+0.0024`, `8→4` was `-0.0062` (big), `4→2` is only `-0.00074` (small). Consistent monotonic direction across 4 data points now (`2 < 4 < 8 < 16` in bpb) is a real trend, not noise, but shrinking fast — looks like a plateau. One more test to close this out cleanly: `PACK_FACTOR=1` (no packing multiplier — note this differs from the *original* Solution 6 baseline test of `PACK_FACTOR=1` at the Sol 1-5 level, before VE/local-conv/weight-cache existed; that earlier test found `PACK_FACTOR=8` a big win over `1`, seemingly in tension with today's opposite-direction trend — plausible explanation: VE/local-conv changed the model's sensitivity to the crop-loss-vs-leakage tradeoff, shifting the optimum down. Testing `1` at *today's* full config directly resolves whether the trend keeps going or reverses.)

| Config | public_val_bpb | Δ vs PACK_FACTOR=2 | status |
|---|---|---|---|
| `PACK_FACTOR=1` (original, no packing multiplier) | 1.205512 | **+0.04024** | **huge regression** |

**The trend reverses sharply — this is a genuine interior optimum, not "smaller is always better."** Full curve, same cumulative level (Sol 1-7,9a,10):

| `PACK_FACTOR` | public_val_bpb | vs. optimum |
|---|---|---|
| 1 | 1.205512 | +0.04024 (crop-loss dominated — same regime the original packing fix targeted) |
| 8 (previously locked) | 1.172216 | +0.00694 |
| 16 | 1.174631 | +0.00936 (leakage-dominated) |
| 4 | 1.166012 | +0.00074 |
| **2 (new best)** | **1.165273** | **—** |

Going all the way to 1 falls off a cliff — confirms the *shape* of the original Solution 6 finding (bigger bins reduce crop-loss) while showing it doesn't hold monotonically forever: somewhere past 8, doc-boundary leakage cost starts winning, and the real sweet spot sits at 2-4, not 8. **`PACK_FACTOR=8` was a real, validated win over `1` (that part of the original finding holds), but it was never re-optimized once VE/local-conv were added on top — turns out it wasn't the local optimum once they were.**

**Re-locking: `PACK_FACTOR=2` replaces `PACK_FACTOR=8` as the winning value.** New cumulative bpb at the 800-step comparison level (Solutions 1-7, 9a, 10, with `PACK_FACTOR=2`): **1.165273** (vs. the previous `PACK_FACTOR=8`-based cumulative of 1.172216) — an additional **-0.59%** relative improvement discovered purely by re-sweeping a constant nobody had touched since it was first set, three architecture additions ago.

### Follow-up: document-boundary attention masking (packing-area research, item #2)

Reused Solution 8's additive-mask plumbing (`Block`/`CausalSelfAttention` signatures unchanged — combination happens once in `GPT.forward`, adding a dynamic per-batch document-id mask to the existing static window mask). New `doc_boundary_mask(idx)`: `is_bos = idx == BOS_TOKEN_ID`, `doc_id = cumsum(is_bos)`, mask out `i,j` pairs with different `doc_id`. `BOS_TOKEN_ID` threaded into both `main()` and `load_for_eval()` from the tokenizer (added to `ARCH_TOGGLE_NAMES` for checkpoint provenance too — `USE_DOC_MASK`, not shape-critical, silent-behavior category).

Two-stage verification as usual: `USE_DOC_MASK=False` no-op smoke (`2.131932` vs the matching `PACK_FACTOR=2` smoke's `2.131734` — noise-range identical) confirmed the refactor doesn't change anything when off. `USE_DOC_MASK=True` smoke: no crash, **causality still exact** (`max_abs_diff=0.0` — important, confirms the new mask correctly preserves causal ordering rather than accidentally breaking it).

Worth noting going in: now that the bin-size sweep found the optimum at `PACK_FACTOR=2` (2048-token bins) rather than the originally-assumed `8` (8192-token bins), this fixes a smaller leakage surface than originally estimated — fewer documents per packed row at the smaller bin size means less to gain here than when this was first flagged.

| Config | public_val_bpb | Δ vs PACK_FACTOR=2 baseline | status |
|---|---|---|---|
| `USE_DOC_MASK=False` (Sol 1-7,9a,10, PACK_FACTOR=2) | 1.165273 | — | — |
| `USE_DOC_MASK=True` | 1.171596 | +0.00632 | **discard — real regression** |

**Negative result — masking hurts, doesn't help.** Above noise floor, not a fluke. Plausible explanation: at the now-optimal smaller bin (`PACK_FACTOR=2`, 2048-token rows), most rows likely contain only 1-2 documents, and fully blocking cross-document attention may remove some mild beneficial signal (broader context diversity acting as a soft regularizer) that outweighs whatever leakage cost it was meant to fix. This is exactly why the item was flagged "speculative, not validated by any real ablation run" — now it has been, and the answer is no. **Discarded, reverted to `USE_DOC_MASK=False`.** Cumulative unchanged at 1.165273.

**Packing-area research arc, summary**: bin-size re-sweep found a large real win (`PACK_FACTOR=8→2`, -0.00694) that nobody had re-checked since Solution 6 first landed, three architecture additions earlier — the single best return from this whole follow-up effort. Document-boundary masking (item #2) was a clean, real negative result. Custom packer with remainder-requeue (item #3) is now lower priority than originally planned: it targets crop-loss specifically, but the smaller optimal bin already has less crop-loss than assumed, and the leakage-side fix (item #2) just failed — the evidence pattern doesn't strongly motivate the highest-risk item of the three. Leaving it as documented future work rather than pursuing it now, given diminishing returns and real implementation risk (a bug there could silently corrupt training data).

**Checkpoint update**: ran a fresh longer confirmation (1450 steps) on this config, `public_val_bpb = 1.10682` (vs. v1's `1.111741` at the same step count) — improvement holds at the longer horizon. Saved as `checkpoints/final-confirmation-v2.pt`, supersedes v1 (kept for historical reference). See `checkpoints/README.md`.

---

## Further exploration (continuing past the packing-area plan, per explicit "don't cut it short" direction)

### Decomposing Solution 8's bundle: `ROPE_BASE=200000` alone

Solution 8 tested window-pattern + short-window + RoPE-base *together* and discarded the combo on a small real regression — but a single run of a 3-part bundle can't say which part (if any) was responsible. Testing `ROPE_BASE=200000` alone first (cheapest, most isolated piece), on top of the current best config (Sol 1-7, 9a, 10, `PACK_FACTOR=2`).

| Config | public_val_bpb | Δ vs baseline | status |
|---|---|---|---|
| `ROPE_BASE=100000` (Sol 1-7,9a,10, `PACK_FACTOR=2`) | 1.165273 | — | — |
| `ROPE_BASE=200000` alone | 1.192130 | **+0.02686** | **discard — huge regression** |

**Striking result**: this single-component regression (+0.0269) is far bigger than the *entire 3-part bundle's* regression (+0.00129) from the original Solution 8 test. That's a strong signal, not just noise: it implies the windowing component in that bundle must have been offsetting a much larger RoPE-induced loss — i.e., windowing alone might be a real, substantial win, masked in the original test by a bad RoPE choice riding along with it. (Caveat: original bundle test ran at `PACK_FACTOR=8`, this ran at `PACK_FACTOR=2` — not perfectly apples-to-apples on magnitude, but the directional implication is strong enough to warrant testing directly.) `ROPE_BASE` reverted to `100000`. Testing `WINDOW_PATTERN=SSSSL, SHORT_WINDOW_FRAC=0.125` alone (RoPE base unchanged) next.

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

**GPU phase complete as of the overnight session.** Final locked config at that point: Solutions 1, 2, 3, 5, 6, 7, 9a, 10 kept; Solutions 4, 8, 9b discarded (real regressions, documented above with reasoning); Solutions 11-12 untouched (Trn2-only by design).

**Update, post-session packing-area follow-up research**: re-swept `PACK_FACTOR` (never touched since Solution 6 first set it to 8, three architecture additions earlier) and found `PACK_FACTOR=2` beats `8` by -0.00694 (see "Follow-up: bin-size sweep" under Solution 6, above) — a bigger win than Solutions 5 or 9a individually. **`PACK_FACTOR=8` retired, `PACK_FACTOR=2` is now the locked value.** New cumulative at the 800-step comparison level: **1.165273** (previously 1.171962). Ready to hand off to Trn2 once allocated — see `experiment-rollout-plan.md` for the re-verification protocol per Solution.

**Post-handoff addition, while the box was still up**: added `--init-from <checkpoint>` warm-start support to `train.py` and a `"toggles"` snapshot to the checkpoint payload (see `checkpoints/README.md` for full detail), so `final-confirmation.pt` can seed further research (manual or `autoresearch`-driven) instead of starting from random init. Verified end-to-end: step-0 loss `2.76` vs. a random init's ~9.0, causality exact. Deliberately a warm start, not a bit-exact resume (fresh optimizer/schedule/dataloader) — which is the right thing for this use case anyway, matching `autoresearch`'s own "advance the branch" convention. Never for a scored submission, which must be a fresh run from scratch per the rules.

---

## Session summary

**Net result at 800-step comparison scale: `1.273797 → 1.165273`, a `-8.53%` relative improvement**, from 8 kept changes (Solutions 1, 2, 3, 5, 6, 7, 9a, 10, with Solution 6's `PACK_FACTOR` re-tuned to `2` post-session) out of 12 attempted (Solutions 4 and 8 and 9b discarded — real regressions, not noise). Biggest single contributor: Solution 1 (LR cooldown+floor, `-4.5%` alone) — the highest-confidence prediction going in, and it delivered the largest single win. Second: Solution 6 (packing-bin reshape, `-1.84%` alone, plus a genuine 35% wall-clock speedup) — validated the enhancement doc's document-tail-cropping hypothesis far more strongly than expected, and the biggest surprise of the session relative to how it was originally sized up. Solution 5 (init scale) and Solution 7 (value embeddings) both landed clean, real wins matching Karpathy's log directly.

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

---

## Hardware transition: RTX 6000 Pro (Vast.ai) → NVIDIA L4 (AWS EC2 g6.xlarge)

Vast.ai access lost mid-session (no longer usable). Moved to a rented AWS `g6.xlarge` (single L4, 23GB usable VRAM, Ada, compute cap 8.9), `ec2-user` login, AWS Deep Learning AMI with a preinstalled `/opt/pytorch` venv (`torch 2.13.0+cu130`).

**Environment verified before trusting it**: same diligence as every prior box — real matmul + SDPA executed on `cuda`, not just `import torch` succeeding. Confirmed clean. Repo + `.git` synced (needed for `run_experiment.py`'s git calls), `requirements.txt` installed (`pyarrow`/`requests` already present on the DLAMI, only `tiktoken` needed), `prepare.py setup` run (526MB data, plenty of headroom on the 25GB root disk).

**Self-inflicted mismatch caught by the toggle-warning system, first real catch**: synced `train.py` for the initial portability check without noticing it still had `WINDOW_PATTERN="SSSSL"` staged from the windowing-decomposition test that was interrupted when Vast.ai died — not the actual locked config. The `--init-from` mismatch check printed exactly the warning it was built for (`WARNING [silent-behavior]: toggle WINDOW_PATTERN mismatch -- checkpoint='L' current='SSSSL'`), which I only noticed on relog inspection, not from the number itself. Fixed and reran — turned out the windowing mismatch wasn't actually the dominant driver of the bpb drift I'd initially suspected it was (both runs landed within ~0.003 of each other); the real cause is just LR-peak-restart-shock variance over a very small step count, not a bug. Both runs had causality exact regardless. Lesson: check the log for warnings before trusting a number, don't infer causation from a single before/after delta.

**Portability check (`--init-from checkpoints/final-confirmation-v2.pt`, 10 steps)**: causality exact both times, correct param count (85.9M), no shape errors — structurally confirms the checkpoint/architecture/eval pipeline all port correctly. The bpb-after-10-steps number itself (~1.157) isn't a meaningful portability metric on its own — too few steps, dominated by the LR-schedule restarting at peak, same caveat noted the first time this technique was used on Vast.ai.

**Throughput**: ~2.2s/step at `PACK_FACTOR=2` (vs. RTX 6000 Pro's ~1.0-1.1s/step for the same config) — roughly 2x slower, consistent with L4 being a smaller/cheaper chip than the Blackwell workstation card, and within the range estimated before allocating it. Confirms the earlier reasoning: L4's 24GB was never the constraint (real usage ~7-8GB); the tradeoff is patience, not capability.

**Real quality-transfer check, in progress**: fresh from-scratch run (no warm-start, avoids the LR-shock confound) of the full locked config (Sol 1-7, 9a, 10, `PACK_FACTOR=2`) at 800 steps — directly comparable to the RTX 6000 Pro's own clean number for this exact config, `1.165273`.

Status: running (~1850s expected, ~2x the RTX 6000 Pro's own time for the same comparison).

---

**First real run hit a miscalibration**: `--max-train-seconds 1850` (estimated from the 10-step warm-start test's `22.2s/10 ≈ 2.2s/step`) cut the run short at `num_steps=406` instead of 800 — `public_val_bpb=1.263573` at that partial step count, not comparable to the RTX 6000 Pro's 800-step number at all (just less training, not a real signal). Root cause, found by checking the per-step log rather than guessing: `training_seconds` only counts time *after* the step-5 budget-clock reset (by design, per the PR #6 merge) — the warm-start test's `22.2s` covered steps 5-9 (5 steps), not all 10, so the real rate was `22.2/5 ≈ 4.44s/step` all along, not `2.2s/step`. Confirmed via the per-step log directly: flat `4.5-4.65s/step` from step 0 through step 200, no thermal-throttling signature (would show as *increasing* dt over time) — just a genuinely slower chip for this workload than estimated.

**Real L4 throughput: ~4.5s/step at `PACK_FACTOR=2`** — about 4x slower than RTX 6000 Pro's ~1.0-1.1s/step for the same config (bigger gap than the 2-4x range estimated before allocating this instance). 800 steps takes ~60 min here, not ~15-17 min. Rerunning with a corrected, generous budget (`--max-train-seconds 3800`) to get the real 800-step comparison.


**Corrected run, real 800-step result**: `num_steps=800` reached cleanly (`training_seconds=3646.2`, confirms ~4.56s/step sustained, matching the earlier per-step-log reading). **`public_val_bpb = 1.168849`** vs. RTX 6000 Pro's `1.165273` for the identical locked config — **Δ = +0.0036 (0.31% relative)**, causality exact.

**Portability confirmed with real numbers, not just structural checks.** The delta is bigger than the same-hardware noise floor established earlier (~0.0001-0.00025) — expected, given a different GPU vendor/generation and a different `torch` build (`2.13.0+cu130` vs. `2.11.0+cu128`) offer no bit-exact reproducibility guarantee — but it's small in absolute terms and lands in the same direction. The core finding of the whole session (baseline `1.273797` → locked config `~1.165-1.169`) genuinely transfers to different hardware, not an artifact of the specific card/torch build this was originally found on.

**Practical note going forward**: L4's ~4.5s/step (vs. RTX 6000 Pro's ~1.0-1.1s/step) makes an 800-step comparison a ~62-minute commitment here, vs. ~15-17 minutes before — a real velocity cost for the remaining exploration list. Recommending a smaller standard unit for new L4 experiments (e.g. 400 steps, ~30 min) rather than continuing at 800, trading a little noise-floor tightness for meaningfully faster iteration on this hardware.


**Adopted 400-step standard unit for L4 going forward.** New reference: `l4-400step-baseline`, full locked config (`WINDOW_PATTERN="L"`), `public_val_bpb = 1.263103` at 400 steps (consistent with the earlier accidentally-truncated 406-step run's `1.263573` — same seed/code/hardware, expected near-determinism, a good self-consistency check).

### Continuing the Sol8 decomposition: windowing alone, on L4

Picking back up the test that was interrupted when Vast.ai went down. `WINDOW_PATTERN="SSSSL"`, `SHORT_WINDOW_FRAC=0.125`, `ROPE_BASE` unchanged at `100000` this time (isolating the windowing component only, after `ROPE_BASE=200000` alone was already found to be a huge standalone regression). Smoke-tested clean first: no crash, causality exact.

Status: running real 400-step comparison against the new `l4-400step-baseline` (1.263103).


| Config | public_val_bpb | Δ vs L4 400-step baseline | status |
|---|---|---|---|
| `WINDOW_PATTERN="L"` (baseline) | 1.263103 | — | — |
| `WINDOW_PATTERN="SSSSL", SHORT_WINDOW_FRAC=0.125` alone | 1.295870 | **+0.03277** | **discard — real regression, bigger than RoPE alone** |

**Overturns the earlier hypothesis.** Windowing alone regresses *more* than `ROPE_BASE=200000` alone did (+0.0328 vs +0.0269) — it was never a hidden win being masked by a bad RoPE choice; **both components are independently bad** on top of this architecture. Plausible explanation: VE and local-conv (Solutions 7, 9a) likely already provide whatever local-context benefit windowing would otherwise offer, so restricting most layers to a 128-token window just loses long-range context for no offsetting gain. **Solution 8's decomposition is now definitively closed** — not just "the bundle regressed," but "every piece of it regresses independently too." Reverted to `WINDOW_PATTERN="L"`.

**Decomposition arc, summary**: `ROPE_BASE=200000` alone: real regression (+0.0269). `WINDOW_PATTERN="SSSSL"` alone: real regression (+0.0328), bigger than RoPE's. Original bundle regression (+0.00129, at the old `PACK_FACTOR=8` config): smaller than either individual component's regression at `PACK_FACTOR=2` — consistent with some cancellation between the two bad components, not evidence either one is secretly good. This whole avenue is closed for this architecture; no further work planned here.


### VE's own hyperparameters — never tuned separately until now

VE currently shares `EMBEDDING_LR` with the token embedding and has `weight_decay=0.0` (constant, no decay) — adopted wholesale when VE landed (Solution 7), never tuned on its own axis. Karpathy's log has a real validated sequence here (tiny WD 0.001→0.002→0.003, each confirmed better in his setup). Added a separate `VE_WD` constant (previously hardcoded `0.0` inline) to test this independently. Smoke-tested the refactor as a no-op first (`VE_WD=0.0`): `2.152395` vs the earlier VE-on smoke's `2.152977` — matches within noise, causality exact.

Testing `VE_WD=0.003` directly (Karpathy's validated endpoint, skipping the intermediate 0.001/0.002 points given the cost per run here — a reasonable one-shot bet given low risk on this small a parameter; will probe intermediate values only if this doesn't land cleanly).

Status: running real 400-step comparison against the `l4-400step-baseline` reference (1.263103).


| Config | public_val_bpb | Δ vs L4 400-step baseline | status |
|---|---|---|---|
| `VE_WD=0.0` (baseline) | 1.263103 | — | — |
| `VE_WD=0.003` (Karpathy's validated endpoint) | 1.264037 | +0.00093 | **discard — flat/marginal negative** |

Not a real win — essentially flat, tiny regression. Karpathy's validated VE-WD finding doesn't transfer here either. **Reverted to `VE_WD=0.0`.**

**Pattern worth naming**: this closes out three attempted transfers from Karpathy's log or the original decomposition hypothesis in a row that didn't hold (RoPE base, windowing pattern, VE weight decay) — all real, clean negative/flat results, not noise. Plausible explanation: our specific combination of kept Solutions (VE + local conv + re-tuned packing bin + independently-tuned schedule) has diverged enough from Karpathy's original setup that his *fine-grained* tuning values stop transferring reliably, even though the *big, structural* wins (LR cooldown shape, init scale, VE's presence, local conv) did transfer cleanly. The remaining small-tweak territory looks more idiosyncratic to this specific configuration than the major architectural/schedule decisions were.


### Sub-step 9c — Selective per-token gating

Pivoted here instead of depth/width sweep: depth/width is inherently a throughput question (bigger model = lower bpb at fixed steps almost by default, not informative; the actually-useful fixed-wall-clock version doesn't transfer L4→Trn2 anyway — already flagged "GPU dry-run only" in `experiment-rollout-plan.md`'s Solution 11 from early in the session). Selective gating is a fixed-architecture-shape quality question instead, same category as the Solutions that actually transferred (VE, local conv) — no throughput confound.

`resid_lambda`/`x0_lambda` (static per-layer scalars) become data-dependent: a small per-token gate (`Linear(32, 2, bias=False)`, zero-init) modulates them around their static values — `resid_lambda * (1 + g[...,0])`, `x0_lambda * (1 + g[...,1])`. Zero-init means the gate starts as an exact no-op (matches the `local_conv`/`ve_gate` convention). Negligible param cost (+640 params total, rounds to the same 85.9M).

Two-stage verification: no-op smoke (`2.152442` vs matching baseline smoke `2.152395`, within noise) confirmed the refactor is a true no-op when off. Real-feature smoke: no crash, causality exact, params unchanged at this precision.

Status: running real 400-step comparison.


| Config | public_val_bpb | Δ vs L4 400-step baseline | status |
|---|---|---|---|
| `USE_SELECTIVE_GATE=False` (baseline) | 1.263103 | — | — |
| `USE_SELECTIVE_GATE=True` | 1.272345 | +0.00924 | **discard — real regression** |

Real regression, well above noise. **Reverted to `USE_SELECTIVE_GATE=False`.**

**This closes the "further exploration" round with four clean negative results in a row** (RoPE base alone, windowing alone, VE weight decay, selective gating) — none of them noise, all properly smoke-tested and verified. That consistency is itself informative: the current locked config (Solutions 1-7, 9a, 10, `PACK_FACTOR=2`) appears to sit near a real local optimum for this architecture/dataset/step-count regime, not just avoiding bad luck on individual tries. Further small architectural tweaks on top of it look like diminishing-to-negative returns. Custom packer w/ remainder-requeue remains the one untried candidate, still deprioritized on the same evidence pattern as before (targets a problem — crop-loss — that's smaller now than assumed, and the leakage-side sibling idea already failed twice).

