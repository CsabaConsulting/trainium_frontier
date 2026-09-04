# Checkpoints — GPU proxy session (RTX PRO 6000 Blackwell Max-Q, Vast.ai)

Saved before releasing the rented box. Plain PyTorch `.pt` files (via `torch.save`/`torch.load(weights_only=True)`, matching `train.py`'s own `save_checkpoint`/`load_for_eval` convention already in use — not converted to safetensors, since these are private single-user artifacts only ever loaded by our own `train.py`, and `weights_only=True` already gives the safe-loading guarantee safetensors would otherwise be for).

Pulled back out of many more checkpoints generated across the sessions — every other config is either superseded, a smoke test, or a discarded Solution, and is already fully described (exact constant values) in `../research/gpu-experiment-results.md`, cheap to reproduce from scratch if ever needed again.

## `final-confirmation-v2.pt` — current best, use this one

Supersedes `final-confirmation.pt` below. Same Solutions 1-7, 9a, 10 as v1, but with the packing-area follow-up research applied: `PACK_FACTOR` re-swept and re-locked from `8` to `2` (real win, -0.00694 at the 800-step comparison scale), document-boundary attention masking tried and discarded (real regression). Trained 1450 steps, `public_val_bpb = 1.10682` (vs. v1's `1.111741` at the same step count — improvement holds at this longer horizon too), causality exact.

**Loads correctly against the `train.py` currently checked into this repo** — module-level constants match exactly what produced it (`PACK_FACTOR=2`, `USE_DOC_MASK=False`, everything else same as v1 below).

## `final-confirmation.pt` — superseded, kept for historical reference

The final locked cumulative config from the overnight session, before the packing-area follow-up: Solutions 1, 2, 3, 5, 6, 7, 9a, 10 kept (Solutions 4, 8, 9b discarded — real regressions, see the results doc). Trained 1450 steps, `public_val_bpb = 1.111741`, causality exact.

**Will NOT load against the current `train.py` as checked in** — it used `PACK_FACTOR=8` (now `2`) and predates `USE_DOC_MASK` entirely (harmless mismatch, not shape-critical, since that toggle adds no new state_dict keys — but it's still not the winning config, don't use this one for anything except historical comparison). Its module-level constants otherwise: `WARMDOWN_RATIO=0.75`, `FINAL_LR_FRAC=0.05`, `WARMUP_STEPS=0`, `WD_DECAY_TO_ZERO=True`, `INIT_SCALE=0.68`, `USE_VE=True`, `USE_LOCAL_CONV=True`, `USE_WEIGHT_CACHE=True`, `WINDOW_PATTERN="L"`, `USE_GATED_MLP=False`. Self-documented — see "Warm-start support" below.

## `baseline.pt`

Unmodified baseline, trained 800 steps, `public_val_bpb = 1.273797`, for A/B reference.

**Will NOT load against the current `train.py` as checked in.** It was trained with every toggle at its original off/default value (`WARMDOWN_RATIO=0.0`, `WARMUP_STEPS=20`, `PACK_FACTOR=1`, `INIT_SCALE=1.0`, `USE_VE=False`, `USE_LOCAL_CONV=False`, `WINDOW_PATTERN="L"` — same as now, `USE_WEIGHT_CACHE=False`) — since the architecture toggles live in `train.py`'s module constants, not in the saved payload, `load_state_dict(strict=True)` will fail on a shape/key mismatch (e.g. no `value_embeds`/`local_conv` weights in this state dict) unless those constants are reverted to their originals first. Trying to `--init-from` this one against the current file will now get a loud `WARNING [SHAPE-CRITICAL]` per mismatched toggle before it hard-errors on the actual `load_state_dict`.

## Warm-start support (added after the initial handoff)

`train.py` now has an `--init-from <checkpoint_path>` flag (`warm_start_from()`), and every checkpoint's payload carries a `"toggles"` dict — a snapshot of all `ARCH_TOGGLE_NAMES` at save time (`PACK_FACTOR`, `INIT_SCALE`, `USE_VE`, `WINDOW_PATTERN`, `SHORT_WINDOW_FRAC`, `ROPE_BASE`, `USE_LOCAL_CONV`, `LOCAL_CONV_KERNEL`, `USE_GATED_MLP`, `USE_WEIGHT_CACHE`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`, `WARMUP_STEPS`, `WD_DECAY_TO_ZERO`, `MUON_MOMENTUM_WARMUP_STEPS`). On load, every saved toggle is compared against the live one; mismatches print a `WARNING` tagged `[SHAPE-CRITICAL]` (would also crash `load_state_dict(strict=True)` — `USE_VE`/`USE_LOCAL_CONV`/`USE_GATED_MLP`) or `[silent-behavior]` (loads fine, quietly wrong — everything else, since those live in non-persistent buffers/plain constants excluded from `state_dict()` entirely).

Both `.pt` files here were patched (not retrained) with the correct toggle snapshot for the exact config that produced them, matching what's documented above.

**This is a warm start, not a bit-exact resume**: fresh optimizer (no Muon/AdamW momentum), fresh `step=0`, fresh LR/WD schedule (restarts the warmup/cooldown curve from the top), fresh dataloader position. Verified end-to-end on the GPU box before release: `--init-from checkpoints/final-confirmation.pt`, step 0 loss `2.76` (vs. a random-init's ~9.0 — confirms the weights genuinely loaded), causality exact, `public_val_bpb=1.1198` after 10 more steps (close to the checkpoint's own `1.1117` — the small bump is the LR schedule restarting at peak rather than continuing the cooldown tail, exactly the expected warm-start-not-resume behavior).

This is deliberately *not* a bit-exact resume (no optimizer/RNG/dataloader state saved) — for the intended use (starting new research/`autoresearch` exploration from this checkpoint's weights rather than random init), a fresh optimizer is the *correct* thing anyway, same convention `autoresearch` itself uses ("advance the branch"). And to be explicit: this warm-start path is for our own dev iteration only — never for an actual scored submission, which must always be a fresh run from scratch per the competition rules.
