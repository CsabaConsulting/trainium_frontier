# Leaderboard gap analysis and "what are the other teams doing" — open questions for research

Written 2026-09-13, from a conversation, not yet acted on. **This does not change the current
plan** (`.claude/plans/valiant-wandering-pascal.md` — capacity/throughput track stays the main
push). This file exists to hand the "how are others ahead of us, and is there a simple
formula/path we're missing" question to a dedicated research pass, separate from the
box-experiment track.

## 1. Where we actually stand (verified numbers, not guesses)

- **Our board score**: 1.0384 (sub-c23baa39, 2026-09-12 1:05 PM), from the T3.1 last-layer n-gram
  VE config. Predicted 1.039 beforehand — off by 0.0006, our tightest calibration match yet.
- **Public leaderboard** (snapshot given by the user, 2026-09-12): #1 0.9683, #2 0.9692,
  #3 0.9733, #4 0.9747, #5 0.9863, #6 0.9898, #7 0.9943, #8 0.9945, #9 1.0069, #10 1.0303.
  This will have moved since — it's a live race, several entries updated same-day as the
  snapshot. Treat as approximate, not current.
- **Distance**: to 10th (1.0303) — 0.0081, plausibly one more real win (the queued n-gram-layers
  cost-reduction alone is estimated ~0.007). To 1st (0.9683) — 0.0701, needs several more
  structural wins.
- **Full submission history** (all real, all ours): 1.1811 → 1.1590 → 1.1394 → 1.1495 (the
  `NUM_STEPS=1000` bug) → 1.0548 (bug fixed) → 1.0384 (T3.1 win). Each step is a real, verified,
  hardware-confirmed cause — see `research/trn2-experiment-results.md` and
  `research/submission-gap-analysis-2026-09-10.md`.

## 2. Reference points that are NOT this competition, but are the closest public comparisons

All three of these are the **nanochat autoresearch** task (Karpathy's newer codebase, not the
classic 2019/2022 nanoGPT repo), same `val_bpb` metric, same ClimbMix-lineage data family as us
(see §3), but a **different hardware/time/shape regime**: B200 GPU, 5 minutes, 10-seed mean,
32,768-vocab tokenizer (ours is 8,192), DEPTH=8×768-wide with 6 heads (ours is DEPTH=6×640-wide
with 5 heads), FlashAttention-4, everything bf16.

| Solution | val_bpb |
|---|---|
| Karpathy seed `train.py` | 0.9751 |
| Karpathy's overnight-tuned solution (commit `438a26e`) | 0.9646 |
| autoresearch@home community best | 0.9372 |
| Recursive `optimized_from_vanilla.py` (from a plain transformer) | 0.9344 |
| **Recursive `optimized_from_karpathy.py`** (code public) | **0.9109** |
| Aster (code not public; 3 changes described) | 0.9098 |

Source: `github.com/recursive-org/first-steps-toward-automated-ai-research`, cited in full in
`research/frontier-research-plan-2026-09-09.md` §11.

**Why this matters**: the current #1 on OUR leaderboard (0.9683) sits almost exactly at
Karpathy's own *fully public, thoroughly documented* tuned baseline (0.9646) — not at Recursive's
improved 0.9109. That's reassuring in one sense (no need to assume a secret trick to explain the
top of the board) and a real target in another (there is a well-documented ~0.05-0.07 bpb of
further improvement, from Recursive/Aster's techniques, that nobody on our board has clearly
banked yet — or if they have, they haven't shown it in a score, or their board number is not
directly comparable to the B200/32k-vocab numbers above for other reasons, e.g. our 8192-vocab
scorer or the 30-minute/1-chip budget).

**Open question 1**: is it plausible any team here has already ported *most* of Recursive/Aster's
full recipe (not just n-gram VE, which we now have) — batch-size ramp done correctly for their
setup, per-depth LR/momentum scaling, a Demon β1 schedule, NorMuon β2 ramp, K=3 routed output
heads — and that's the realistic explanation for anyone below ~0.99? We ported the single biggest
piece (n-gram VE, -0.015) but not the rest; some of what we tried from that list (batch ramp, EMA)
came back negative *on our hardware/regime*, which doesn't mean it's negative for a team running
a different shape/budget.

## 3. Corpus identity — checked, not assumed

`prepare.py:52`: `BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/..."`
— this is **Karpathy's own hosted shuffle of ClimbMix**, confirmed in `README.md:322` too
("the ClimbMix dataset (publicly available, shuffled web text)"). So our data and the nanochat
reference runs above are the same source family, likely the same shuffle. The real difference is
the *slice and amount*: we get exactly 5 train shards + 1 pinned eval shard of 6543 total, in a
30-minute/1-chip budget (~288M tokens, ~0.9 epochs); the nanochat comparisons use a different
slice and a very different compute/time budget. Not "different data," a different cut of it.

## 4. Our own GPU-proxy-phase numbers, for context (not comparable to the above)

- RTX 6000 Pro Blackwell Max-Q (Vast.ai), our own locked config @ 800 fixed steps:
  `public_val_bpb = 1.165273`.
- AWS L4 (g6.xlarge), identical config @ 800 fixed steps: `public_val_bpb = 1.168849`
  (Δ +0.0036, causality exact — L4 matched quality, ~4x slower per step).

**Why these are not informative about the Karpathy/Recursive gap**: this was our *own* early,
unoptimized checkpoint (384-wide/86M params) — before `torch.compile`, before the Muon NKI kernel,
before the T0.4 hyperparameter resweep, before T3.1's n-gram VE. Essentially none of this
competition's real wins existed yet. It also measures a fixed step count (800), not a fixed
wall-clock budget, so it isn't even the same kind of number as a "5 minutes" comparison. Three
confounds are stacked in any naive comparison of `1.165` to `0.9646`: hardware tier (B200 ≫ H100 >
RTX 6000 Pro Max-Q > L4, roughly, and Max-Q specifically means a power/thermal-capped workstation
variant), recipe maturity (early snapshot vs. Karpathy's fully-tuned final recipe), and
measurement basis (fixed-step vs. fixed-wall-clock). None of this is about NKI vs. CUDA kernel
style specifically, even though that's a real, separately-measured effect (see §5).

## 5. Kernel-fusion style (NKI vs. CUDA) — a real, separately-measured effect, smaller than the confounds above

We run at **~8% MFU** on Trainium (dispatch-bound: one NEFF per op, no cross-op fusion by default
in the native eager backend) vs. GPU+CUDA with flash-attention and mature fused kernels commonly
reaching 30-50%+ MFU. This is real and well-documented in `research/trn2-experiment-results.md`
and is *why* so much of this competition's effort went into `torch.compile`, the Muon NKI kernel,
and the rope/softcap NKI kernels. But it's a smaller-magnitude, separately-characterized effect
from the GPU-proxy-vs-Karpathy gap above — don't conflate the two when reasoning about either.

## 6. The excluded startup window — could another team be using it to train?

Already researched by us (`research/frontier-research-plan-2026-09-09.md` T4.3) and discussed
live. Facts, not speculation:
- The window is capped at 900s, real compile times measured 40-150s here — bounded slack, same
  cap for every team.
- README explicitly bans "exploitation of the evaluation infrastructure," and top entries get
  **manual human review**, not just the automated LLM check — the teams most likely to be
  reviewed are exactly the ones at the top of the board, which is a real disincentive against
  disguising training as "compile warmup" there, not just an assumption that everyone's honest.
- Bounded to a few hundred seconds even if pushed hard — not large enough alone to explain a
  0.07-0.08 bpb gap.

**Open question 2**: is there a way to get *any* real signal on this (forum posts, other teams'
public writeups, devpost activity) without assuming bad faith, before ruling it fully out as a
factor for any specific team?

## 7. What this file is asking a research pass to do

Not to change the current plan. To think about, with fresh eyes and possibly new web research:
1. Is there a simple, well-evidenced "formula" or short list of techniques that plausibly explains
   most of the gap to the top of the board, beyond what's in §2's table? (E.g., is there a newer
   nanochat-autoresearch or modded-nanogpt record we haven't mined yet that's bigger than anything
   currently in our queue?)
2. Any public information (forum posts, team writeups, devpost activity, GitHub repos) about what
   specific techniques other Trainium Frontier teams — not the nanochat/GPU reference teams, the
   actual competitors on *this* leaderboard — have used. Treat with appropriate skepticism; we
   have no way to verify unverified claims, and don't assume bad faith without evidence.
3. Whether the confound analysis in §4 changes if we ever run our *current* (mature) recipe on a
   GPU for a real apples-to-apples comparison — is that worth doing at all, or is it purely
   academic curiosity given we can only submit from Trn2?
4. Anything else in the Recursive/Aster/Karpathy playbook (§2) not yet cross-checked against our
   own `research/frontier-research-plan-2026-09-09.md` §11.1/§11.2 lists, in case something was
   missed the first time.

## 8. Sources

- `research/frontier-research-plan-2026-09-09.md` §11 (Recursive/Aster table and recipe detail).
- `research/gpu-experiment-results.md` (RTX 6000 Pro and L4 numbers, hardware-transition section).
- `research/trn2-experiment-results.md` (T3.1 win, MFU/dispatch-overhead measurements, submission
  history context).
- `research/submission-gap-analysis-2026-09-10.md` (the NUM_STEPS bug root-cause).
- `README.md` (data source, rules on evaluation-infrastructure exploitation, manual review for top
  entries), `prepare.py` (ClimbMix source URL).
- This conversation, 2026-09-13.
