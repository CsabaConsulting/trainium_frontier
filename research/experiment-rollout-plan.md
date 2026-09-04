# Experiment Rollout Plan — H100 First, Then Trainium

This is the execution companion to `enhancement-opportunities.md`. That doc establishes *what* to change and *why* (rationale, evidence, portability bucket A/B/C) — this doc only adds *sequencing, measurement protocol, and migration notes*. Rationale/evidence for each item lives in the source doc; follow the `#N` cross-references there, not here.

**Test harness**: every H100 step below means running the actual competition `trainium_frontier/train.py` / `prepare.py` with the CUDA device path (`detect_device_type` already checks `torch.cuda.is_available()` first — `train.py:73`). This is **not** the separate `autoresearch/` single-NeuronCore dev-loop script, which is a different, smaller file with its own baseline. Using the wrong one invalidates the comparison.

**Environment gap on a fresh H100 box**: `requirements.txt` has zero Neuron-only pins, but its header comment says it's layered on top of a Trainium DLAMI that *already provides torch* — a plain H100 instance needs `pip install torch` (CUDA build) done separately first, then `pip install -r requirements.txt`. No other code changes needed to run the unmodified baseline; imports in `train.py`/`prepare.py` are all plain PyTorch, nothing unconditionally Neuron/XLA-specific.

**Run every experiment through `research/run_experiment.py`**, not `train.py` directly — it wraps the run, parses `train.py`'s own final summary block, and appends one row to a persistent, git-committed `research/h100-experiment-log.jsonl` (durable across instance loss — see spot-durability note below). Example: `python research/run_experiment.py --label sol1-warmdown75 --solution 1 --description "warmdown=0.75 final_lr_frac=0.05" --num-steps 800 --max-train-seconds 900`.

**Protocol — both axes are native to `train.py`, no new code needed:**
- **Primary axis, all Solutions: `--num-steps`** (`train.py:588`) — fixed step count isolates the quality effect, independent of H100-vs-Trn2 throughput differences. Pass `--max-train-seconds` (`:589`) alongside it too, but generously — here it's only an **operational safety ceiling** (bounds cost, catches a hung/pathological run, bounds spot-loss exposure), not the control variable.
- **Secondary axis, [SPEED]-tagged Solutions only (#4, #5, #9, #10)**: run a *second* invocation with the roles flipped — a generous `--num-steps` and a real, tight `--max-train-seconds` as the actual control. This gives "bpb reached at matched wall-clock time," i.e. the net effect once the technique's own speed cost is included — closer to what actually matters for the real time-boxed Phase 1 objective, still using the exact same flags. Magnitude won't transfer to Trn2 either way; direction (does it remain net-positive once speed is included) is the useful signal here.
- Record step time from the free per-step log lines (`train.py:684-689`) directionally only — never trust the absolute number on Trn2.

**Spot-instance durability**: keep every `--max-train-seconds` short (10-15 min) so a reclaim costs at most one cheap re-run, not hours of accumulated work. `train.py` only checkpoints at the end (`out/final.pt`) with no mid-run resume — acceptable given short experiment durations, not worth adding. What must survive an instance loss is the **results log**, not any individual run's checkpoint: `run_experiment.py` git-commits `h100-experiment-log.jsonl` after every completed experiment by default, so a terminated/reclaimed instance never loses already-gathered history, even if the in-flight experiment is lost.

**Cumulative, not full-factorial.** Each Solution starts from the previous Solution's winning config — same greedy hill-climbing convention as Karpathy's `results.tsv` (keep/discard, advance the branch). Solutions 1-4 all touch the same schedule-function region but stay as separate measurement checkpoints.

**Noise floor.** Karpathy's own log has a seed-variance entry (`889dbed random seed 42→137, discard`) — treat that as a standing rule: any borderline keep/discard call gets a repeat-seed check before being trusted, not just a single run.

---

## Solution 1 — LR cooldown + floor (time-based progress)

Cross-ref: `enhancement-opportunities.md` #1. Bucket **A**.

- **Change**: rewrite `lr_multiplier` (`train.py:574`, called at `:660`) to compute `progress = elapsed_wall_time / target_time_budget` and apply warmdown + a nonzero floor instead of the current flat-after-warmup shape.
- **H100 protocol**: fixed step count, compare final/average val_bpb against current baseline (flat LR).
- **Sub-sweep**, in this order (Karpathy's own validated values first): `WARMDOWN_RATIO` ∈ {0.5, 0.7, 0.75} → `FINAL_LR_FRAC` ∈ {0.0, 0.05} once warmdown ratio is locked.
- **Keep/discard gate**: lower val_bpb at equal steps → keep, advance to Solution 2. Any regression → revert before continuing (don't stack an unverified change under the next one).
- **Trainium migration**: trivial copy — pure schedule math, no device-specific code (bucket A). Re-verify under real 30-min timing on Trn2 once ported, since step count achieved in exactly 30 min differs from whatever fixed count was used on H100.

## Solution 2 — Remove/shrink warmup

Cross-ref: #2. Bucket **A**.

- **Change**: `WARMUP_STEPS` (`train.py:58`), consumed in `lr_multiplier` (`:574-576`).
- **H100 protocol**: same fixed step count, starting from Solution 1's winning schedule.
- **Sub-sweep**: `WARMUP_STEPS` ∈ {0, 1-2 (numerical safety only), current default}.
- **Keep/discard gate**: Karpathy's log found warmup hurts every time it was tried — expect 0 to win; if it doesn't, re-check for a numerics issue (first-step NaN/inf) before assuming warmup is actually needed here.
- **Trainium migration**: trivial copy, bucket A.

## Solution 3 — Weight-decay decay-to-zero schedule

Cross-ref: #3. Bucket **A**.

- **Change**: `WEIGHT_DECAY` (`train.py:57`) currently applied as a constant (`:664-665`) — make it a function of `progress` (linear decay to 0), same `progress` variable introduced in Solution 1.
- **H100 protocol**: fixed step count, starting from Solutions 1-2's winning config.
- **Sub-sweep**: linear decay-to-zero (Karpathy's validated shape) vs. current constant, as a straight A/B — no further sub-values needed unless linear wins and you want to try a sqrt schedule as a stretch goal (Karpathy tried this too and linear won).
- **Keep/discard gate**: decaying WD should beat constant per Karpathy's log; if not, check the decay is actually reaching the optimizer's `weight_decay` group value (not just computed and discarded).
- **Trainium migration**: trivial copy, bucket A.

## Solution 4 — Muon momentum-warmup step count

Cross-ref: #6. Bucket **A**.

- **Change**: `muon_momentum` (`train.py:580`, called `:664`) — currently ramps over a step count matching Karpathy's *original* baseline (300), not his validated-better value.
- **H100 protocol**: fixed step count, starting from Solutions 1-3's winning config.
- **Sub-sweep**: momentum-warmup length ∈ {300 (current), 200 (Karpathy's keep), 150 (Karpathy's discard — confirms the shape, don't go lower)}.
- **Keep/discard gate**: expect 200 to win narrowly; this is a low-impact tune, don't over-invest time here.
- **Trainium migration**: trivial copy, bucket A.

## Solution 5 — Init scale sweep

Cross-ref: #11. Bucket **A**.

- **Change**: `init_weights` (`train.py:322`) — scale down the uniform-init bounds on attention/MLP matrices.
- **H100 protocol**: fixed step count, starting from Solutions 1-4's winning config. This one interacts with training dynamics more than the schedule tweaks, so run it after the schedule is locked, not before.
- **Sub-sweep**: overall init scale multiplier ∈ {1.0 (current), 0.8, 0.7, 0.68} — Karpathy's own validated sequence, each confirmed better than the last down to 0.68.
- **Keep/discard gate**: expect monotonic improvement down to ~0.68-0.7x, then flattening/reversal — stop at the first value that doesn't improve, don't assume smaller is always better (his log shows 0.66 lost to 0.68).
- **Trainium migration**: trivial copy, bucket A. Because the run is short (~500-1500 steps), this lever has outsized leverage relative to a normal-length pretrain — worth a slightly wider sweep than the others if time allows.

## Solution 6 — Packing-bin reshape trick

Cross-ref: #9. Bucket **A**.

- **Change**: in `train.py`, change the `make_dataloader(` call (`:633`) to request `seq_len = 8 * SEQ_LEN` and reshape the returned `(inputs, targets)` to `(batch*8, SEQ_LEN)` before feeding the model. **`prepare.py` itself stays untouched** (`make_dataloader` def at `prepare.py:431` is read-only, per the competition's own rule).
- **H100 protocol**: fixed step count (note: this changes effective batch size, so hold *total tokens processed* fixed across the comparison, not raw step count, to isolate the packing effect from a batch-size effect).
- **Sub-sweep**: none needed — this is a binary change (on/off). Verify the reshape's target alignment holds (spot-check a few token positions across the chunk boundary) before trusting the bpb number.
- **Keep/discard gate**: expect a small quality improvement (fewer document-tail crops) plus a step-time change from the effective `DEVICE_BATCH_SIZE` increase (directional only on H100).
- **Trainium migration**: trivial copy, bucket A — the reshape sits outside any compiled graph, same one-time-compile cost as any other constant change.

## Solution 7 — Value embeddings

Cross-ref: #4. Bucket **A**.

- **Change**: new module — a per-layer `nn.Embedding` + gating `Linear`, added into `CausalSelfAttention`'s V stream (`train.py:254`, wired through `Block.forward` `:300` and `GPT.__init__` `:307`). This is greenfield — no existing VE code in the baseline to anchor to; use Karpathy's `autoresearch/train.py` `value_embeds` ModuleDict as the reference implementation to adapt (already confirmed present there, absent here).
- **H100 protocol**: fixed step count, starting from Solutions 1-6's winning config. First verify shape correctness and `causality_check` (`prepare.py:638`) still passes before trusting any bpb number.
- **Sub-sweep**: VE coverage — all layers (Karpathy's winning config) vs. every-3rd-layer vs. last-3-layers-only (both confirmed *worse* in his log — include them anyway as a sanity check that the Trainium port reproduces the same ordering). Then VE's own LR/WD, starting from Karpathy's validated small values (tiny WD ~0.001→0.003 range).
- **Keep/discard gate**: full-coverage VE should beat partial coverage; if partial coverage wins here, something is likely wrong in the port, not a genuine Trainium-specific finding.
- **Trainium migration**: trivial copy, bucket A. Do a manual A/B on Trn2 before handing to `autoresearch` — see "Rollout plan" in the source doc for why (blind keep/discard can't distinguish "genuinely better" from "accidentally not broken").

## Solution 8 — Sliding-window attention

Cross-ref: #5. Bucket **B**.

- **Change**: replace `is_causal=True` in `CausalSelfAttention.forward` (`train.py:266`) with an explicit static per-window-size attention mask, plus a `window_pattern` config string (e.g. `SSSSL`) controlling which layers get the short window. New code, no existing anchor for the mask builder itself.
- **H100 protocol**: fixed step count, starting from Solutions 1-7's winning config. **Force the same masked-eager SDPA path Trainium will actually run** — explicitly disable flash-attn/mem-efficient backends (e.g. `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False)` or the current `torch.nn.attention.sdpa_kernel` context-manager equivalent) so the H100 bpb number isn't contaminated by a GPU-only fused kernel we can't use on Neuron anyway. Record step time separately with flash-attn *enabled* just to see the ceiling, but don't use that number for anything Trainium-related.
- **Sub-sweep**: window pattern ∈ {current (full dense), `SSSL`, `SSSSL` (Karpathy's keep)}; short-window size ∈ {1/2 context (baseline-equivalent), 1/8 context = 256 tokens (Karpathy's keep)}; then RoPE base frequency ∈ {10K (current), 200K (Karpathy's keep)}.
- **Keep/discard gate**: expect both quality and step-time improvement per Karpathy's log; on the *masked-eager* H100 path specifically, the step-time win may be smaller than his (he had flash-attn's native windowing) — that gap is exactly what a follow-up NKI kernel on Trn2 would close.
- **Trainium migration**: mask-building code copies as-is (bucket A logic), but this is tagged bucket **B** overall — profile on real Trn2 after porting to check whether eager masked SDPA is already adequate or whether a custom NKI windowed-attention kernel is warranted (see Solution 12).

## Solution 9 — SSM-inspired add-ons (three ordered sub-experiments)

Cross-ref: #13. Bucket **A**, with a bucket-B caveat on the conv (GPU has a hand-tuned CUDA kernel for this that Trainium lacks — irrelevant at this small kernel size, noted for completeness).

- **Change (sub-step 9a — local causal conv)**: add a small depthwise causal `conv1d` (kernel size ~4) ahead of attention in `Block.forward` (`train.py:300`). Purely additive, static-shape-safe.
- **Change (sub-step 9b — gated MLP)**: restructure `MLP` (`train.py:281`) to a SwiGLU/GEGLU-style gate (`y * silu(gate(x))`) instead of the current plain ReLU² MLP. Adds one weight matrix — weigh against the model's simplicity criterion.
- **Change (sub-step 9c — selective per-token gating)**: make `resid_lambda`/`x0_lambda` (currently static per-layer scalars in `Block`) a function of the hidden state via a small projection. Higher effort, lower priority — only attempt if 9a/9b both land and there's time left.
- **H100 protocol**: fixed step count, run 9a → 9b → 9c in sequence, each cumulative on the previous sub-step's winning result (same convention as Solutions 1-4).
- **Sub-sweep**: 9a — conv kernel size ∈ {2 (RWKV-style shift), 4}. 9b — none of Karpathy's log applies here (untested territory); just A/B gated vs. ungated. 9c — gate architecture is open-ended; treat as exploratory, not a fixed sweep.
- **Keep/discard gate**: none of these three have ablation precedent — treat every result as a genuine new finding, not a confirmation. Real risk each one just doesn't help or actively hurts; discard freely.
- **Trainium migration**: trivial copy for all three (bucket A) — plain PyTorch ops, no device-specific code. Good first candidates to hand to `autoresearch` on Trn2 specifically because they're untested and bounded.

## Solution 10 — Weight-cast caching (free sub-fix under #10)

Cross-ref: #10 (the non-kernel part). Bucket **A**.

- **Change**: `Linear.forward` (`train.py:242`) currently casts `self.weight.to(dtype=x.dtype)` on every forward call, even though weights only change once per optimizer step. Cache the cast after `optimizer.step()`, reuse across the `grad_accum_steps` inner loop.
- **H100 protocol**: fixed step count, measure step time before/after (this is a pure speed fix, bpb should be unchanged — use bpb-unchanged as a correctness check, not the target metric).
- **Sub-sweep**: none — single change.
- **Keep/discard gate**: step-time should improve or stay flat; any bpb change signals a caching bug (stale weights), not a real effect — investigate, don't keep.
- **Trainium migration**: trivial copy, bucket A. Land this one early/opportunistically — it's independent of everything else here and has zero interaction risk with the other Solutions.

## Solution 11 — Batch size + depth/width resweep

Cross-ref: #7 / #8. Bucket: values are Trn2-only, no code is device-specific.

- **Change**: `DEPTH` (`train.py:42`), `ASPECT_RATIO` (`:43`), `DEVICE_BATCH_SIZE` (`:46`), `TOTAL_BATCH_SIZE` (`:47`).
- **H100 protocol**: **dry-run the sweep script/procedure only** — confirm the bisection logic (vary depth/width/batch, measure tokens/sec, find where more steps stop compensating for a smaller/shallower model) runs end-to-end and produces sane output. Do **not** trust the resulting optimal values — H100 throughput has nothing to do with Trn2 throughput.
- **Sub-sweep**: exercised only as a dry run here; the real sweep (concrete depth/width/batch grid) happens in the Trn2 phase, after all of Solutions 1-10 have landed there (the optimum shifts once schedule/VE/windowing/packing changes are in).
- **Keep/discard gate**: N/A on H100 — this Solution's H100 phase is a tooling check, not a measurement.
- **Trainium migration**: this Solution *starts* on Trn2 in the meaningful sense. Run the real sweep against the actual 30-min wall-clock budget once everything else is ported.

## Solution 12 — NKI kernel work

Cross-ref: #10 (the kernel-writing part). Bucket **C** — Trn2-native from the start, no H100 phase.

- **Change**: candidates in priority order (per the source doc's call-frequency/tensor-size reasoning): fused RMSNorm (`norm()`, ~42 calls/forward), fused rotary embedding, fused logit softcap (ideally fused into cross-entropy), and — only if profiling shows it's warranted — a flash-attention-style kernel for `CausalSelfAttention`. Template: `_relu_sq_fwd`/`_relu_sq_bwd`/`_ReluSquaredNKI` (`train.py:169-222`).
- **H100 protocol**: none — NKI doesn't exist on CUDA. This Solution begins directly on Trn2.
- **Process**: profile first with `neuron-nki-profiling`/`neuron-nki-profile-analysis-agent` (deployed this session) to confirm the actual bottleneck before writing anything — don't write a kernel based on source-reading assumptions alone. Author with `neuron-nki-writer-agent`, debug compile errors with `neuron-nki-debugger-agent`, validate numerical equivalence against the eager reference with `neuron-framework-equivalence` before trusting any kernel in a scored run.
- **Keep/discard gate**: numerical equivalence check must pass before a bpb comparison means anything — a shape-correct but numerically-off kernel can still pass `causality_check` while quietly moving val_bpb the wrong way.
- **Trainium migration**: N/A — already Trn2-native. This is the last Solution to attempt, after every portable win (Solutions 1-10) is already banked and re-verified on real hardware, and after Solution 11's resweep, since kernel-writing effort should target whatever the *final* architecture's actual bottleneck is, not an intermediate one.

---

## After both hardware phases

Once Solutions 1-10 are ported to Trn2 and re-verified under real 30-minute timing, and Solutions 11-12 are done (they only ever happen on Trn2), hand the resulting checkpoint to `autoresearch` to mine the local hyperparameter neighborhood — see `enhancement-opportunities.md`'s "Rollout plan" section for the full rationale (why sequencing this way, the 6x-more-expensive-per-experiment argument for Trn2 vs. Karpathy's GPU cycle). Only spend real, rate-limited (5/week) Phase 1 submissions once local search on the fully-updated baseline has converged, not before.

Item #12 (Mamba2/hybrid architecture swap) stays excluded from this plan — see its own verdict in `enhancement-opportunities.md` #12.
