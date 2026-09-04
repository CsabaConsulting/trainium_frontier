# Experiment Rollout Plan — GPU Proxy Now, Trn2 When Allocated

This is the execution companion to `enhancement-opportunities.md`. That doc establishes *what* to change and *why* (rationale, evidence, portability bucket A/B/C) — this doc only adds *sequencing and measurement protocol*. Rationale/evidence for each item lives in the source doc; follow the `#N` cross-references there, not here.

> **Current hardware situation**: no Trn2 allocation yet, so experiments run on a rented CUDA GPU (RTX 6000 Pro Max-Q) in the meantime, and migrate to Trn2 the moment it's allocated. `detect_device_type` (`train.py:73`) already picks CUDA before Neuron, so **the same `train.py` and the same `research/run_experiment.py` run unmodified on either box** — nothing here is GPU-only or Trn2-only code, only the launch *shape* and a couple of measurement caveats differ. Those differences are called out inline below wherever they apply, tagged **[GPU]** or **[Trn2]**.

**Test harness**: every step below means running the actual competition `trainium_frontier/train.py` / `prepare.py`. This is **not** the separate `autoresearch/` single-NeuronCore dev-loop script, which is a different, smaller file with its own baseline. Using the wrong one invalidates the comparison.

**Environment**:
- **[GPU]** `requirements.txt` has zero Neuron-only pins but assumes torch/torch_neuronx/neuronx-cc are already provided by a Trainium DLAMI — on the rented GPU box, `pip install torch` (CUDA build) separately first, then `pip install -r requirements.txt`.
- **[Trn2]** launch from the README-specified DLAMI (`ami-0d5703e99c1589f8d`, venv `neurips-torchneuronx-native-beta-2.11.3-20260824`) — torch/torch_neuronx/neuronx-cc already present, no separate install step.
- Either way: `python prepare.py setup` once to populate `~/.cache/...` before any experiment.

**Run every experiment through `research/run_experiment.py`**, not `train.py` directly — it launches via `torchrun` (matching the real submission launch-command format, `NEURON_LOGICAL_NC_CONFIG=<lnc> torchrun --standalone --nproc_per_node=<n> train.py ...`), captures the log, parses `train.py`'s own final summary block, and appends one row to a persistent, git-committed `research/experiment-log.jsonl`. Example: `python research/run_experiment.py --label sol1-warmdown75 --solution 1 --description "warmdown=0.75 final_lr_frac=0.05" --num-steps 800 --max-train-seconds 900`.

**Launch shape**: the harness defaults to `--nproc-per-node 1 --lnc 1` — a single-device run, right for the rented single-GPU box. **[Trn2]** once allocated, pass `--nproc-per-node 8 --lnc 1` explicitly to match README's default Phase 1 submission shape — distributed batch math (`grad_accum_steps`, effective micro-batch) depends on `world_size`, so any keep/discard decision meant to carry over to the real submission needs re-confirming at that shape, not just at the GPU box's single-device default.

**[Trn2] NEFF compile overhead**: first run at any given tensor shape pays a ~60s one-time compile (per `program.md`). Batch experiments that share a shape (same `DEPTH`/`ASPECT_RATIO`/`DEVICE_BATCH_SIZE`) back-to-back where possible. Not applicable on the GPU box. **Update (upstream PR #6, merged into our `train.py`/`prepare.py`)**: the official runner now excludes compilation time from the charged 30-minute budget entirely — the first 5 completed optimizer steps are treated as a startup/compilation allowance, capped at 900s, and `train.py` restarts its own budget clock at that boundary to match. This substantially de-risks the compile-overhead concern for Solution 11's real sweep: a slow first-shape compile no longer eats into the charged training window. Still worth batching same-shape runs for iteration-speed reasons, just no longer a budget-correctness concern. Also now **mandatory for a valid submission**: `train.py` must call `prepare.record_step_boundary(step)` after every completed optimizer step, or the submission checker rejects it outright (static AST check) — already wired into our `train.py`, verified against the exact check upstream's `check_submission()` runs.

**[GPU] flash-attn contamination**: CUDA's `F.scaled_dot_product_attention` auto-selects an efficient fused backend (flash/mem-efficient) that Neuron doesn't have. This only matters for Solution 8 (windowed attention) — see its note below; nothing else here is affected, since everywhere else the baseline is already just eager math/elementwise ops with no fused-kernel backend to worry about on either device.

**Protocol — both axes native to `train.py`, no new code needed:**
- **Primary axis, all Solutions: `--num-steps`** (`train.py:588`) — fixed step count isolates the quality effect, independent of which box's throughput you're currently on. Pass `--max-train-seconds` (`:589`) alongside it too, generously — here it's an **operational ceiling** (bounds cost, catches a hung run), not the control variable.
- **Secondary axis, [SPEED]-tagged Solutions only (#4, #5, #9, #10)**: run a *second* invocation with the roles flipped — generous `--num-steps`, a real, tight `--max-train-seconds` as the actual control. This gives "bpb reached at matched wall-clock time." **[GPU]** treat the resulting number as directional only — GPU throughput ≠ Trn2 throughput, magnitude won't transfer, direction (does the technique remain net-positive once its own speed cost is included) is the useful part. **[Trn2]** once run there, this number is directly meaningful, not just directional.
- Record step time from the free per-step log lines (`train.py:684-689`) — **[GPU]** directional only; **[Trn2]** trustworthy as-is.

**Cumulative, not full-factorial.** Each Solution starts from the previous Solution's winning config — same greedy hill-climbing convention as Karpathy's `results.tsv` (keep/discard, advance the branch). Solutions 1-4 all touch the same schedule-function region but stay as separate measurement checkpoints. **Every Solution measured on the GPU box gets a cheap re-confirmation run on Trn2 once allocated** — bucket A items (see `enhancement-opportunities.md`'s portability table) should reproduce directly; only bucket B/C items (#5, #10's kernel part) are expected to need real re-tuning on the new hardware.

**Noise floor.** Karpathy's own log has a seed-variance entry (`889dbed random seed 42→137, discard`) — treat that as a standing rule: any borderline keep/discard call gets a repeat-seed check before being trusted, not just a single run.

**Durability**: `run_experiment.py` git-commits `experiment-log.jsonl` after every completed experiment by default, so the results history from the GPU-box phase survives the switch to Trn2 intact — one continuous log across both hardware phases, not two separate ones. Record which box each row came from (the harness's `launch_command` field plus `world_size` in the parsed summary are enough to distinguish after the fact; add a plain `--description` note like "GPU box" if you want it more obvious at a glance).

---

## Solution 1 — LR cooldown + floor (time-based progress)

Cross-ref: `enhancement-opportunities.md` #1. Bucket **A**.

- **Change**: rewrite `lr_multiplier` (`train.py:574`, called at `:660`) to compute `progress = elapsed_wall_time / target_time_budget` and apply warmdown + a nonzero floor instead of the current flat-after-warmup shape.
- **Protocol**: fixed step count (full-chip shape), compare final val_bpb against current baseline (flat LR).
- **Sub-sweep**, in this order (Karpathy's own validated values first): `WARMDOWN_RATIO` ∈ {0.5, 0.7, 0.75} → `FINAL_LR_FRAC` ∈ {0.0, 0.05} once warmdown ratio is locked.
- **Keep/discard gate**: lower val_bpb at equal steps → keep, advance to Solution 2. Any regression → revert before continuing (don't stack an unverified change under the next one).

## Solution 2 — Remove/shrink warmup

Cross-ref: #2. Bucket **A**.

- **Change**: `WARMUP_STEPS` (`train.py:58`), consumed in `lr_multiplier` (`:574-576`).
- **Protocol**: same fixed step count, starting from Solution 1's winning schedule.
- **Sub-sweep**: `WARMUP_STEPS` ∈ {0, 1-2 (numerical safety only), current default}.
- **Keep/discard gate**: Karpathy's log found warmup hurts every time it was tried — expect 0 to win; if it doesn't, re-check for a numerics issue (first-step NaN/inf, or a NEFF-compile-step artifact) before assuming warmup is actually needed here.

## Solution 3 — Weight-decay decay-to-zero schedule

Cross-ref: #3. Bucket **A**.

- **Change**: `WEIGHT_DECAY` (`train.py:57`) currently applied as a constant (`:664-665`) — make it a function of `progress` (linear decay to 0), same `progress` variable introduced in Solution 1.
- **Protocol**: fixed step count, starting from Solutions 1-2's winning config.
- **Sub-sweep**: linear decay-to-zero (Karpathy's validated shape) vs. current constant, as a straight A/B — no further sub-values needed unless linear wins and you want to try a sqrt schedule as a stretch goal (Karpathy tried this too and linear won).
- **Keep/discard gate**: decaying WD should beat constant per Karpathy's log; if not, check the decay is actually reaching the optimizer's `weight_decay` group value (not just computed and discarded).

## Solution 4 — Muon momentum-warmup step count

Cross-ref: #6. Bucket **A**.

- **Change**: `muon_momentum` (`train.py:580`, called `:664`) — currently ramps over a step count matching Karpathy's *original* baseline (300), not his validated-better value.
- **Protocol**: fixed step count, starting from Solutions 1-3's winning config.
- **Sub-sweep**: momentum-warmup length ∈ {300 (current), 200 (Karpathy's keep), 150 (Karpathy's discard — confirms the shape, don't go lower)}.
- **Keep/discard gate**: expect 200 to win narrowly; this is a low-impact tune, don't over-invest time here.

## Solution 5 — Init scale sweep

Cross-ref: #11. Bucket **A**.

- **Change**: `init_weights` (`train.py:322`) — scale down the uniform-init bounds on attention/MLP matrices.
- **Protocol**: fixed step count, starting from Solutions 1-4's winning config. This one interacts with training dynamics more than the schedule tweaks, so run it after the schedule is locked, not before.
- **Sub-sweep**: overall init scale multiplier ∈ {1.0 (current), 0.8, 0.7, 0.68} — Karpathy's own validated sequence, each confirmed better than the last down to 0.68.
- **Keep/discard gate**: expect monotonic improvement down to ~0.68-0.7x, then flattening/reversal — stop at the first value that doesn't improve, don't assume smaller is always better (his log shows 0.66 lost to 0.68). Because the run is short (~500-1500 steps), this lever has outsized leverage relative to a normal-length pretrain — worth a slightly wider sweep than the others if time allows.

## Solution 6 — Packing-bin reshape trick

Cross-ref: #9. Bucket **A**.

- **Change**: in `train.py`, change the `make_dataloader(` call (`:633`) to request `seq_len = 8 * SEQ_LEN` and reshape the returned `(inputs, targets)` to `(batch*8, SEQ_LEN)` before feeding the model. **`prepare.py` itself stays untouched** (`make_dataloader` def at `prepare.py:431` is read-only, per the competition's own rule).
- **Protocol**: fixed step count (note: this changes effective batch size, so hold *total tokens processed* fixed across the comparison, not raw step count, to isolate the packing effect from a batch-size effect).
- **Sub-sweep**: turned out this was wrong the first time — assumed binary (on/off) with no bin size worth sweeping. A post-session follow-up swept `PACK_FACTOR ∈ {1,2,4,8,16}` and found a real interior optimum at `2` (not `8`), beating it by -0.00694 — see `gpu-experiment-results.md`'s "Follow-up: bin-size sweep." Sweep the bin size, don't assume the first value tried is optimal, especially after other architecture changes land on top of it. Verify the reshape's target alignment holds (spot-check a few token positions across the chunk boundary) before trusting any bpb number.
- **Keep/discard gate**: expect a small quality improvement (fewer document-tail crops) plus a step-time change from the effective `DEVICE_BATCH_SIZE` increase — this is a bucket-A change so the shape stays the same as long as `8*SEQ_LEN` doesn't itself trigger a different compile path than whatever else is running; check the compile-overhead note above if timing looks off on the first run.

## Solution 7 — Value embeddings

Cross-ref: #4. Bucket **A**.

- **Change**: new module — a per-layer `nn.Embedding` + gating `Linear`, added into `CausalSelfAttention`'s V stream (`train.py:254`, wired through `Block.forward` `:300` and `GPT.__init__` `:307`). This is greenfield — no existing VE code in the baseline to anchor to; use Karpathy's `autoresearch/train.py` `value_embeds` ModuleDict as the reference implementation to adapt (already confirmed present there, absent here).
- **Protocol**: fixed step count, starting from Solutions 1-6's winning config. First verify shape correctness and `causality_check` (`prepare.py:638`) still passes before trusting any bpb number.
- **Sub-sweep**: VE coverage — all layers (Karpathy's winning config) vs. every-3rd-layer vs. last-3-layers-only (both confirmed *worse* in his log — include them anyway as a sanity check that the port reproduces the same ordering). Then VE's own LR/WD, starting from Karpathy's validated small values (tiny WD ~0.001→0.003 range).
- **Keep/discard gate**: full-coverage VE should beat partial coverage; if partial coverage wins here, something is likely wrong in the implementation, not a genuine finding. Do a manual A/B before handing to `autoresearch` — see "Rollout plan" in the source doc for why (blind keep/discard can't distinguish "genuinely better" from "accidentally not broken").

## Solution 8 — Sliding-window attention

Cross-ref: #5. Bucket **B**.

- **Change**: replace `is_causal=True` in `CausalSelfAttention.forward` (`train.py:266`) with an explicit static per-window-size attention mask, plus a `window_pattern` config string (e.g. `SSSSL`) controlling which layers get the short window. New code, no existing anchor for the mask builder itself.
- **Protocol**: fixed step count, starting from Solutions 1-7's winning config. **[GPU]** explicitly disable flash-attn/mem-efficient SDPA backends (e.g. `torch.nn.attention.sdpa_kernel` context manager, or the older `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False)`) so the measured bpb reflects the same masked-eager path Trainium will actually run — otherwise the GPU number is contaminated by a fused kernel Neuron doesn't have. **[Trn2]** no action needed — eager masked SDPA *is* the only path there, so the measurement is already exactly what will ship.
- **Sub-sweep**: window pattern ∈ {current (full dense), `SSSL`, `SSSSL` (Karpathy's keep)}; short-window size ∈ {1/2 context (baseline-equivalent), 1/8 context = 256 tokens (Karpathy's keep)}; then RoPE base frequency ∈ {10K (current), 200K (Karpathy's keep)}.
- **Keep/discard gate**: expect both quality and step-time improvement per Karpathy's log. If the step-time win is smaller than expected, that's the profiling signal Solution 12 (NKI kernel work) should act on — a custom windowed-attention kernel is the natural follow-up if eager masked SDPA turns out to be the bottleneck.

## Solution 9 — SSM-inspired add-ons (three ordered sub-experiments)

Cross-ref: #13. Bucket **A**.

- **Change (sub-step 9a — local causal conv)**: add a small depthwise causal `conv1d` (kernel size ~4) ahead of attention in `Block.forward` (`train.py:300`). Purely additive, static-shape-safe.
- **Change (sub-step 9b — gated MLP)**: restructure `MLP` (`train.py:281`) to a SwiGLU/GEGLU-style gate (`y * silu(gate(x))`) instead of the current plain ReLU² MLP. Adds one weight matrix — weigh against the model's simplicity criterion.
- **Change (sub-step 9c — selective per-token gating)**: make `resid_lambda`/`x0_lambda` (currently static per-layer scalars in `Block`) a function of the hidden state via a small projection. Higher effort, lower priority — only attempt if 9a/9b both land and there's time left.
- **Protocol**: fixed step count, run 9a → 9b → 9c in sequence, each cumulative on the previous sub-step's winning result (same convention as Solutions 1-4).
- **Sub-sweep**: 9a — conv kernel size ∈ {2 (RWKV-style shift), 4}. 9b — none of Karpathy's log applies here (untested territory); just A/B gated vs. ungated. 9c — gate architecture is open-ended; treat as exploratory, not a fixed sweep.
- **Keep/discard gate**: none of these three have ablation precedent — treat every result as a genuine new finding, not a confirmation. Real risk each one just doesn't help or actively hurts; discard freely. Good first candidates to hand to `autoresearch` specifically because they're untested and bounded.

## Solution 10 — Weight-cast caching (free sub-fix under #10)

Cross-ref: #10 (the non-kernel part). Bucket **A**.

- **Change**: `Linear.forward` (`train.py:242`) currently casts `self.weight.to(dtype=x.dtype)` on every forward call, even though weights only change once per optimizer step. Cache the cast after `optimizer.step()`, reuse across the `grad_accum_steps` inner loop.
- **Protocol**: fixed step count, measure step time before/after (this is a pure speed fix, bpb should be unchanged — use bpb-unchanged as a correctness check, not the target metric).
- **Sub-sweep**: none — single change.
- **Keep/discard gate**: step-time should improve or stay flat; any bpb change signals a caching bug (stale weights), not a real effect — investigate, don't keep. Land this one early/opportunistically — it's independent of everything else here and has zero interaction risk with the other Solutions.

## Solution 11 — Batch size + depth/width resweep

Cross-ref: #7 / #8.

- **Change**: `DEPTH` (`train.py:42`), `ASPECT_RATIO` (`:43`), `DEVICE_BATCH_SIZE` (`:46`), `TOTAL_BATCH_SIZE` (`:47`).
- **Protocol**: **[GPU]** dry-run the sweep script/procedure only here — confirm the bisection logic (vary depth/width/batch, measure tokens/sec, find where more steps stop compensating for a smaller/shallower model) runs end-to-end and produces sane output. Do not trust the resulting optimal values — GPU throughput has nothing to do with Trn2 throughput. **[Trn2]** this is where the real sweep happens: run after Solutions 1-10 have landed, since the optimum shifts once schedule/VE/windowing/packing changes are in; measure tokens/sec and val_bpb at the actual 30-minute wall-clock budget (`--max-train-seconds` = the real `TRAIN_TIME_BUDGET_SECONDS`).
- **Sub-sweep**: **[Trn2]** each distinct `(DEPTH, ASPECT_RATIO, DEVICE_BATCH_SIZE)` combination is a different compiled shape (~60s NEFF compile each) — group runs by shape and batch them to amortize that cost rather than round-robining across shapes. Not applicable on the GPU box.
- **Keep/discard gate**: N/A on the GPU box (tooling check only). **[Trn2]** the winning combination is whichever reaches the lowest val_bpb at the real 30-minute ceiling, not the most steps or the biggest model in isolation.

## Solution 12 — NKI kernel work

Cross-ref: #10 (the kernel-writing part). Bucket **C**.

- **Change**: candidates in priority order (per the source doc's call-frequency/tensor-size reasoning): fused RMSNorm (`norm()`, ~42 calls/forward), fused rotary embedding, fused logit softcap (ideally fused into cross-entropy), and — only if profiling shows it's warranted — a flash-attention-style kernel for `CausalSelfAttention` (see Solution 8's note on this). Template: `_relu_sq_fwd`/`_relu_sq_bwd`/`_ReluSquaredNKI` (`train.py:169-222`).
- **Process**: profile first with `neuron-nki-profiling`/`neuron-nki-profile-analysis-agent` (deployed this session) to confirm the actual bottleneck before writing anything — don't write a kernel based on source-reading assumptions alone. Author with `neuron-nki-writer-agent`, debug compile errors with `neuron-nki-debugger-agent`, validate numerical equivalence against the eager reference with `neuron-framework-equivalence` before trusting any kernel in a scored run.
- **Keep/discard gate**: numerical equivalence check must pass before a bpb comparison means anything — a shape-correct but numerically-off kernel can still pass `causality_check` while quietly moving val_bpb the wrong way.
- **Sequencing**: last Solution to attempt, after every other win (Solutions 1-11) is already banked, since kernel-writing effort should target whatever the *final* architecture's actual bottleneck is, not an intermediate one.

---

## After Solution 12

Once Solutions 1-12 have all landed and been measured, hand the resulting checkpoint to `autoresearch` to mine the local hyperparameter neighborhood — see `enhancement-opportunities.md`'s "Rollout plan" section for the full rationale (why sequencing this way, the 6x-more-expensive-per-experiment argument for a 30-minute Trn2 run vs. Karpathy's 5-minute GPU cycle). Only spend real, rate-limited (5/week) Phase 1 submissions once local search on the fully-updated baseline has converged, not before.

Item #12 in the source doc (Mamba2/hybrid architecture swap) stays excluded from this plan — see its own verdict in `enhancement-opportunities.md` #12.
