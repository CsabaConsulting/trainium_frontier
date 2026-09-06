# Experiment Rollout Plan — GPU Phase Complete, Trn2 Readiness Plan

This is the execution companion to `enhancement-opportunities.md`. That doc establishes *what* to change and *why* (rationale, evidence, portability bucket A/B/C) — this doc only adds *sequencing and measurement protocol*. Rationale/evidence for each item lives in the source doc; follow the `#N` cross-references there, not here.

> **Status as of this update**: Solutions 1-10 are **done and validated on two independent GPUs** (RTX 6000 Pro Blackwell, then NVIDIA L4 Ada — different vendor generation, different `torch`/CUDA build), not just planned. Full detail, every number, every keep/discard call: `gpu-experiment-results.md`. This section below is now a status summary, not a to-do list — jump to "Trn2 readiness plan" for what's actually left.

## GPU phase — final status

Locked config: Solutions 1, 2, 3, 5, 6 (with `PACK_FACTOR` re-swept to `2`, not the originally-locked `8`), 7, 9a, 10 kept. Solutions 4, 8, 9b discarded (real regressions). Post-lock exploration (RoPE-base-alone, windowing-alone, VE weight decay, selective per-token gating) — four further clean negative results, closing out low-effort GPU-testable options. Custom packer w/ remainder-requeue remains the one deliberately-untried candidate (weak motivation, high risk — see `gpu-experiment-results.md`).

**Cross-hardware portability confirmed with real numbers**: `1.165273` (RTX 6000 Pro) vs `1.168849` (L4) for the identical config — 0.31% relative difference, both causality-exact. This is real, independent evidence the architecture/schedule findings are properties of the data/model, not an artifact of one specific card.

**Solutions 11 (batch/depth resweep), 12 (NKI kernels), and 13 (Trn2 compiler/runtime config wins) were never GPU-testable** — all three inherently need real Trn2 (throughput-dependent, Neuron-only, or Neuron-runtime-only respectively). Solution 13, added after the Trn2-specific-optimization research pass (see `trn2-optimization-research.md`), is the fastest and lowest-risk of the three and is sequenced first. They're the actual remaining work, detailed below.

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

## Solutions 1-10 — done, see `gpu-experiment-results.md` for full detail

| # | Item | Cross-ref | Final status |
|---|---|---|---|
| 1 | LR cooldown + floor | #1 | **Kept.** `WARMDOWN_RATIO=0.75, FINAL_LR_FRAC=0.05` — biggest single win of the session, -4.55% alone |
| 2 | Remove warmup | #2 | **Kept.** `WARMUP_STEPS=0` — tied/marginal on its own, kept on Karpathy's independent replication + simplicity criterion |
| 3 | WD decay-to-zero | #3 | **Kept.** `WD_DECAY_TO_ZERO=True` — small real win |
| 4 | Muon momentum-warmup steps | #6 | **Discarded.** `200` regressed vs `300` here — real divergence from Karpathy's finding |
| 5 | Init scale sweep | #11 | **Kept.** `INIT_SCALE=0.68` — matched Karpathy's monotonic trend exactly |
| 6 | Packing-bin reshape | #9 | **Kept, re-tuned post-hoc.** `PACK_FACTOR=2` (not the originally-locked `8` — a full bin-size sweep `{1,2,4,8,16}` found a real interior optimum at `2`, -0.00694 vs `8`) — second-biggest win of the session |
| 7 | Value embeddings | #4 | **Kept.** Alternating layers, matches Karpathy directly |
| 8 | Sliding-window attention | #5 | **Discarded, decomposed.** Bundle regressed; later decomposition found *both* sub-components (`ROPE_BASE=200000` alone, `WINDOW_PATTERN="SSSSL"` alone) independently regress — fully closed, not just bundle-level |
| 9a | Local causal conv | #13 | **Kept.** Novel finding, no Karpathy precedent, real win |
| 9b | Gated MLP | #13 | **Discarded.** Worse quality, +72% params, +19% slower — clean discard on every axis |
| 9c | Selective per-token gating | #13 | **Discarded** (tested post-session on L4). Real regression |
| 10 | Weight-cast caching | #10 (non-kernel) | **Kept** (no-op in practice — Solution 6 already absorbed the benefit this targets, but harmless and free) |

Also tested post-session, both discarded: `VE_WD=0.003` (Karpathy's validated value, flat/marginal-negative here), document-boundary attention masking (real regression). **Cross-hardware portability confirmed**: this exact locked config scores `1.165273` (RTX 6000 Pro) and `1.168849` (L4) — 0.31% relative difference, both causality-exact.

**Not pursued, deliberately**: custom packer with remainder-requeue (targets a smaller problem now than originally assumed, since the re-tuned `PACK_FACTOR=2` already has less crop-loss than `8` did, and the sibling leakage-fix idea already failed) — left as documented future work, not current work.

## Solution 13 — Trn2 compiler/runtime config wins (do first, before 11 and 12)

Cross-ref: `enhancement-opportunities.md` #14. New from the Trn2-specific-optimization
research pass (6 PDFs + 4 cloned repos), not from Karpathy's ablation (GPU-only, no
Neuron runtime). Full evidence trail: `trn2-optimization-research.md`. All four items
are pure [SPEED] — judge on tokens/sec at fixed steps, then on val_bpb at the real
30-minute budget, not at fixed steps. Sequenced *before* Solutions 11/12 because both a
batch/depth resweep and any kernel-profiling decision should target the fastest correct
config, not a slower one.

**13a. Neuron compiler/runtime env flags — implemented, crash found and fixed, now clean.**
Landed in `train.py` gated behind `USE_TRN2_COMPILER_FLAGS` (default `False`):
```python
os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")  # unconditional
if USE_TRN2_COMPILER_FLAGS:
    os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
    os.environ.setdefault("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "3")
    os.environ.setdefault("NEURON_CC_FLAGS",
                          f"--model-type=transformer --distribution-strategy=llm-training "
                          f"--lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1')}")
```
**Real incident, worth reading before touching this block again** (full account:
`trn2-experiment-results.md`): the original version included
`--internal-hlo2tensorizer-options='--fast-relayout=true'` (from the Gemini Deep Research
cross-check, claiming ~2x faster tensor transposition) as an **unconditional** env-var
block, reasoning "it's just config, safe by default." Both assumptions were wrong. First,
the embedded quotes broke `neuronx-cc` outright the moment it synced to the box —
`neuronx-cc compile --help` on the installed SDK (2.26.6360.0) confirms
`--internal-hlo2tensorizer-options`/`--fast-relayout` **do not exist as flags at all** on
this version; the claim didn't hold on real hardware. Second, being unconditional meant
the very next run silently used it with no isolated smoke test first. Fixed: the
fabricated flag is dropped, `--model-type`/`--distribution-strategy`/`--lnc` (all
confirmed present in `--help`) are kept, and the whole block now sits behind
`USE_TRN2_COMPILER_FLAGS` like every other Solution 13 toggle.

Decomposed and smoke-tested individually plus combined (single device, 15 steps, seed 42,
matching every other smoke test this phase): `NEURON_FUSE_SOFTMAX` alone, `NEURON_CC_FLAGS`
alone, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` alone, and all three combined all land
in the same tight `~2.1294-2.1296` bpb cluster as the established 15-step baseline — no
crash, no numerical divergence. (One early combined-run result, `2.682`, didn't reproduce
on an identical retest — treated as a non-reproducing anomaly, not a real effect.) **Still
untested: the actual throughput benefit** — a real fixed-step-count A/B is needed before
enabling this for anything that matters; HLAT measured these flags in a TP/PP-sharded 7B
context we don't share, so the speedup (if any) here is unverified either way.
`NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` stays unconditional — a plain filesystem path,
not compiler-flag syntax, verified safe across every run this phase.

**13b. BF16 + hardware stochastic rounding.** We already run `COMPUTE_DTYPE=torch.bfloat16`
(`train.py:217`) but set no rounding mode. HLAT measured higher throughput and lower
memory with equal convergence at 7B (diverged only at 70B). Enable via
`NEURON_RT_STOCHASTIC_ROUNDING_EN=1` (+ `NEURON_RT_STOCHASTIC_ROUNDING_SEED` for
reproducibility) — confirm the exact env-var name against the installed SDK's docs on
the box before relying on it.

**13c. QKV coalescing.** `train.py:356-358` has three separate
`Linear(n_embd, n_embd)` (`c_q`/`c_k`/`c_v`), called separately at `:364-366`. Fuse to
one `Linear(n_embd, 3*n_embd)` + `.split()`. Add `USE_FUSED_QKV` to
`ARCH_TOGGLE_NAMES` **and** `SHAPE_CRITICAL_TOGGLES` — it changes `state_dict` keys, so
`final-confirmation-v2.pt` will not warm-start into it (the existing toggle-mismatch
guard catches this by design). Apply the per-tensor `uniform_(-s, s)` init (`:449-451`)
slice-wise to the fused weight so the parameter distribution is unchanged (RNG draw
order differs — expected, not a bug). `refresh_weight_cache` (Solution 10) is unaffected,
it's per-`Linear`. Pure-throughput change — do not judge on quality.

**13d. Drop the redundant additive mask, use `is_causal=True`.** With the locked config
(`WINDOW_PATTERN="L"`, `USE_DOC_MASK=False`) the precomputed `mask_full`/`mask_short`
passed as `attn_mask=` (`train.py:376`) is purely causal — redundant with SDPA's own
`is_causal` fast path. Pass `is_causal=True, attn_mask=None` only when both windowing
and doc-masking are off; keep the mask path alive behind those toggles for when either
is revived. Verify numerical equivalence against the mask path before trusting it —
`causality_check` alone won't catch a subtly wrong mask; compare val_bpb at matched
steps too.

## Solution 11 — Batch size + depth/width resweep — all axes resolved, final combo confirmed at full budget

**Final locked architecture and result, up front**: `DEPTH=6, ASPECT_RATIO=96` (n_embd=640, n_head=5, 55.7M params), `TOTAL_BATCH_SIZE=131072`, rope+QK-norm NKI kernel default ON. Full-1800s confirmation (`sol11-final-combo-fullbudget`): **500 steps, `public_val_bpb = 1.176151`, `causality_passed: True`** — the final headline number for the entire Trn2 phase, beating the prior headline (`n_embd=384`, 581 steps, `1.197595`) by a real `0.021444` despite 81 fewer steps. Ready for competition submission: `train.py` + `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`. Full detail below and in `trn2-experiment-results.md`.

Cross-ref: #7 / #8. Full detail, every number: `trn2-experiment-results.md`.

- **Change**: `DEPTH` (`train.py`), `ASPECT_RATIO`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`.
- **Depth sweep — done, solid.** Fixed wall-clock (`--max-train-seconds 900`, half budget), real submission shape (`--nproc-per-node 8 --lnc 1`), `ASPECT_RATIO=64` held fixed, `DEPTH ∈ {4,6,8,10,12}`. Result: a clean U-shape, **`DEPTH=6` (n_embd=384, 26.4M params) is a real interior optimum** — `public_val_bpb` 1.297186, vs. 1.330846 (depth4), 1.342754 (depth8), 1.368452 (depth10, prior baseline), 1.365736 (depth12). Every gap here is 100x+ the established GPU-proxy noise floor. **`DEPTH=6` locked in `train.py`** — the single largest quality finding of the Trn2 phase, something the GPU proxy could never have found. Independently re-verified clean against the data-integrity bug below (each run's own `grad_accum_steps × micro_tokens` matches its label).
- **Batch-size sweep — a real data-integrity bug was found and corrected, converged to a consistent final answer.** Original sweep (`TOTAL_BATCH_SIZE ∈ {32768, 65536, 131072, 262144, 524288}`) reported `262144` winning. **Found later, while investigating an unrelated throughput anomaly in a Solution 12 kernel smoke test**: the `262144`-labeled run had actually executed with `grad_accum_steps=8` — i.e. `TOTAL_BATCH_SIZE=131072`, a duplicate of the row above it, not a real measurement. Verified via each run's own printed config (the only reliable method — git-history reconstruction doesn't work, since every sweep point deliberately ran from an uncommitted edit). The other four points re-verified genuinely correct. **Corrected re-run of the real `262144`: 172 steps, `1.310506` — worse than `131072` (`1.290674`).** The real 900s-sweep U-shape minimum is `131072`.
- **Full-1800s head-to-head — decisive, and agrees with the corrected sweep.** `131072`: **560 steps, `1.199914`**. Genuine `262144`: **362 steps, `1.200143`**. Gap `0.000229` — a statistical tie at the established noise floor, not a real quality difference. `131072` completes 55% more steps for that tied result. **`TOTAL_BATCH_SIZE=131072` is the final locked value** — chosen for the step-count margin once quality is a wash, and reassuring that the 900s-sweep correction and the full-budget head-to-head landed on the same answer independently.
- **Toggle-stack re-validation — corrected, conclusion unchanged.** Was compared against the mislabeled `262144` point; redone against the genuine one (`172 steps, 1.310506`). Full stack still beats raw baseline (`1.310506` vs. `1.321078`), margin corrected from a claimed `0.033` down to the real `0.0106`. Run at genuine `262144`, not the final `131072` — not considered urgent to redo, since the qualitative conclusion now holds at three distinct shapes this phase and `131072` sits between two tested points, not off in new territory.
- **Width-only resweep at fixed `DEPTH=6` — done, real second finding, optimum shifted.** Same 900s protocol, `DEPTH=6` fixed, `ASPECT_RATIO ∈ {32,64,80,96,128}` → `n_embd ∈ {256,384,512,640,768}`. Clean U-shape: `256` (1.309278) → `384` (1.297186, prior default) → `512` (1.270811) → `640` (**1.269647 — winner**) → `768` (1.273621). **`ASPECT_RATIO=96` (n_embd=640, 55.7M params) locked in `train.py`**, replacing `64`. `384→640` gap is `0.0275` — two orders of magnitude above noise floor, a real second major finding. Full table and mechanism discussion: `trn2-experiment-results.md`.
- **Batch-size axis re-validated at `n_embd=640` — done, same winner.** Kernel default ON throughout (first batch sweep ever run with the kernel on). `65536` (375 steps, `1.279503`) → `131072` (240 steps, **`1.269318` — winner**) → `262144` (160 steps, `1.307378`). `131072` wins by real margins both directions (`0.0102`, `0.0381`) — the width change did not shift the batch-size optimum. `params_M=55.7` confirmed consistent across all three points. `TOTAL_BATCH_SIZE=131072` re-locked (unchanged value, updated comment).
- **Full 1800s confirmation of the complete combo — done, new final headline.** `sol11-final-combo-fullbudget`: `DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, kernel ON, real 8-way shape, full budget. **500 steps, `public_val_bpb: 1.176151`, `causality_passed: True`, `num_params_M: 55.7`** (matches, no config drift). Beats the old headline (`n_embd=384`, 581 steps, `1.197595`) by `0.021444` despite 81 fewer steps — the wider model's per-step quality gain outweighs the reduced step count at the real full budget. **This is the final headline number for the Trn2 phase.**
- **Keep/discard gate**: the winning combination is whichever reaches the lowest val_bpb at the real 30-minute ceiling — settled here by the full-budget confirmation above, with the 900s sweeps as a (now-verified-consistent) proxy, not a substitute.

## Solution 12 — NKI kernel work — Step 2 (rope+QK-norm fusion) done, kept, defaulted ON

Cross-ref: #10 (the kernel-writing part). Bucket **C**. Full evidence trail, exact
`git show` recovery commands, and what was checked and found empty:
`trn2-optimization-research.md`. Sequenced *after* Solution 13 — profile against the
fastest correct config, not a slower one.

**Result, up front**: the fused rope+QK-norm kernel below (`--nki-rope-norm`, now
**default ON**) delivers a real, measured **~15-18% per-step speedup** at the final
locked config (`DEPTH=6, TOTAL_BATCH_SIZE=131072`, world_size=8), verified to 1-bf16-ULP
numerical equivalence against eager (forward and backward), causality exact. Full-1800s
confirmation with the kernel ON at this config is done: **581 steps, `public_val_bpb =
1.197595`** — vs. the old kernel-OFF number (560 steps, `1.199914`), both more steps and
a lower bpb. **This is the new final headline number for the Trn2 phase.** Full detail,
every number and the two real bugs caught while building it: `trn2-experiment-results.md`.

**Step 1 — profile before writing anything. Done, real data, changed the plan.**
`neuron-explorer inspect` only gave coarse `SystemProfile*` event counts (no per-engine
breakdown); a real `Instruction`/`DmaPacket`-level capture (via the
`neuron-nki-profile-analysis-agent`, `torch_neuronx.profiling.NeuronConfig` externally
driven, `train.py` never touched) found the real bottleneck is **launch/dispatch
serialization, not any single engine** — 72.2% of the profiled window is the gap to
`perfect_pipeline`; DMA/TensorE/GpSimd/Vector/Scalar are all within a 2.4x band of each
other (28% down to 12%). Root cause: the native TorchNeuron eager backend compiles one
NEFF per aten op, zero cross-op fusion — 689 NEFF launches in one micro fwd+bwd, only
61.9% device busy. Full detail, every number: `trn2-experiment-results.md`.

**Step 2 — fused rope + QK-norm, forward and backward. Done, kept, defaulted ON.** The
original call-frequency reasoning (RMSNorm ~42 calls/forward, highest priority) was wrong — it didn't account for the fact that `F.rms_norm` already lowers to
a *single* fused NEFF per call, while `apply_rotary_emb` (`train.py:372`) has **no fused
lowering at all** (15 separate NEFF launches per call: slice+mul+add+cat+neg). Rope
costs 2.1x more than norm() in the real profile (12.63ms vs. 6.00ms of a 91.445ms
window). Fusing exactly `q = norm(apply_rotary_emb(q, cos, sin))` /
`k = norm(apply_rotary_emb(k, cos, sin))` (`train.py:411-412`) collapses 204 of 689
launches, saving ~16% of the profiled window — **but honestly size this as ~8% of real
step time**, since the profiled device window only covers ~50% of the real 2.90s/step
(the rest is host-side dispatch/dataloader, out of scope for this pass). Also cuts fp32
elementwise HBM traffic (`q`/`k` currently emerge fp32 post-`norm()` even though the
model is bf16-compute) — a bf16 kernel roughly halves that further.

References for the two halves, both still forward-only (backward is ours to write, same
as before — simplified for the norm half since it's parameter-free, only `dx`):
- RoPE: `git show 8370e11^:contributed/neuron-team-kernels/RoPE.py` — four
  `nisa.tensor_tensor` ops for the rotate-half form. Layout warning: expects `[d_head, S]`
  (head-dim on partitions, `d_head ≤ 128` — our `HEAD_DIM=128` fits), transposed vs. our
  `[B,S,H,D]`; `transpose_and_RoPE(...)` in the same history's `llama3_transformer.py:44`
  shows the fused transpose+RoPE variant — the natural place to also fold in the norm.
- RMSNorm half: `git show 8370e11^:contributed/neuron-team-kernels/rmsnorm_quant.py` —
  `nisa.activation_reduce(op=nl.square, ...)` fuses square+reduce into one instruction,
  `nisa.activation(op=nl.rsqrt)`. Also try the builtin `nl.rms_norm(x, w, axis=-1, n=D,
  epsilon=...)` first.
- Template for the fwd/bwd + `torch.autograd.Function` structure: the shipped
  `_relu_sq_fwd`/`_relu_sq_bwd`/`_ReluSquaredNKI` (`train.py:281-330`).
- Interacts with Solution 13c: if QKV is fused, this kernel has a cleaner single input.

**Step 2b — RMSNorm alone: demoted, not a separate kernel target.** Already lowers to
one fused NEFF via `F.rms_norm` — no launch-count win from fusing it in isolation.
Only relevant as the norm half of the combined rope+QK-norm kernel above.

**Confirmed NOT targets, from the real profile**: `aten::linear`/`linear_backward` (the
real compute, TensorE already hits 80.4% of peak FLOPS when active — nothing to fix via
kernel work) and attention itself (`_scaled_dot_product_fused_attention_overrideable`,
already fused, only ~12% of its FLOPs are transposes) — consistent with Solution 8's
earlier finding that a custom windowed-attention kernel wasn't motivated either.

**Second-priority candidate if rope+QK-norm doesn't fully pan out**:
`aten::embedding_dense_backward` — 4 executions, 5.86ms (6.41% of the profiled window),
GpSimd-dominated.

**Step 4 — logit softcap fused into cross-entropy.** The `nl.multiply`+`nl.add`+`nl.exp`
→ single `nisa.activation(nl.exp, data, bias, scale)` combining trick applies directly
to our `tanh` softcap + CE log-sum-exp. **Honest gap**: no standalone softmax/CE kernel
exists anywhere in `nki-samples`, HEAD or history — this is from scratch unless the
`nkilib` Cross Entropy Kernel (Release Notes) is importable on the real SDK. Our vocab
(8192) is small, so the win here is instruction-count, not memory.

**Step 5 — attention kernel, gated on Step 1, not attempted by default.** If profiling
justifies it: the only fwd+bwd flash attention in any cloned repo is
`git show c4b806e^:src/nki_samples/reference/attention.py` (`flash_fwd`/
`flash_attn_bwd`); everything in HEAD (`tutorials/attention_fwd_performance/attention_kernels.py`'s
`attn_fwd_v1`…`v10` ladder) is forward-only. Structure: SBUF ~2MB forces K/V tiling in
128-column chunks; TensorE 128×128 matmuls; VectorE online-softmax rescale in-place; one
`@nki.jit` function to avoid multi-launch overhead. Engine mapping: matmul→TensorE,
exp→ScalarE, sum→VectorE, causal masking→GpSimdE. Solution 8's GPU finding (both
windowed-attention sub-components independently regressed) means only a *plain* causal
kernel is on the table, for throughput only.

**Cross-cutting constraints**: namespace moved `neuronxcc.nki.*`→`nki.*` in SDK 2.28,
`nl.par_dim` removed from internals in 2.29 — check the installed version before copying
any API verbatim (recovered history is ≤SDK 2.32). SBUF buffers declared inside the
innermost loop, not hoisted. Tiles want ≥128 elements/partition; DMA transfers >32 KiB.
NKI runs on a single NeuronCore — verify (don't assume) it behaves correctly invoked
from 8 parallel DDP processes. Check `nkilib` prebuilt kernels before hand-rolling
(RoPE, `rmsnorm_tkg`, Cross Entropy, Attention CTE/BWD/Block-TKG — caveat: TKG is
inference-only, likely doesn't apply to training). Vendor any adapted AWS sample code
into the submission (compliant per README's "optional custom NKI kernels" clause; pip
installs beyond the provided environment are not) — attribute in a comment.

- **Process**: author with `neuron-nki-writer-agent`, debug compile errors with
  `neuron-nki-debugger-agent`, validate numerical equivalence against the eager
  reference with `neuron-framework-equivalence` before trusting any kernel in a scored
  run.
- **Keep/discard gate**: numerical equivalence check must pass before a bpb comparison means anything — a shape-correct but numerically-off kernel can still pass `causality_check` while quietly moving val_bpb the wrong way.
- **Sequencing**: after Solution 13, before or interleaved with Solution 11 depending on profiling results — kernel-writing effort should target whatever the *current* architecture's actual bottleneck is.

---

## Trn2 first-hour checklist — do this before spending any real budget

Everything below is genuinely new — none of it could be tested on either GPU this session. Do it in this order.

**1. Environment.** Launch from the README-specified DLAMI (`ami-0d5703e99c1589f8d`, venv `neurips-torchneuronx-native-beta-2.11.3-20260824`). `pip install -r requirements.txt`, `python prepare.py setup`. Sync `train.py`/`prepare.py` (checksummed against local before trusting), the checkpoints (`checkpoints/final-confirmation-v2.pt` at minimum), and `research/` (needed for `run_experiment.py`'s git calls — `.git` must come along too, same lesson learned twice this session with GPU rentals).

**2. Real-hardware verification, same discipline as every GPU box this session.** Don't assume `torch_neuronx` works just because it imports — run an actual op:
```python
import torch
x = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)
y = x @ x
torch.neuron.synchronize()
```
Then the same portability check already proven twice on GPU: `--init-from checkpoints/final-confirmation-v2.pt --num-steps 10`, confirm `causality_passed=True` and no `WARNING [SHAPE-CRITICAL]` toggle mismatch in the log (check the log, not just the number — a real self-inflicted mistake this session came from trusting the bpb number without checking for a printed mismatch warning first).

**3. `record_step_boundary` — verified locally on L4 this session, but only in single-device mode.** Confirmed it correctly appends one line per completed step to `NEURON_COMPETITION_R1_STEP_LOG` when set (previously an unverified assumption from reading the PR #6 diff — now real). Re-confirm once on Trn2 too, cheap and quick (10-step run with the env var set), since Neuron's process/threading model could in principle behave differently than CUDA's.

**4. The one thing that has never been tested at all, on any hardware, all session: multi-core (`nproc_per_node > 1`).** Every GPU experiment this whole session ran single-device (`--nproc-per-node 1`) — the combined bad-loss/predictive-stop `dist.all_reduce` logic (`train.py`, added as part of the PR #6 merge) has never been exercised with `world_size > 1`. Before trusting any multi-core result: run a short (~20-step) smoke test at the **real submission shape** (`--nproc-per-node 8 --lnc 1`), confirm it completes without a distributed-hang or mismatched-stop-decision issue (one rank deciding to stop while others continue would be exactly the bug this combined-tensor design was meant to prevent — verify it actually does, don't just trust the code read). This is squarely the "utilizes all NeuronCores" data-parallel behavior discussed earlier — same model per core, gradients synced once per step, per-core memory footprint independent of core count — but it's still a real code path that has literally never run.

**5. Once 1-4 pass, re-run the locked config from scratch (fresh init, no warm-start) at the real submission shape and a real step count, and compare against the GPU numbers (`1.165273` / `1.168849`).** This is the third independent hardware confirmation, and the first one that's actually target-hardware, not a proxy — treat any large deviation as a real finding to investigate, not noise to explain away (same standard applied to every GPU→GPU and Karpathy→GPU comparison this whole session).

**6. Warm the Neuron persistent cache before any timed run, on local disk, not a network FS.** Evidence: the EKS doc's separate ahead-of-time compile job producing NEFFs into a persistent cache, and HLAT's report that shared FSx caused communication bottlenecks/timeouts. PR #6 already excludes compile time from the charged budget, so this is now a dev-iteration-speed win, not a budget-correctness one — but still the single most valuable operational idea from that source. Exact env var (Gemini Deep Research cross-check): `NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` (already added in Solution 13a). Use `--compile-only` to force the warm-up. Open question surfaced by the same cross-check: the SDK also ships `neuron_parallel_compile`, an official CLI that traces the graph across all ranks and multi-threads the lowering — check on the real box whether it works against our custom `train.py` directly, or whether our own `--compile-only` path is the right tool and `neuron_parallel_compile` doesn't apply here.

**7. Run `neuron-top` alongside every real run**, not just as a profiling step — it's the standing "are we actually using all 8 cores" check, which matters for the correctness gate ("submissions must run on the full chip"), not just performance.

**8. Then, the three genuinely Trn2-only Solutions, in order: 13 (compiler/runtime config wins, near-zero risk, do first), 11 (batch/depth resweep, at the real 30-minute budget — also test the two HLAT quality toggles here per the deferred-to-Trn2 decision: AdamW β2=0.95 vs. our current 0.995, and output-projection init scaled by 1/sqrt(2·n_layer) — record the result honestly even if it doesn't transfer from 7B/1.8T-token scale), and 12 (NKI kernel work, profiling-first)**. All detailed above. **Coupling flagged by the Gemini Deep Research cross-check**: if 13b (hardware stochastic rounding) is enabled, test it *together with* the β2=0.95 toggle, not independently — the claimed mechanism is that stochastic rounding's added gradient variance is specifically what makes a long β2 window (0.99+) unstable, so enabling SR without also lowering β2 may reproduce that instability rather than testing two unrelated toggles.

**9. LNC=1 stays the default; only revisit if Solution 11 pushes the model meaningfully larger.** The Gemini cross-check's own hardware table puts the LNC=1-vs-LNC=2 crossover around ~100M params (LNC=1/pure-DP=8 optimal below that, LNC=2/hybrid TP+DP optimal above ~500M, memory-bound) — confirms, doesn't change, our current default. ZeRO-1 optimizer-state sharding across the 8 ranks was also surfaced as available (`neuronx_distributed`) but is **not worth adopting at our scale**: optimizer state for 85.9M params is ~1.0-1.4 GB/rank against 12 GiB/rank available under LNC=1 — no memory pressure to justify the added dependency and complexity. Revisit only if 11's resweep changes that math.

**Researched and found not applicable — don't re-explore**: `neuron-workshops`' only training code is LoRA SFT via `optimum.neuron` with `tensor_parallel_size=2`, not DDP. `nanochat-contribution-fork` has zero Neuron code (CUDA/FlashAttention-3-targeted); useful only as an architectural lineage reference, already covered in `gpt-model-comparison.md`. Full detail: `trn2-optimization-research.md`.

---

## After Solution 12/13

Once Solutions 1-13 have all landed and been measured, hand the resulting checkpoint to `autoresearch` to mine the local hyperparameter neighborhood — see `enhancement-opportunities.md`'s "Rollout plan" section for the full rationale (why sequencing this way, the 6x-more-expensive-per-experiment argument for a 30-minute Trn2 run vs. Karpathy's 5-minute GPU cycle). Only spend real, rate-limited (5/week) Phase 1 submissions once local search on the fully-updated baseline has converged, not before.

Item #12 in the source doc (Mamba2/hybrid architecture swap) stays excluded from this plan — see its own verdict in `enhancement-opportunities.md` #12.
