# Experiment Rollout Plan — GPU Phase Complete, Trn2 Readiness Plan

This is the execution companion to `enhancement-opportunities.md`. That doc establishes *what* to change and *why* (rationale, evidence, portability bucket A/B/C) — this doc only adds *sequencing and measurement protocol*. Rationale/evidence for each item lives in the source doc; follow the `#N` cross-references there, not here.

> **Status as of this update**: Solutions 1-10 are **done and validated on two independent GPUs** (RTX 6000 Pro Blackwell, then NVIDIA L4 Ada — different vendor generation, different `torch`/CUDA build), not just planned. Full detail, every number, every keep/discard call: `gpu-experiment-results.md`. This section below is now a status summary, not a to-do list — jump to "Trn2 readiness plan" for what's actually left.

## GPU phase — final status

Locked config: Solutions 1, 2, 3, 5, 6 (with `PACK_FACTOR` re-swept to `2`, not the originally-locked `8`), 7, 9a, 10 kept. Solutions 4, 8, 9b discarded (real regressions). Post-lock exploration (RoPE-base-alone, windowing-alone, VE weight decay, selective per-token gating) — four further clean negative results, closing out low-effort GPU-testable options. Custom packer w/ remainder-requeue remains the one deliberately-untried candidate (weak motivation, high risk — see `gpu-experiment-results.md`).

**Cross-hardware portability confirmed with real numbers**: `1.165273` (RTX 6000 Pro) vs `1.168849` (L4) for the identical config — 0.31% relative difference, both causality-exact. This is real, independent evidence the architecture/schedule findings are properties of the data/model, not an artifact of one specific card.

**Solutions 11 (batch/depth resweep) and 12 (NKI kernels) were never GPU-testable** — both inherently need real Trn2 (throughput-dependent, or Neuron-only respectively). They're the actual remaining work, detailed below.

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

## Solution 11 — Batch size + depth/width resweep

Cross-ref: #7 / #8.

- **Change**: `DEPTH` (`train.py:42`), `ASPECT_RATIO` (`:43`), `DEVICE_BATCH_SIZE` (`:46`), `TOTAL_BATCH_SIZE` (`:47`).
- **Protocol**: **[GPU]** dry-run the sweep script/procedure only here — confirm the bisection logic (vary depth/width/batch, measure tokens/sec, find where more steps stop compensating for a smaller/shallower model) runs end-to-end and produces sane output. Do not trust the resulting optimal values — GPU throughput has nothing to do with Trn2 throughput. **[Trn2]** this is where the real sweep happens: run after Solutions 1-10 have landed, since the optimum shifts once schedule/VE/windowing/packing changes are in; measure tokens/sec and val_bpb at the actual 30-minute wall-clock budget (`--max-train-seconds` = the real `TRAIN_TIME_BUDGET_SECONDS`).
- **Sub-sweep**: **[Trn2]** each distinct `(DEPTH, ASPECT_RATIO, DEVICE_BATCH_SIZE)` combination is a different compiled shape (~60s NEFF compile each) — group runs by shape and batch them to amortize that cost rather than round-robining across shapes. Not applicable on the GPU box.
- **Keep/discard gate**: N/A on the GPU box (tooling check only). **[Trn2]** the winning combination is whichever reaches the lowest val_bpb at the real 30-minute ceiling, not the most steps or the biggest model in isolation.

## Solution 12 — NKI kernel work

Cross-ref: #10 (the kernel-writing part). Bucket **C**.

- **Change**: candidates in priority order (per the source doc's call-frequency/tensor-size reasoning): fused RMSNorm (`norm()`, ~42 calls/forward), fused rotary embedding, fused logit softcap (ideally fused into cross-entropy), and — only if profiling shows it's warranted — a flash-attention-style kernel for `CausalSelfAttention` (see Solution 8's note on this — both its sub-components independently regressed on GPU, so a custom windowed kernel is *not* motivated by that finding; only pursue this specific one if profiling shows attention itself is the bottleneck for a different reason). Template: `_relu_sq_fwd`/`_relu_sq_bwd`/`_ReluSquaredNKI` (`train.py:169-222`).
- **Process**: profile first with `neuron-nki-profiling`/`neuron-nki-profile-analysis-agent` (deployed this session) to confirm the actual bottleneck before writing anything — don't write a kernel based on source-reading assumptions alone. Author with `neuron-nki-writer-agent`, debug compile errors with `neuron-nki-debugger-agent`, validate numerical equivalence against the eager reference with `neuron-framework-equivalence` before trusting any kernel in a scored run.
- **Keep/discard gate**: numerical equivalence check must pass before a bpb comparison means anything — a shape-correct but numerically-off kernel can still pass `causality_check` while quietly moving val_bpb the wrong way.
- **Sequencing**: last Solution to attempt, after every other win (Solutions 1-11) is already banked, since kernel-writing effort should target whatever the *final* architecture's actual bottleneck is, not an intermediate one.

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

**6. Then, and only then, the two genuinely Trn2-only Solutions**: 11 (batch/depth resweep, at the real 30-minute budget) and 12 (NKI kernel work, profiling-first). Both detailed above.

---

## After Solution 12

Once Solutions 1-12 have all landed and been measured, hand the resulting checkpoint to `autoresearch` to mine the local hyperparameter neighborhood — see `enhancement-opportunities.md`'s "Rollout plan" section for the full rationale (why sequencing this way, the 6x-more-expensive-per-experiment argument for a 30-minute Trn2 run vs. Karpathy's 5-minute GPU cycle). Only spend real, rate-limited (5/week) Phase 1 submissions once local search on the fully-updated baseline has converged, not before.

Item #12 in the source doc (Mamba2/hybrid architecture swap) stays excluded from this plan — see its own verdict in `enhancement-opportunities.md` #12.
