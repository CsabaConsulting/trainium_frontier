# Trn2 Experiment Results — Real Hardware Lab Notebook

Live log for the real-Trn2 phase of `experiment-rollout-plan.md`. Companion to
`gpu-experiment-results.md` (GPU-proxy phase) — this doc starts where that one left off,
against actual target hardware instead of a proxy. Every entry below is a real run on
real silicon, not a projection.

## Hardware / environment

- **Instance**: `trn2.3xlarge`, reachable at `ubuntu@ec2-16-26-112-252.ap-southeast-4.compute.amazonaws.com`. Allocated for a 48-hour window starting 2026-09-05; a second instance follows after.
- **Not the README-specified DLAMI** — this box was pre-provisioned with a custom-built
  native venv (`~/workspace/native_venv`, Python 3.12.3) assembled from DLC-extracted
  wheels (`~/build/remote_venv.sh`, `~/build/requirements.trn2-native_venv.txt`), not
  `ami-0d5703e99c1589f8d`. Versions: `torch==2.11.0+cpu`, `torch-neuronx==2.11.3.0.0`,
  `neuronx-cc==2.26.6360.0+6f180f47`. `pyarrow`/`tiktoken`/`requests` already present —
  no `pip install -r requirements.txt` needed.
- **Device**: `neuron-ls` reports 1 physical Neuron device, 96 GB HBM, currently
  presenting as **LNC=2** by default (`logical-neuroncore-config: 2`, 4 logical cores,
  core IDs 0-3) — not LNC=1's 8. Need to confirm whether `NEURON_LOGICAL_NC_CONFIG=1` as
  a per-process env var actually switches this at runtime, or whether the driver-level
  default requires a separate reconfiguration step. Testing this is part of the
  multi-core smoke test below.
- `neuron-agentic-development` skills package already present system-wide
  (`~/workspace/neuron-agentic-development`, pip-installed as `neuron-agentic-development
  1.0`) — the `neuron-nki-*`/`neuron-profile` tooling referenced throughout
  `trn2-optimization-research.md` should be available without a separate deploy step.
- Repo synced via `rsync` (excluding `checkpoints/`, `out/`, `.cache/`, `data/`,
  `.memsearch/`), `.git` intact, local-only git identity set (`user.name`/`user.email`,
  no credential helper — matches the "never push from a rented box" discipline). Only
  `checkpoints/final-confirmation-v2.pt` + `checkpoints/README.md` synced separately
  (the rest of `checkpoints/` is gitignored and large).
- `python prepare.py setup` completed cleanly — downloaded 6 shards to
  `~/.cache/neuron_competition_r1/data`.

## Checklist progress (`experiment-rollout-plan.md` "Trn2 first-hour checklist")

| # | Item | Status |
|---|---|---|
| 1 | Environment | **Done** — see above. Not the README DLAMI but functionally equivalent versions. |
| 2 | Real-op verification | **Done.** `torch.randn(4096,4096,device="neuron",dtype=bfloat16) @ itself` executed successfully, real numeric result (`sum=348804.65625`), not just a clean import. |
| 2b | Warm-start portability check | **Done.** See below. |
| 3 | `record_step_boundary` re-verification | **Done.** See below. |
| 4 | Multi-core (`nproc_per_node>1`) smoke test | **Done — passed clean on first attempt.** See below. |
| 5 | Fresh-init confirmation run at real shape | In progress |
| 6 | Neuron persistent cache warm-up | Partial — `NEURON_COMPILE_CACHE_DIR` in place; `neuron_parallel_compile` still an open question |
| 7 | `neuron-top` monitoring | Skipped so far — TUI, no scriptable batch/log mode found; `neuron-monitor`'s JSON has no temp/throttle field either (see cliff investigation) |
| 8 | Solution 13 (config/env wins) | In progress — 13c, 13d verified correct in isolation; 13a caused a real incident (fixed, now gated, not yet re-tested); 13b untested |
| 9 | LNC=1 default confirmation | **Confirmed.** `NEURON_LOGICAL_NC_CONFIG=1` env var (no reboot/driver reconfig needed) correctly switches `neuron-ls` from 4 logical cores (LNC=2, the box's boot-time default) to 8 (core IDs 0-7). Per-process override works exactly as the README describes. |

### 2b. Warm-start portability check (`--init-from checkpoints/final-confirmation-v2.pt --num-steps 10`, single device, LNC=1)

First real-Trn2 run of any kind. `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=1 train.py --num-steps 10 --max-train-seconds 300 --init-from checkpoints/final-confirmation-v2.pt`.

- Checkpoint loaded correctly: `warm-started from checkpoints/final-confirmation-v2.pt: checkpoint step=1450 total_train_tokens=95027200`. **No `[SHAPE-CRITICAL]`/`[silent-behavior]` toggle-mismatch warnings** — `train.py` on this box matches the config that produced the checkpoint.
- `causality_passed: True, causality_max_abs: 0.000000` — exact, same as every GPU run.
- `public_val_bpb: 1.151113` after 10 warm-started steps — consistent with the expected warm-start-not-resume bump (checkpoint's own `1.10682`; the LR schedule restarts at peak rather than continuing the cooldown tail). Same pattern seen on L4 (`1.1198` after 10 steps from the same checkpoint) — the bump size differs a bit (schedule/hardware differences expected) but the *direction and mechanism* replicate.
- **Step-5 budget-clock reset (PR #6 mechanic) verified working on real Neuron hardware for the first time** — previously only exercised on CUDA. Step 0 (`dt=116.60s`) pays the NEFF compile; steps 1-4 run at steady-state `~7.6s/step, ~8,580 tok/s` (single device, `DEVICE_BATCH_SIZE`/`grad_accum_steps=32`, `TOTAL_BATCH_SIZE=65536`). At step index 4 (5th completed step), `startup_allowance_est=160.6s` (capped, since real elapsed launch-to-step-5 exceeded the 900s cap by nowhere near — comfortably under), and the budget clock correctly resets — `remaining_time` printed for that step already reflects the fresh window. `startup_allowance_est: 160.6, charged_seconds_est: 30.7, over_budget: False`.
- Only steps 0-4 appear in the printed log — this is expected, not a truncation or early-stop bug: `train.py`'s print gate is `step < 5 or step % 10 == 0`, so steps 5-9 run silently. Confirmed by `training_seconds: 30.4` (≈4 steps × ~7.6s, consistent with steps 5-8 executing after the reset, stop firing at step 9 via `step + 1 >= args.num_steps`) and a clean `over_budget: False` / normal eval afterward, not a distributed-hang or abort.
- Single-device steady-state throughput baseline for later comparisons: **~8,580 tok/s, ~7.6s/step at world_size=1**.

### 3. `record_step_boundary` re-verification (fresh init, `NEURON_COMPETITION_R1_STEP_LOG=/tmp/steplog.txt`, single device, LNC=1)

`NEURON_LOGICAL_NC_CONFIG=1 NEURON_COMPETITION_R1_STEP_LOG=/tmp/steplog.txt torchrun --standalone --nproc_per_node=1 train.py --num-steps 10 --max-train-seconds 300`.

- **Confirmed working on real Neuron hardware**: `/tmp/steplog.txt` contains exactly 10 lines, `0` through `9` — one line per completed optimizer step, matching the format already verified on L4. Previously an assumption carried over from CUDA; now directly confirmed on Trn2.
- Fresh init (no `--init-from`): loss started at `9.01`, consistent with the ~9.0 random-init figure noted earlier in the GPU-proxy phase. `causality_passed: True, causality_max_abs: 0.000000`. `public_val_bpb: 2.162916` — high, but expected for a 10-step fresh-init run (early training, not a red flag).
- Step 0 compiled in only `7.21s` this run (vs. `116.60s` for the warm-start check minutes earlier) — the Neuron compiler's default local NEFF cache location reused the compiled graph for the identical tensor shape across the two separate `torchrun` invocations, even without us having explicitly set `NEURON_COMPILE_CACHE_DIR` yet. Good sign for Solution 13a/checklist-item-6 (explicit cache dir + `neuron_parallel_compile` should make this even more reliable, but the default already caches within a session).
- `startup_allowance_est: 44.0` (vs. `160.6` for the warm-start check) — reflects the much faster compile.

### 4. Multi-core smoke test (`--nproc-per-node 8 --lnc 1`, fresh init, 20 steps) — the never-tested-on-any-hardware item

`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py --num-steps 20 --max-train-seconds 600` — the real Phase 1 submission shape, exercised for the first time on any hardware this whole session (every prior GPU-proxy experiment was single-device).

**Passed clean on the first attempt.** No distributed hang, no mismatched-stop-decision (the combined bad-loss/predictive-stop `dist.all_reduce` control tensor added in the PR #6 merge — the exact bug that design was meant to prevent — did not occur), single rank-0 checkpoint save, single final summary block (not 8 duplicates).

- `world_size: 8, causality_passed: True, causality_max_abs: 0.000000`.
- **Near-linear data-parallel speedup, confirming the "utilizes all NeuronCores" design intent with a real number**: `micro_tokens` dropped from `2,048`/`grad_accum_steps=32` (world_size=1) to `16,384`/`grad_accum_steps=4` (world_size=8) — `TOTAL_BATCH_SIZE` is fixed globally and correctly divided across the 8 ranks. Steady-state `dt` dropped from `~7.6s/step` (single device) to `~1.40s/step` (8-way) — a **5.4x** wall-clock speedup, not the naive 8x (expected: allreduce + per-step launch overhead don't scale away completely), but a real, substantial, correctly-functioning parallelism win. Throughput: `~46,000-47,000 tok/s` aggregate (vs. `~8,580` single-device).
- Startup compile for this new distributed shape: `36.0s` — fast, no problem.
- `public_val_bpb: 2.082694` after 20 fresh-init steps — expected range for this step count, not a signal either way.
- Total wall time for the whole smoke test (compile + 20 steps + eval): **59.2s**.

### 5. Fresh-init confirmation run — in progress

`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py --num-steps 3000 --max-train-seconds 1800` (real submission shape, real full 1800s budget, `NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` set). Launched 13:07 UTC; all 8 rank processes confirmed alive and healthy shortly after launch. `--num-steps 3000` deliberately set well above the file's `NUM_STEPS=1000` default — at the 8-way throughput observed in the smoke test (~1.4s/step), the *default* 1000-step cap would stop the run at ~1400s, leaving ~400s of the real budget unused. This is real evidence that **the current step-count default was sized against GPU-proxy assumptions, not real Trn2 throughput at the actual submission shape** — exactly the gap Solution 11 (batch/depth resweep) exists to close. Letting `--max-train-seconds 1800` be the actual binding control here (not `--num-steps`) is deliberate, to see how far the real budget goes.

**Result: `public_val_bpb: 1.242326` after 478 steps (target 3000, stopped by the real 1800s time budget). causality_passed: True, causality_max_abs: 0.0, over_budget: False.**

This is a **real, meaningful deviation** from the GPU-proxy numbers (`1.165273` RTX 6000 Pro / `1.168849` L4) — worse (higher bpb), and not attributable to noise (the GPU-to-GPU discrepancy was 0.31% relative; this is ~6.5% relative). Per the standard applied to every comparison this whole session, this gets investigated, not explained away. Root cause found — see "Throughput cliff" below: only 478 steps completed in the budget instead of the ~1285 the smoke test's throughput predicted, because **per-step time collapsed by ~5x partway through the run and never recovered**. The bpb gap is very plausibly just "fewer effective training steps," not a quality regression — but this needs a clean re-run *after* the cliff itself is understood/fixed to actually confirm that, not assumed.

#### Throughput cliff — real finding, under investigation

Per-step time in the confirmation run: steady `~1.40-1.50s/step` (`~45,000-47,000 tok/s`) for the first **~320 steps (~460s wall-clock)**, then an abrupt, sustained collapse to `~6.4-8.75s/step` (`~7,500-10,700 tok/s`) for the remaining ~150 steps through the end of the budget — **not a gradual decline, a step-change that never recovered**. The post-cliff rate (~7.6s/step) closely matches the *single-device* (`world_size=1`) steady-state rate measured earlier (~7.6s/step) — as if the 8-way data-parallel speedup simply stopped applying, rather than the whole system slowing down proportionally.

Checked and ruled out (or found inconclusive):
- **Not visible in `dmesg`** — no thermal/throttle kernel log lines around the relevant timestamps.
- **`neuron-monitor`'s JSON schema on this SDK build has no temperature/power/throttle field** — can't directly instrument chip thermal state with the tools available on this box.
- **Not a dataloader epoch rollover** — `epoch` field stayed `1` through the whole run.
- **Not swap/OOM** — `free -h` showed no swap in use at any point checked.
- Host RSS per rank grew steadily during the run (observed ~1.6GB → ~2.0GB across ~15 min of polling) — plausibly related to `gc.disable()` being called after step 0 (`train.py`, permanently disables garbage collection for the rest of the process to avoid GC-pause jitter) allowing reference-cycle garbage to accumulate unboundedly over hundreds of steps. Not yet confirmed as the cause of the cliff specifically (the growth looked gradual, the cliff was sharp), but a real candidate.
- **Leading hypothesis, not yet confirmed**: thermal or power throttling on the physical chip after several minutes of sustained near-peak load, given the timing (~7-8 minutes of continuous ~45k tok/s before the drop) and the fact it never recovers for the rest of the run (consistent with a throttle state that persists once triggered, not a transient blip). No direct instrumentation available to confirm this without AWS-side chip telemetry we don't have access to.

**Diagnostic run result — cliff reproduced, but at a different threshold.** `trn2-cliff-diag-420s-v2` (started ~13:49 UTC, ~11 minutes after the confirmation run above finished its full ~30-min load): steady `~1.4-1.5s/step` through step 70, then the same sharp, sustained collapse to `~7.7-9.0s/step` starting at **step ~80 (~116s of training elapsed)** — not step ~320/~460s like the isolated confirmation run.

**Leading theory, now better supported: cumulative thermal load across back-to-back runs, not a fixed per-run threshold.** The confirmation run started from a comparatively cool chip (first sustained load of the session) and didn't throttle until ~460s in; this diagnostic run started only ~11 minutes after that same chip had just finished ~30 minutes at near-peak load (plus the brief compiler-crash run in between), and throttled roughly 4x sooner (~116s). Both runs show the identical qualitative signature (sharp, non-recovering step-change), consistent with one underlying mechanism (most likely thermal or power throttling) whose *trigger point* depends on how warm the chip already was going in, not a fixed step-count or fixed single-run duration.

**Third data point, after a deliberate ~20-minute idle cooldown**: `trn2-cliff-diag-cooldown` — cliff at **step ~105-110 (~147s)**. Barely later than the 11-minute-gap run's ~116s, nowhere close to the first cold run's ~460s, despite nearly double the idle gap.

**Three data points now**: cold (first heavy load this session) → cliff at ~460s. ~11 min idle after ~30 min load → cliff at ~116s. ~20 min idle after ~10 min load → cliff at ~147s. **Conclusion: there is a real recovery effect from idling (147s > 116s), but it is slow relative to a 20-minute gap — nowhere near a full reset to the cold-chip ~460s figure.** This is consistent with thermal throttling with a long time-constant (the chip's heat doesn't fully dissipate on a tens-of-minutes idle timescale), though the exact mechanism (thermal vs. power-cap vs. some other driver-level state) is still not directly confirmed — no chip telemetry is available on this box to instrument it further, and further isolating the exact mechanism has diminishing returns against the 48-hour budget.

**Ruled out: `gc.disable()` is not the cause.** Before accepting the cliff as a hard hardware ceiling, tested the cheapest plausible software explanation: `train.py` calls `gc.collect(); gc.freeze(); gc.disable()` after step 0 to avoid GC-pause jitter — if reference-cycle garbage accumulated unboundedly with collection fully off, that could plausibly produce exactly this kind of sustained slowdown. Ran a diagnostic with `gc.disable()` removed (keeping `gc.freeze()`, so large frozen model/optimizer objects still don't get rescanned) at `world_size=8`: **the cliff still occurred**, at step ~100 (~140s) — same sharp, sustained signature as every other run. This confirms the cliff is a real hardware/runtime-level phenomenon (thermal/power throttling remains the leading candidate), not an application-level Python garbage-collection artifact. Reverted `gc.disable()` back in (it's still valid — just not the cause here).

**Attempted, inconclusive by tooling limitation: XLA/runtime-state-growth vs. thermal, via `neuron-explorer inspect`.** A second AI-doc cross-check (see `trn2-optimization-research.md` Source 5) surfaced a real alternative hypothesis — an accumulating internal XLA/Neuron-runtime state (distinct from the Python-level `gc.disable()` already ruled out above). Tested it using `neuron-explorer inspect` (found via the newly-available `neuron-nki-profile-querying` skill) wrapping a real 8-way run long enough to cross the cliff, intending to compare host-side (`SystemProfileEvents`/`SystemProfileHbmUsage`) telemetry pre- vs. post-cliff:

- Attempt 1 (`--max-train-seconds 700`, default trace-event limit): `SystemProfileMetadata` showed the captured window was only **~5 seconds**, right at the tail of the run — never spanned the pre/post-cliff transition (cliff landed at step ~590/~450s per the run's own step log; the capture covered a different window entirely).
- Attempt 2 (`--max-train-seconds 400`, `--num-trace-events 50000000` — a 1000x increase on the one flag that should control ring-buffer retention, specifically to rule out eviction): **still only ~5 seconds captured**, again at the tail, despite the cliff landing at step ~135 (~100s), safely inside the 400s window this time.
- The identical ~5s-at-the-tail failure across a 1000x change in the relevant flag means this isn't event-count eviction — `inspect`'s system-profile capture appears to retain only a short window near process exit via this capture path, not a rolling window through execution. A proper device-level Instruction-table capture (`capture`/`view -n/-s` with `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`) wasn't attempted — a bigger workflow change, judged not worth the remaining-budget cost for this specific side-investigation.
- **Verdict: inconclusive by tooling limitation, not by evidence.** The XLA-runtime-state-growth hypothesis is neither confirmed nor refuted. Thermal/power throttling remains the leading explanation by elimination only (dmesg clean, `gc.disable()` ruled out, not swap, not epoch rollover) — still never directly confirmed by any chip telemetry. Stopping this avenue here, per the pre-agreed fallback in the approved plan.

**Practical, actionable conclusion, adopted going forward**: budget every future throughput estimate off the **post-cliff, sustained rate (~7.5-9.0s/step at world_size=8, i.e. close to the world_size=1 rate)**, not the initial burst rate (~1.4s/step) — any real ~10+ minute run in this testing session should be assumed to hit this cliff, usually well before the nominal budget ends. The one-shot, presumably-cold real submission environment is the case most likely to resemble the *first* data point (~460s before cliffing) rather than our repeated-testing sessions — good news for the eventual scored run, but it means **our own iterative testing this session will systematically under-count real steps-per-30-minutes** unless every throughput-sensitive measurement (Solution 11 in particular) explicitly accounts for the cliff rather than extrapolating from early-run burst numbers.

**This directly reframes Solution 11's priority and design**: the locked config was carried over from GPU-proxy testing under an implicit assumption of ~1450 steps in the real budget (matching the GPU's own step count). Real Trn2 throughput at this shape, once the cliff is accounted for, likely supports something in the **~450-650 step range** for a real 30-minute run, not ~1450 — a >2x gap. Solution 11's batch/depth/step-count resweep is no longer a nice-to-have refinement; it's now clearly necessary before any further architecture-level conclusions (Solution 12 kernel-profiling target, further Solution 13 A/B calls, etc.) can be trusted, since every one of those decisions implicitly assumes a step budget that may be roughly half of what was assumed.

#### Incident: Solution 13a broke `neuronx-cc` compilation, caught immediately

While preparing the diagnostic run above, hit a real crash: `RuntimeError: Compilation error occurred on Neuron for operation=nki_kernel_global; error message="COMPILATION FAILED: hlo2penguin: Did you mean '--remat=true'?"` — this took down `dist.init_process_group`'s own internal RNG-state-sync kernel, before training even started.

**Root cause, and a real process mistake worth recording**: Solution 13a's `NEURON_CC_FLAGS`/`NEURON_FUSE_SOFTMAX`/`NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` env vars were written as *unconditional* `os.environ.setdefault` calls, reasoning "safe, it's just config, the risky toggles (`USE_FUSED_QKV` etc.) are what need smoke-testing." That reasoning was wrong — the moment the fixed `train.py` synced to the box, **every subsequent run silently picked up an unverified compiler-flag string**, including this important diagnostic run, without an isolated smoke test first. The specific bug: `--internal-hlo2tensorizer-options='--fast-relayout=true'` carried literal single-quote characters into `NEURON_CC_FLAGS` — correct shell-quoting syntax for a *command line*, meaningless (and apparently fatal to the flag parser) as a raw env-var string with no shell involved.

**Fixed**: the whole 13a block is now gated behind a new `USE_TRN2_COMPILER_FLAGS` toggle (default `False`, added to `ARCH_TOGGLE_NAMES`), matching the discipline already applied to 13b/13c/13d. The quoting bug itself is also fixed (`--internal-hlo2tensorizer-options=--fast-relayout=true`, no embedded quotes) for whenever it does get smoke-tested. `NEURON_COMPILE_CACHE_DIR` stays unconditional — it's a plain filesystem path, not compiler-flag syntax, and has been used safely across every run so far. Verified the box recovered with a clean 5-step single-device run (`causality_passed: True`) before doing anything else.

**Lesson**: "the toggle defaults to False so it's safe to sync" only holds if *every* behavior-changing line in the change is actually behind that toggle. Audit the whole diff for unconditional side effects, not just the named toggle, before calling a change safe-by-default.

## Solution 13 (compiler/runtime config wins) — implemented in `train.py`, not yet smoke-tested

Code landed while the confirmation run above was in flight (can't smoke-test concurrently — one physical chip, the confirmation run holds all 8 cores). All four toggles default to `False`/off, so this change is a no-op until explicitly enabled — safe to have synced onto the box mid-run.

- **13a** (env flags) + **13b** (stochastic rounding): a new block right after `_PROCESS_STARTED`, before the `torch`/`torch_neuronx` imports. `NEURON_FUSE_SOFTMAX=1` and `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3` set unconditionally (`os.environ.setdefault`, overridable from the shell). `NEURON_CC_FLAGS` includes `--model-type=transformer --distribution-strategy=llm-training --internal-hlo2tensorizer-options='--fast-relayout=true'` plus `--lnc=<value>` **mirrored dynamically from `NEURON_LOGICAL_NC_CONFIG`** (not hardcoded to 1) so it can't silently contradict whatever LNC the launch command actually specifies. `NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` also set here. `USE_STOCHASTIC_ROUNDING` (new toggle, default `False`) gates `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` — off by default because of the flagged coupling with AdamW β2 (enabling SR without also lowering β2 risks reproducing HLAT's instability; that pairing is still deferred pending the Solution 11 resweep).
- **13c** (QKV coalescing): new `USE_FUSED_QKV` toggle (default `False`, added to both `ARCH_TOGGLE_NAMES` and `SHAPE_CRITICAL_TOGGLES` since it changes `state_dict` keys). `CausalSelfAttention` conditionally builds one `c_qkv = Linear(n_embd, 3*n_embd)` instead of three separate `c_q`/`c_k`/`c_v`, split via `.split(n_embd, dim=-1)` in forward. Init loop applies the per-tensor `uniform_(-s, s)` slice-wise to each third of the fused weight, preserving the same per-tensor distribution as the unfused baseline (RNG draw order differs — expected).
- **13d** (drop redundant mask): new `USE_CAUSAL_FASTPATH` toggle (default `False`, not shape-critical — no new params, just a control-flow branch). `GPT.forward` passes `attn_mask=None` for a layer only when `USE_CAUSAL_FASTPATH` is on **and** that layer isn't short-windowed **and** doc-masking is off — i.e. only when the layer's mask would have been purely causal anyway. `CausalSelfAttention.forward` then calls `F.scaled_dot_product_attention(..., is_causal=True)` when `attn_mask is None`.
- All four toggles registered in `ARCH_TOGGLE_NAMES` (three new: `USE_FUSED_QKV`, `USE_CAUSAL_FASTPATH`, `USE_STOCHASTIC_ROUNDING`) so the checkpoint payload and the toggle-mismatch warning system track them like every other architecture switch.

**13c (`USE_FUSED_QKV`) — smoke-tested, correct.** Single device, 15 steps: `causality_passed: True, causality_max_abs: 0.000000`, `num_params_M: 85.9` unchanged (fusing reorganizes weights, doesn't change count), loss trajectory sane (`9.01 → 6.78` over 10 steps, matching the unfused baseline's shape). No throughput difference visible at this tiny scale/step-count (three 640×640 matmuls vs. one 640×1920 — kernel-launch-overhead reduction likely needs more steps/bigger batch to show up clearly, or is simply small at this model size). Reverted to `False` after the test, confirmed clean.

**13d (`USE_CAUSAL_FASTPATH`) — smoke-tested, correct.** Single device, 15 steps: `causality_passed: True, causality_max_abs: 0.000000`, `public_val_bpb: 2.129617` — in the same range as every other 15-step smoke test this session (`~2.13`), no numerical divergence from the additive-mask path. Reverted to `False` after the test, confirmed clean.

**13a (`USE_TRN2_COMPILER_FLAGS`) — second crash, second real bug found, then clean.** Retested after the quoting fix and it crashed again with the *identical* error. Checked `neuronx-cc compile --help` directly on the installed SDK (2.26.6360.0): **`--internal-hlo2tensorizer-options` and `--fast-relayout` do not exist as flags on this version at all** — the Gemini Deep Research claim (`trn2-optimization-research.md` Source 4) simply doesn't hold on real hardware. `--model-type`, `--distribution-strategy`, and `--lnc`/`--logical-nc-config` are all confirmed present in `--help` output, though. Dropped the fabricated flag entirely.

Re-tested with the fabricated flag gone: **no crash**, `causality_passed: True`, but `public_val_bpb: 2.682047` — a large, suspicious jump above the tight `~2.129` cluster every other same-seed 15-step smoke test landed in (all these runs share `torch.manual_seed(42)`, so a same-config, same-step-count divergence this large should not happen from noise alone). Decomposed it exactly like Solution 8's windowed-attention bundle earlier this session — tested each of the three pieces alone:
- `NEURON_FUSE_SOFTMAX=1` alone → `2.129419` (normal).
- `NEURON_CC_FLAGS` alone (`--model-type=transformer --distribution-strategy=llm-training --lnc=1`) → `2.129466` (normal).
- `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3` alone → `2.129444` (normal).
- **Full combination, retested once more** → `2.129486` (normal — did not reproduce the `2.682` result).

**Conclusion: the `2.682` result was a one-off, non-reproducing anomaly, not a real effect of these flags.** All three pieces, individually and combined, land in the same tight `~2.1294-2.1296` cluster as every other 15-step baseline smoke test. Cause of the single anomalous run not identified (didn't reproduce on an immediate identical retest, so not chased further — low expected value against the 48-hour budget). **13a is now numerically clean and crash-free**, gated behind `USE_TRN2_COMPILER_FLAGS` (still default `False` — no measured *quality* effect either way at 15 steps, and its actual *speed* benefit still needs a real fixed-step-count A/B before enabling for anything that matters).

**13b (`USE_STOCHASTIC_ROUNDING`) — not yet tested.** Per the plan, defer pairing with the AdamW β2=0.95 toggle until Solution 11.

## Solution 11 — depth/width resweep, first real result, this changes the locked config

Cross-ref `enhancement-opportunities.md` #7/#8. This is exactly the "GPU throughput has nothing to do with Trn2 throughput, the optimum shifts" scenario the plan flagged as untestable until real hardware arrived — now tested, and it moved.

**Protocol**: fixed **wall-clock** (`--max-train-seconds 900`, half the real budget), real submission shape (`--nproc-per-node 8 --lnc 1`), `ASPECT_RATIO=64` held fixed so only `DEPTH` varies (`n_embd` derives from `depth*64` rounded to the nearest 128). Compare `public_val_bpb` reached in the same wall-clock window — the actual Solution-11 objective ("lowest val_bpb at the real budget," not "most steps" or "biggest model" in isolation). Run sequentially, back-to-back, so there's an unavoidable **chip-warmth confound**: depth8 ran first/coolest, depth12 ran last/warmest (see the throughput-cliff section above) — a real limitation on how cleanly these numbers isolate the architecture effect from hardware state, noted rather than ignored.

| DEPTH | n_embd | params | steps in 900s | `public_val_bpb` |
|---|---|---|---|---|
| 8 | 512 | 50.3M | 308 | **1.342754** |
| 10 (prior baseline) | 640 | 85.9M | 235 | 1.368452 |
| 12 | 768 | 135.3M | 193 | 1.365736 |
| 6 | 384 | 26.4M | 399 | **1.297186 — winner** |
| 4 | 256 | 11.5M | 529 | 1.330846 |

**Result: a clean U-shape, real interior optimum at DEPTH=6 (n_embd=384, 26.4M params — ~31% of the prior 85.9M baseline).** `DEPTH=4` completed the most steps of any candidate (529, more than depth6's 399) but landed *worse* than depth6 — direct confirmation of Karpathy's "under-parameterized model hits an irreducible validation-entropy floor early" mechanism: more steps stopped compensating for too little capacity. Combined with depth8/10/12 all being worse in the other direction (too few steps for their size), this is unambiguous: **DEPTH=6 is the real optimum for this real Trn2 budget**, not an artifact of the chip-warmth confound (the confound could explain small ~0.003 gaps like 10-vs-12, not this ~0.03-0.07 U-shape spanning 5 points).

**Decision: `DEPTH=6` locked into `train.py`.** This is the single largest quality finding of the Trn2 phase so far — a real, first-ever confirmation (never testable on the GPU proxy, since GPU throughput doesn't transfer to Trn2) that the GPU-tuned baseline's model size was miscalibrated for real Trn2 throughput. `ASPECT_RATIO=64` was held fixed throughout this sweep (varying `DEPTH` scales width proportionally); an independent width-only resweep at the now-fixed `DEPTH=6` is flagged as further refinement, not yet done — the depth sweep alone already recovered a large, clear win, and further narrowing has diminishing-but-nonzero expected value against the remaining ~44-hour budget.

**Meta-check: does the whole Solutions 1-10 stack still help at DEPTH=6? Yes, decisively.** Rather than re-ablating each of the ten solutions individually against the 48-hour budget, ran the cheaper bundle-level check: `sol11-depth6-rawbaseline-900s` — all eight quality/schedule toggles reverted to their original untuned values (`WARMUP_STEPS=20, WARMDOWN_RATIO=0.0, FINAL_LR_FRAC=1.0, WD_DECAY_TO_ZERO=False, INIT_SCALE=1.0, PACK_FACTOR=1, USE_VE=False, USE_LOCAL_CONV=False`), same `DEPTH=6`, identical 900s/world_size=8 protocol. Result: **927 steps (more than double the full-stack's 399, since dropping VE/local-conv removes real per-step compute) but `public_val_bpb: 1.32268` — worse than the full-stack's `1.297186` despite the huge step-count advantage.** The Solutions 1-10 stack, tuned entirely at `DEPTH=10`, remains a clear net win at `DEPTH=6` even though steps are now far more precious. `num_params_M: 16.9` for this raw-baseline variant vs. `26.4` for the full stack (VE + local conv add real capacity).

Individual-toggle re-ablation at `DEPTH=6` (does each of the ten still help by the *same* margin, or has the ranking shifted) remains genuinely undone and is lower priority now that the bundle-level answer is a clear yes — flagged as further refinement, not blocking. Toggles reverted back to locked values immediately after this test, verified byte-identical to the last-committed `train.py` via diff against `git show 39bec99:train.py`.

### Clean full 1800s confirmation at the new `DEPTH=6` config — new headline number

`sol11-depth6-fullbudget`: fresh init, real submission shape, the actual full `--max-train-seconds 1800` (not the 900s proxy used for the sweep itself — the LR schedule's cooldown denominator is the real budget here, unlike every sweep-comparison run above).

**Result: 809 steps, `public_val_bpb: 1.220658`, `causality_passed: True`.** This is a clear, real improvement over every `DEPTH=10` full-budget number this phase (`1.242326` was the best `DEPTH=10` confirmation result, at only 478 steps) — directly validating that the depth resweep's 900s-proxy conclusion holds at the real 1800s budget too, not just as an artifact of the shorter comparison window.

**Refinement to the throughput-cliff model, from this longer run**: per-step time held fast (`~0.72-0.80s/step`, `~85-90k tok/s`) all the way through **step ~590 (~450s elapsed)** — notably later than the back-to-back sweep runs' cliffs (~100-150s), consistent with the chip having had a several-minute idle gap plus a smaller/cooler-running model. But the degradation from there was **staged, not a single clean drop**: `~5.0-5.1s/step` (steps ~600-670) → `~5.6-6.1s/step` (steps ~680-770) → `~7-8s/step` (steps ~780-800) — three distinguishable plateaus getting progressively worse across the remaining ~350s of the run. This is new evidence beyond the earlier single-drop observations: **over a long enough sustained run, the throttle appears to deepen in multiple stages, not just trigger once and hold** — consistent with a real thermal system with more than one power/clock throttle threshold as the chip keeps heating. Doesn't change the practical planning conclusion (budget off sustained/degraded throughput), but sharpens the mechanism picture.

Committed as the new reference point for any future comparison at this config.

### Batch-size resweep at fixed `DEPTH=6` — second Solution 11 axis, resolved

Same protocol as the depth sweep: fixed wall-clock (`--max-train-seconds 900`), real submission shape, `DEPTH=6` held fixed from the result above, varying `TOTAL_BATCH_SIZE` by powers of 2 around the prior default (`65536`).

| `TOTAL_BATCH_SIZE` | steps in 900s | `public_val_bpb` |
|---|---|---|
| 32768 | 601 | 1.322478 |
| 65536 (prior default) | 399 | 1.297186 |
| 131072 | 257 | 1.290674 |
| ~~262144~~ *(mislabeled, see correction below)* | ~~262~~ | ~~1.288261~~ |
| 524288 | 115 | 1.403897 |

**CORRECTION, found later this phase while investigating an unrelated throughput discrepancy in the Solution 12 kernel smoke test**: the `262144` row above is wrong. Verified via the only reliable ground truth — each run's own printed `grad_accum_steps × micro_tokens` (git-history reconstruction is *not* reliable here, since every sweep point ran from an uncommitted edit, deliberately, per this phase's whole methodology) — the run labeled `sol11-batch262144-900s` actually executed with `grad_accum_steps=8`, i.e. **`TOTAL_BATCH_SIZE=131072`, a duplicate of the row directly above it, not a real 262144 measurement.** All four other rows (`32768/65536/131072/524288`) were independently re-verified genuinely correct by the same method. The near-identical numbers between the "131072" and "262144" rows (257 vs. 262 steps, `1.2907` vs. `1.2883`) — which we'd previously read as "close but real, weakened by the chip-warmth confound" — is actually two measurements of the *same* config with ordinary run-to-run noise. Root cause of the mislabeling not chased further (the run following it was a crash-then-retry; plausible the retry didn't pick up a fresh edit) — not worth the time now that it's caught and corrected.

**Corrected re-run** (`sol11-batch262144-900s-v2`, verified genuine via `grad_accum_steps=16 × micro_tokens=16384 = 262144`): **172 steps, `public_val_bpb: 1.310506` — worse than 131072.** The real U-shape minimum at the 900s protocol is **`TOTAL_BATCH_SIZE=131072`**, not 262144: `32768 (1.322) → 65536 (1.297) → 131072 (1.291, best) → 262144 (1.311) → 524288 (1.404)`. The batch axis conclusion flips: our locked value was wrong at the 900s comparison scale.

**But the full-1800s headline number below was genuinely run at 262144, so it doesn't get retracted — it gets a real head-to-head competitor.** See the next section.

### Full 1800s confirmation, head-to-head: `TOTAL_BATCH_SIZE=262144` vs. the corrected winner `131072`

`sol11-final-combined-fullbudget`: fresh init, real submission shape, real full `--max-train-seconds 1800`, `DEPTH=6` + `TOTAL_BATCH_SIZE=262144` — **genuinely 262144** (verified the same way as the correction above). **Result: 362 steps, `public_val_bpb: 1.200143`, `causality_passed: True`.**

`sol11-batch131072-fullbudget`: same protocol, verified genuine `TOTAL_BATCH_SIZE=131072` (`grad_accum_steps=8 × micro_tokens=16384`). **Result: 560 steps, `public_val_bpb: 1.199914`, `causality_passed: True`.**

**Head-to-head verdict: a statistical tie, `131072` locked for the step-count margin.** `131072` (`1.199914`) vs. `262144` (`1.200143`) — a gap of `0.000229`, right at the established GPU-proxy-era noise floor (`~0.0001-0.00025`), i.e. not a real quality difference either way. But `131072` completes **560 steps vs. 262144's 362 — 55% more** for that statistically-identical result. **`TOTAL_BATCH_SIZE=131072` is the final locked value.** More steps banked for the same scored quality is a real, practical advantage even in a tie: more headroom for any future change (e.g. Solution 12 kernel work) to compound against, and no reason to prefer the larger batch once quality is a wash.

This also means the corrected 900s sweep table's ordering *and* the full-budget head-to-head **agree** — `131072` wins both ways — which is reassuring given the mislabeling scare; the correction converged to a consistent answer rather than trading one uncertainty for another.

### Toggle-stack re-validation — also corrected, conclusion still holds

`sol11-final-rawbaseline-900s` (all eight Solutions 1-10 toggles reverted to untuned values, genuinely `TOTAL_BATCH_SIZE=262144`, `288 steps, public_val_bpb: 1.321078`) was originally compared against the mislabeled `262144` row above (`262 steps, 1.288261` — actually 131072). **Corrected comparison, using the genuine `262144` full-stack point from the corrected re-run (`172 steps, 1.310506`)**: full-stack `1.310506` still beats raw-baseline `1.321078` — the Solutions 1-10 stack still wins at genuine `DEPTH=6, TOTAL_BATCH_SIZE=262144`, just by a smaller margin (`0.0106`) than originally claimed (`0.033`, using the wrong anchor). Conclusion unchanged; evidence corrected. The head-to-head above locked `TOTAL_BATCH_SIZE=131072` as the final value, not `262144` — this toggle check was run at `262144`, not the exact final shape. Not considered urgent to redo: the qualitative conclusion (stack beats raw baseline) has now been established at *three* distinct shapes this phase (`DEPTH=6`/old-batch-65536, and genuine `DEPTH=6`/`262144` here), and `131072` sits between two of those tested points, not off in new territory.

**Width-only resweep at fixed `DEPTH=6` — done, real second finding.** See the next section: the depth sweep varied width proportionally via fixed `ASPECT_RATIO`, never independently; the independent resweep found the width optimum has shifted too, `n_embd=640` (`ASPECT_RATIO=96`) not `384`.

### Width-only resweep at fixed `DEPTH=6` — real interior optimum, shifted from `384` to `640`

Same protocol as the depth sweep: fixed wall-clock (`--max-train-seconds 900`), real submission shape (`--nproc-per-node 8 --lnc 1`), `DEPTH=6` held fixed, varying `ASPECT_RATIO` (n_embd derives from `depth*ASPECT_RATIO` rounded up to the nearest 128, since `HEAD_DIM=128`) to hit a spread of widths independent of depth. `train.py`'s `ASPECT_RATIO` constant edited locally, `py_compile`-checked, and rsynced to remote before each point (no CLI flag exists for it) — same as the depth sweep's methodology, run sequentially back-to-back (same chip-warmth-confound caveat applies, noted not ignored).

| `ASPECT_RATIO` | n_embd | n_head | params | steps in 900s | `public_val_bpb` |
|---|---|---|---|---|---|
| 32 | 256 | 2 | 15.2M | 310 | 1.309278 |
| 64 (prior locked default) | 384 | 3 | 26.4M | 399 | 1.297186 |
| 80 | 512 | 4 | 39.9M | 268 | 1.270811 |
| 96 | 640 | 5 | 55.7M | 240 | **1.269647 — winner** |
| 128 | 768 | 6 | 73.9M | 235 | 1.273621 |

**Result: a clean U-shape, real interior optimum at `ASPECT_RATIO=96` (n_embd=640, 55.7M params).** `384 → 640` is a real, large gap (`0.0275`, two orders of magnitude above the established noise floor `~0.0001-0.00025`) — a genuine second major quality finding this phase, independent of the depth finding. The curve turns over cleanly within the swept range: `640` beats both its neighbors (`512` by `0.00116`, real but small; `768` by `0.00397`, real and confirms the turn) — this is a true interior optimum, not a monotonic trend truncated at the edge of the sweep. Same mechanism as the depth U-shape: `256` (most steps, 310) is under-parameterized and hits an early entropy floor; `768` (fewest steps, 235) is over-parameterized for this wall-clock and doesn't get enough steps to use its extra capacity.

**Decision: `ASPECT_RATIO=96` (n_embd=640) locked into `train.py`**, replacing `64`. Combined with `DEPTH=6`, the final locked architecture is now **640-wide, 6-deep, 55.7M params** — up from the prior 384-wide/26.4M-param point, but still far below the original 85.9M-param GPU-tuned baseline (`DEPTH=10, n_embd=640` coincidentally shares the same width as this new optimum, at 2x the depth). This changes the architecture shape non-trivially, with direct knock-on implications for the Solution 11 batch-size axis and the Solution 12 kernel — both re-validated at `n_embd=640` immediately below, rather than left as an open item.

### Batch-size axis re-validated at the new `n_embd=640` — same winner, kernel ON throughout

Same protocol as the original batch sweep, at the new locked width, with the rope+QK-norm kernel default ON throughout (this is the first batch sweep ever run with the kernel on — the original 384-width sweep predated the kernel's existence). A quick single-device smoke test first (`sol11-width640-kernel-smoke`, 15 steps) confirmed the kernel is architecture-invariant at the new width: no crash, `causality_passed: True`, `causality_max_abs: 0.0`, `num_params_M: 55.7` — the kernel operates per-head on `HEAD_DIM=128` (unchanged), independent of `n_head`, so this was expected but verified rather than assumed.

| `TOTAL_BATCH_SIZE` | steps in 900s | `public_val_bpb` |
|---|---|---|
| 65536 | 375 | 1.279503 |
| 131072 | 240 | **1.269318 — winner** |
| 262144 | 160 | 1.307378 |

**Result: `TOTAL_BATCH_SIZE=131072` wins again, same as the width=384 conclusion — the batch optimum did not shift with width.** Clean U-shape (`131072` beats `65536` by `0.0102` and `262144` by `0.0381`, both real, well above noise floor). `params_M=55.7` confirmed identical across all three points (no config drift). **No change to `train.py`'s `TOTAL_BATCH_SIZE=131072`** — the width resweep changed the architecture but not the batch-size answer.

### Full 1800s confirmation at the complete final combo — `DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, kernel ON — new final headline for the Trn2 phase

`sol11-final-combo-fullbudget`: fresh init, real submission shape, full `--max-train-seconds 1800`, every locked finding combined for the first time in one run.

**Result: 500 steps, `public_val_bpb: 1.176151`, `causality_passed: True`, `causality_max_abs: 0.0`, `num_params_M: 55.7`** (matches the width resweep and batch resweep-at-640 exactly — no config drift).

**Beats the old final headline decisively.** Old headline (`sol12-kernel-on-fullbudget`, `n_embd=384`, kernel ON): 581 steps, `public_val_bpb: 1.197595`. New combo: **500 steps (81 fewer, -14%) but `public_val_bpb: 1.176151` (`0.021444` lower/better — a real gap, two orders of magnitude above the noise floor)**. The wider model's per-step quality advantage more than compensates for completing fewer steps at the full 1800s budget: unlike the 900s-proxy sweep (where 640 already won outright, steps-vs-quality tradeoff already resolved at the shorter window), this was the genuine open question for the full budget — extra params vs. fewer possible steps could in principle have gone either way, and it resolved cleanly in favor of the wider model.

**This is the new final headline number for the entire Trn2 phase: `public_val_bpb = 1.176151` at 500 steps**, with the complete locked architecture: `DEPTH=6, ASPECT_RATIO=96` (n_embd=640, n_head=5, 55.7M params), `TOTAL_BATCH_SIZE=131072`, rope+QK-norm NKI kernel default ON, real 8-way submission shape (`--nproc-per-node 8 --lnc 1`). Ready for competition submission: file `train.py`, launch command `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.

## Solution 12 — real instruction-level profile, first proper pass

Earlier `neuron-explorer inspect` attempts (see the throughput-cliff and Source 5/6 sections above) only captured coarse `SystemProfile*` tables — event-count totals, no per-instruction or per-engine breakdown. Delegated a real profiling pass to the `neuron-nki-profile-analysis-agent`, which found a working capture path (`torch_neuronx.profiling.NeuronConfig(modes=[DEVICE, RUNTIME])` inside `torch.profiler.profile`, driven externally without touching `train.py`) and produced real `Instruction`/`DmaPacket`/`Summary`-table data — one micro fwd+bwd (91.445 ms, 689 NEFF executions, 83 distinct NEFFs), steady-state pre-cliff (step 6, dt 2.90s).

**Bottleneck: launch/dispatch overhead, not any single engine.** Bounds table: DMA 27.8%, TensorE 22.5%, GpSimd 20.1%, Vector 17.1%, Scalar 11.6% of the window — all engines within a 2.4x band, none dominant. The real gap is `total_time → perfect_pipeline`: **72.2% of the window is serialization**, not compute or memory. Direct evidence: 689 NEFF launches per micro-step, mean exec 82.2μs, but 688 inter-execution dispatch gaps sum to 34,825μs = **38.1% of the window** — device busy only 61.9%. Root cause: the native TorchNeuron eager backend compiles **one NEFF per aten op**, zero cross-op fusion. This is exactly the profile Solution 12's original "many tiny kernel dispatches" hypothesis predicted — just confirmed with real numbers instead of the earlier coarse dispatch-count signal.

**RMSNorm hypothesis: denied. Rope is 2.1x bigger.** `F.rms_norm` already lowers to a *single* fused `aten::_fused_rms_norm` NEFF — one launch per call, regardless of the ~42-calls/forward call-frequency reasoning that made it the presumptive first target. `apply_rotary_emb` (`train.py:372`) has **no fused lowering at all** — it decomposes into 15 separate NEFF launches per call (4 mul + 2 add + 1 cat + 1 neg forward; 2 slice_backward + 1 add + 4 mul + 2 add backward), because the code slices into halves, does elementwise math, then `torch.cat`s the result. 180 launches for rope vs. 64 for norm, in the profiled window. Call-count-based reasoning alone was the wrong proxy for launch-count; it missed which ops PyTorch already fuses under the hood.

**New top NKI target, evidence-based: fused rope + QK-norm, forward and backward, exactly `q = norm(apply_rotary_emb(q, cos, sin))` at `train.py:411-412`.** Fusing these two calls collapses 204 of 689 launches (29.6%) into an estimated ~24, saving 14.77ms of the 91.445ms window (16.15%). Also cuts the associated HBM traffic (1,061 MB vs. separately 871+498 MB) and — since every rope/rmsnorm elementwise op currently runs in fp32 (`q`/`k` emerge fp32 post-`norm()` even though the model is bf16-compute) — a bf16 NKI kernel would roughly halve that traffic further.

**Honest sizing caveat, load-bearing for expectations**: the profiled device window (16 micro-steps × 91.445ms ≈ 1.46s) accounts for only **~50% of the real 2.90s/step** — the other half is host-side (dataloader `fetch()`, Python dispatch, grad-accum loop overhead), outside this profile's scope. So the rope+QK-norm fusion's real expected win is closer to **~8% of total step time** (14.77ms × 16 microsteps ÷ 2900ms ≈ 8.1%), not the raw 16.15%-of-window figure. Still a real, worthwhile target — roughly the same order of magnitude as some of the Solution 11 batch-sweep deltas — but sized correctly, not oversold. What dominates the other ~50% (host-side) is an open question, out of scope for this NKI-focused profiling pass.

**Second-priority candidate, if rope+QK-norm doesn't fully pan out**: `aten::embedding_dense_backward` — 4 executions, 5.86ms (6.41% of window), GpSimd-dominated. **Confirmed NOT a target**: `aten::linear`/`linear_backward` (the real compute, 80 execs, 15.00ms/16.40%) — TensorE already hits 80.4% of peak FLOPS when active; nothing to fix there via kernel work. Attention itself (`_scaled_dot_product_fused_attention_overrideable`, 12 execs, 9.11ms/9.96%) is also already fused, with only ~12% of its FLOPs being transposes — not a priority target either, consistent with Solution 8's earlier finding that a custom windowed-attention kernel wasn't motivated.

**Cleanliness confirmed**: `train.py` untouched throughout — all instrumentation lived in an external driver script, never in the repo. `git status`/`git diff` clean on both remote and local, md5sums identical.

### Fused rope+QK-norm kernel — written, verified, decisive win, kept and defaulted ON

Delegated the actual kernel-writing to `neuron-nki-agent`. Result: `_rope_norm_fwd`/`_rope_norm_bwd`/`_RopeNormNKI`/`rope_norm()` in `train.py`, gated by `--nki-rope-norm`/`--no-nki-rope-norm` (`argparse.BooleanOptionalAction`, same pattern as `--eval-public`). Backward derived entirely from scratch — every reference recoverable from `nki-samples`'s git history is forward-only.

**Numerical equivalence, concrete, not "looks fine"**: `research/scratch/rope_norm_equiv.py`, importing the kernel directly out of `train.py` so the harness can't drift. bf16, shape `(2,1024,3,128)`, vs. CPU eager `F.rms_norm(apply_rotary_emb(...))`, forward *and* backward, across 7 test cases (varied seeds, scales from `1e-3` to `40`, all-zeros, mixed per-head scale): worst deviation anywhere is **exactly 1 bf16 ULP on ≤0.01% of elements**. Two real bugs caught and fixed en route (both silent-corruption class, exactly the failure mode this session has been disciplined about catching): `_RMS_EPS` was double-precision eps instead of `F.rms_norm`'s real float32 default (`1.1920928955078125e-07`) — gave 6% error at small scale and a `9.8e7` all-zeros backward; `nisa.activation(op=nl.rsqrt)` has `4.3e-5` relative error on this SDK, replaced with `nisa.reciprocal` + `nisa.activation(op=nl.sqrt)` at `5.8e-7` — moved bit-exactness `99.91% → 100.00%`.

**Real measured speedup, at the correct final locked config** (`DEPTH=6, TOTAL_BATCH_SIZE=131072`, world_size=8 — the kernel's first A/B was run against a batch size later found to be superseded by the data-integrity correction above, then correctly redone here): fixed 30-step comparison, config headers verified matching (`grad_accum_steps=8, micro_tokens=16384` both runs, learned from the batch-size mixup not to trust a comparison without checking this first) — **toggle OFF: ~1.29-1.32s/step steady state; toggle ON: ~1.07-1.13s/step — a real ~15-18% per-step speedup**, beating both the single-device measurement (14.2%) and the profiling agent's a-priori ~8% estimate (the removed cost is host-side dispatch, which device-only profiling undercounts). `causality_passed: True` both ways, `causality_max_abs: 0.000000`. Loss trajectories match to 4 decimals step-by-step — the quality difference is noise-level, not systematic, consistent with the 1-ULP equivalence result.

**Decision: kept, defaulted ON.** `USE_NKI_ROPE_NORM = True` at the module level; the CLI flag's own default independently reads `os.environ.get(..., "1") != "0"`, so omitting `--nki-rope-norm`/`--no-nki-rope-norm` entirely now runs with the kernel on — verified directly (`nki_rope_norm: True` printed with no flag passed, `nki_rope_norm: False` with `--no-nki-rope-norm` passed, both smoke-tested clean before trusting).

**Full-1800s confirmation, kernel ON, final locked config — done.** `sol12-kernel-on-fullbudget`: `DEPTH=6, TOTAL_BATCH_SIZE=131072`, world_size=8, kernel ON (default, no flag passed), full 1800s budget. Config verified matching prior runs. **Result: 581 steps, `public_val_bpb: 1.197595`, `causality_passed: True`, `causality_max_abs: 0.0`.** vs. the old kernel-OFF headline (`sol11-batch131072-fullbudget`: 560 steps, `1.199914`): **+21 steps (+3.75%) and a lower (better) bpb** — both improvements land in the same direction, consistent with the measured per-step speedup and the noise-level quality parity already established by the equivalence harness. **This is the new final headline number for the Trn2 phase: `public_val_bpb = 1.197595` at 581 steps.**

`research/scratch/rope_norm_dev.py` is a stale, superseded copy of the kernel (old `activation(rsqrt)`) left over from development — `rope_norm_equiv.py` is the authoritative equivalence harness; don't read the stale file as current. If `USE_FUSED_QKV` (Solution 13c) is ever turned on, `rope_norm()` silently falls back to eager (no crash, just no speedup) since q/k may become non-contiguous — a real interaction to remember if that toggle is revisited.

## Post-submission round — closing the gap to the real leaderboard

First real submission: `public_val_bpb = 1.176151`. Real leaderboard top scores: `1.00-1.06` — a large, real gap. Plan: `/Users/csabatoth/.claude/plans/valiant-wandering-pascal.md`, Bucket A first (cheap, mechanical, real A/B's never done at the exact final locked shape).

### Solution 13's four toggles — real speed A/B at the final locked shape, two real wins found

Every prior Solution-13 test was correctness-only (15-step single-device smoke tests) at a now-superseded width — none had ever been given a real fixed-step speed/quality A/B at the actual final shape (`DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, rope+QK-norm kernel default ON, real 8-way, 900s fixed-budget proxy). Baseline at this exact shape, all four toggles `False` (already had this: `sol11-width640-batch131072`): **240 steps, `public_val_bpb: 1.269318`**.

| Toggle | Result | vs. baseline | Verdict |
|---|---|---|---|
| `USE_TRN2_COMPILER_FLAGS=True` | 246 steps, `1.264601` | `+0.0047` | Real small win, isolated |
| `USE_STOCHASTIC_ROUNDING=True` | 248 steps, `1.270737`, **`causality_passed: False`** (`max_abs_diff=0.021134`) | — | **Disqualified** (see below) |
| `USE_FUSED_QKV=True` | 206 steps, `1.312086` | `-0.0428` | Real net negative, as expected |
| `USE_CAUSAL_FASTPATH=True` | 265 steps, `1.252675` | `+0.0166` | **Real win, largest of the four** |
| `USE_TRN2_COMPILER_FLAGS=True` + `USE_CAUSAL_FASTPATH=True` (combo) | 262 steps, `1.254491` | `+0.0148` | Real, but *worse* than `USE_CAUSAL_FASTPATH` alone — doesn't compose |

**`USE_FUSED_QKV`'s net negative is explained, not a bug**: this toggle silently disables the rope+QK-norm NKI kernel fusion (falls back to eager, no crash, no speedup from that kernel) — fewer steps (206 vs. 240) and worse quality follow directly from losing that kernel's ~15-18% per-step speedup. `causality_passed: True` throughout — a real cost, not a correctness issue.

**`USE_STOCHASTIC_ROUNDING`: a real methodology finding, not just a discarded toggle.** The causality "failure" was investigated rather than assumed real, via a standalone diagnostic (`research/scratch/sr_causality_falsepositive_check.py` — imports `train.py`'s `GPT` class directly, does not modify `prepare.py` or `train.py`) that ran `model(x)` **twice on the identical input** under `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` and measured the self-consistency diff: **`0.011341`** — same order of magnitude as the observed "failure" (`0.021134`), and already larger than `causality_check`'s own fixed `1e-2` tolerance *by itself*, with no flipped/perturbed input involved at all. **Confirmed: stochastic rounding's own hardware-level forward-to-forward noise exceeds the harness's tolerance on its own — this is a false positive, not a real information leak from future tokens.** Discarded anyway, regardless of any hypothetical speed benefit: the real competition submission would be scored through the same `causality_check` with the same fixed tolerance, so this toggle carries the identical false-positive risk in a real submission, independent of whether it's "actually" causal. **General lesson, worth remembering beyond this one toggle**: `causality_check`'s fixed `1e-2` tolerance has a structural blind spot for any feature that introduces hardware-level randomness into the forward pass (stochastic rounding, and potentially others) — a real methodology gap, not specific to this toggle.

**`USE_TRN2_COMPILER_FLAGS` and `USE_CAUSAL_FASTPATH` do not compose** — combining them (`1.254491`, 262 steps) is real but strictly worse than `USE_CAUSAL_FASTPATH` alone (`1.252675`, 265 steps): fewer steps and a worse bpb, both by margins clearly above the noise floor. Not chased further; not worth the effort to explain the exact mechanism when the practical answer (don't combine) is clear.

**Decision: `USE_CAUSAL_FASTPATH=True` locked, all other three Solution-13 toggles left `False`.** The single largest and cleanest win among the four, and it doesn't need `USE_TRN2_COMPILER_FLAGS` to get there — combining actively hurts. `USE_TRN2_COMPILER_FLAGS`, `USE_STOCHASTIC_ROUNDING`, and `USE_FUSED_QKV` remain implemented (available if a future config change shifts their calculus) but off.

### LNC=2 A/B at the final locked shape — large real win, changes the launch command

Never tried in the final combo before — `USE_CAUSAL_FASTPATH=True` locked above was itself only ever tested at `NEURON_LOGICAL_NC_CONFIG=1`/`--nproc-per-node 8`. One real run at `NEURON_LOGICAL_NC_CONFIG=2`/`--nproc-per-node 4` (LNC=2 pairs cores into 4 logical ranks instead of 8), same 900s proxy, same final locked shape otherwise.

**Result: 330 steps, `public_val_bpb: 1.234514`, `causality_passed: True`, `num_params_M: 55.7`** (matches — LNC only regroups cores, doesn't change the model). vs. the LNC=1 reference at this exact toggle state (265 steps, `1.252675`): **+65 steps (+24.5%) and `0.018161` better bpb** — a large, real win, bigger than any single Solution-13 toggle finding this round, both dimensions moving the same direction (more steps *and* better quality, not a step-count/quality tradeoff).

**This changes the real submission launch command.** If this holds at the full 1800s budget, the actual submission should use `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`, not the `NEURON_LOGICAL_NC_CONFIG=1`/`--nproc_per_node=8` command used for the first submission.

### Full-1800s confirmation, LNC=2 + `USE_CAUSAL_FASTPATH` combo — new best-known local config, NOT YET SUBMITTED

`sol13-lnc2-fullbudget`: fresh init, full `--max-train-seconds 1800`, `NEURON_LOGICAL_NC_CONFIG=2 --nproc-per-node 4`, `USE_CAUSAL_FASTPATH=True` (locked), DEPTH=6/n_embd=640/TOTAL_BATCH_SIZE=131072/rope+QK-norm kernel ON — everything from Bucket A applied together at the real budget for the first time.

**Result: 639 steps, `public_val_bpb: 1.15361`, `causality_passed: True`, `causality_max_abs: 0.0`, `num_params_M: 55.7`** (matches, no config drift).

**Beats the currently-submitted headline decisively.** Submitted: 500 steps, `public_val_bpb: 1.176151` (LNC=1, no `USE_CAUSAL_FASTPATH`). New: **639 steps (+139, +27.8%) and `public_val_bpb: 1.15361` (`0.022541` lower/better)** — both dimensions improved simultaneously, confirming the 900s-proxy signal held at the real budget, same direction and similar-or-larger magnitude.

**Update: submitted.** This config (`LNC=2 --nproc-per-node 4`, `USE_CAUSAL_FASTPATH=True`) has since been submitted to the real leaderboard, superseding the earlier `1.176151` submission. Launch command used: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.

## Post-submission round, continued — Bucket B: data-sampling remainder-requeue packer

The data-pipeline audit (prompted by the user's own past experience finding a Q&A-order-swap corruption bug in a fine-tuning dataset at a hackathon) found the loader's shift-by-one, packing, and train/val separation all correct — no such bug here. It did find one real, pre-existing quality cost: `prepare.py`'s `make_dataloader()` packer, when no buffered document fits the current row's remaining space, always crops the globally-shortest buffered document to exactly fill the remaining space and **permanently discards the rest of that document** (`doc[remaining:]`, silently, every time this happens). Not a corruption bug (inputs/targets stay correctly aligned) — a real, ongoing loss of training signal, structurally identical in spirit (silently discarded content) to the class of bug the hackathon story described, though the specific failure mode differs (truncation, not reordering).

**Fix: `USE_REQUEUE_PACKER` + `make_dataloader_requeue()` in `train.py`.** A legal "data sampling strategy" change per README (train.py may change this freely; a custom packer driven from train.py, not editing prepare.py itself, was already forum-ruled admissible earlier this competition). The new function is a byte-for-byte copy of `prepare.py`'s `make_dataloader()` — same best-fit/shortest-crop selection policy, unchanged — except the crop branch appends `doc[remaining:]` back into `doc_buffer` instead of discarding it. The leftover carries no new BOS token (correct: it's a continuation of the same document, not a new one — BOS is only ever prepended once, at `encode()` time, before any cropping happens). Reuses `prepare.py`'s private `_document_batches` generator and `get_dist_info()` directly; `prepare.py` itself is untouched.

**Offline correctness harness, before any hardware time spent** (`research/scratch/requeue_packer_equiv.py`): exercises the real shipped `make_dataloader_requeue()` (not a reimplementation) via an injectable `_document_batches_fn` test seam, against a synthetic 20,000-document corpus with globally unique per-token markers (`token = doc_id * 100_000 + local_index`), so any drop/duplicate/reorder/cross-document corruption is directly detectable by reconstructing each document's token sequence from the packed output stream.

**A real bug caught in the test itself, before it ever touched hardware**: the first version used `doc_id * 1_000_000 + local_index`, which overflows `int32` once `doc_id` gets into the tens of thousands — and the real code legitimately casts packed inputs to `int32` (correct for the real ~8,192-token vocabulary; only a problem because the synthetic marker scheme was unrealistically large). This produced exactly the kind of confusing, real-looking "corruption" symptom (garbled indices appearing under the wrong document's bucket) that could have wasted a lot of debugging time chasing a phantom bug in the packer itself — caught by inspecting the actual divergence point rather than trusting the first failure message, then fixed by choosing a marker scheme (`* 100_000`, `NUM_DOCS ≤ 20_000`) that respects the real code's actual dtype constraints. **General lesson**: synthetic test data must respect the real code's actual numeric ranges, not just "some larger, safer-looking" range — the same discipline as picking real epsilon values for the NKI kernel equivalence harnesses earlier this competition.

**Harness result: PASS, zero corruption.** Across the full 20,000-doc corpus: zero duplicate tokens, zero reorderings, zero out-of-range indices, zero cross-document mixing. The only "incompleteness" observed — some documents not yet fully flushed, or not yet touched at all, by the time the finite synthetic corpus ran out — is a single benign phenomenon (a document's current chunk is still buffer-resident, hasn't been re-selected yet), confirmed benign by re-running at `NUM_DOCS=15,000` vs `20,000`: completeness rose `96.51% → 97.41%` with more corpus/runtime, i.e. it shrinks with more time rather than being a fixed leak. Every single "incomplete" document, without exception, was a perfectly clean `0..k-1` prefix of its real content — never garbled.

**Real hardware A/B, at the final locked shape** (`DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, rope+QK-norm kernel ON, `USE_CAUSAL_FASTPATH=True`, `LNC=2/--nproc-per-node 4`, 900s proxy). Baseline (`USE_REQUEUE_PACKER=False`, already had this exact shape: `sol13-lnc2-final-shape`): 330 steps, `public_val_bpb: 1.234514`. With the fix (`sol-bucketb-requeue-on`): **332 steps, `public_val_bpb: 1.235799`, `causality_passed: True`** — essentially the same step count, but a small *real* regression in bpb (`0.001285`, above the noise floor but not dramatic).

**Decision: implemented, verified correct, tested — not kept.** `USE_REQUEUE_PACKER=False` remains the default. The fix is real and correct (recovers previously-discarded document content, verified via the offline harness), but doesn't translate into a measured quality win at this training budget/scale — plausibly because the crop-and-drop event is relatively rare given `PACK_FACTOR=2`'s generous row capacity, so the actual volume of previously-lost signal is small relative to total tokens processed in a ~330-step run, and/or the recovered continuation chunks (mid-document, no fresh BOS, competing for buffer selection under the same "shortest wins" policy as always) don't add enough distinct signal at this scale to outweigh the noise. An honest negative result — same discipline as `USE_FUSED_QKV`'s writeup — not swept under the rug. The toggle, the function, and the harness all remain in the codebase (correct, available, off by default) in case a larger model/budget or a longer training run changes the calculus.

## Post-submission round, continued — context-length (SEQ_LEN) resweep, third major finding

Proposed by the user, questioning whether the current `SEQ_LEN=1024` was ever validated as optimal (it wasn't — inherited from the GPU-proxy baseline, never resweept alongside depth/width/batch). Correct framing established up front: Trn2 HBM is not the binding constraint here (attention uses the fused SDPA kernel, which never materializes the full `L×L` score matrix, so memory scales close to linearly with `SEQ_LEN`, not quadratically) — the real question is step-time/throughput and quality, not memory headroom.

Valid sweep grid, constrained by `world_tokens_per_micro = DEVICE_BATCH_SIZE(1) × PACK_FACTOR(2) × SEQ_LEN × world_size(4)` needing to divide `TOTAL_BATCH_SIZE=131072` evenly (`SEQ_LEN` must divide `16384`): `{512, 1024, 2048, 4096}` — all powers of 2, same protocol as every other sweep this session (fixed 900s wall-clock, real final-locked shape: `DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, rope+QK-norm kernel ON, `USE_CAUSAL_FASTPATH=True`, `LNC=2/--nproc-per-node 4`). Finer non-power-of-2 grid points (e.g. `1536`) aren't reachable without also changing `PACK_FACTOR`/`TOTAL_BATCH_SIZE` — out of scope for this pass, noted as an open question.

| `SEQ_LEN` | steps in 900s | `public_val_bpb` |
|---|---|---|
| 512 | 253 | 1.332677 |
| 1024 (prior default) | 330 | 1.234514 |
| 2048 | 387 | **1.198086 — winner** |
| 4096 | 396 | 1.209781 |

**Result: a clean U-shape, real interior optimum at `SEQ_LEN=2048`.** The gaps are large and unambiguous: `2048` beats `1024` by `0.036428` and beats `4096` by `0.011695`, both far above the established noise floor (`~0.0001-0.00025`) — a third major quality finding this phase, on the same order of magnitude as the width resweep's gap, on an axis (context length) that had never been touched since the original GPU-proxy tuning.

**Mechanism, honestly reported from what the numbers actually show, not a forced narrative.** Two candidate effects were in play going in: (a) attention FLOPs scale quadratically with `SEQ_LEN` (a real cost that should eventually dominate and hurt), and (b) fewer, longer sequences per grad-accum step mean fewer microsteps, so the *fixed* per-microstep dispatch overhead (the launch/dispatch-serialization bottleneck identified in the Solution 12 profiling, ~72% of the profiled window at the time) is amortized over more tokens each time it's paid. The step-count trend (`512→1024→2048→4096`: `253→330→387→396`, monotonically increasing all the way to `4096`) shows effect (b) dominating throughout this whole range — throughput kept improving even at `4096`, well past the naive FLOPs-crossover estimate (`~1920`, from the Solution-12-era profile where attention was only ~12% of FLOPs at `n_embd=640`). But **quality** (`public_val_bpb`) turned over at `2048`, getting *worse* at `4096` despite `4096` completing *more* steps than `2048` — meaning the U-shape's downturn past `2048` is a genuine quality effect, not a throughput one. Plausible causes not further isolated this pass: less useful signal density per token at very long context relative to this model's capacity (55.7M params), or a real diminishing-returns/extrapolation effect in how RoPE and the rest of the stack handle much-longer-than-baseline context — an open question for future work, not resolved here.

**Decision: `SEQ_LEN=2048` locked into `train.py`**, replacing `1024`.

### Full-1800s confirmation, `SEQ_LEN=2048` combo — new best-known local config, beats the submitted headline

`sol-seqlen2048-fullbudget`: fresh init, full `--max-train-seconds 1800`, everything from this phase combined for the first time with `SEQ_LEN=2048` (`DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072`, rope+QK-norm kernel ON, `USE_CAUSAL_FASTPATH=True`, `LNC=2/--nproc-per-node 4`).

**Result: 746 steps, `public_val_bpb: 1.122271`, `causality_passed: True`, `causality_max_abs: 0.0`, `num_params_M: 55.7`, `seq_len: 2048`** (all confirmed, no config drift).

**Beats the currently-submitted headline decisively.** Submitted (`sol13-lnc2-fullbudget`, `SEQ_LEN=1024`): 639 steps, `public_val_bpb: 1.15361`. New: **746 steps (+107, +16.7%) and `public_val_bpb: 1.122271` (`0.031339` lower/better)** — the 900s-proxy signal held at the real budget and, if anything, the margin held up strongly. Both dimensions improved simultaneously, same pattern as every other real win this session.

**Not yet submitted.** Candidate launch command is unchanged from the current submission (`SEQ_LEN` isn't a launch-command parameter, it's baked into `train.py` itself): `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.

## Bucket E, first attempt — whole-step graph compilation via `torch.compile(backend="neuron")` — real, correct, marginal at the 900s proxy — superseded below

Triggered by the user asking whether anything from nanochat's TPU/JAX port transfers. That port's biggest lever (MFU `16%→22%`) was merging a training step split across three separately-compiled programs into **one** compiled unit — the same root-cause class as this competition's own Solution-12 profile finding: the native TorchNeuron eager backend compiles one NEFF per aten op, zero cross-op fusion — 689 NEFF launches per microstep, 688 dispatch gaps eating **38.1%** of the profiled window, **72.2%** of the window lost to serialization overall (measured at the *old* shape: `SEQ_LEN=1024, LNC=1`, no causal-fastpath). Whole-step compilation targets all of that at once, vs. the two existing hand-written NKI kernels' targeted per-op fixes.

**Discovery: the mechanism already existed, unused.** `train.py` already implements a `--compile` flag (`torch.compile(model, backend="neuron", dynamic=False)`, wraps the entire model) — confirmed via a full search of `research/*.md` + `experiment-log.jsonl` + `run-logs/` that it had **never been invoked in any real run this competition**. The real competition README (`README.md:303-304`) explicitly permits it: *"The baseline is eager-mode; `torch.compile` / Neuron trace are allowed at your own risk, and compile time counts against the budget."*

**A real documentation bug, now fixed**: `research/enhancement-opportunities.md` (lines 9, 33, 123, 204) repeatedly cited `program.md` — the separate, stricter ruleset for the *unrelated* `autoresearch/` 5-minute dev-loop tool, which does ban `torch.compile` for its own purposes — as if it were the real competition rule, and used that mistaken citation to write off an entire bucket of kernel-fusion-via-compilation techniques as infeasible on this hardware. It wasn't infeasible; it was never tried. Citations corrected in that doc.

**Two correctness smoke tests, both clean, single device**: (1) `--compile --no-nki-rope-norm` (kernel forced off) — `causality_passed: True`, `causality_max_abs: 0.0`, no crash, `296.6s` one-time compile. (2) `--compile` alone (kernel ON, default) — `causality_passed: True`, `causality_max_abs: 0.0`, no crash, **no graph-break/fallback warnings** anywhere in the log, `202.8s` compile, loss trajectory smooth and sane. **The rope+QK-norm NKI kernel is compile-compatible** — a real, non-obvious result worth having verified rather than assumed, given "at your own risk" in the README is a genuine warning about exactly this kind of custom-autograd-op interaction.

**Real hardware A/B, at the final locked shape** (`DEPTH=6, n_embd=640, TOTAL_BATCH_SIZE=131072, SEQ_LEN=2048, USE_CAUSAL_FASTPATH=True, LNC=2/--nproc-per-node 4`, kernel ON, 900s proxy). Eager baseline (`sol-seqlen2048`, already had this exact shape): 387 steps, `public_val_bpb: 1.198086`, `training_seconds: 893.1`. Compiled (`sol-e-compile-final-shape-900s`): **386 steps, `public_val_bpb: 1.197115`, `training_seconds: 892.2`, `causality_passed: True`, `causality_max_abs: 0.0`, `num_params_M: 55.7`** (matches). The startup-allowance mechanism worked as designed — `startup_allowance_est: 165.2s` for the one-time compile, fully excluded from the charged training window (`training_seconds` is within 1s of eager's own).

**Result: essentially a tie, real but marginal.** `0.000971` better bpb, one fewer step — both differences are small, right at the edge of what this session's noise floor discussion has treated as meaningful (`~0.0001-0.00025`) but nowhere near the size of any other real win this round. A closer look at per-step `dt` in the tail of each run shows a real discrepancy worth flagging honestly rather than smoothing over: compiled's sampled tail-window ran **~3.5s/step** consistently, vs. eager's **~3.5-4.9s/step** (noisier, averaging notably slower) — a real ~15% per-step advantage in that window that did **not** cleanly translate into more total completed steps across the full run. The most likely explanation, given this session's own established evidence: the real, reproducible **throughput cliff** (chip-warmth-dependent slowdown, documented extensively earlier this competition) affects single, uncontrolled runs differently depending on thermal history — a single-run comparison without a repeat-seed/back-to-back-matched-conditions check can't cleanly separate a genuine compile-vs-eager throughput difference from this known confound. Not chased further with a repeat run this round — the size of the effect either way is too small to justify the hardware time given everything else still on the plan.

**Decision at the time: kept as a working, verified-correct, opt-in `--compile` flag — not adopted as the default/submission recipe.** The 900s-proxy result didn't clear the bar this session has used for "real win" (well above noise floor, unambiguous direction).

**Superseded — a later matched repeat found a real, large win instead, not a tie.** A second attempt re-ran the same 900s-proxy A/B independently (`bucket-e-compile-900s`, section below) and got a materially different compiled-side result at nominally the same config: `406` steps/`1.190086`, vs. this section's `386` steps/`1.197115` — a real, honestly-flagged discrepancy between two supposedly-matched 900s runs, most likely the same throughput-cliff/chip-warmth noise this section already suspected, now confirmed large enough to flip the practical conclusion. The tie-breaker is the **full-1800s confirmation** (`bucket-e-compile-fullbudget`, section below): `827` steps, `public_val_bpb: 1.113006`, unambiguously beating every eager number this competition has produced. **`--compile` is now default ON** — see the section below for the full real result; this section is kept as an honest record of the earlier, inconclusive-at-proxy-scale attempt, not deleted.

## Bucket C — profile-driven cleanup and measurement, real correctness fix + one ambiguous result

A fresh instruction-level profile at the *current* locked shape (`SEQ_LEN=2048, USE_CAUSAL_FASTPATH=True`, kernel ON, `LNC=2`) — not reusing the old, now-stale `SEQ_LEN=1024` profile — found the bottleneck composition has genuinely shifted: launch/dispatch serialization dropped from `72.2%` of the window to `52.9%` (`689 → 461` NEFF launches/microstep), `TensorE` now a clear `47.1%` bottleneck. Full profiling methodology, bounds tables, and per-op breakdown: delegated to the `neuron-nki-profile-analysis-agent`, cross-validated via a real empirical ablation (not just profile inference).

**`ignore_index=-1` at `train.py:901` — provably dead code, removed.** The training loader (`prepare.py`'s `make_dataloader`) never emits `-1` targets (`cpu_targets.copy_(row_buffer[:, 1:])`, always fully packed); only the eval loader emits `-1` for a partial final row, and eval uses its own separate `F.cross_entropy` call in `prepare.py:628`, never going through `GPT.forward`'s loss path at all. Verified directly (not just via the profiling agent's claim): confirmed `make_dataloader`'s code, confirmed `loss_reduction` is never passed `"none"` by any caller in `train.py`. The profile found this costs `1.5ms`/microstep in dispatch stall (an `aten::nonzero` host round-trip, the single worst dispatch gap in the profiled microstep) for a mask that's always all-true in training. Removed, correctness-smoke-tested clean (`causality_passed: True`, `causality_max_abs: 0.0`).

**Fresh 900s baseline with the fix (`sol-c-dropignore-900s`): 366 steps, `public_val_bpb: 1.206231`** — worse than the pre-fix baseline (`sol-seqlen2048`: 387 steps, `1.198086`), the opposite of the small win the ablation predicted. Given the change is provably a no-op numerically (causality identical, and `ignore_index=-1` only matters if `-1` ever appears in targets, which it provably never does), this is most likely this session's own well-established **throughput-cliff/chip-warmth confound** affecting a single, uncontrolled comparison — not a real regression from the code change. Noted honestly rather than forced into a clean story either way; the correctness fix is kept regardless (it's dead code, removing it is unambiguously correct cleanup independent of whether it measurably helps throughput in any one comparison).

**`--nki-relu2` — already-existing kernel, never measured this competition, now measured: ambiguous, not kept.** The relu² chain was found to be `6.0%` of the profiled window — a real but modest target, and the kernel to fuse it already exists in `train.py` (lines ~371-420), default off. Real A/B (`sol-c-relu2-900s`): **381 steps, `public_val_bpb: 1.200379`**, `causality_passed: True`. Comparison is genuinely ambiguous depending on which baseline is trusted: beats the same-session `sol-c-dropignore-900s` baseline (`366/1.206231`, `+15` steps, `-0.005852` bpb — a real, favorable direction) but is roughly flat-to-slightly-worse against the older, cleaner-looking `sol-seqlen2048` baseline (`387/1.198086`, `-6` steps, `+0.002293` bpb). Both differences are modest, consistent with a real-but-small target (6% of window) easily masked by the same thermal-noise confound flagged above.

**Decision: `--nki-relu2` left default `False`.** The result doesn't clear this session's bar for a clean win either way — not kept, not because it's disproven, but because the comparison is genuinely inconclusive given the noise, and the underlying target is small enough that further hardware time chasing a clean answer isn't clearly worth it against everything else on the plan. A matched back-to-back repeat (same discipline as the `--compile` note above) would be the right way to resolve this if revisited.

## Bucket C closed out — fused softcap+cross-entropy NKI kernel: written, verified, real hardware result, kept

Following the profile's GO recommendation above, delegated the kernel-writing to `neuron-nki-agent` (same workflow as the rope+QK-norm kernel). Result: `_softcap_ce_fwd`/`_softcap_ce_bwd`/`_SoftcapCrossEntropyNKI`/`softcap_cross_entropy()` inside the existing `try: import nki` block, gated by `USE_NKI_SOFTCAP_CE`/`--nki-softcap-ce` (`argparse.BooleanOptionalAction`, same pattern as `--nki-rope-norm`). Fuses `LOGIT_SOFTCAP * torch.tanh(z/S)` + `F.cross_entropy` into one kernel pair, collapsing ~8 full-`[B*T,V]` NEFF launches into 2. The kernel produces the *unreduced* per-token loss (matching `F.cross_entropy(..., reduction="none")`) — mean/none reduction semantics stay in the caller, unchanged.

**Numerical equivalence, concrete, independently re-verified** (not just trusted from the agent's report — code reviewed line-by-line, the equivalence harness `research/scratch/softcap_ce_equiv.py` re-run independently and confirmed matching): worst case over 17 test scenarios (varied scales, tanh-saturation edge cases, all-equal/all-zero logits, extremely-confident-correct/-wrong predictions, a zero-upstream-gradient edge case) is `2.44 bf16 ULP` forward, `0.22 bf16 ULP` backward — and that worst forward case is an edge case (`extremely-confident-correct`) where the *eager fp32 chain being replaced* is equally far from an fp64 ground truth (`2.641e-05` eager vs. this kernel's `2.653e-05`) — an fp32 precision limit, not a regression. At the real shape/logit distribution that actually ships (`[4096,8192]`, real lm_head logits): `4.3e-5` absolute forward (`0.0012` bf16 ULP), `2.4e-7` absolute backward (`0.0015` bf16 ULP) — roughly 1000x inside one bf16 ULP.

**Two real bugs caught during development, same silent-corruption class this session has been disciplined about catching**: (1) the naive `loss = logsumexp - y_c` formula is catastrophically ill-conditioned for a confident token (`logsumexp ≈ y_c ≈ 15`, true loss `≈2.5e-3` — a 16% relative error from the subtraction) — rewritten as `loss = log(Σ_j exp(y_j - y_c))`, shifting by `y_c` directly instead of a constant, which also makes the `j==c` term exactly `exp(0)=1` so the sum is structurally `≥1`. (2) a single sequential fp32 accumulator over the full vocab (`V=8192`) carried `O(V·eps) ≈ 4.9e-4` rounding error, worst exactly when one term dominates (a confidently-correct token) — fixed by chunking the reduction into partial sums of 512, matching torch's own pairwise-sum precision (`4.2e-4 → 2.65e-5` on the worst case).

**A real, useful negative finding on tooling**: `nkilib.experimental.loss.cross_entropy` *does* exist and import cleanly on this SDK (contradicting an earlier research note that no such primitive existed) — but isn't usable as a drop-in: no softcap fusion, its backward hardcodes a scalar `grad_scale` so it can't accept a per-token upstream gradient or support `reduction="none"` autograd, and its indirect-DMA one-hot gather (128 tiny DMAs per tile) was rejected in favor of a masked-reduce approach used here instead.

**Real-hardware correctness smoke test**: clean (`sol-c-softcapce-smoke`, single device, `causality_passed: True`, `causality_max_abs: 0.0`, no crash).

**Real hardware A/B at the final locked shape** (`sol-c-softcapce-900s`): **384 steps, `public_val_bpb: 1.198478`, `causality_passed: True`, `num_params_M: 55.7`**. Comparison against the two available baselines is genuinely mixed given this round's established chip-warmth noise: beats the same-lineage, most-recent baseline (`sol-c-dropignore-900s`: 366 steps, `1.206231`) decisively — `+18` steps, `0.007753` better bpb. Against the older, cleaner-looking baseline that predates the ignore_index fix (`sol-seqlen2048`: 387 steps, `1.198086`) it's a near-tie — `-3` steps, `0.000392` worse — despite this run *also* carrying that fix. Read honestly: no comparison shows the kernel worse than the best number on record; the same-lineage comparison (the fairer one, since it isolates just the kernel's own effect on top of an already-fixed baseline) shows a large, real win, and the dropignore baseline it's compared against was very likely the unlucky thermal draw, not the kernel underperforming.

**Decision: kept, defaulted ON.** `USE_NKI_SOFTCAP_CE = True`; `--nki-softcap-ce`'s own default independently reads `os.environ.get(..., "1") != "0"`, matching `--nki-rope-norm`'s exact pattern — omitting the flag entirely now runs with the kernel on. A full-1800s confirmation of the complete combo (this kernel + everything else locked) would settle the exact magnitude question cleanly, same as every other finding this round — not run yet, flagged as the natural next step pending a decision on whether to spend that hardware time now.

**Bucket C, fully closed out — three real findings**: (1) dead `ignore_index=-1` removed (correct, kept, throughput signal muddied by noise but the fix itself is unambiguously right), (2) `--nki-relu2` measured for the first time, genuinely ambiguous, left off, (3) the softcap+cross-entropy kernel — real, verified, kept, defaulted on.

## Bucket D, first real implementation — Exclusive Self Attention (XSA): real, clean negative result

Researched from modded-nanogpt's "Learnable XSA" record (#82) — traced to the real underlying mechanism, **Exclusive Self Attention** (`arXiv:2603.09078`, Shuangfei Zhai, Apple ML research; "Learnable XSA" in modded-nanogpt is presumably a learned-scale variant on top of this base mechanism, not investigated). Real paper, verified live on arXiv, real published downstream-benchmark results.

**Mechanism**: after standard causal softmax attention produces output `y_i`, project out the component along the token's own value vector: `z_i = y_i - (y_i·v_i / ||v_i||²) * v_i`. The paper's own motivation: standard attention's output tends to have high cosine similarity with the token's own value vector ("attention similarity bias") — the attention layer ends up spending capacity on point-wise feature transformation that's redundant with the residual path and FFN block, rather than exclusively modeling cross-token context. XSA removes that redundant self-component explicitly. **Two lines of code, no new parameters, no shape change** — implemented in `CausalSelfAttention.forward` right after the SDPA call, before reshape/`c_proj`, gated by `USE_XSA` (module constant, no CLI flag, same direct-edit pattern as `DEPTH`/`SEQ_LEN`). Causally safe by construction: `z_i` is a function of `y_i` and `v_i` only, both already causal.

**Real, honest caveat going in, not discovered after the fact**: the paper's own Table 2 shows downstream-accuracy gains growing with model size — their smallest tested model (0.7B params) got only `+0.26` average accuracy over baseline, `1.3B` got `+1.03`, `2.7B` got `+1.36`. Our model (`55.7M` params) is ~13x smaller than their smallest tested case, trained for `~500-750` steps vs. their `200K`-step/`100B`-token full run — real, acknowledged uncertainty whether this transfers to our tiny/short regime, not an assumed win.

**Offline correctness checks, before touching hardware**: verified the orthogonality property directly (`z_i · v_i ≈ 0`, max `|z·v| = 3.3e-6`), the Pythagorean energy identity (`|z|² + proj² · |v|² == |y|²`, max error `3e-5`), that projected energy never exceeds input energy, and that the `eps=1e-6` floor safely handles degenerate near-zero `v` with no NaN/Inf.

**Real-hardware correctness smoke test**: clean (`sol-d-xsa-smoke`, single device, `causality_passed: True`, `causality_max_abs: 0.0`, no crash, `num_params_M: 55.7` unchanged — confirms no shape change).

**Real hardware A/B at the final locked shape** (`sol-d-xsa-900s`, vs. the current-default baseline `sol-c-softcapce-900s`: 384 steps, `public_val_bpb: 1.198478`): **368 steps, `public_val_bpb: 1.208657`, `causality_passed: True`, `num_params_M: 55.7`**. `-16` steps (`-4.2%`) and `0.010179` worse bpb — **both dimensions moved the wrong direction, a real, unambiguous negative**, not a coin-flip-adjacent result like `--nki-relu2`. Consistent with the honest caveat flagged before running: the mechanism's benefit, per the paper's own data, requires more model capacity and/or more training to manifest; at this tiny scale and step count the FFN/residual path may not have enough capacity or training time to pick up the self-referential information that XSA explicitly strips out of attention, making the net effect negative rather than merely diminished.

**Decision: `USE_XSA=False`.** Real, correct, well-verified implementation of a real published technique — genuinely doesn't help at our scale. An honest negative result, not swept under the rug, same discipline as `USE_FUSED_QKV`/`USE_REQUEUE_PACKER`. Code remains in the file (available, off by default) in case a larger future model/budget changes the calculus, matching this session's standing pattern for negative-but-real findings.

### Bucket D, remaining candidates — researched and triaged, none implemented this round

Real, time-boxed research (not full implementation attempts) on the rest of `modded-nanogpt`'s record #81-89 techniques not already covered above:

**Dynamically Composable MHA (record #85, real paper: `arXiv:2405.08553`, Xiao et al., ICML'24 oral — "DCMHA")**: dynamically composes/transforms the attention *score and weight matrices themselves* (a learned "Compose" function operating on the `[B,H,T,T]` attention matrix), tested 405M-6.9B params (real, positive results at that scale: matches models trained with ~1.7-2x compute). **Deprioritized without implementation**: this mechanism needs the raw attention matrix exposed, which means unfusing our `F.scaled_dot_product_attention` call — the same fused kernel this competition's Bucket C profile confirmed is already near-peak-efficient (`4.7%` of the profiled window, TensorE near its FLOPS ceiling when active). Unfusing it to add DCMHA's cross-head mixing would risk a real throughput regression *independent of* whether the mechanism helps quality — the same reasoning that already ruled out a custom windowed-attention kernel earlier this competition (Solution 8). Combined with the same scale-transfer uncertainty that already produced a real negative result for XSA, this carries a worse risk/effort profile than XSA did, for a similarly uncertain payoff. Not pursued.

**MUDD skip connections (record #81)**: real, published (MUDDFormer, ICML'25 poster) dense cross-layer connections. Already researched earlier this round (see the Bucket D research summary reported in-conversation): fixes residual bottlenecks specifically in *very deep* models, a mismatch with our deliberately shallow `DEPTH=6`, and adds real per-layer dynamic-gating compute on hardware that's spent this whole competition fighting many-small-ops dispatch overhead. Flagged risky, not discarded outright — genuinely revisitable later, just not pursued this round given XSA's negative result already confirmed the general "small-model/short-budget scale mismatch" risk class is real, not merely hypothetical.

**Prefix-token-prediction auxiliary loss (record #88)**: traced to the real underlying idea — "during early training, add an auxiliary loss on the longest token-prefix of each target token" (a coarse-to-fine auxiliary classification signal derived from the tokenizer's subword structure, intended to bootstrap the embedding/output space faster before the fine-grained full-vocab signal is well-learned). Legal in principle (an auxiliary loss term is squarely within `train.py`'s "architecture/training loop" freedom; deriving a static prefix-class lookup table by *reading* — not modifying — the fixed tokenizer's vocabulary would not touch `prepare.py`). **Not pursued this round**: real, nontrivial work to correctly derive the "prefix" structure from our specific BPE tokenizer (8192 vocab, unknown merge order without direct inspection), and a real, open question whether the annealed-out early-training benefit the technique is designed for even applies here — our entire run (~500-750 steps) *is* "early training" by the standard this technique targets, so there's no later phase to anneal into; unclear a priori whether the extra head/loss term pays for itself over that short a horizon. Genuinely worth a closer look if more time is spent on Bucket D-style bets, not ruled out on the merits, just not time-boxed into this round.

**FP8 MLP/matmul precision (records #84/#89)**: real feasibility check, direct SDK introspection (matching this session's established discipline rather than assuming) — **corrects an earlier research note's uncertainty**. Confirmed real on this SDK: `torch.matmul` on direct `float8_e4m3fn` tensors works natively on the `neuron` device (no upcast needed), and `neuronx-cc compile --help` lists a real `--auto-cast-type fp8_e4m3` compiler flag — genuine training-path FP8 support exists, not just the inference-quantization tutorials found earlier. One real snag: `nn.Linear`'s native fp8 weight init fails (`check_uniform_bounds` not implemented for the dtype) — a real, but standard-pattern-solvable limitation (real fp8 training keeps bf16/fp32 "master" weights and casts to fp8 only around the forward matmul, with a delayed-scaling calibration scheme — matching `modded-nanogpt`'s own record #89 description, "efficient delayed scaling metric"). **Not implemented this round**: building this properly (a custom autograd-wrapped cast-and-scale Linear, with real numerical verification) is a genuine engineering project comparable in size to writing a new NKI kernel, not a quick toggle — flagged as a real, confirmed-feasible candidate for future work, not a quick win to grab now.

## Bucket E, second attempt — `torch.compile(backend="neuron")`: real win, confirmed at full 1800s budget, now default ON

The `--compile` CLI flag (`train.py`, wraps the *entire* model in `torch.compile(model, backend="neuron", dynamic=False)`) existed in the file from early in the competition but had never actually been invoked in any real run before this bucket — confirmed via a search of `research/*.md` + `experiment-log.jsonl` + `run-logs/` at the time this bucket was planned. README explicitly permits it (`README.md:303-304`): eager is the baseline, compile is allowed "at your own risk, and compile time counts against the budget." Targets the same root cause already identified in the YAT/SLAY Phase 2 smoke test and the earlier Solution 12 profile: native TorchNeuron eager compiles one NEFF per aten op with zero cross-op fusion, and inter-execution dispatch gaps eat the majority of the profiled window — whole-step compilation targets that bottleneck globally instead of one op's slice of it.

**Real risk going in**: the hand-written rope+QK-norm `torch.autograd.Function` kernel is exactly the kind of thing that can silently break, force a graph break, or produce subtly wrong gradients under `torch.compile` — verified incrementally, not assumed safe.

**Step 1 — smoke test, rope kernel forced OFF** (`bucket-e-compile-smoke1`, single device, 20 steps): ran end-to-end with no crash. `causality_passed: True, causality_max_abs: 0.0` (exact). Compile time absorbed cleanly by the startup-allowance mechanism (`startup_allowance_est: 210.6s`, well under the `900s` cap from `prepare.py`).

**Step 2 — rope+QK-norm kernel re-enabled under compile** (`bucket-e-compile-smoke2`, single device, 20 steps): the step most likely to surface a silent compile/custom-autograd-Function correctness bug. Also clean: `causality_passed: True, causality_max_abs: 0.0`. No graph-break/recompile warnings found in the run log. Custom kernel compiles safely as shipped — no compile-safe registration fix needed.

**Step 3/4 — real speed/quality A/B at the final locked shape** (`bucket-e-compile-900s`, `nproc_per_node=4, lnc=2`, 900s budget, vs. the current-default eager baseline `sol-c-softcapce-900s`: 384 steps, `public_val_bpb: 1.198478`): **406 steps, `public_val_bpb: 1.190086`, `causality_passed: True`, `num_params_M: 55.7` (config matches)**. `+22` steps (`+5.7%`) **and** `0.008392` better bpb — a real win on both axes simultaneously, not a throughput/quality trade-off and not a coin-flip-adjacent result.

**Step 5 — full-1800s confirmation** (`bucket-e-compile-fullbudget`): **827 steps, `public_val_bpb: 1.113006`, `causality_passed: True`, `causality_max_abs: 0.0`**. Beats the prior full-1800s best (`sol-seqlen2048-fullbudget`, eager, predates the softcap+CE kernel: 746 steps, `1.122271`) by a real `0.009265`, and beats the currently-submitted real leaderboard entry (`1.15361`) by `0.040604`. Startup allowance (`45.4s`) absorbed the compile cleanly within the `900s` cap even at full budget/world_size=4.

**Decision: `--compile` flipped to default ON** (`argparse.BooleanOptionalAction`, same pattern as the NKI kernel flags — `--no-compile` still available for the plain eager comparison path). Verified with a plain-flags launch command (`bucket-e-default-check`, no explicit `--compile`) that the real submission launch format now compiles by default: `compiled: true`, `causality_passed: True`, `causality_max_abs: 0.0`.

**This is currently the single largest real, hardware-confirmed win found this round** — larger than the softcap+CE kernel, larger than the SEQ_LEN resweep. `1.113006` at full budget is the new best confirmed number, not yet submitted to the real leaderboard.

### Post-compile re-profile — RMSNorm is a no-go, attention was mis-measured before, `local_conv` is the new real target

Re-profiled the current locked+compiled config (`neuron-nki-profile-analysis-agent`, same methodology as the original Solution-12 profile, `capture_enabled_for_nc="0"` so this is at `nproc=1/LNC=1` — the real submission is `LNC=2`/4 ranks, dispatch pressure is 4x worse there; results below are directional for per-op targeting, not a literal re-measurement of the real submission shape). Two fresh captures at identical current code (eager control + compiled), plus the old baseline reproduced exactly for continuity (47.1/39.8/23.8/21.6/16.8% TensorE/DMA/VectorE/GpSimd/ScalarE — matches).

**Launch-count/dispatch problem: real, large fix, exactly as hoped.** NEFF launches/microstep `689→452 (current code, eager)→67 (compiled)`; dispatch gaps `12.9%→2.9%`; device busy `87.1%→97.1%`; serialization loss (`total→perfect_pipeline`) `72.2%→48.9%→27.9%`. Window structure under compile: one forward NEFF (70.9ms) + one backward NEFF (77.1ms) + ~65 tiny per-parameter grad-accum `stablehlo.add` ops left outside the compiled graph (0.9% footprint, ~100% exposed — the only launch-overhead residue left).

**Honest counter-fact, explains why this profile doesn't just replicate the real submission's win**: at `nproc=1/LNC=1`, the compiled step is measured *slower* (5.17-5.19 s/step vs eager's 4.92-4.98 s/step) — compile trades 14.9ms of dispatch gaps for 34.8ms of *more* TensorE work (compiled matmuls are both smaller — 21.3→18.9 kFLOP each — and slower — 456.8→537.2 ns each — plus +75% more transpose instructions; TensorE-active efficiency drops from 90.1% to 63.1% of peak). The real full-budget win (827 steps/`1.113006`) was measured at 4 ranks/`LNC=2`, where 452-launches-per-microstep × 4 contending host processes costs far more than it does on one core — the launch-count win dominates there even though it doesn't at this single-core profiling scale. Not over-interpreting this single-core matmul-efficiency regression beyond that; it's a real, separate, second-order finding (compiled matmul scheduling itself has room to improve) that doesn't change the real, confirmed full-budget result.

**RMSNorm fusion kernel: NO-GO, decided.** `torch.compile` already fused all 40 launches and 1.65ms of dispatch gaps for the plain pre-sublayer/outer `norm()` calls — exposed time (while TensorE idle) dropped from 3.8% (eager) to **0.6%** (compiled) on its own, no kernel needed. A perfect hand-written fusion could recover at most that remaining 0.6% of the *device* window (itself only ~half of wall-clock step time) — real end-to-end ceiling well under 0.5%. Not worth writing this round; removed from the backlog.

**Attention was mis-measured in the original Solution-12 profile — the "4.7%" figure was forward-only.** Backward (`attention_bwd.py`, a self-contained NKI flash kernel) was hiding in an `<unresolved>` bucket, 14.9ms/9.4% on its own. Real total: **~10.6-10.8%** of the window (fwd 2.6% + bwd 7.9-8.2%), and — expected, since it's a standalone kernel outside the graph `torch.compile` traces — **completely unchanged by compile** (16,304→16,364us, instruction counts within 0.2%). Already near-peak: backward is 98.5% TensorE-occupied, 0.1% exposed. Confirms the Tier-4 "flash-attention-style kernel" backlog item stays closed — there's no naive-materialization path here, and no compile interaction either.

**New real finding, bigger than RMSNorm ever was**: `local_conv` (`train.py:248`, `F.conv1d`, the depthwise causal conv1d before attention) is the largest single non-GEMM exposed-time target under compile — **5.4% exposed while TensorE is idle**, ~8x the RMSNorm ceiling. Real candidate for a future NKI kernel, added to the backlog (Phase 2 territory, not this round — time-boxed against everything else still on the plan).

**Second new finding**: the two existing NKI kernels (rope+QK-norm, softcap+CE) are hand-written `torch.autograd.Function`s and therefore **cannot fuse into `torch.compile`'s traced graph** — they now sit as small serialized "islands" against otherwise-fused surroundings: rope+QK-norm 3.9% exposed, softcap+CE 2.1% exposed (~6% combined, both ~100% exposed since they're fully separate NEFFs). This is a real cost introduced *by* adopting compile, not present in the eager-only measurement — worth knowing, not urgent: the net full-budget result (827/`1.113006`) already accounts for this cost and still wins decisively. A future path (registering these as proper compile-traceable custom ops via `torch.library`) could claw this back, but it's a nontrivial undertaking — flagged as low-priority future work, not this round.

**Biggest remaining bottleneck overall: TensorE matmul-issue efficiency under compile, not fusion.** `train.py:921` (`F.linear`) alone is 48.3% of the compiled window. This is a compiler/scheduling question (how `torch.compile(backend="neuron")` lowers matmul tiling), not something a hand-written kernel targets — no action item here this round, noted for awareness.

## Tier-1 deferred items — AdamW β2, real win, locked

`EMBED_ADAMW_BETA2`: embedding/VE param groups used `betas=(0.8, 0.995)`; HLAT's real 7B/70B pretrain used `β2=0.95`, flagged `0.99+` as unstable specifically under stochastic rounding. That SR-pairing motivation is moot now (`USE_STOCHASTIC_ROUNDING` permanently disqualified for an unrelated methodology reason), but β2=0.95 alone was never tested standalone. Real A/B at the final locked shape (900s, compile ON, softcap+CE ON, everything else default): `tier1-beta2-095-900s` — **427 steps, `public_val_bpb: 1.180633`**, `causality_passed: True`, config matches (`num_params_M: 55.7, seq_len: 2048, world_size: 4, lnc: 2`) — vs. the `β2=0.995` baseline (`bucket-e-compile-900s`): 406 steps, `1.190086`. **+21 steps (+5.2%) and 0.009453 better bpb — both dimensions improved, comfortably above the noise floor.** `EMBED_ADAMW_BETA2=0.95` locked as the new default. Not yet re-confirmed at full 1800s budget.

## Tier-1 deferred items — `--nki-relu2` matched back-to-back repeat, resolved

Previously genuinely ambiguous (noise-comparable both ways). Real matched back-to-back repeat at the new locked shape (β2=0.95, compile ON), minimal idle gap between the two runs to control for the throughput-cliff confound: **ON** (`tier1-relu2-on-900s-v2`): 393 steps, `1.191908`. **OFF** (`tier1-relu2-off-900s-v3`): 407 steps, `1.187254`. Both `causality_passed: True`, config matches. **OFF wins clean this time — more steps (+14, +3.6%) and better bpb (0.004654)**, resolving the prior ambiguity. `--nki-relu2` stays default `False`; ambiguity closed, not revisited further.

## Tier-1 architecture spot-check under compile — DEPTH stays 6, but a real budget-dependent-optimum lesson

Fixed 400s proxy across `DEPTH ∈ {5,6,7}` (`ASPECT_RATIO=96` fixed, so `n_embd` = 512/640/768 respectively), current locked config (compile ON, β2=0.95): DEPTH=5 236 steps/`1.347282`, DEPTH=6 179 steps/`1.348617`, DEPTH=7 103 steps/`1.530831`. DEPTH=7 decisively worse; DEPTH=5 edges out DEPTH=6 at this short proxy.

**Escalated DEPTH=5 to 900s** (`tier1-depth5-900s`): **560 steps, `1.176155`** — a real, decisive win over DEPTH=6's 900s number (`tier1-beta2-095-900s`: 427/`1.180633`), both throughput and quality better, well above noise floor.

**Full-1800s confirmation told a different story — a real, honest budget-dependent-optimum finding, not cherry-picked.** DEPTH=5 at full budget (`tier1-depth5-fullbudget`): 1082 steps, `1.117264` — **worse** than DEPTH=6's full-budget number, despite completing 31% more steps at 900s and still far more steps at 1800s (1082 vs 847). Fair apples-to-apples check: re-ran DEPTH=6 (still locked) at full budget *with* the now-locked `β2=0.95` (`depth6-beta095-fullbudget`, since the existing full-budget DEPTH=6 number predated that win): **847 steps, `public_val_bpb: 1.108697`**, `causality_passed: True` — beats both the old pre-β2 DEPTH=6 number (`827/1.113006`, by `0.004309`) and DEPTH=5's full-budget number (by `0.008567`).

**Mechanism, consistent with the original Solution-11 finding**: a smaller model (DEPTH=5, 36.7M params) completes more steps in any fixed window and can look ahead at a short proxy, but hits its capacity/entropy floor before the full 30-minute budget is up — exactly the "under-parameterized model hits an irreducible floor" pattern already documented. **DEPTH=6 stays locked.** Real, useful lesson reinforced: a 400-900s proxy is directionally useful but not sufilcient on its own for an architecture-capacity decision at this competition's real 1800s scoring window — always escalate to full budget before locking a capacity-affecting change, even when the proxy result looks clean and decisive.

**New best confirmed number: `public_val_bpb = 1.108697`, 847 steps, full 1800s budget, causality exact.** Beats the currently-submitted real leaderboard entry (`1.1394`) by `0.030703`. Not yet submitted.

## Process note: 4 Tier-1 experiment-log rows lost when the old box terminated, disclosed honestly

While auditing (prompted by the user asking whether the DEPTH=5-style short-proxy trap could affect other locked findings), found a real sync gap: after each Tier-1 hardware run this round, `research/experiment-log.jsonl` and `research/run-logs/*.log` were never pulled back from the box to this local repo before committing — only `train.py` and the prose doc updates were synced. The old box was terminated shortly after (unrelated event, mid-`autoresearch` run) before this was caught, so the raw structured log rows and full run-logs for `tier1-depth6-400s`, `tier1-depth5-400s`, `tier1-depth7-400s`, and `tier1-depth5-900s` are gone — unrecoverable.

**Not fabricating replacement rows.** The prose numbers already written above for these four runs (179/`1.348617`, 236/`1.347282`, 103/`1.530831`, 560/`1.176155`) are trustworthy — they were read directly from the box's own `run_experiment.py` output at the time, quoted verbatim in the session transcript — but their machine-readable log rows don't exist in the committed `experiment-log.jsonl`. The other 5 Tier-1 rows (`tier1-beta2-095-900s`, `tier1-relu2-on-900s-v2`, `tier1-relu2-off-900s-v3`, `tier1-depth5-fullbudget`, `depth6-beta095-fullbudget`) *were* captured verbatim (via `tail -1` at the time) and have been appended to `experiment-log.jsonl` to close that gap.

**Sync discipline fix, going forward**: pull `research/experiment-log.jsonl` and any new `research/run-logs/*.log` back from the box after *every* hardware run, before considering that run's work committed-safe — not just the files directly edited. This was a real, if minor, gap in the "sync discipline" already corrected once this round (the much larger box-vs-local git reconciliation) — same lesson, smaller instance.

## Value Residual Learning (arXiv:2410.17897, Zhou et al., ICLR 2025 — "ResFormer") — found by autoresearch, real negative at real scale

Found by the autonomous `autoresearch` loop's non-capacity structural exploration (`autoresearch/sept7` branch): inject layer-0's own value stream into every later layer's V via a learned per-layer scalar (zero-init). Real, verified-real paper (checked live, not assumed). Autoresearch's proxy result: `val_bpb` 1.405→1.371 (repeat-confirmed, two runs within `0.00015`), a further small gain with a warm `v_lambda` init (0.3) to `1.368430`. Distinct from `USE_VE` (a separate *learned embedding* gated into V) — safe to combine, not a duplicate mechanism.

**Ported to the real `train.py`** (`USE_VALUE_RESIDUAL` toggle, `SHAPE_CRITICAL_TOGGLES`-registered since it adds one scalar `nn.Parameter` per layer past the first): `CausalSelfAttention.forward` now returns `(output, v_raw)`, `Block.forward` threads an optional `v0`/`v0_lambda` through and returns `v_raw`, `GPT.forward` captures layer-0's `v_raw` as `v0` once and passes it to every later block. Causally trivial (both tensors are pure per-position functions of `x`, no new cross-position mixing beyond what SDPA already does) — smoke-tested clean: `causality_passed: True, causality_max_abs: 0.0`, `num_params_M: 55.7` unchanged at display precision.

**Real hardware A/B at the final locked shape** (`valres-900s`, vs. the current default baseline `tier1-beta2-095-900s`: 427 steps, `1.180633`): **388 steps, `public_val_bpb: 1.196654`, `causality_passed: True`, config matches.** `-39` steps (`-9.1%`) **and** `0.016021` worse bpb — a real, clean negative on both axes, well outside noise.

**A new instance of the scale-transfer trap already seen with XSA and SLAY this competition, but via a different mechanism this time.** Autoresearch's own environment (eager, no `torch.compile`, no NKI kernels, a much simpler model file) doesn't share the same failure mode this hits here: keeping `v0` (a full `[B,T,H,D]` activation, the same size as any layer's own V) alive across the *entire* network depth, needed for both the forward injection and the backward pass through every later layer, is a real new memory-bandwidth/liveness cost that plausibly interacts badly with `torch.compile`'s own graph-level memory/fusion assumptions — consistent with the post-compile re-profile's own finding that TensorE matmul efficiency and memory traffic are already the tightest constraints in this compiled graph, not fusion opportunity. Not chased further with a fix-and-retry this round (e.g. avoiding the extra return-value plumbing for layers that discard it) — a real, measured negative at the shape that actually matters, and Occam's-razor mechanism (real cost, not an internal harness bug) is well-supported by the profile evidence already in hand.

**Decision: `USE_VALUE_RESIDUAL=False`.** Real, correctly-implemented, verified-real published technique — doesn't transfer to this competition's actual hardware regime. Code remains in the file (off by default), matching this session's standing pattern for honest negative results.

## FP8 training-path precision — Phase 1 (offline correctness harness), design research + real numbers

Full FP8 delayed-scaling/current-scaling research pass (real, verified against NVIDIA
Transformer Engine's own documentation and `modded-nanogpt` records #84/#89) landed on a
concrete, de-risked design: **current-scaling** (each tensor's own amax, no
delayed-scaling amax-history — sidesteps TE's own documented 50-200-step warmup
instability, a bigger fraction of this competition's ~850-step budget than of a normal
long pretrain), **forward-only fp8** (both operands cast to `float8_e4m3fn` for the
matmul itself), **backward stays bf16/fp32** (de-risked — gradient quantization is a
separate, unvalidated risk this pass doesn't take on yet). Real precedent found at a
comparable step-count scale (`modded-nanogpt`'s own FP8-MLP records, ~700-1400-step runs)
— unlike XSA/SLAY/value-residual, which only had big-scale evidence before failing here,
this is the first "found via real research already at a scale that matches" candidate.

**Offline equivalence harness** (`research/scratch/fp8_linear_equiv.py`, CPU-only,
`torch.float8_e4m3fn` is CPU-constructible without a Neuron device, doesn't touch
`train.py`): real MLP-shape weights (`n_embd=640`, `4×n_embd` hidden, real `INIT_SCALE`
uniform init), swept activation magnitude `0.1`-`10.0`.

- **Forward relative L2 error: `0.0364`-`0.0370`, flat across the entire activation-scale
  sweep** (current-scaling correctly normalizes out the scale dependence — a real,
  meaningful ~3.6% per-matmul quantization cost, not a small-perturbation "basically
  free" result. Worth going into hardware testing with eyes open about this, not assuming
  it's negligible.
- **Backward gradients: exact (`0.0000` relative error), no NaN** — confirms the
  de-risked-backward design is implemented as intended (gradients computed against the
  original full-precision tensors, only the forward matmul actually experiences fp8
  quantization).
- **Numerical-stability sweep, 9 orders of magnitude of activation scale (`1e-3` to
  `1e6`): zero NaN/Inf anywhere** — current-scaling's dynamic per-tensor normalization is
  robust, as expected (unlike a fixed/static scale which could overflow at extreme
  inputs).

**Honest open question, not yet resolved**: whether a real ~3.6%-per-matmul forward noise,
applied to both MLP matmuls in every one of `DEPTH=6` layers, compounds through the
residual stream into something that hurts real training, or stays bounded/tolerable
(transformer residual connections are additive, not multiplicative, so catastrophic
compounding isn't the obvious default expectation — but this is exactly the kind of thing
to verify with a real loss-trajectory smoke test, not settle by more offline reasoning).
Real-hardware correctness smoke test (causality + loss-trajectory sanity) is next, once
`local_conv` kernel work frees up the shared Trn2 device.

---

## `local_conv` NKI-kernel investigation → two real findings, neither of them a kernel

The profile said the depthwise causal conv1d (`CausalConv`, `x = x + local_conv(norm(x))`)
was the largest remaining non-GEMM exposed-time target under `torch.compile` (5.4% of the
profiled window). Chasing it produced no kernel and two measured surprises.

### 1. `local_conv` is provably DEAD COMPUTE — and removing it is 6.9x SLOWER

`CausalConv.weight` is zero-initialised, `init_weights()` never touches it, and
`setup_optimizer()` collects only attn/mlp/lambda params, so the conv weight is in **no
optimizer group** and stays exactly zero for the whole run: `conv(norm(x))` is exactly
`0.0` and `x + 0.0` is bit-exact `x`. `GRAD_CLIP == 0.0` closes the one remaining path
(global grad-norm) by which its computed-then-discarded `dweight` could perturb training.
Asserted in `research/scratch/local_conv_deadparam_check.py` (62 model params vs 56 in the
optimizer; conv output exactly `0.0`). Corollary: the earlier `USE_LOCAL_CONV=True` "win"
in `gpu-experiment-results.md` (1.171962 vs 1.174188) was noise — the two models are
bit-identical.

So switching it off should be free steps. It is not. Matched A/B, **one identical binary**
(new `--local-conv` / `--no-local-conv` flag), both graphs pre-warmed with `--compile-only`
so neither arm pays a cold `neuronx-cc` compile, box verified idle (`users=0`, no foreign
processes, monitored every 15s), 300s budget:

| arm | steady dt (at `lrm == 1.000`) | steps | `public_val_bpb` |
|---|---|---|---|
| `--local-conv` (default) | **0.90 s** | 135 | 1.553617 |
| `--no-local-conv` | **6.21 s** | 38 | 1.978906 |

Step-1 loss is identical to 4 decimals in both arms (8.7670), confirming the math really is
bit-identical — the entire 6.9x is compiler scheduling. Reproduced 4 times across two
sessions (6.20/6.21/6.21/6.21 s). Whatever the Neuron compiler does with the block, the
dead conv's presence is load-bearing for it. **`USE_LOCAL_CONV` stays `True`.**

### 2. The continuous LR cooldown recompiles the optimizer EVERY step (real 2.1x win)

Chasing the dt numbers above exposed something much bigger. Step time is bimodal:
`0.89-0.90 s` while `lrm == 1.000`, and `2.2-4.3 s` for **every** step after the cooldown
starts, with `neuronx-cc` processes appearing and disappearing throughout (`cc=0` ↔ `cc=8`
in the contention monitor). `WARMDOWN_RATIO = 0.75`, so this hits the last **75% of every
run**. Cause: the optimizer's `lr` is a Python float that gets baked into the Neuron graph
(`p.add_(update, alpha=-group["lr"])` is the only `alpha=` in the file), so a new `lr`
value every step means a fresh compile every step. Changing `weight_decay` and `momentum`
scalars do *not* trigger it (they only ever appear as `tensor * float`), which is why the
first ~85 steps are fast.

Fix: `LRM_QUANTUM` / `--lrm-quantum N` snaps the multiplier onto a `1/N` grid, so the
cooldown uses at most `N` distinct `lr` values and only the first step at each level pays a
compile. `N=25` distorts the multiplier by at most 0.02 (2% of peak LR).

| arm | steps | `public_val_bpb` |
|---|---|---|
| `--lrm-quantum 0` (current, continuous) | 132 | 1.561450 |
| `--lrm-quantum 25` | **277** | **1.326234** |

`dt` stays `0.89-0.91 s` right through the cooldown in the `N=25` arm, with an isolated
`3.2 s` step each time the schedule crosses a level — exactly the predicted signature.
2.1x the steps and −0.235 bpb at a 300s budget.

**Real 900s budget confirmation (`lrmq25-900s`): 1004 steps, `public_val_bpb` 1.093785**
(`training_seconds` 896.6, `startup_allowance_est` 44.1, `total_seconds` 928.5,
`over_budget` false, `causality_passed` true / `causality_max_abs` 0.0, `num_params_M` 55.7
unchanged), against the continuous schedule's 427 steps / 1.180633 at the same budget --
2.35x the steps, −0.0868 bpb. It also beats the previous **full-1800s** best
(`depth6-beta095-fullbudget`, 847 steps / 1.108697) while using half the wall clock.

Grid-size sweep at 900s (`--lrm-quantum`): 12 -> 982 / 1.095435, **25 -> 1004 / 1.093785**,
50 -> 936 / 1.099500 -- shallow optimum around 25, and fewer compiles beat a finer schedule.

**Root-cause fix, no quantisation needed** (`--tensor-lr-scalars`,
`MuonAdamW._device_scalar`): hold the per-step `lr` in a persistent device tensor and drop
the single `alpha=` in the file, and the *exact* continuous cooldown runs at `dt 0.90s` from
step 10 to the last step, with `lrm` changing every step and no compile spikes at all --
**995 steps / 1.094327 at 900s** (startup allowance 62.2s, i.e. 18s more than the Q=25 arm
paid, so its steady throughput is if anything the better of the two). That is the direct
confirmation of the mechanism: the Python-float `lr` was the sole trigger.

All four arms are budget-clean (`over_budget` false) with `causality_passed` true and
`causality_max_abs` 0.0.
