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

**Not yet done**: an independent width-only resweep at fixed `DEPTH=6` (this session's depth sweep varied width proportionally via fixed `ASPECT_RATIO`, never independently).

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
