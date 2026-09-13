# Throughput-to-capacity track — a self-contained programme for the second Trn2 box

Written 2026-09-13 (local box, no hardware). Purpose: give a second instance, and whichever
agent drives it, a track that is independent of the main box's locked config and submission
cadence. The thesis, the probes, the decision gates and the coordination rules are all here so
the track can start the hour the instance comes up.

## 0. Thesis in four sentences

The corpus is a hard wall at ~288M tokens ≈ 2195 steps, so extra throughput cannot buy more
data, only a bigger model over the same data (notebook "data wall", 2026-09-13). Our step runs
at ~8% of tensor-engine peak on the beta native-eager runtime (one NEFF per operator), while
our own sweeps show the capacity to buy: width 768 nearly tied width 640 on 20% fewer steps,
depth 8 lost only through step count. A loop 2–3× faster therefore trains a 2–3× larger plain
model for the full epoch, which is the size of the gap to the sub-1.0 cluster (board 0.97 ≈
local 0.95) and needs no exotic technique. The main box's "capacity per step" pivot is the
right conversion but is spending a 9% speedup; this track goes after the 200% one.

## 1. Coordination rules (read first)

- **Never change a locked default in `train.py` on this track.** Everything is a flag, default
  OFF, registered in `ARCH_TOGGLE_NAMES`/`SHAPE_CRITICAL_TOGGLES` as the repo already does.
  Submissions continue to come from the main box's config only.
- **One process on the box at a time** (notebook: two concurrent `eval-public`/`train.py`
  processes deadlock on device/compile-cache locks).
- Separate `--out-dir out/tc-*` and label prefix `tc-` in `research/experiment-log.jsonl`; write
  results into `research/trn2-experiment-results.md` under dated `## … (track: throughput→capacity)`
  headers; commit the memsearch log separately from real changes (both machines append to it).
- Before every timed run: `md5sum train.py` equals local HEAD; clear `/tmp/hlo_cache
  /tmp/local_cache /tmp/neff_cache /tmp/neuron_cache` if any compile exceeds ~2× its usual time
  (plan §9.x).
- Price every result with the asymmetric slope: −0.077 bpb per ln(steps) below the wall, ~0
  above it; throughput that pushes past ~2195 steps is only useful once converted into width or
  depth at a fixed step count. Decompose every delta into (steps) + (per-step residual) before
  believing it (T3.9 lesson: both 900 s winners inverted at 1800 s).
- Compile time is uncharged below the 900 s allowance; bigger graphs and more shapes are free
  to compile, not free to run.

## 2. Probe 0 — environment (30 minutes, no training)

Record on the new box, verbatim into the notebook:
- `pip list | grep -iE 'torch|neuron'`; `python -c "import torch_xla"` (present or not);
  `neuron-ls`; `neuronx-cc --version`. Same AMI as the main box is the expectation; confirm.
- `python -c "import torch; print(torch.device('neuron'))"` works (native backend) and, if
  `torch_xla` imports, `python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"`.
- Run the **unmodified upstream starter kit** (`aws-neuron/trainium-frontier` `train.py`, LNC=2,
  4 ranks, `--num-steps 60`) and record steady-state s/step and `micro_tokens`. This is the
  reference every other team started from.

## 3. Probe 1 — the execution backend (the decisive one; half a day)

Hypothesis: the mature torch-neuronx **XLA graph path** compiles a whole micro-step into one
graph with the production compiler, versus native eager's one-NEFF-per-op design that leaves
us dispatch-bound. README: "torch.compile / Neuron trace are allowed at your own risk", so it is
in-bounds; the scorer's `load_for_eval(path, device)` receives the native `neuron` device and
a plain `state_dict`, so a model *trained* under XLA is scored exactly like ours.

Protocol, on the starter kit first (small, no NKI kernels, nothing to break):
1. Port: `device = xm.xla_device()`; replace `synchronize()` with `xm.mark_step()` after each
   micro-step's backward and after `optimizer.step()`; `xm.optimizer_step` or plain
   `optimizer.step()` + `mark_step`; DDP via `torch_xla.distributed.xla_backend` (`init_process_group("xla")`)
   or `xm.all_reduce` on grads; keep `record_step_boundary(step)` after the post-step `mark_step`.
   Static shapes only (the loader already gives them).
2. Measure at identical shape (DEPTH 6, 640, SEQ_LEN 2048, batch 131072, PF 4): steady-state
   s/step native vs XLA, plus compile time. Report the ratio.
3. **Gate:** ratio ≥ 1.5× → port the locked model (`train.py` `--backend xla` flag; expect work:
   the three NKI kernels need the `torch_neuronx.xla_impl.ops.nki_jit` wrapper or a plain-torch
   fallback for the probe; `torch._foreach_*` is eager-only here — check under XLA). Ratio
   < 1.2× → close the hypothesis with numbers and move to Probe 2.
4. Also try `torch.compile(backend="openxla")` on the XLA device if available; and, on the
   native side, confirm whether `torch.compile(backend="neuron")` can wrap the **optimizer step**
   (T1.7) once `_foreach_*` is removed — that is the native path's own whole-graph option.

Hazards: per-step Python scalars recompile under XLA exactly as under native (keep the tensor
LR/momentum scalars); `.item()` forces a sync (the control all-reduce does one per step — keep
it after `mark_step`); bf16 autocast is `XLA_USE_BF16=1` there, and stochastic rounding is on by
default under it — turn it off (`NEURON_RT_STOCHASTIC_ROUNDING_EN=0`) or the causality check
fails at eval.

## 4. Probe 2 — launches per token: one micro-step per optimizer step (half a day)

Our step is 4 micro-steps of 8192 tokens per rank, ~67 launches each after compile. Hypothesis:
launch count per token, not FLOPs, sets the step time, so one 32768-token micro-step per rank
cuts launches ~4× at equal FLOPs. The only evidence against is a synthetic microbenchmark
saying PACK_FACTOR 8 is past the optimum; it was never a real run.

Protocol: `--pack-factor 8` and `16` (grad_accum 2 and 1) at the locked model; `phase_timer`
first for `step_total_ms` and launch count, then a 900 s pair against a same-day control.
Watch HBM (an eval batch of 64×1024 rows once overflowed; training at 16×2048 rows per rank is
the same order — if it overflows, `DEVICE_BATCH_SIZE`/PF combinations that fit). Expected
outcome if the hypothesis holds: 15–30% fewer s/step. Gate: ≥ 10% → adopt as the track's
baseline for Probe 3; < 5% → the launch-count story is wrong, record it.

## 5. Probe 3 — the plain block: ablate down, then widen (one day)

Hypothesis: every feature in the current block adds ops (local conv 5.4% exposed, value
embeddings and their gates, per-layer lambdas, two NKI kernel islands outside the compiled
graph, last-layer n-gram gathers), and op count is the binding constraint (T3.2 finding). A
plain block — RMSNorm, attention with the causal fast path, ReLU² MLP, tied or untied head — may
compile into a much better-scheduled graph.

Protocol:
1. Ablate-down arm on the fastest backend/micro-step from Probes 1–2: `--no-local-conv`,
   `--no-ve`, `--no-ngram-ve`, lambdas fixed, `--no-nki-rope-norm --no-nki-softcap-ce` (let the
   compiler fuse the plain ops). Measure s/step only; quality will be worse and that is expected.
2. Re-add features one at a time and record each one's s/step cost on the fast stack (the costs
   on the eager stack are not transferable — the rope kernel win, for instance, was a
   launch-count win).
3. **The payoff experiment:** with the throughput gained, sweep width and depth at a **fixed
   ~2150 steps** (shorten `--max-train-seconds` so every arm stops at the same step, as the
   parallel-block test did): 640×6 (control), 768×6, 896×6, 640×8, 768×8. Price each by bpb at
   equal steps and by whether it fits the 1800 s budget on the fast stack. Then re-add the
   features that paid on the fast stack (T3.1's tables first).

Gate for the whole track: a configuration that reaches ≤ 1.010 local (2048-context in-process
eval), i.e. ≈ 0.99–1.00 on the board, inside 1800 s with the bare launch command and a clean
dress rehearsal. That is the point at which it replaces the main box's config.

## 6. What this track must not do

- Chase FP8 (no torch path; NKI double-row is a kernel project — only after Probes 1–3 show
  the step is TensorE-bound), MTP, diffusion, byte-level, ternary, linear attention, KAN,
  n-gram drafters or count mixers — all closed with numbers this week (plan T4.4).
- Re-run the T3.9 one-liners or Engram-lite; the main box has those.
- Submit. The main box owns submissions; this track hands over a config, a dress-rehearsal log
  and an `md5sum train.py`.

## 7. Reporting

One notebook section per probe with: commands, s/step table, launch counts, compile times,
the gate decision, and the exact commit. Update plan §12 (a new §12.6 "throughput→capacity
track") with the running status so the two boxes never duplicate a run. Expected calendar:
Probes 0–2 in two days, Probe 3 in two more; the payoff sweep is where the remaining ~15 days
should go if any probe shows a ≥ 1.5× gap.
