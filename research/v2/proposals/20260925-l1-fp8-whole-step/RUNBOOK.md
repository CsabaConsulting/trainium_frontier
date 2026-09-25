# L1 FP8: exact-source native gate and matched whole-step packet

Prepared execution recipe, **not launch authority or a performance result**.
No production code, existing checker, observer, data, evaluator, compiler flag,
power mode or launcher is changed. Use the already installed serial launch
route and shared lock. No second launcher and no interruption of active jobs.
The current source registry may not contain921271. Supervisor/Operator must bind
these exact bytes through the existing checked route or proven manual path;
registry membership is not assumed and this packet adds no launcher.

This packet supersedes the original lane README's source pin and generic timing
example **for this experiment only**. The original README and builder remain
unchanged historical implementation evidence.

This revision addresses independent packet HOLD `0a8247e7`: the initial test is
one OFF 100-step run followed by one ON 100-step run. There are no debug30 runs
or mandatory ABBA block. A repeat needs a concrete uncertainty, not ceremony.

## Exact source and the remaining gate

Use `50a901374ae6f76b35b8556d32fd72b1ece28fc6`, whose
`workflow/proposals/research-v2/train.py` hashes to
`921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
OFF and ON use these **same bytes at the same absolute source path**. OFF passes
`--no-fp8-proj`; ON passes `--fp8-proj`. Do not compare against old c5c,
original FP8 `dea479`, Recursive `6ef7`, or L2 architecture bytes.

`SOURCE_MANIFEST.json` pins the unchanged support files. The source has the same
two startup-predictor edits in both arms: clear excluded samples after step5,
and permit an empty predictor. Charging, startup cap900, launch margin30,
checkpoint reserve and1800-second cap are unchanged. There is no schedule or
architecture stack here.

Independent `20ec99c2` clears **original dea479 source and reduced CPU only**.
It does not clear exact6x1024, native kernels or speed. The completed independent
timer-composition review is **e9109dec72e41f0d481bd55df49aaedf7655bb6a**, at
`research/v2/verification/20260925-lane-timer-ports-and-gather-review.md`.
It directly covers exact921271 source/timer CPU checks; retain its stated scope.
Distinct Verifier disposition must still cover three-step default-OFF identity
and exact parameter-size ON/OFF G2. The laptop
must not run exact-size tests: use an idle host's CPU under separate Operator
preflight, sequentially, OMP/MKL/OpenBLAS threads at most2. No Neuron job may
overlap those CPU checks. No local tensor test was run for this packet.

## Source-bound CPU recipe (separate Verifier/Operator)

Operator first binds `FP8_ROOT` to the **new clean checked-out source root**,
`FP8_SOURCE` to its exact candidate, `FP8_RECEIPTS` to a new receipt directory,
and `FP8_PYTHON` to the installed native environment's Python. Do not substitute
the mutable canonical file. All absolute paths and hashes enter the receipt.
Below, `<common>` means the literal ordered common argv in the proposal.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$FP8_PYTHON" "$FP8_ROOT/workflow/tools/preflight_dryrun.py" --train-py "$FP8_SOURCE" --prepare-dir "$FP8_ROOT" --steps 3 --micro-rows 1 --micro-tokens 16 --nproc 4 --timeout 900 --out "$FP8_RECEIPTS/g2-off.json" -- <common> --num-steps 100 --no-fp8-proj
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$FP8_PYTHON" "$FP8_ROOT/workflow/tools/preflight_dryrun.py" --train-py "$FP8_SOURCE" --prepare-dir "$FP8_ROOT" --steps 3 --micro-rows 1 --micro-tokens 16 --nproc 4 --timeout 900 --out "$FP8_RECEIPTS/g2-on.json" -- <common> --num-steps 100 --fp8-proj
```

These retain **all six1024-wide layers and parameters**, but G2 deliberately
uses a1x16 synthetic forward microbatch; record the shims. They are not a native
pack32 test. Do not reduce width/depth and label it exact-size. Use the same
native runtime and bounded shims for the original c5c control at
`submissions/research_v2/train.py`, verifying SHA
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` first,
with the same common argv but **without the nonexistent FP8 flag**. Compare its
receipt to921271 OFF via `preflight_dryrun.py --compare`. Require all ranks'
three losses,62 parameter digests and ownership plan bit-identical. An already
accepted exact-size c5c receipt can be reused only by an explicit independent
same-runtime/argv/shim disposition. Any further source edit changes this pin
and requires a new packet; do not silently repair on the host.

## Host binding and comparability (before every native run)

Supervisor chooses exactly one complete host label set from the proposal.
Materialize a separate per-label record with exact argv, source SHA, output and
step-log paths, safe environment whitelist, start/end UTC and monotonic times,
host boot identity and the existing runner's process/terminal receipts. Ensure
all labels and paths are unused. Start only after exact-source review/G2 and
fresh STOP, clean source, disk >=6.5GiB **plus predicted footprint**, idle,
device-holder, existing shared-lock and one-job checks. Preserve all logs,
partial outputs and failures. Never clear a breaker or delete caches/artifacts.

Reuse the selected host's actual installed native environment and persistent
cache roots from its current approved launch receipt. Record rather than guess:

- Python/Torch/torch-neuronx/NKI/compiler/runtime versions; LNC2/world4;
  `NEURON_CC_FLAGS` exactly `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
- `TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`,
  `TORCH_NEURONX_HLO_CACHE_DIR`, `TORCH_NEURONX_DEBUG_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `NEURON_COMPILE_CACHE_DIR`, and effective
  `TORCH_NEURONX_HLO_CACHE`/`TORCH_NEURONX_NEFF_DISABLE_CACHE` state.
- `NKI_VENV_PATH`, executable/PYTHONPATH resolution, seed42, training data-cache
  location, identical organizer shards/tokenizer hashes and default data order.
  Use the pinned real prepare.py, never the test-only PyArrow/prepare stubs.
- No unexplained `NEURON_COMPETITION_R1_*` model override; resolved config must
  be depth6,width1024,sequence1024,pack32,total batch131072,one accumulation step,
  matrixLR0.015,FP32 masters,BF16 backward,no tables,no loss scaling/no EMA.
  All other configuration equals the exact source defaults in both arms.

Use one source root throughout. Only arm flag, unique label, output, step-log
and receipt paths differ in the timed argv/environment. No cache reset, SDK
change, debug profiler or NKI persistence flag. Installed evidence in canonical
`research/v2/operator/20260925-trn21-live-cache-diagnosis/FINDINGS.md` shows the
native eager backend actually consumes TORCH_NEURONX NEFF/local paths; a generic
`NEURON_COMPILE_CACHE_DIR` directory is not cache-hit proof. NKI dump persistence
defaults OFF for a correctness reason. Do not enable it for this experiment.

Capture existing read-only host telemetry at the same cadence for every arm
(before/after and, if already available, during the measured window): power
policy/limit, clocks, temperatures, throttle/error events, Neuron/CPU utilization
and competing processes. Use only supported read-only interfaces already on the
host; do not invent telemetry flags or change power/governor modes. Equal watts
are **not** required: FP8 may legitimately change workload power. Unchanged
policy and absence of differential throttling/background contention are needed.
Record unavailable sensors as unavailable; never write "power matched" without
evidence. Unavailable optional telemetry alone **does not block** this diagnostic.
Known differential contention/throttling or documented drift can invalidate a
pair; disclose the actual evidence and the limits, without an invented sensor gate.

## Native component check once, then the matched timing pair

1. Under the checked full-chip containment route, run the existing pinned checker:

   ```sh
   NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 research/v2/lanes/fp8-projections/check_native.py --train-py "$FP8_SOURCE" --out-dir "$FP8_RECEIPTS/native-component"
   ```

   `PYTHONPATH` must resolve real pinned prepare.py and the installed SDK.
   Require four receipts with exact921271 SHA and clean exits. Its declared
   <=1% relative-RMS gates cover actual FP8 packed bytes/scales/saved operands,
   forward and BF16 VJP, including zero/tiny/outlier rows and padding. This is
   **limited-shape component correctness**, never throughput/full-model proof.

2. The OFF100 and ON100 runs below are also the production-shape native training
   gate. Require each to complete all 100 steps with finite logged losses,
   all four clean worker exits, quiescence, a finite checkpoint, exact source and
   <=1800 charged seconds through save/exit. Record the logged loss coverage;
   do not claim that every gradient was inspected during timing. The supplied
   component check supplies the native VJP evidence. **No extra debug30 runs,
   diagnostic gradient inspections or speed-invalidating flags are required.**

3. After that pair, in **one** fresh full-chip process, strict-load the ON timed checkpoint
   through the **unchanged** `load_for_eval`, with no training/install shim.
   Distinct Verifier/Operator checks exact6x1024 config, saved FP8 toggle ON,
   FP32 finite model state, exact checkpoint-to-loaded parameters, frozen
   parameters, repeated-forward determinism, suffix perturbation (prefix logits
   unaffected), changed-input sensitivity, and row0 invariance when other batch
   rows or batch size change. Use fixed synthetic token IDs with seed42 and
   context1024; max absolute prefix/batch difference <=0.01, repeat exactly equal.
   No validation training or alternate evaluation model. Archive raw per-rank
   differences, not only PASS. An official fresh eval alone does not prove
   causality: current prepare.py eval-public does not call causality_check.
   A supplied test-only recipe is below; it must be independently checked before
   the Operator runs it. It does not modify the model or evaluator.

   ```sh
   NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 --no-python "$FP8_PYTHON" -c '
   import hashlib, importlib.util, json, os, pathlib, sys
   import torch, torch_neuronx
   p = pathlib.Path(os.environ["FP8_SOURCE"]).resolve()
   expected = "921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858"
   if hashlib.sha256(p.read_bytes()).hexdigest() != expected: raise ValueError("source mismatch")
   spec = importlib.util.spec_from_file_location("fp8_checkpoint_gate", p)
   m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
   _, rank, _, world, device = m.init_runtime("neuron", seed=42)
   try:
       if world != 4: raise ValueError("full-chip world4 required")
       checkpoint = os.environ["FP8_CHECKPOINT"]
       if os.environ["FP8_ARM"] not in {"off", "on"}: raise ValueError("unknown arm")
       expected_on = os.environ["FP8_ARM"] == "on"
       payload = m.torch_load_weights(checkpoint, map_location="cpu")
       if bool(payload["toggles"]["USE_FP8_PROJ"]) != expected_on: raise ValueError("wrong saved arm")
       net = m.load_for_eval(checkpoint, device)
       if net.config.n_layer != 6 or net.config.n_embd != 1024: raise ValueError("wrong shape")
       marked = sum(bool(getattr(layer, "fp8_projection", False)) for layer in net.modules())
       if marked != (36 if expected_on else 0): raise ValueError("wrong projection inventory")
       if any(t.requires_grad or t.dtype != torch.float32 for t in net.parameters()): raise ValueError("not frozen FP32")
       for name, value in net.state_dict().items():
           cpu = value.detach().cpu()
           if not torch.isfinite(cpu).all() or not torch.equal(cpu, payload["model_state"][name]): raise ValueError("state mismatch: " + name)
       torch.manual_seed(42 + rank)
       x = torch.randint(0, net.config.vocab_size, (2, 1024)).to(device)
       with torch.no_grad():
           a = net(x).float().cpu(); repeat = net(x).float().cpu()
           suffix = x.clone(); suffix[:, 512:] = torch.flip(suffix[:, 512:], [1])
           future = net(suffix).float().cpu()
           companion = x.clone(); companion[1] = torch.flip(companion[1], [0])
           other = net(companion).float().cpu()
           single = net(x[:1]).float().cpu()
           changed = x.clone(); changed[:, 0] = (changed[:, 0] + 1) % net.config.vocab_size
           different = net(changed).float().cpu()
       if not all(torch.isfinite(v).all() for v in [a, repeat, future, other, single, different]): raise ValueError("nonfinite logits")
       errors = {"causal": float((a[:, :512] - future[:, :512]).abs().max()), "companion": float((a[:1] - other[:1]).abs().max()), "batch_size": float((a[:1] - single).abs().max())}
       if not torch.equal(a, repeat) or any(v > 0.01 for v in errors.values()) or torch.equal(a, different): raise ValueError("frozen forward failure: " + json.dumps(errors))
       print(json.dumps({"rank": rank, "world": world, "arm": os.environ["FP8_ARM"], "source_sha256": expected, "checkpoint": checkpoint, "errors": errors, "repeat_exact": True, "input_sensitive": True, "verdict": "PASS_NATIVE_CHECKPOINT_ONLY"}), flush=True)
   finally:
       m.cleanup_runtime()
   '
   ```

   Bind/export only `FP8_SOURCE`, `FP8_CHECKPOINT` and `FP8_ARM=on` plus the
   already approved launch environment. Hash the checkpoint separately and
   bind the hashes to this receipt. Never use this check as a speed observation.

## Initial timed pair: OFF1, then ON1

Run the two proposal timing labels back-to-back, same host/day/boot and environment,
after exact-size CPU and the one native component check. No intervening other job, checkpoint evaluation or
cache cleanup. Same checkpoint-start rule: **fresh initialization, seed42 for
every run**, never resume/warm-start from a prior checkpoint. All runs begin
at the same data position and use the same real training loader.

Literal timed argv (set `FP8_ARM_FLAG` to exactly `--no-fp8-proj` or
`--fp8-proj`; `FP8_OUT` is that label's new output):

```sh
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 "$FP8_SOURCE" --depth 6 --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 --max-train-seconds 1800 --num-steps 100 "$FP8_ARM_FLAG" --out-dir "$FP8_OUT"
```

Before each launch, start the **unchanged** observer in the existing driver:

```sh
"$FP8_PYTHON" "$FP8_ROOT/research/v2/lanes/fp8-projections/whole_step_benchmark.py" observe --train-py "$FP8_SOURCE" --step-log "$FP8_STEPS" --out "$FP8_TIMING" --label "$FP8_LABEL" --arm "$FP8_ARM" --warmup 20 --steps 100 --poll-seconds 0.005 --timeout 2700
```

The driver keeps the observer concurrent with its **one** training child;
this recipe is not a second launcher. The step-log and timing files must be
absent and their parents exist. Set the existing organizer
`NEURON_COMPETITION_R1_STEP_LOG` to exactly `FP8_STEPS`. Boundaries are decimal
lines0..99, not JSON despite a possible `.jsonl` suffix. Capture observer exit
and complete output. Its timeout is only an observer wait, not extra training
allowance. Do not run a diagnostic/debug-numerics/profiler in the timed child.

The primary interval is boundary19 to99: exactly80 complete consecutive steps
after20 warmup steps. Boundaries occur after optimizer, cache refresh and device
synchronization; their intervals also include data acquisition, preceding
loss/control work and collectives. This measures complete training steps, not
only FP8 projections. Printed `dt` omits some between-boundary work and is rounded;
do not substitute it. Keep all80 intervals, including slow ones. No cherry-picked
window, median-only promotion, dropping outliers, or changing20/80 after seeing data.

Require no compilation/recompilation activity in the measured window from the
existing logs/process evidence and matching warm graph/cache state. A populated
directory alone does not prove reuse. Any measured compilation remains in the
raw80 intervals and makes the steady-state comparison inconclusive; do not
delete it from the mean. One OFF/ON pair cannot establish repeatability or remove
run-order drift. Preserve available telemetry and block trends. An extra matched
pair is conditional on documented uncertainty, not mandatory conditioning.

Startup is **reported separately, not forgiven**: process launch to fifth
completed boundary, all first20 warmup durations, actual compiler activity,
applied capped allowance, charged startup excess, process-to-terminal wall time,
and charged elapsed through checkpoint/exit. Reuse the proven runner's timing
receipts/observer; the L1 timing observer alone has no process-launch timestamp.
Startup above870 seconds does **not** automatically disqualify this warm research
diagnostic. Charge its excess exactly as production does, and record each arm's
charged-time/schedule phase. LR is initially flat, but default weight decay varies
continuously with charged time: a flat LR is not proof of identical phases.
This comparison measures throughput, not per-step quality. All100 steps must
still fit the1800 cap through save/exit. Do not forgive excess, freeze schedules
or alter timing arithmetic to force comparability. Cold submission feasibility
is a later separate gate, not an additional cold run in this packet.
Checkpoint time is outside the speed interval but inside the budget/terminal gate.

## Analysis and >=3% gate

After both runs' four-worker clean exits/quiescence, rehash source and checkpoints;
read both checkpoint payloads on CPU for finite model state and saved arm/config.
Run the one native ON checkpoint recipe above after the pair, not between arms.
No additional OFF native frozen-forward run is required. Retrieve raw run/step/
timing/terminal/cache/power records before any performance claim.

```sh
python research/v2/proposals/20260925-l1-fp8-whole-step/compare_pair.py off1.json on1.json
```

The original observer is unchanged; its old `compare` subcommand requires four
receipts and **must not** be passed two. The small packet-local comparator above
consumes two receipts and requires the matched pair to achieve
`1 - (ON_mean + ON_poll_error)/(OFF_mean - OFF_poll_error) >= 0.03`.
It is a **3% step-time reduction** (at least3.0928% throughput uplift), not merely
`OFF/ON-1 >=0.03`. Its maximum-observed-poll-gap bound is measurement uncertainty, **not a confidence
interval for run variance**. The comparator pins921271, same host/source path,
100 contiguous boundaries,20/80 window, summary agreement with raw events,
poll bounds and observed OFF-before-ON boundary chronology. It does not establish
actual arm argv, complete process non-overlap, environment, boot, power, startup
or terminal state; Operator records establish those separately.

The same comparator reports each run's mean, median, population standard
deviation, min/p10/p90/p95/max and eight consecutive10-step block means, keeping
all80 intervals. It exits0 for timing PASS,1 when even the optimistic polling
bound is below3%,2 when polling bounds straddle3%, and3 for invalid input.
No arbitrary1% same-arm drift threshold or mandatory second pair remains.
Read available telemetry and trends alongside the distributions, not just PASS.
Run its tiny CPU synthetic checks with:

```sh
python research/v2/proposals/20260925-l1-fp8-whole-step/test_compare_pair.py
```

Final outcomes:

- **FAIL correctness/budget:** preserve exact error and partial artifacts; no
  further native/quality work until applicable diagnosis/review/authority.
- **INCONCLUSIVE comparability:** no quality promotion. Any repeat needs new
  labels and a documented cause: polling uncertainty around3%, measured compile
  contamination, documented drift/contention or another concrete ambiguity.
  Only then run one additional matched OFF100/ON100 pair with the same20/80
  window. Archive both pairs; do not pool a clearly failing valid first pair
  into a win. No repeat is required after an accepted unambiguous first PASS.
- **Valid timing FAIL (optimistic bound below3%):** stop this promotion; profile before
  another kernel revision. A component gain does not override the failure.
- **Accepted timing PASS plus native correctness:** Supervisor may materialize
  a separate full1800 proposal on these exact dense bytes, same environment and
  same-host matched OFF control; remove the100-step diagnostic cap by restoring
  the existing high step cap. Run fresh public20M/context1024 afterwards and use
  unchanged decide.py. This packet does not authorize or label the full run,
  claim a BPB benefit, change a threshold, or permit a submission. Cold-start
  feasibility remains separately required before any submission candidate.
