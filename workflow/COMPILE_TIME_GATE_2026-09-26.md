# Latest human instruction: compile time is a launch gate

Poojith's direct instruction after the 00:30 IST cutoff amendment controls
over earlier full-architecture-rerun and host-queue orders. Read together with
FRESH_REVIEW_AND_CUTOFF and ALL IN. No legality or human-only action changes.

## Interpretation of the original architecture result

The original8x1024 run is compile-limited: it completed2246 versus3633 control
steps (61.82%). It is not evidence that8x1024 is intrinsically worse at matched
training work. It did compile and exit successfully, so do not invent a compiler
crash. Keep the actual1.017340669168314 BPB and mechanically returned fixed-budget
DISCARD in history; that result cannot establish a fair architectural-capacity
verdict. No history rewriting or decide.py override.

## Architecture gate, effective immediately

Before any full compile-fix rerun, run a unique-label genuine cold startup-only
test stopping after approximately5steps. Preserve the1800-second accounting and
first-five model/optimizer graph shapes. Measure uncapped launch-to-boundary
startup explicitly; a capped900 allowance is not the measured number.
Full run ONLY if measured cold startup<=900seconds. Otherwise report the
number and park the architecture. Target remains<=870seconds.

## CT lane: priority1 with K1/K2, two independent racers

Targets: base cold startup<=600seconds;8x1024<=870seconds. Every tested result
reports cold startup AND steady-state tokens/s on base and architecture.
The first5step gate may leave too few steady samples: state that limitation;
do not present two postcompile steps as a robust throughput benchmark.

Candidate directions:

1. Exact compiler flags set inside train.py: optlevel1 versus default and
   --model-type=transformer. Current launch already explicitly sets optlevel1;
   do not claim adding an existing setting is an optimization. Record actual
   compiler arguments. A >2% steady-step cost is acceptable only where it
   offsets charged startup above the cap, judged at the fixed budget.
2. Establish whether4ranks compile identical graphs concurrently on12vCPUs;
   if so test safe single-rank compile/cache reuse without collective deadlock.
3. Count distinct first-step, steady, schedule-switch and evaluation graphs;
   remove avoidable specializations while preserving trained/eval arithmetic.
4. Verify actual installed native torch-neuronx support for a layer loop;
   do not assume torch_xla scan/scan_layers works on the native backend.
5. If supported, export HLO once and compare compiler flags on CPU with the
   same SDK/compiler/target. No new machine or external-data authority.

Every patch must report cold startup; any patch that pushes the base above
870seconds needs a compile offset before a full run. CPU/static/exactness,
fresh review, one-host-job and native correctness gates remain unchanged.

## Host queue until K1/K2 land

- Preserve already-running trn21 control and its attached fresh20M eval.
- trn22: CT startup-only tests on base/8x1024, then gated full architecture
  only if qualified. Previous3e79 full binding remains parked.
- trn21: actual-loop profile P, then prospective262k-token batch test on the
  adopted schedule base (>=5%tokens/s requirement), then seed checks.
- K1/K2 timing/full runs take priority when their complete packets are ready.

## No-host-time analyst task

Compare original8x1024 and its declared same-host base training loss at matched
steps through500. Report LR multipliers as well; time-based cooldown and the
startup penalty confound this weak signal. It is not BPB, a fair equal-schedule
comparison, or adoption evidence.
