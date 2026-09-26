# Exact36: batch262144 port and matched whole-step screen

Completed Researcher design; **NON-LAUNCHABLE**. This is one useful fallback
behind ready K2 work and clean/cold submission qualification, not six ready jobs.
No code, host, queue, adoption or launch change is made here.

## Evidence and selected change

Parent is exactly `7419927c7417b5666a984ec9c2b849feadabaaa9`:
`workflow/proposals/research-v2/train.py`, SHA256
`36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Its confirmed profile is 8x1024 U-Net, attention0,2,4,6,7, initial-cache refresh,
startup-adjusted schedule, pack32/context1024/world4/LNC2, LR.015/seed42. It has
113256724 parameters, TOTAL_BATCH_SIZE131072 and no --total-batch-size flag.
Recursive is absent from this source/profile; the unadopted Recursive child and
LR.012 must not enter this arm. Authority/profile: `adoptions/20260926-clock36-confirmed.json`.

Port only the five-line CLI exposure from `4e361d15baef94f041501f558620422c55ae8c88`
into an isolated copy of exact36: global TOTAL_BATCH_SIZE in main, integer
--total-batch-size default131072, positive-value parser guard, then assignment.
Preserve the existing divisibility check and all loop/model/optimizer/cache/
schedule/data/checkpoint/evaluator code byte-for-byte. No new environment override,
automatic LR scaling, batch ramp, microshape change or default change. Removing
those additions must reconstruct exact parent bytes. Freeze the new SHA; none is
invented by this proposal.

The old a489f7/6ef7 six-layer Recursive arm ran105 steps and achieved
282266.5 versus268831 tokens/s on its predeclared printed-median statistic:
4.99774951549%, below5% by6.05 tokens/s. Preserve SPEED_SCREEN_BELOW_THRESHOLD.
Its cold status was unknown; observed startup794.928466952988s. Its100-step
aggregate281488.70004656207 had no matched control aggregate and was not the
decision statistic. Evidence:
`operator/20260926-trn21-batch262144-timing/completed/speed-screen.json` and
`verification/20260926-batch-result-seed43-launch-independent/REVIEW.md`.
Neither a rounded PASS nor a proven loss on the new eight-layer base follows.
Old source/CPU117453580-parameter receipts are implementation precedents only.

Mechanism: two sequential microsteps amortize one optimizer/reduction/cache-
refresh cost across twice as many tokens. In the simple T=B+O model the gain is
O/(2B+O); >=5% requires fixed per-update O to be >=9.5238% of current step time.
That fraction on exact36 is unmeasured. The old near-threshold result makes this
worth one matched screen, not a high-confidence speed/BPB forecast.

## Batch legality and exact semantics

| Frozen profile | World tokens per microstep | Accumulations/update | Decision |
| --- | ---: | ---: | --- |
| device1, pack32, context1024, world4; total131072 |131072|1|control|
| Same; total262144 |131072|2|selected arm|
| Same; total196608 |131072|1.5|reject by existing modulus check|
| pack16; total196608 |65536|3|mathematically valid different microshape; outside this proposal|

196608 is not intrinsically prohibited by competition rules. It is invalid for
this frozen pack32 profile. Making it work by pack16, context/world changes or
fractional accumulation would alter the experiment; do not silently do that.

Both selected arms keep each rank's forward input32x1024. Arm262144 consumes two
successive unchanged-loader microbatches per optimizer update. Zero once, backward
each mean micro-loss divided by2 in original order, unscale once, rank AVG once,
clip the averaged gradients once, optimizer update once, then refresh caches once.
The step0 initial differentiable cache refresh stays before the first microstep.
Both backward passes must contribute through those caches; no detach or graph-
lifetime workaround is authorized. Loss scale1, ADAM_EVERY_N1 and batch ramp OFF
stay unchanged. No double division, per-micro clipping/update/reduction or LR
compensation. Tokens/update must be observed as accum*world_micro, not inferred
from a filename. Historical mislabelled262144 runs in `research/trn2-experiment-results.md`
are an explicit reason to authenticate this arithmetic.

Batch changes the tokens per optimizer/momentum-warmup step, data reached at a
given update, and time-based LR/WD trajectory. Those are inherent effects of the
single batch factor; no retuning accompanies it. The unchanged logged loss is
the last micro-loss, not a two-micro loss average. Do not use early loss or
step-count equality to decide quality.

## Required preparation, then one two-arm native screen

Separate Implementer produces the isolated source port; a genuinely fresh
independent reviewer receives exact diff/spec/receipts and executes its own CPU
reference. Required new-source checks: positive/default/explicit131072/262144
parsing; zero/negative/noninteger and frozen-profile196608 rejection; default
three-step bit identity versus exact36; and two distinct microbatch contributions,
normalization, one reduction/clip/update, cache gradient flow and token counts.
Use the existing G2 framework and independently checked tiny actual-model tests.
Do not demand arm262144 equals131072 weights: the batch is deliberately different.

Full-parameter CPU/gloo4 three-step probes are parent, port-default and port262144,
serially on an independently allocated idle CPU slot with >=64GiB MemAvailable.
Compare parent/default exactly before arm execution. Preserve all65 parameter
records, per-rank losses/gradients/updates exposed by the checked harness, and
trained-checkpoint ordinary-load/finite/deterministic/causal evidence. The existing
synthetic2x256 micro-inputs represent2048 versus4096 actual CPU tokens/update;
they are full parameter size, not production input geometry or native proof.
No full model is allocated on this workstation by this Researcher. Old G2 results
and reviewer approvals do not transfer to the port.

Native screen: one same-host pair, control first then arm, each fresh seed42
process, same new port source, exact adopted argv and environment, differing only
--total-batch-size131072 versus262144. Both use --num-steps105 and retain
--max-train-seconds1800. No profiler, debug-numerics, isolated-kernel benchmark,
LR changes or public eval. New labels/roots; serial lifecycle under the common
lock with complete prior-job quiescence and fresh in-lock checks before each arm.
Use the existing reviewed rehearsal/observer route with identical STEP_LOG
instrumentation on BOTH arms; rebind it independently, not the old packet's pins.

Give each arm six absent job-private compile/cache roots (plus URL/DIR agreement
and captured effective worker/compiler environment) using the proven cold path.
Do not claim cold from empty-directory declarations alone. Record uncapped
launch-to-fifth-boundary startup on each arm; observed allowance capped900 is a
separate value. This105-step cold pair incorporates the approximately-five-step
cold gate without a redundant third/fourth cold launch. If either startup>900s,
park full progression even if timing later completes; >870s is the established
planning warning. Source and through-exit charged1800 accounting remain intact.

Use all100 completed optimizer intervals after the fifth boundary:
duration=observed_boundary104-observed_boundary4. Control tokens=13107200;
arm tokens=26214400. **Primary gain=2*duration_control/duration_arm-1; require
unrounded gain>=0.05.** This is a newly predeclared matched aggregate, not a
reinterpretation of the old printed-median verdict. Same-definition printed
10,20,...100 medians are secondary checks only. Do not discard stalls, slow
steps or post-startup recompiles from the100-step denominator.

Require raw sequential0..104 boundaries, both authenticated source/argv/worker
sets, actual accum1/2 and micro131072, finite observations, identical LR formulas,
recorded actual LR multipliers, retained loadable finite checkpoints, exit0,
budget compliance and clean process/device postflight. The existing observer
polls around50ms; archive actual observations and disclose sampled endpoint
uncertainty and single-pair/run-order limitations. The sleep interval is not a
hard error bound. Missing/ambiguous endpoint timing is INCOMPLETE, not a rounded
speed pass. A threshold-crossing point estimate is only
SPEED_SCREEN_PASS_PENDING_INDEPENDENT_REVIEW; no automatic full launch.

An arm gets one process with existing1800s outer timeout plus60s own-process
termination grace; no restart under a consumed label. Each actual training arm
separately retains the source's1800s budget and capped startup accounting.
Before each arm, preserve the proven cold packet's >=10.5GiB free requirement
PLUS >=3GiB projected incremental reserve (>=13.5GiB); remeasure after control,
include all retained control artifacts/caches, and increase reserve if measured
footprint needs it. Floor6.5GiB remains; no cleanup or threshold lowering is
authorized. CPU checkpoints/native artifacts are retained/hash-accounted without
assuming the pair fits either current host. No host is reserved here.

## Shapes, cold scope, results and next authority

Unchanged forward dimensions do NOT prove unchanged compiled graphs: loss/2,
first versus subsequent gradient accumulation and cache/autograd state may
specialize. The new arm must execute its real32x1024 inputs, all113256724
parameters, both accumulation positions, initial/first-five and steady steps
on allfour ranks. Record emitted graph/compiler evidence and actual startup.
No arbitrary other-width,196608/pack16 or old6x shape sweep is required by this
proposal. The105-step screen does not establish every later LR/momentum/stop/eval
graph; the separate full1800 run and attached fresh20M must expose those costs.
No unsupported claim of complete graph inventory, cache hits or zero spills.

Below5%, any correctness/budget/cold failure, incomplete pair or missing evidence
means no full recommendation. Preserve the exact reason and logs; no automatic
repeat, alternate196608 arm or post-hoc window. If fully accepted, prepare ONE
separate262144 full1800/fresh20M proposal against a genuine same-host131072 control
on the same new default-preserving source. Old adopted architecture scores and
this105-step checkpoint are not formal control rows. BPB selection uses unchanged
decide.py/context1024/20971520tokens rules and replication; timing is not a win.
Final submission still requires whole cold1800 replay, ordinary checkpoint
qualification, fresh20M, Claude via Poojith and human upload. This document
authorizes only proposal preparation and requests separate Supervisor selection.
