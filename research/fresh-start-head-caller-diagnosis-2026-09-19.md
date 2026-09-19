# Cached head caller regression and bounded allocation-scope trial

The cached BF16 head/exact CE is22.56% faster as a complete isolated autograd
component but37.791% slower in the same-host complete step:689.956164 versus
500.726543ms. Native/candidate source112, five warmup and30 timed updates,
T1024/pack16/GA2/four ranks, identical flags except cached head. Both commands
pass all four native postchecks. Candidate's disk-only postflight failure and
verified maintenance recovery remain in the evidence. No quality promotion.

The source118 local profile analysis matches ALL45 runtime dispatches and one
preserved startup barrier. It compares with the prior source97 native analysis:

| Profile quantity | Native body/head | Cached head in same graph |
|---|---:|---:|
| Forward device interval, ms |68.001178|161.379941|
| Backward device interval, ms |138.211172|135.365241|
| Forward reported spill reload bytes |6,997,243,392|14,260,823,552|
| Linear.forward reported reload bytes |1,835,008|6,520,836,096|
| MLP up Tensor active ms, core0 |6.738853|57.531466|
| MLP up Tensor active ms, core1 |6.652610|57.763615|
| MLP down Tensor active ms, core0 |6.583619|16.748693|
| MLP down Tensor active ms, core1 |6.631959|16.884104|

These profiles came from different hosts/sources with matching architecture and
training geometry; their purpose is attribution, not speed certification.
Instruction/source activity unions overlap. Missing dynamic DMA metadata means
reported counters do not establish exact physical transfer volume or bandwidth.
All source hashes, warnings, dispatch matches and installed schema definitions
are retained in the analysis JSON. FP32-cost matmuls are separately accounted;
the large body cost is not established as a precision-conversion accident.

The16MiB resident head weight cache fits the kernel's isolated SBUF estimate,
but changes to compiled allocation/layout can hurt unrelated upstream operations.
This is a stronger lead than trying to save another fraction of a millisecond
in head tanh. A prior per-layer FP8 compilation boundary lost, but it created
multiple body boundaries and does not test one boundary at the end of the body.

Trial: keep the same head algorithm and every charged transpose; compile the
entire head/loss autograd function separately from the transformer. All source112
numerical/evaluation checks remain exact, plus verify at least two compiled
backward nodes so an absent boundary cannot pass unnoticed. Permit a graph break
only in this explicitly requested boundary variant. If it passes, run native,
monolithic cached head and separate cached head at the same immutable source and
on the same clean host. Promote only on complete-step gain, then recheck pack32
and the optional separately gated saved-softcap variant. No allocation mechanism
or training quality improvement is established until those tests complete.

Evidence: run-logs/fs-w118-head-loss-cached-caller-local-analysis-r1.analysis.json,
run-logs/fs-w97-native-bf16-caller-analysis-r3.analysis.json,
run-logs/fs-w112-head-loss-cached-whole-step-comparison-r1.json.

## First boundary gate fails; precision diagnosis before timing

Source120's BF16 head/body interface produces the intended two compiled backward
nodes and identical first two losses. The ORIGINAL exact all-master gate fails
on the256-wide token embedding gradient after two microbatches: max absolute
difference2.86102294921875e-6. Command1, clean postflight04:07:19. This is not a
passed numerical gate and no full-step run is dispatched from it.

Source121 adds a separate diagnostic with native/cached heads in monolithic
and split graphs. Exact native/cached agreement within each allocation scope
would localize the discrepancy to the boundary rather than cached-head math;
all cross-scope differences are reported, never accepted by a looser tolerance.

A separately gated candidate carries BF16-valued activations in an FP32 head
interface and returns an FP32 input-gradient accumulator. This tests whether
materializing the BF16 gradient at the new boundary introduced rounding that
the original compiler elides. The head still consumes BF16 inputs/weights;
all extra conversion/traffic must be charged. Require the20component checks and
then the unchanged196GPT checks before complete timing. Diagnosis and repair
are hypotheses until those results exist. The failed source120 gate is retained.
