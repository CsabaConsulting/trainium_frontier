# Cheapest falsifier and evidence contract

Research plan only. A separate Implementer writes the default-OFF patch; a
different Verifier checks it. This Researcher neither runs nor verifies it.

## Stage 0: reject the wrong boundary before hardware

Review exact shared6ef7 timer-fixed parent diff. OFF must preserve the prior route, RNG draws,
model/parameter ordering and optimizer. ON must gather original narrow table
rows with autograd enabled in an uncompiled caller, using unchanged on-device
int32 hashes, and pass them into one compiled core. No detached leaf, copied
Parameter, custom backward, CPU hash, sparse gradient or row cache is allowed.

Capture the core with a CPU counting backend in a separate synthetic probe
(diagnostic code belongs in the implementation lane, not the production file).
Inspect all captured graph inputs and nodes: no table-sized weight input or
n-gram table gather/scatter operation may remain in the intended core. It is
normal for token/unigram embedding operations to remain. The raw table input
gradients must exist and flow back through eager gather to the original weights.
Count forward/backward captures across three calls; do not mistake a CPU capture
for Neuron HLO, native graph count, layout or timing evidence.

Avoid placing one disabled gather call inside each transformer layer, which can
fragment the dense core. The proposal is an outer eager caller plus one compiled
model entry point, with a fixed tuple order. The backend can still partition that
core internally; report what it does.

## Stage 1: CPU mathematical rejection gate

Run the existing four-rank synthetic `preflight_dryrun.py` through the exact
candidate's real main with flag OFF/ON and `--trace-rmsprop`; compare OFF with
the pinned parent for at least three steps. Include exact A1/A2 shapes when
resources permit; reduced geometry is a cheap discriminator, not exact-size G2.
The verifier must inspect the harness: replacing `torch.compile` with identity
does not exercise the actual compiled boundary, so Stage 0 remains separate.

Use synthetic token patterns with repeated rows within/across microbatches and
ranks. Force a nonzero table-gradient case in a small explicit comparison too:
all-zero initialized tables/projection branches alone can hide a broken link.
Include collisions, a zero-touched row, an absent-table key rejection, and both
bigram and trigram keys. Preserve pad/hash behavior; compare against the existing
function, not a newly invented hash reference.

Require bit equality of ON/OFF logits, losses, every dense parameter gradient,
post-update parameters and RMS state/counter through three steps on CPU. Confirm
each table receives one RMS update and is rank-identical after AVG reduction.
Check original parameter IDs/order and state_dict keys are unchanged. Failure
means reject or repair the boundary; do not relax to a convenient tolerance.

Save the normal model-only checkpoint and reload with the unmodified eval hook.
Require identical ON/OFF eval logits, repeated-call identity, prefix causality,
batch decomposition, and no state/RNG mutation. Test that ON is rejected with
Engram/no-tables, and OFF preserves those existing configurations. CPU G2 appends
--no-compile: allow ON in that diagnostic mode so the real pre-gathered-input
arithmetic is exercised without a compiler. It makes no boundary/performance
claim. Keep the separate counting-backend compiled graph probe mandatory.

Positive result: independent diff review + CPU G2 + three-step OFF identity
make a native diagnostic eligible for Supervisor scheduling and Operator
preflight. They are not native, BPB or submission PASS.

## Stage 2: native boundary and startup discriminator

Use one full-chip world4/LNC2 host, one job at a time, seed42, exact A2 profile.
Freeze source/hash/argv/environment/unique labels for both ON and OFF. Leave
`NEURON_CC_FLAGS` and SDK files unchanged. Existing warm research jobs have
priority until terminal and their common lock is released. No new active
launcher, cache deletion or interruption is implied by this proposal.

First obtain a bounded native ON/OFF diagnostic under the established research
cache policy. Record loss/gradient/RMS equivalence at the earliest boundaries,
graph hashes/call sites/shapes, compile count/durations, actual cache hit/miss
evidence, and stable whole-step timing. Preserve the exact source's charged clock,
checkpoint reserve and startup cap. A too-short time budget can stop at the
five-step boundary and cannot meet a 30-step diagnostic goal. Prefer an explicit
30-step diagnostic with the ordinary 1800-second budget, labeled as diagnostic,
after the separately reviewed predictor is used identically on both arms if
needed. It cannot supply a fixed-budget quality claim.

Native graph proof must distinguish:

- compiled model forward/backward regions;
- eager gather and its dense scatter-add backward;
- rank-owned Muon projection graph specializations;
- dense table reduction/clipping/RMS work and any layout/reshard operations.

Historical four-rank `research/run-logs/reshard-a/` already falsified the
steady-state dynamic-address resharding explanation on the older shape. This
experiment earns continuation by reducing **cold compilation**, not by repeating
the old one-rank narrative. Python hoisting/merged gathers inside the same compiled
forward were already null; verify the boundary itself before using host time.

Full-table traffic merely relocating to eager execution is a possible failure.
CPU equality plus missing graph attribution is inconclusive, never a claim that
cross-core resharding has been removed. A claim of graph-count reduction needs
unique graph/hash evidence across the complete step, not fewer Python functions.

## Stage 3: cold fixed-budget proof, only for an eligible candidate

The Supervisor must select the candidate for the cold submission-development
gate under the newest directive. The Operator allocates new uniquely named cache
roots without deleting existing caches, captures which roots the installed SDK
actually uses and proves they were empty. Warm cache results cannot replace this.
Measure process start to **the fifth completed step boundary**, compile phases
and process exit independently; record raw values and the 900-second capped
allowance separately. Bind the observation in the cold-start registry.

Run matched same-host ON/OFF with identical reviewed candidate bytes and flags
except the boundary switch, independent empty cache roots and 1800-second budget.
Faster startup alone does not prove better BPB: run fresh public-20M/context1024
evaluation on each valid checkpoint and retain source/logs/checkpoint hashes.
The old five-step A2 is not a control. Do not compare A1/trn21 timing directly
against A2/trn22 to assign a causal cost to table count.

Mechanism target: at least 10% less measured startup and <=3% stable whole-step
regression versus the matched OFF run. Planning target: startup <=870 seconds;
actual official cap remains 900. If startup still exceeds the planning target,
record the actual charged excess and full-run quality; do not convert warm BPB
with an assumed penalty or call a margin failure solved. Submissions remain
human-only and require the applicable cold replay/audit gates.

Negative boundary result closes this specific approach. Next is an evidence-led
distinct-graph proposal; compiler flags remain last. No failed gate is hidden by
changing thresholds, timing, table size, optimizer math or evaluator behavior.
