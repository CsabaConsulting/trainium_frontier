# Bounded review: reuse the FP8 pair for timing-before-fix diagnostics

Scope: newest Supervisor-relayed human instruction to time whole steps before
fixing gradients. Reviewed existing packet30177711824b6d82feb8e4706eb57e41784f0414,
prior independent reviewb880cc4a, `NATIVE_BINDING-v2.json`, unchanged observer,
pair comparator, and native failure receipt67ccc0a4. No launch, source/packet
edit, ledger change or authority grant.

**The existing measurement design is reusable; the old acceptance/prerequisite
wording is not an as-is binding for this new diagnostic ordering.** Exact
921271 remains backward-broken. Its all-zero gradients are a real bug, not
quantization tolerance. Removing a component-PASS prerequisite for this one
explicit diagnostic must not create a component/performance/quality PASS.

## Minimum sufficient measurement remains one pair

Keep one fresh-seed42 OFF100 then ON100, identical exact921271 bytes at the same
path, full6x1024/pack32/context1024/LNC2x4, same real loader and start position,
existing compiler flags/cache roots/SDK, matrixLR0.015 and all other defaults.
No architecture or Recursive stack. Keep the fixed20warmup/80measured window:
boundary19→99, all80 intervals included. This wraps the actual complete loop,
including backward call, gradient reduction/clipping, optimizer, cache refresh
and synchronization—not an FP8-forward microbenchmark or rounded printed `dt`.

Retain100-step completion, finite logged losses/checkpoint state, correct saved
ON/OFF toggles/config, exits, source hashes, process non-overlap, shared lock,
unchanged1800 charged cap and process/fifth-boundary timing. Start the existing
observer before each child with unique absent step-log/output paths and existing
parents. Pair comparator is the packet-local two-receipt version, not the
observer's four-receipt legacy compare. Compilation during the80-step interval
stays in the raw data and makes steady-state attribution inconclusive. Optional
sensor unavailability is a caveat, not a new gate. No mandatory ABBA or extra
debug/native-causality run is needed to collect this diagnostic. The native ON
checkpoint-forward correctness check can remain pending until the repaired
candidate is considered for acceptance; it cannot rescue a zero-gradient path.

## Required truthful interpretation

The new per-run/combined diagnostic receipt should preserve these facts (field
names illustrative, not a new control-plane schema):

- `component_zero_gradients_observed=true`, bound to the four-rank256cube
  diagnostic—not a claim that whole-model gradients were observed.
- `whole_model_gradient_status=unmeasured`, unless directly measured by
  separately described evidence; source loop completion does not establish it.
- `component_correctness=HOLD`, `work_equivalence=UNPROVEN`,
  `quality_authorized=false`, `accepted_complete_work_speedup=false`.
- Preserve the unchanged comparator output as **raw timing threshold result**,
  not an acceptance/launch decision. Its conservative criterion stays
  `1 - (ON_mean + ON_poll_error)/(OFF_mean - OFF_poll_error) >= .03`.

Boundary counts prove the unchanged call sequence progressed; they do not prove
the compiled backward executed the required math. Existing graph-stat execution
counts and compiler/cache logs can aid attribution, but neither graph invocation
nor checkpoint parameter changes prove valid VJP (weight decay can change a
weight even with zero gradient). Never count missing backward work or compiler
dead-code elimination as a valid speedup. The raw diagnostic can measure what
the broken program costs without claiming equal useful work. A raw>=3% result
can motivate further repair/investigation under the human's ordering; valid
speed acceptance requires real gradients and a corrected-path matched timing
result. A raw sub3% result does not promote this version.

Original1% component ceilings and original failure remain unchanged. The newly
requested per-parameter cosine>=.99 over three matched BF16-comparison steps is
a separate future gradient criterion, not permission to call zero vectors
acceptable or silently erase the old component failure. No full quality run
follows automatically from either a diagnostic or comparator exit0.

## Driver status and checks

The current `NATIVE_BINDING-v2.json` already has both timing argv, observer
commands and `graph_stats`; its old `requires_prior` says componentPASS and
`native_authorized` remainsfalse. The only existing executable in that packet
is `component_manual.py`, not a matched-pair driver. A concrete proposed timing
driver was not yet supplied at this audit cut; it must preserve the above
semantics rather than merely treating comparator exit0 as a terminal PASS.
This is a specific pending implementation review, not a request for another
approval ceremony or native correctness replay before collecting diagnostics.

Re-ran all10 packet comparator synthetic tests: PASS, no tensors or native work.
Exact SHA256:

- Comparator `eaf0c1fa53545e4eca40057e7c2e9b9b38d23ae3c527776da6f614df806cc6d0`.
- Tests `0b2d5eb31d94d067654478a80b6feecf651313ee53ac641438d3de1e8f66a958`.
- Observer `886c5241273b9752320ac0020229228f8579f9f85d1eecf9c88f3192a4e436d5`.
