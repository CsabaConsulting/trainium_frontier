# Independent verifier receipt: repaired conditional c5c A2/trn22 package

Reviewed 2026-09-24 22:30 UTC by a Verifier independent of the Implementer
and Operator, in a separate clean worktree. The package under review is
Implementer commit 633299fab9bc7092a389551a736c5cd1e0a09d8f. This is a
new review of the repaired bytes, not an edit to the historical FAIL receipt
for controller commit 516e2c862717e362f8db41d68c8dcefb91c319ad.

**Verdict: STATIC PACKAGE PASS; DISPATCH HOLD.** The two identified package
defects are repaired, and the complete conditional proposal has no remaining
static contradiction found in this review. This verdict does not pass G2,
native smoke, host staging, the disk breaker, same-host A0, the scored run, or
candidate adoption. It grants no AWS, SSH, training, evaluation, protected
cleanup, breaker-reset, or leaderboard authority.

## Repaired failure checks

1. **Literal ordered argv now matches the queue.** I derived the base argv
   from research/v2/queue.json's base_command after its launcher prefix, then
   replaced the single --no-ngram-ve token with A2 argv_delta.with *in place*.
   The independent ordered-array comparison returned true for the 15-token
   dense control and true for the 27-token A2 array in GATE_CONTRACT.json.
   The A2 profile now occurs immediately after --n-embd 1024, before
   --no-qk-shift, and the payload ends with --matrix-lr 0.015. Rendered
   contract G2 and full commands occur exactly in RUNBOOK.md. The smoke
   extension is separately specified as --num-steps 30; the full scored argv
   has no --seed, --max-train-seconds, or --num-steps override. This repairs
   the exact-argv failure rather than assuming parser equivalence.

2. **Trusted boundary-file binding now permits observation without a science
   override.** The clean-environment contract allows exactly
   NEURON_COMPETITION_R1_SEED, NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS,
   NEURON_COMPETITION_R1_STEP_LOG, and optional NEURON_COMPETITION_R1_OUT_DIR
   under that prefix, and rejects other science overrides. STEP_LOG is
   required for native smoke and full runs, must be a unique absolute
   label-specific path absent before launch, append-only, identical to the
   external boundary observer's path, and bound by the final immutable
   request SHA-256. Optional OUT_DIR must resolve to the CLI --out-dir and
   also be hash-bound. RUNBOOK.md and IMPLEMENTATION.md now state the same
   rule. Unchanged prepare.py reads STEP_LOG and otherwise makes boundary
   recording a no-op; exact c5c train.py calls the recorder after completed
   optimizer steps. The archived trn22 A0 observer used that same mechanism.
   Final path values and request hashes are still null, correctly making
   this package non-dispatchable until a clean host stage binds them.

## Other static checks

- I rechecked root AGENTS.md, README.md, CURRENT_STATE.md, the experiment
  protocol, DECISION_LOG.md, the newer human controller and table-gate
  amendments, Verifier role, package, queue, source call sites, G1 evidence
  and current controller status. The repaired commit changes only
  GATE_CONTRACT.json, IMPLEMENTATION.md and RUNBOOK.md against the prior
  package. It does not edit train.py, prepare.py, tokenizer, evaluator,
  scoring, data, launcher, queue, G2 tool or decision rule. Both package
  JSON files parse, git diff --check passes, and a simple private-key/token
  signature search found no match in the package. Exact proposed G2,
  smoke and full labels had no collision outside the package in this local
  worktree; host/output/step-file absence is not yet proved.
- Recomputed SHA-256 identities at 633299fa match the source contract:
  research-v2 train.py
  c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586;
  prepare.py
  5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb;
  queue.json
  1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00;
  preflight_dryrun.py
  6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8;
  decide.py
  a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66.
  The proposal, repaired contract, Implementer handoff and repaired runbook
  hashes are, respectively,
  9f406cc97e4d0ce8fbc1e5f01a623dd4441a05bec02f90560cbe068f6b4d9d8f,
  a40723151099e0fec912d670b702ef9f2774fe14853025660a9ffc116d2fc364,
  2ae81dc03b64b6bb5a37f9ea5d0da7b0e795b1139f5e2bb942cc13c57261d740,
  and a343a901f62583013acb1e76686a4e0f87543859c5988c95733360febe9c883a.
- The predeclared A2-versus-dense-A0 factor is the complete table module
  profile, not a layer-1-only claim. On depth-6 c5c, 1,-1 selects VE layers
  1 and 5 and first/last-layer trigram tables, so expected keys b1, t1, b5,
  t5 are correct. Each 524288 x 128 BF16 table is 134217728 bytes; four
  tables are 536870912 parameter bytes per rank. Equal-sized BF16 dense
  gradients and RMSProp second moments give a simple 1610612736-byte
  (1.5-GiB) per-rank tensor estimate, not a measured HBM or disk peak.
- Static source review still finds the RMSProp table group excluded from
  owned AdamW/Muon planning, averaged table gradients with separate clipping,
  and one replicated RMSProp group step per optimizer update. With default
  ADAM_EVERY_N=1, the specified exact-size four-rank, two-step CPU G2 should
  yield two group calls and eight per-table checks per rank. It has not been
  run here. Every rank's trace, table change, equality, finite math, strict
  checkpoint load, containment and quiescence require independent evidence;
  the aggregate tool verdict alone is insufficient. CPU G2 cannot prove
  native compilation, device equality or scored quality.
- The trn22 G1 evaluator-calibration receipt hash
  d78803b0ed21f9f4470560840ae7f78bb2b8ccaf2c82d4d8d9a472c4a6445a57
  still matches. Its 0.992968300659387 at context 1024 over 20971520
  tokens is a retained-checkpoint calibration only. The consumed trn22 c5c
  A0 r1 stopped at zero optimizer boundaries without checkpoint or BPB;
  A0 r2 remains a separate held retry package. No valid same-host c5c A0
  result exists to authorize A2 full or score a delta.
- The newer human direction permits conditional A2/trn22 preparation in
  parallel with A1/trn21 after each run's own gates, superseding the older
  A1-first science scheduling recommendation. A2 versus same-host trn22 A0
  tests the profile; attributing an incremental layer-1 effect still needs
  a same-host trn22 A1 result. The unchanged decision rule requires a
  comparable same-host control, fresh public20M/context1024, a nominated
  delta <= -0.0006, and for adoption two negative host deltas, mean
  <= -0.0010, disagreement <= 0.0015 and human acknowledgement. Its
  big-win HOLD rule remains unchanged.

## Runtime and authority hold

The controller's later post-resize audit reports trn22 guest/root storage
expanded and 58454982656 free bytes at 2026-09-24 22:26:14 UTC, clearing
the **storage-only** block. The package's older 6448918528-byte observation
is historical and must not be treated as current capacity. At this review,
controller state still has launch_authorized=false and trn22
launch_allowed=false with the disk breaker tripped. A recorded human
conditional acknowledgement permits a separate independent re-arm review,
not automatic reset or a launch. The A0 10-GiB provisional floor and the
1.5-GiB table-tensor estimate do not bound A2 compile/cache/checkpoint growth.

Before any A2 host action, the Supervisor/Operator must bind fresh host,
idle/lock/device, six cache, unique checkout/output/step-file and exact
env/argv identities; a conservative A2-specific disk/HBM projection with
reserve above the unchanged 6979321856-byte floor; the independently
reviewed breaker state; a contained full-size G2 PASS; then a distinct
authorized 30-step native smoke with sequential boundaries and strict
postflight. Before a scored full run, also bind a valid archived same-host
trn22 c5c A0 seed-42 full-budget fresh public20M control and issue a new
one-attempt immutable request and separate authorization. The proposal's
runtime gates, final request and receipts remain unset. I performed no host,
AWS, G2, native, training, evaluation, staging or breaker action.
