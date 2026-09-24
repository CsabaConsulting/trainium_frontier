# Independent verifier receipt: conditional c5c A2/trn22 package

Reviewed 2026-09-24 22:24 UTC by a Verifier independent of the package
Implementer and Operator. Reviewed controller commit
516e2c862717e362f8db41d68c8dcefb91c319ad in a separate clean worktree.

**Verdict: STATIC PACKAGE FAIL; DISPATCH HOLD.** This is a package-contract
failure, not a G2 result, a table-quality result, or a rejection of the A2
hypothesis. The current four-file package must not be staged as an exact
queue-derived job or used for a host action. An Implementer should repair the
two mismatches below, and a different review of the new package hash is
required. This receipt grants no G2, native, training, AWS, breaker, cleanup,
adoption, or leaderboard authority.

## Failures and exact repair criteria

1. **The ordered A2 argv does not equal the queue expansion.** The control
   argv equals the queue base argv exactly. The queue's A2 delta says to
   replace the single --no-ngram-ve token immediately after --n-embd 1024
   with this ordered profile:

       --ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64
       --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop
       --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude

   Consequently the literal queue expansion places that profile before
   --no-qk-shift and ends with --matrix-lr 0.015. The contract's a2_argv
   instead places --no-qk-shift immediately after --n-embd 1024 and appends
   the profile *after* --matrix-lr 0.015. An independent ordered-array
   comparison returned false. The same flags and values occur once each,
   but parser equivalence is not an exact-argv receipt. Repair by making
   GATE_CONTRACT.json a2_argv the literal ordered expansion of
   research/v2/queue.json's A2 argv_delta. Make the RUNBOOK's rendered G2
   and full command order match that array. Then recheck G2, smoke, full
   payloads against the repaired array and issue a new package hash.

2. **The environment rule blocks the required optimizer-boundary log.**
   GATE_CONTRACT.json sets
   environment_binding.no_other_NEURON_COMPETITION_R1_variables=true, and
   RUNBOOK.md says no other variable with that prefix may be present besides
   SEED and MAX_TRAIN_SECONDS. But unchanged prepare.py reads
   NEURON_COMPETITION_R1_STEP_LOG at import and record_step_boundary is a
   no-op when it is unset (lines 45 and 96-109). The exact c5c train.py calls
   record_step_boundary after each completed step (line 4010). The archived
   trn22 A0 external observer authenticated that variable and its boundary
   file. The package requires 30 sequential smoke boundaries and full-run
   boundary evidence, so the absolute ban conflicts with its stated gate.
   Repair by allowing the non-science STEP_LOG path in the clean env, binding
   its exact value and output identity in the immutable request and observer,
   while continuing to reject unapproved science overrides. The explicit
   --out-dir argv can supply the output directory without an OUT_DIR env
   override. No threshold or source change is needed.

## Checks that passed in this static review

- Read root AGENTS.md, README.md, CURRENT_STATE.md,
  research/experiment-protocol.md and DECISION_LOG.md in the prescribed
  order, then the active controller/table briefs, workflow Verifier role,
  proposal template, package files, source call sites, G1 and host-state
  evidence. The newer human brief permits A2/trn22 and A1/trn21 in parallel
  once each has its own gates; A2 versus same-host A0 tests the predefined
  profile. A same-host trn22 A1 result is still required to attribute the
  incremental layer-1 effect.
- Commit 516e2c86 adds only the proposal, gate contract, Implementer handoff
  and runbook. It edits no train.py, prepare.py, tokenizer, evaluator, data,
  launcher, queue, or decide.py. Both package JSON files parsed with jq and
  git diff --check passed. A simple private-key/credential signature search
  found no match in the package files.
- Independently recomputed SHA-256 values match the contract: research-v2
  train.py c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586;
  prepare.py 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb;
  queue.json 1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00;
  preflight_dryrun.py 6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8;
  decide.py a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66.
  The package's proposal and contract SHA-256 values are
  9f406cc97e4d0ce8fbc1e5f01a623dd4441a05bec02f90560cbe068f6b4d9d8f
  and f69d6a17186e369e9f244f00fdb8ce4de51658ad57f02c97b57e085d4f9ae56c.
- The queue selects A2 kind table_arm on trn22 with the profile above,
  predicted delta -0.008, interval [-0.017,+0.003], P 0.6 and two host-hours.
  The proposal's EV 0.0024 is arithmetically consistent; all are priors, not
  exact-c5c results. The profile changes the table module as one predeclared
  factor against dense A0. It cannot isolate layer 1 against A1.
- The depth-6 c5c source has VE layers 1, 3, 5; 1,-1 resolves to 1 and 5,
  with first/last-VE trigram tables. Expected keys b1, t1, b5, t5 are correct.
  At vocabulary 8192 and multiplier 64, each BF16 table is
  524288 x 128 = 67108864 elements = 134217728 bytes (128 MiB).
  Four tables total 268435456 elements and 536870912 parameter bytes
  (512 MiB) per rank. Equal-sized BF16 dense gradients and RMSProp moments
  bring the simple estimate to 1610612736 bytes (1.5 GiB) per rank.
  This excludes projections, other model state, activations, temporary
  tensors, compiler memory and all disk growth; it is not a measured peak.
- Static source review finds that make_plan excludes the known RMSProp table
  group from AdamW/Muon ownership; the training path averages gradients,
  clips the tables separately, and calls the replicated RMSProp group once
  on each update step. ADAM_EVERY_N defaults to 1, so the proposed two-step
  G2 trace should show two group calls and eight per-table checks per rank.
  The exact-size G2 was not run or passed here. Its per-rank trace, strict
  checkpoint load, finite values and cross-rank hashes need independent
  review before native smoke; native compilation and device equality remain
  unproved.
- The trn22 G1 operator receipt hash
  d78803b0ed21f9f4470560840ae7f78bb2b8ccaf2c82d4d8d9a472c4a6445a57
  matches. It reproduced 0.992968300659387 at context 1024 and 20971520
  tokens on a retained checkpoint, strictly as evaluator calibration. The
  only c5c trn22 A0 ledger row is the consumed r1 disk failure at zero
  optimizer boundaries, with no checkpoint or BPB; r2 is still a held
  package. There is no valid same-host c5c A0 control yet.
- The three proposed G2/smoke/full labels occur locally only in this
  package. Remote checkout and output-path absence is unverified and must
  be checked again before staging. The package correctly has no final job
  or request, no one-shot authorization, and no G2/native/full receipt.
  The recorded trn22 free space is 6448918528 bytes, below the unchanged
  6979321856-byte floor; state.json has launch_authorized=false and the
  trn22 disk breaker tripped. A2's root and HBM peaks remain unknown.
- The decision tool remains unchanged: one-host nomination delta
  <= -0.0006, two-host adoption only with both deltas negative, mean
  <= -0.0010, disagreement <= 0.0015, human acknowledgement, and its
  unchanged big-win HOLD rule. Neither this receipt nor the package supplies
  a quality verdict.

## Still required after repair

A fresh independent static review of the repaired exact argv and environment
contract; human root-volume/guest-filesystem expansion and trn22 breaker
acknowledgement/reset; a fresh read-only host and A2-specific disk/HBM
projection; clean exact-source staging with six cache identities and unique
labels; a contained full-size four-rank, two-step c5c CPU G2 with independent
inspection of every RMSProp rank trace; a separately authorized and reviewed
30-step native smoke; a valid archived trn22 c5c A0 full-budget control with
fresh public-20M/context-1024; and a separate Supervisor one-attempt request
and Operator preflight. No G2, host, AWS, training or evaluation action was
performed in this review.
