# Proposal: exact-profile c5c table and RMSProp rank hashes at 50 Neuron steps

**Status: RESEARCH ONLY — NON-DISPATCHABLE.** Researcher proposal
`20260925-c5c-table-native50-hash`; no diagnostic source, reviewed request,
host stage, one-shot authority, native result, or BPB verdict exists. This is a
correctness experiment with predicted BPB delta **0 by design**, not a quality
arm. A diagnostic checkpoint cannot serve as an A0 control, an A1/A2 scored
result, or a submission candidate.

## Evidence and question

- `research/v2/queue.json` IMPL-1 accepts a **50-step Neuron smoke with equal
  per-rank table hashes**; it separately asks for CPU lockstep against 35c9 and
  a five-step gloo/AVG-gradient reference. The old IMPL-1 CPU evidence used
  multiplier 8 and an earlier source. It cannot certify c5c/m64 native state.
  This proposal addresses the native rank-hash line only; it cannot by itself
  mark the queue's separate lockstep/reference requirements or all of IMPL-1
  accepted.
- The independently reviewed c5c A2 native package is for **30 steps** only.
  Its contract and the independent
  `research/agentic/verification/20260925-v2-c5c-a2-native30-r2-static-review.md`
  explicitly leave the older at-least-50-step table **and RMSProp-moment**
  equality gate open. The A1 package likewise has only an unrun 30-step smoke.
- Exact c5c `submissions/research_v2/train.py` SHA-256 is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  c5c keeps RMSProp table groups replicated, AVG-reduces and separately clips
  their gradients, then calls `_rmsprop_step` on every rank. Its normal log
  does not emit rank-local table/moment hashes, and `final.pt` saves only the
  rank-0 model state. A 30-step boundary log or rank-0 checkpoint cannot
  establish four-rank optimizer-state equality.
- Both host-specific G1 evaluator calibrations passed, but no valid scored
  c5c A0 control, exact m64 table G2 result, 30-step native table result, or
  50-step rank-hash result is recorded in the current handoff. G1 is an
  evaluator check, not a new control or table score.

**Hypothesis:** after 50 completed, full-chip Neuron optimizer steps under
each exact c5c m64/d128 table profile, all four ranks hold byte-identical BF16
table weights **and** RMSProp `exp_avg_sq` moments, with each table's
optimizer-state step equal to 50.

| Profile | Host | Expected table keys | New proposed diagnostic label |
|---|---|---|---|
| A1, last VE layer | trn21 | `b5,t5` | `fs-20260925-v2-c5c-a1-m64d128-trn21-r1-hash50` |
| A2, layers `1,-1` | trn22 | `b1,t1,b5,t5` | `fs-20260925-v2-c5c-a2-m64d128-trn22-r1-hash50` |

Each table is `524288 x 128` BF16 and uses the queue's RMSProp, flat table
LR, and separate clipping flags. Use the exact ordered A1/A2 table argv in
their existing gate contracts, full-chip LNC2/world4, seed 42, the same
provided train shards/data order, and the control's
`NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
The only diagnostic argv extensions are `--num-steps 50` and a unique
`--out-dir` matching the label; `--no-eval-public` is already in both
profiles. No launch command is bound here. The labels must be checked
against both local evidence and live host paths before any later request;
any failed attempt consumes its label.

## Smallest implementation and proof

A separately assigned Implementer must make a **diagnostic-only copy of
`train.py`** from c5c. It necessarily has a **different SHA-256**, to be
recorded after implementation. Keep the c5c model, initialization, forward,
loss, data loader, gradient AVG/clipping, optimizer update, schedule,
checkpoint and `load_for_eval` behavior unchanged. Do not edit
`prepare.py`, tokenizer, evaluator, scoring, `decide.py`, or the scored
source. No optimizer-state serialization/resume is needed.

The narrow observational hook should fire only after the step-49
`optimizer.step` and device synchronization, before the step-49 boundary and
checkpoint. On **each** rank it records one atomic, fsynced, rank-specific
manifest in the unique private diagnostic output path: rank and source
identity; profile, table names/shapes/dtypes; the optimizer group/owner
inventory; each table's nonmissing gradient, finite table and moment status,
nonzero table and moment counts, `state["step"] == 50`; and full SHA-256 of the raw, contiguous bytes of
each table weight and its `exp_avg_sq` tensor. Hash tensors in bounded CPU
copies, never emit table contents, and keep the moment/state dtype exact
(BF16 for this profile). A missing/extra table, moment, state step, manifest,
or rank is a failure. Probe overhead contaminates final-step timing, so
report ordinary step rate only from earlier, unprobed steps and make **no**
throughput or BPB selection claim.

One exact-profile run per host is the minimum direct proof for **both**
profiles. A2's four-table equality can support the common IMPL-1 mechanism,
but cannot itself prove equality for A1's separate b5/t5 execution. Do not
infer A1 from A2 or the reverse.

## Independent gates and sequencing

1. **Source and CPU equivalence.** A different Verifier checks that the
   diagnostic diff is observational and default-off outside this diagnostic,
   computes both source hashes, and performs an independent four-rank CPU
   lockstep against unmodified c5c on identical seed-42 synthetic inputs and
   clock. At exact A1/A2 m64/d128 profiles, steps 0 and 1 must have identical
   losses, complete parameter digests, table gradients/updates, RMSProp
   moments/state counters, and AdamW/Muon ownership plans. Also preserve
   the already-required no-table c5c-vs-35c9 G2 identity. No m8 or
   different-source receipt substitutes.
2. **Exact source gates.** Each host's full-size c5c four-rank/two-step G2
   RMS trace must independently PASS for its own table keys, equal ranks,
   expected calls/checks, finite update math, unchanged untouched rows and
   strict checkpoint load. Then its separately reviewed **30-step unmodified
   c5c** Neuron smoke must pass compile, all 30 sequential boundaries,
   finite four-rank evidence, HBM/disk projection, strict model-only load,
   synthetic causality/batch-independence, and clean postflight. The 30-step
   smoke remains an initial gate, not the 50-step proof.
3. **Native diagnostic equivalence and equality.** A separate Verifier
   reviews the new diagnostic package, then an Operator may request an
   independently reviewed, one-attempt host authorization. On native,
   steps 0 and 1 must match the same-host unmodified c5c smoke's emitted
   losses and step sequence; a mismatch is HOLD for diagnosis, not an
   assumed instrumentation effect. The boundary observer must witness
   `0..49` exactly once and four workers must exit cleanly. Compare all
   four rank manifests **per table**: full weight hashes identical, full
   moment hashes identical, correct shapes/dtypes and state step 50, finite
   values, nonzero table change, replicated RMSProp excluded from the owned
   AdamW/Muon plan. A rank-0-only record, sampled rows, approximate
   numerical equality, or 50 boundaries without moment hashes is FAIL.
   The final diagnostic checkpoint must strict-load via unchanged
   `load_for_eval` in a fresh process and pass deterministic, causal and
   batch-independent tests on synthetic tokens only; no validation eval.
4. **Host and evidence gate.** Before each attempt require the protocol's
   fresh idle/device-holder/lock/disk/clean-source/environment/seed checks,
   one job per machine, six authenticated warm caches, an A1/A2-specific
   disk/HBM headroom projection above the immutable 6.5-GiB floor, and the
   independently reviewed contained launcher, live disk monitor and external
   boundary observer. Stage a unique trusted writable STEP_LOG parent
   outside the clean checkout and bind its absent file identically in
   environment, observer, guard and immutable request; no unreviewed
   launcher or hidden override. Retrieve and hash every rank manifest,
   boundary/log/guard receipt, checkpoint and postflight, record the
   diagnostic source/commit/argv/environment/box/steps, and obtain an
   independent result verdict. Any crash, NaN, missed boundary, unequal
   hash, budget/disk failure, dirty tree, missing checkpoint or incomplete
   quiescence is a failed gate.

The current campaign can prepare this package while A0 controls proceed,
but **native diagnostic dispatch follows its own exact m64 G2 and 30-step
smoke PASS**. A valid same-host c5c A0 full-budget, fresh public20M control
is independently required before either **scored** A1/A2 run; it is not a
substitute for these native hashes. A later full arm needs a new one-shot
authorization and uses the unmodified c5c science bytes. A2 may be compared
with trn22 A0 as its predefined profile; a layer-1 increment still requires
a same-host A1 comparison. Neither diagnostic may feed `decide.py` as a
quality result.
