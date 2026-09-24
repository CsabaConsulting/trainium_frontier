# Conditional A2 native-30 gate on trn22

**HOLD — no host dispatch from this file.** This is an additive Implementer
package for the exact `c5c410af…` A2 table profile. It leaves the independently
reviewed [A2 gate contract](GATE_CONTRACT.json) byte-for-byte unchanged
(`a4072315…`) and is itself awaiting a different Verifier. It is a correctness
smoke, **not** an 1,800-second scored experiment, control, public-BPB result,
or permission to run the later full A2 arm. No model, `prepare.py`, data,
tokenizer, evaluator, launcher or decision-rule source is changed.

## Exact proposed smoke

- Host: `trn22`, expected hostname `ip-172-31-32-253`; one job on that host.
- Fresh label: `fs-20260925-v2-c5c-a2-m64d128-trn22-r1-smoke`. This is the
  **planned but unused** label in the static-PASS A2 contract; search the host
  as well as local ledgers again before staging. A prior failure consumes the
  label, and any retry must be separately named and diagnosed.
- Source: stage a new, clean pinned checkout and copy exact
  `submissions/research_v2/train.py` to its root `train.py`; both must hash to
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  `prepare.py` must remain
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Full chip: `NEURON_LOGICAL_NC_CONFIG=2`, four `torchrun` ranks, seed 42,
  parser budget default 1,800 s, outer guard at most 3,600 s.
- Effective science environment, from a clean `env -i`:
  `NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`,
  `NEURON_COMPETITION_R1_SEED=42`, and
  `NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800`, plus LNC=2. The final
  request must also bind the unique trusted
  `NEURON_COMPETITION_R1_STEP_LOG=<absolute label>.steps.jsonl` to the
  observer's exact file. Optional `NEURON_COMPETITION_R1_OUT_DIR`, if used,
  must resolve to the CLI output directory. Reject all other unapproved
  `NEURON_COMPETITION_R1_*` overrides, especially step/eval/compile flags.
- Exact `torchrun` argv is the ordered array in
  [NATIVE_SMOKE_GATE_CONTRACT.json](NATIVE_SMOKE_GATE_CONTRACT.json): the
  reviewed A2 table argv, followed **only** by `--num-steps 30` and the unique
  `--out-dir`. It preserves `--no-eval-public`; neither an in-process nor a
  fresh public eval belongs to this smoke. Do not add a science/debug flag
  casually; that would require a distinct proposal, label and review.
- The six warm train-cache paths, executable `PATH`, `HOME`, `PYTHONPATH`,
  venv path, output path and all log/observer paths are deliberately unresolved
  here. The Operator binds and authenticates their path/device/inode, exact
  values and SHA-256 in a separate immutable request. Cold-cache fallback,
  cache pruning, cleanup and hidden inherited exports are forbidden.

The smoke emits `final.pt` after step 30. That checkpoint is a diagnostic
artifact, not a candidate for BPB selection. Preserve and hash it until the
evidence and independent review are durable; do not delete it under this
Implementer package.

## Gates before any native attempt

1. A different Verifier reviews this additive package against the original
   A2 contract (`a4072315…`), its static-r2 receipt (`c1f46653…`), queue,
   source and `prepare.py` hashes; compare the ordered command array to
   `GATE_CONTRACT.native_smoke`, not just its parser-equivalent meaning.
2. Finish the **exact-size** c5c A2 four-rank/two-step G2 under independent
   containment on a memory-qualified idle machine. Review all four rank
   records, not only aggregate `PASS`: keys `b1/t1/b5/t5`, two RMSProp group
   calls and eight per-table update checks **per rank**, changed tables,
   finite loss/gradient/recomputation, untouched rows exact, owner inventory
   with RMSProp excluded from AdamW/Muon ownership, cross-rank equality,
   strict checkpoint load and `max_abs_err_over_lr <= 0.03`. The current m8
   stub or different-source native receipts cannot substitute. Record the
   exact G2 request, tool/hash, temporary-space qualification, containment,
   quiescence, full rank records and independent verdict.
3. Rebind trn22's already-passed G1 evaluator calibration as a source/tool
   identity check only; it is **not** an A0 control or a table result. Confirm
   the human-provenance disk-breaker re-arm remains valid and
   `research/agentic/STOP` is absent. A changed Boolean alone is insufficient.
4. On trn22, fresh read-only preflight verifies expected hostname, one clean
   pinned commit and exact source/prepare/package/observer/guard hashes;
   complete effective environment; six retained warm-cache identities;
   absent label, checkout, output, log, STEP_LOG and observer receipt paths;
   no `torchrun`, training/eval, tagged process, shared-lock holder or
   `/dev/neuron*` holder; and no second active launcher.
5. Root free space must exceed **both** a provisional 20 GiB prelaunch floor
   and the independently reviewed A2-specific projection of compile/cache,
   temporary, smoke checkpoint/log and unknown-growth reserve above the
   immutable 6.5 GiB operational floor (6,979,321,856 B). The old A0 10 GiB
   threshold and the estimated 1.5 GiB of replicated table tensors are not
   A2 disk/HBM bounds. Bind a separate peak-HBM projection/abort limit.
   Monitor actual root free bytes through compile and training; preemptively
   stop only the authenticated owned job if it approaches the floor. A single
   below-floor sample, OOM or unbounded growth fails the gate.
6. Materialize a **new immutable smoke request** that hashes the complete
   `env -i` command, source, stage, cache/observer paths, guard and disk/HBM
   projection. The Supervisor, not this Implementer, issues a separate
   one-attempt authorization after an independent review of the finalized
   request. Only an Operator may then dispatch.

The original [A2 runbook](RUNBOOK.md) permits this non-scored native smoke
before the same-host A0 finishes, but **a full scored A2 may not run** until
there is a valid archived trn22 c5c A0 full-budget/seed-42 public20M control,
fresh A2 preflight/projection and a second one-shot full-run authorization.
The zero-step A0 `r1` failure cannot fill that control field.

## Observer, containment and native acceptance

Use the already-reviewed read-only
`research/a0_boundary_observer.py` (SHA-256
`a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`)
with the **same** label-specific guard-process JSON, STEP_LOG and postflight
JSON from the proven legacy guard. Stage the observer outside the pinned
training checkout. Start it before the guard; prove it is alive and the
STEP_LOG is absent before dispatch. The observer's `.steps.jsonl` file is
actually one fsynced decimal integer per completed optimizer boundary,
not JSONL. The exact path must be in both the guarded `env -i` command and
the observer request. Bind the guard script's own staged hash; it is not
present in this controller checkout. If the guard cannot support an exact
30-step smoke, or its process/postflight contract differs, stop and seek
independent review—do not substitute an ad-hoc launcher or two launchers.

Successful process exit is only the first check. Require all of the following
before an independent Verifier can call the **native smoke gate** PASS:

- The observer saw the live guard/child before the first boundary, then
  exactly the sequence `0,1,…,29` with no duplicate, gap, rewrite, late
  start or partial line. Its first five readings establish startup identity;
  its word `scored_timing_proof` is **not** a scored-BPB verdict for a smoke.
- The guard postflight is clean, the child returned zero, the four Neuron
  workers are gone, `/dev/neuron*` holders are absent, output/log/checkpoint
  files are stable and hash-verified, and containment/quiescence is recorded.
- `train.py` reports world size 4, LNC=2, 30 steps, the expected four BF16
  n-gram tables and RMSProp profile, finite losses and checkpoint weights,
  no NaN, crash, budget violation or scorer-path modification. Record
  compile/startup and peak host-disk/HBM measurements. Native NKI/compiled
  paths must be those of the exact c5c source.
- All four `SUBMISSION_OPTIMIZER_INTEGRATION` rank records must have the
  same expected owner plan (modulo rank), with RMSProp tables replicated,
  not owner-sharded. The G2 rank trace proves CPU update math; it does not
  prove native rank equality. If normal native logs do not provide finite
  per-rank gradients/parameters and table/moment hashes after updates,
  leave this equality item **HOLD** and commission a separately reviewed
  exact-source, new-label diagnostic. Rank-0 logs and a single rank-0
  checkpoint cannot prove four-rank equality. `--debug-numerics-steps`
  may expose per-rank finiteness but does not alone prove table equality.
- Test the diagnostic checkpoint via unchanged `load_for_eval` in a fresh
  process with strict model-only loading, synthetic tokens only: logits
  finite, causal, deterministic and batch-independent. Do not modify the
  scorer or use public/private validation data in the smoke gate.

The original queue's IMPL-1 acceptance mentions a **50-step** Neuron smoke
with equal per-rank table hashes, while the newer A2 package and master
runbook specify **30 steps**. This package implements only the current
30-step smoke. Before promoting A2 to a full experiment, the Supervisor and
independent Verifier must either obtain the outstanding 50-step proof under
a separate unique label or record a legitimate newer human-approved
supersession. Never silently call 30 steps a 50-step pass.

Archive every failure, including an empty boundary log. A crash, NaN,
missing checkpoint, failed equality/correctness proof, disk-floor breach,
bad/dirty checkout, occupied host, budget or quiescence failure is a failed
gate—not a promising A2 result. This Implementer made no SSH/AWS call,
G2/native run, staging, training, eval, breaker change, deletion or
submission.
