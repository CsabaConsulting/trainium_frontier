# Independent Verifier receipt: pending c5c A0 trn21 control r1

Reviewed 2026-09-24 22:15 UTC. Reviewed package commit:
`4c41de520531c7828bcc105b27cbc3f1a0e524db`. The reviewer did not
prepare the package or operate either host.

**Verdict: STATIC PACKAGE PASS; DISPATCH HOLD.** The three documents form a
coherent conditional A0 control package. This receipt is not a completed
host preflight, G1 or G2 pass, final run request, human volume receipt,
Supervisor ratification, one-shot authorization, training result, or BPB
result. No trn21 c5c public20M control exists yet.

## Independent checks

- Read root `AGENTS.md`, then `README.md`,
  `research/agentic/CURRENT_STATE.md`, `research/experiment-protocol.md`,
  and `research/agentic/DECISION_LOG.md` in the required order before the
  package and cited evidence. Reviewed the current human controller brief,
  standing authority, workflow role contract, trn21 G1 pending package,
  trn22 r2 package, disk packet, state, G2 receipt, incident receipt, and
  archived trn21 35c9 control evidence.
- `git diff 4c41de52^ 4c41de52` adds only the proposal, gate contract, and
  runbook under
  `research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/`.
  There is no model, launcher, `prepare.py`, tokenizer, data, evaluator,
  scoring, or decision-engine edit. Both JSON files parse, `git diff --check`
  passes, and a sensitive-pattern scan of the three new files found no key,
  credential, or token assignment. The reviewed file SHA-256 values are:

  | File | SHA-256 |
  | --- | --- |
  | `EXPERIMENT_PROPOSAL.json` | `55c6d500fbc535a45ad3afb6ef32882e30726b8ec90ac412e2ba987b5c71c1b9` |
  | `GATE_CONTRACT.json` | `8d6a6a5cc03f30c05de80f5581cfe628a727d7b0be62e91b18cbed09c3d8dbdf` |
  | `RUNBOOK.md` | `984839b1460741ea1f2affbcd3d35bce7321fda149fc8e0c51973e796f41ff09` |

- The local `submissions/research_v2/train.py` SHA-256 is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The normalized immutable science fields in the trn21 contract and the
  pending trn22 r2 contract have the same SHA-256 fingerprint,
  `e1b50dfbc45b5e2c732e5cce516a882a5b07c55879ae8df06d495d21f852771d`:
  source and evaluator hashes, full-chip LNC2/four-rank launcher, 3,600-second
  outer guard, complete Track-A argv, and four named effective settings.
  The source parses `NEURON_COMPETITION_R1_SEED=42` and
  `NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800` as defaults, and passes the
  seed to `init_runtime`. The training argv adds neither `--num-steps` nor
  `--seed`/`--max-train-seconds`; clean `env -i` materialization must preserve
  this exact effective state and exclude inherited overrides. Compiler flags
  include `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
- The archived trn21 35c9 control at evidence commit `b0b23cbc` used source
  commit `c51f6fdc5ca77793a1524148a563cafac0ab22b0` and SHA-256
  `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`.
  Its retrieved result records 3,636 steps, seed 42, public2M BPB
  `0.9866973734526626`, and **no public20M score**. Its training log records
  step-0/1 losses `9.0112`/`8.7822`; those are numerical anchors, not a
  same-source c5c control. Its preserved checkpoint bytes/hash, and the
  separate retained 116188 confirmation checkpoint bytes/hash, match the
  disk packet. Their present host identities remain unverified. The trn22
  c5c r1 terminal receipt reports zero optimizer boundaries, no checkpoint,
  and no BPB; it cannot stand in for a trn21 control. The old trn22 CPU G2
  comparison found 35c9 and default-off c5c identical, but does not pass
  the new trn21 exact-environment G2 gate.
- Recomputed the provisional disk formula: `6,979,321,856` floor +
  `max(1,846,603,776 trn21 pre/post decline, 1,854,160,896 other-host
  dense footprint, 1,500,942,336 trn22 pre-step compile growth +
  469,831,009 checkpoint example)` + `1,073,741,824` reserve =
  `10,023,837,025` bytes. The required 10 GiB (`10,737,418,240` bytes)
  exceeds that provisional sum by `713,581,215` bytes. The calculation
  does not establish a maximum c5c footprint; continued disk monitoring
  and the unchanged 6.5-GiB floor are essential.
- Searched tracked local records and the active checkout's `runs/` records
  for the exact `fs-20260925-v2-c5c-a0-control-trn21-r1` label. Only these
  three package files matched; no earlier local result, dispatch, or ledger
  record matched. Live trn21 checkout, output, step, and log path absence
  has **not** been checked. The package properly leaves that proof to the
  Operator and treats a later attempted launch as consuming the label.
- The proposal distinguishes a public2M screen, a global numerical
  equivalence band `[0.99095, 0.99295]`, and a future fresh context-1024
  public20M result. It does not call an A0 baseline a quality win or grant
  A1/A2, adoption, or submission authority. `dispatch_authorized=false`,
  `final_run_request_exists=false`, and the proposal's gate booleans remain
  false.

## Dispatch HOLD and required follow-up

The latest archived read-only disk audit is **22:05:10 UTC**, with trn21 at
`6,901,743,616` root-free bytes on a 200-GiB root disk, below the
`6,979,321,856`-byte operational floor. The contract correctly timestamps
its older `6,901,788,672`-byte 21:50:53 observation; the runbook's phrase
“latest archived trn21 inventory” for that older value is stale. Correct
that historical wording in the next package/final-request revision. It
does not change this HOLD or the mandatory fresh post-resize measurement.

Before any separate run authorization, require all of the following:

1. Human-controlled expansion of the existing trn21 root volume and guest
   filesystem, with fresh read-only device/filesystem evidence and at least
   10 GiB free **after staging and immediately before dispatch**. No cleanup,
   protected-checkpoint prune, threshold reduction, or agent breaker reset
   follows from this review.
2. A completed trn21-specific fresh public20M G1 calibration on the retained
   116188 checkpoint, retrieved Operator evidence, and an independent PASS.
   The trn22 G1 pass and the pending trn21 G1 package do not satisfy it.
3. Supervisor ratification of this conditional proposal; a new, clean,
   pinned trn21 checkout with byte-identical c5c root `train.py`, unchanged
   `prepare.py` and launcher hashes; proof of live label/path absence; and
   a newly materialized immutable request with the exact complete `env -i`
   command, output and step paths, and `dispatch_authorized=false`.
4. A new four-rank G2 PASS and declared numerical comparison bound to that
   checkout, argv, and exact effective environment, with independent review.
   Prior trn22 G2 and archived trn21 35c9 training are supporting evidence
   only.
5. Fresh authentication of the historical trn21 warm-cache root and all six
   children without substitution or cleanup; both retained checkpoint
   identities; hostname, host and launcher cleanliness; shared lock;
   stop sentinel; no train/eval/G2/guard process or Neuron holder; and all
   host/global breakers applicable to the selected launcher. The archived
   v2 trn21 breaker is untripped, but that is not a live preflight. The
   preserved inactive legacy-controller breaker must not be silently reset.
6. A read-only external optimizer-boundary observer alive before guard
   launch, a fresh Operator preflight, and a separate Supervisor one-attempt
   authorization binding the final request hash and every required receipt.
   The new `run_job.py` remains pilot-only; no launcher switch is approved.

If a run is later authorized, its own postflight must prove five sequential
external boundaries, budget compliance, finite model-only checkpoint,
quiescence, no disk-floor breach, causality, and a separately authorized fresh
public20M evaluation before the label becomes a valid A0 control. No SSH,
AWS action, guest resize, training, evaluation, deletion, breaker reset,
dispatch, or submission was performed in this review.
