# Independent r2 prelaunch review: two 120-second compile smokes

**Proposal/assignment design PASS for both exact corrected packets at controller
`e69f532e`; dispatch remains HOLD until Supervisor's one-use enablement and
each Operator's fresh host preflight.** This supersedes only the step-evidence
HOLD in independent receipt `8d688b03`. I did not propose, author, SSH,
launch, or approve a submission.

## Shared repaired gate and evidence

- The Researcher proposals at `f8897647` now explicitly permit one job-owned
  `NEURON_COMPETITION_R1_STEP_LOG` exception and define proof as ≥20 distinct
  completed **rank-0** boundaries plus all four workers cleanly completing
  synchronized per-step collectives and the final post-checkpoint
  `dist.barrier()`. The trusted `prepare.py` writes boundaries for rank 0
  only; no direct per-rank boundary file is claimed. The proposal accepts
  either four worker exit statuses or bounded torchrun-all-workers exit proof.
  The assignments at `e69f532e` now pin the **same** proposal paths, under
  each unique run root's private `receipts/step-boundaries-rank0.log`, and
  require exclusive empty precreation and raw log preservation. No other
  `NEURON_COMPETITION_R1_*` training override is permitted. Missing 20
  boundaries after clean exit is INCONCLUSIVE, never PASS.
- Exact S0 and 116188 source SHA-256s remain respectively
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`
  and `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`;
  trusted `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The real-prepare four-rank base CPU G2 receipts remain PASS. Both files'
  argparse accepts `--max-train-seconds 120`; the ordered training commands
  differ from their G2/historical source profiles solely by that appended
  budget flag. The packets explicitly invoke the latest human break-glass
  base-G2-plus-static-CLI-delta allowance and do **not** call an old receipt
  an exact-120s G2. No source/model/evaluator edit is authorized.
- Both retain distinct host, labels, roots, full-chip LNC2×4, seed 42,
  trusted `prepare.py` via SHA-checked `PYTHONPATH` with only `train.py`
  copied, no concurrent launcher, ≥6.5 GiB disk, data/tokenizer/STOP checks,
  and A0 **training plus fresh public-20M eval terminal** before each slot.
  trn21's populated warm paths require an actual hit/path observation;
  trn22's seven private cold paths require prelaunch emptiness and new
  compiler files under the pinned `NEURON_COMPILE_CACHE_URL`. These are
  future Operator checks, not claims about live host state now.
- The earlier 1800-second S0 warm and 116188 cold assignments remain
  `SUPERSEDED_UNLAUNCHED_BY_120_SECOND_COMPILE_SMOKE_INSTRUCTION`, both with
  `launch_authorized:false`. No BPB evaluation, quality/adoption verdict, or
  submission clearance is permitted from either short smoke. A clean short
  result still does not replace S0's own full-budget proven-cold replay.

## Separate exact dispositions

1. **trn21 S0 warm compile smoke: PASS** for proposal
   `20260925-s0-warm-compile-smoke-trn21-r1.json` (SHA-256
   `62565e4d3cec7663175892178b2dfe14c23d845d3ba0d2bd5ca14d1820fabcbc`)
   and prepared assignment
   `20260925-trn21-s0-warm-compile-smoke-r1.json` (SHA-256
   `875fbb774e725f5b54c9f6cefaa944cffd1791d9492f4297a8649e490c4c4d82`).
   Warm-cache attribution is inconclusive absent an observed hit, even if
   S0 reaches native steps.
2. **trn22 116188 cold compile smoke: PASS** for proposal
   `20260925-116188-cold-compile-smoke-trn22-r1.json` (SHA-256
   `5b8abb40c2584a2116207c3c850339c8987287ccda06636354dda9857f8404cf`)
   and prepared assignment
   `20260925-trn22-116188-cold-compile-smoke-r1.json` (SHA-256
   `8e2bc57604902e5eca183c1895e9ebc78b398a9b29747f9456ad540cba992560`).
   True-cold attribution is inconclusive absent effective persistent-URL
   file proof, even if 116188 reaches native steps.

For **each exact prepared assignment independently**, the Supervisor may
change only `assignment_status` from
`PREPARED_PENDING_PROPOSAL_AND_INDEPENDENT_REVIEW` to `ASSIGNED` and
`launch_authorized` from `false` to `true`. The unchanged proposal, file
SHA, launch argv, cache and step-log bindings then go to a distinct Operator.
The Operator must refuse on any failed live preflight, consumed label,
unexpected source/import/env, occupied host, STOP or insufficient disk; no
failed label retry or automatic submission follows this review.
