# Human-authorized break-glass execution mode — 2026-09-25

Source: Poojith's direct message on 2026-09-25 IST. Effective now until the
new staging tool passes one real run. This later human instruction supersedes
older per-run migration and adoption-acknowledgement requirements where they
conflict, but it does not authorize AWS lifecycle changes, protected cleanup,
secret disclosure, evaluator/data changes, or leaderboard submissions.

## Latest human clarification (2026-09-25, before tonight's launch)

- For the **S0 cold replay** and the **research-v2 base control**, Poojith
  explicitly waived independent **pre-launch** review. One Operator performs
  exact-file transfer, host SHA-256 verification, disk/Neuron preflight,
  launch and fresh-process evaluation. A different reviewer checks the
  receipts **after launch and before using either result**. This does not
  waive an exact-byte CPU G2 PASS where one exists, the completed proposal,
  or the role split from the Implementer.
- Both hosts must be **training by 20:00 IST**. If either is not, stop other
  work and report the exact failed command and actual error (secrets redacted).
- The cross-process mount-identity comparison in the new staging tool may be
  relaxed or skipped; it is not a gate for the plain transfer manual path.
  Still check the intended destination is new and the host/source SHA is exact.
- Code lanes continue only insofar as they do not block either launch.

## Timebox and route

- The earlier 20:30-IST staging timebox is superseded by the 20:00-IST
  both-host launch deadline. Use the proven Sep 24 manual LR 0.015 path now;
  continue staging-tool debugging separately, without blocking hardware.
- Prepare the manual path now; tonight's jobs start as soon as their minimum
  gates pass. Do not silently retry a failed unique run label.

## Minimum gates for every break-glass training run

1. Copy the **exact** intended `train.py` to a separate host run location by
   plain `scp` or `rsync`; verify its SHA-256 **on the host** before launch.
2. Have a CPU G2 PASS receipt for the exact bytes and argv wherever one exists;
   a missing receipt is stated explicitly, not invented.
3. Fresh host check: at least 6.5 GiB disk free, no other Neuron job, one job
   per host. Preserve existing worktrees/artifacts; no cleanup is implied.
4. After training, score the checkpoint using a **fresh-process public-20M**
   eval (context 1024) and append a complete result or failure to the v2 ledger.

Containment receipts and the full `run_job.py` pipeline are optional in this
mode. Keep a unique label, exact command, source hash, host, log, checkpoint
status, and result in the repository. Do not call a failed or unscored run a
win. Report failures with their actual error text, redacting only secrets.

## Job order

The first S0 cold replay (`fs-20260925-s0-cold-replay-trn21-r1`)
launched on trn21 but failed before step 1 during Neuron compilation. Its
label is consumed, and its logs remain protected evidence. The trn22 c5c A0
control launched; trn21 c5c A0 launched after the S0 failure.

The newer direct instruction from Poojith keeps S0 as the submission gate:

1. Read the detailed `neuronx-cc` error for the failed S0 module, or record
   that no compiler log exists in the failing debug directory and identify
   the first actual error in the available logs. Do not infer a code or cache
   cause from a missing `.neff` alone.
2. Immediately after trn21 A0 finishes, test S0 under a **new unique label**
   with the normal warm research cache and bare launch. This separates
   possible S0-code failure from the cold-start condition.
3. On the next compatible host slot, cold-test the known-good `116188` file
   with an explicitly created, fresh empty cache directory and enough disk.
   Its exact source, flagged launch, proposal, and unique label must be bound
   before dispatch. If it also fails cold, investigate the setup rather than
   blaming S0 code.
4. Only after those diagnostics and applicable native gates, resume A1/A2;
   keep their preparation moving in parallel so hardware does not idle.

S0 is **not submittable** until it completes a clean cold-cache full run,
fresh-process public-20M evaluation, and independent post-launch review.
No failed label may be retried or erased.

Report to Poojith at first launch, each completed run (steps, public-20M BPB,
delta versus comparable control), and any failure. Research-base adoption is
pre-approved **only** when unchanged `workflow/tools/decide.py` thresholds
pass; leaderboard submissions remain human-only.

This mode does not assert that any run, source transfer, or score has occurred.
