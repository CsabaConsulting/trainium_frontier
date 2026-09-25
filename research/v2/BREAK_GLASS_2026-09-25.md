# Human-authorized break-glass execution mode — 2026-09-25

Source: Poojith's direct message on 2026-09-25 IST. Effective now until the
new staging tool passes one real run. This later human instruction supersedes
older per-run migration and adoption-acknowledgement requirements where they
conflict, but it does not authorize AWS lifecycle changes, protected cleanup,
secret disclosure, evaluator/data changes, or leaderboard submissions.

## Timebox and route

- If the new staging tool has not produced one successful launch by **20:30
  IST today (15:00 UTC)**, stop debugging it as a blocker and use the proven
  Sep 24 manual LR 0.015 path. Continue tool debugging separately, without
  leaving hardware idle for it.
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

1. trn21: S0 cold replay, bare launch, fresh compile cache (submission gate).
2. trn22: research-v2 dense base control with matrix LR 0.015.
3. trn21: A1 last-layer tables on research-v2.
4. trn22: A2 layers 1,-1 on research-v2.
5. Cross-host replicates, then D1 epoch-value in the next free slot.

Report to Poojith at first launch, each completed run (steps, public-20M BPB,
delta versus comparable control), and any failure. Research-base adoption is
pre-approved **only** when unchanged `workflow/tools/decide.py` thresholds
pass; leaderboard submissions remain human-only.

This mode does not assert that any run, source transfer, or score has occurred.
