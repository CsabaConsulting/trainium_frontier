# Poojith's resume, table-budget and builder correction

Received 2026-09-25 15:41 UTC (21:11 IST). This supplements FINAL PUSH and
overrides its conflicting model routing and queue order. The preceding
read-only restriction applied only to that status report; development resumes.

## Exact human instructions

0. "Read-only" was only for that status report. Resume all work now.

1. A1/A2 are table arms: they inherently cost ~10% of steps. Judge them on final
   public-20M at the fixed 1800 s budget; mark them throughput-exempt so the
   step window can't CONFOUND them.
2. Report TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS from prepare.py, and explain why
   A1/A2 startup is 990-1060 s even with the "warm" cache. If startup can exceed
   the cap, the excess is charged to training on the official cold run. Fix
   the cache reuse, and measure the table files' cold startup before any table
   submission.
3. Builder lanes: stop using Fable 5.1 (refuses; do not reword prompts). Run
   L1 FP8 and L2 8x1024 as local Codex implementer subagents (GPT-6 Astra), same
   deliverables. Bedrock only with a non-Fable model, and only if we need more
   capacity.
4. Recursive schedule patch: fix the whitespace, finish review, queue it as a
   high-priority arm tonight.
5. Overnight queue, conditional:
   - If A1 or A2 improves by <= -0.003: next jobs are its cross-host replicate,
     then A3 (tables on all VE layers), then the Recursive schedule ON the
     table base, then LR 0.0125/0.0175 on the table base. Flag-bank runs on the
     old base are cancelled.
   - Otherwise: the Recursive schedule and the architecture arm first, the
     flag bank last. Install the queue so both hosts continue automatically
     after A1/A2.

## Execution semantics

- Keep `kind: table_arm`; annotate A1/A2 and their replicas with explicit
  `throughput_exempt: true` and `training_budget_seconds: 1800`. This changes
  only their step-window gate, not BPB thresholds, budget validation, fresh
  20M/context-1024 evaluation, correctness gates or the big-win audit hold.
- Use valid final same-host deltas, not training loss, to select the branch.
  An incomplete, failed or five-step startup-only attempt is not a scientific
  negative and cannot justify a claimed table quality verdict.
- Conditional jobs and source patches awaiting gates are not falsely counted
  as ready. Existing live A1/A2 jobs are not interrupted or modified.
- L1/L2 are local GPT-6 Astra implementation tasks. No further Fable call or
  disguised retry is authorized. Earlier Fable sessions are terminal failures.
- `prepare.py` remains untouched. Its current constant is **900 seconds**;
  the separate 30-second launch margin in train.py is not an extra allowance.
- Submission remains human-only. A warm research result never substitutes
  for the required cold-start timing and replay of the exact submission file.
