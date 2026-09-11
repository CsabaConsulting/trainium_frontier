# Why the 2026-09-10 submission scored 1.1495 instead of ~1.03

Research pass over the notebook, run logs, experiment log, git history, upstream
competition repo and the Devpost forum, done 2026-09-10 on the local box (no Trn2
instance available). Nothing here was run on hardware; every claim is traceable to a
file, a log line or a commit hash quoted below.

## 0. The answer in one paragraph

The submitted `train.py` (commit `bf1a3ae`, byte-identical at `ba8568d` and at HEAD,
md5 `230055978f8d5eaa4e62a5c3d893e780`) still carries the starter kit's
`NUM_STEPS = 1000` default, and `--num-steps` inherits it. The official launch command
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` passes no
flags, so the official run stopped at optimizer step 1000, about 815 seconds into the
1800-second budget. Because the LR schedule is driven by wall-clock progress and not by
step count, the run ended with the LR multiplier still at ~0.74 and no cooldown at all.
Every local "full-budget" run that produced the 1.034 numbers was launched through
`research/run_experiment.py`, which *requires* `--num-steps` and always passed
`--num-steps 5000 --max-train-seconds 1800`, so the default was never exercised locally.
The cap was harmless for the first three submissions because their throughput was
500 to 750 steps per 1800 s; it became binding the moment `USE_TENSOR_LR_SCALARS`
(2026-09-07) pushed throughput past 1000 steps. A 1000-step, no-cooldown run of the
current config is expected to land at roughly 1.12 to 1.16 bpb; 1.1495 sits inside that
range. No hardware, compiler, cache, pack-factor, or optlevel difference is needed to
explain the gap, and the hedge submissions discussed on 2026-09-10 (optlevel=2,
PACK_FACTOR=2) would not have fixed it.

## 1. Evidence chain

### 1.1 The default and the launch command

- `train.py:191` — `NUM_STEPS = 1000`. `git log -S'NUM_STEPS = ' -- train.py` shows it
  has never been touched since the repo was cloned from the starter kit on 2026-09-03
  (`73a6753`). The upstream baseline has the same value at line 55.
- `train.py:4071` — `parser.add_argument("--num-steps", type=int,
  default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))`.
- `train.py:4072` — `--max-train-seconds` defaults to `MAX_TRAIN_SECONDS`, which equals
  `TRAIN_TIME_BUDGET_SECONDS` (1800). So the *time* default is right; only the *step*
  default is wrong.
- `train.py:4559-4562` — `stop_after_step = (step + 1 >= args.num_steps or (...time
  checks...))`. The step cap is checked first and is unconditional. When it fires the
  loop breaks (`train.py:4606`), saves `final.pt`, runs the in-process public eval, and
  exits normally. From the runner's point of view this is a clean, in-budget, scorable
  run — which is exactly what happened (a score was produced, no crash status).
- Submission launch command (README "Submission Format", plan §12.1): no `--num-steps`.

### 1.2 The schedule is wall-clock based, so a step-capped run never cools down

- `train.py:4008-4020` — `lr_multiplier(step, elapsed, max_train_seconds)`:
  `progress = elapsed / max_train_seconds`; flat until `progress < 1 - WARMDOWN_RATIO`
  (0.25), then linear to `FINAL_LR_FRAC` (0.05) at `progress = 1.0`.
- `train.py:4519` — `wd_progress` and the Muon momentum cooldown use the same
  wall-clock fraction.
- Local full-budget log `research/run-logs/t04-stack-fullbudget.log`:
  `step 01000 | loss 3.0760 | lrm 0.744 | dt 0.82s | ... | remaining_time 985s`.
  Steady-state step time is 0.81 to 0.82 s, so step 1000 falls at ~815 s of charged time,
  progress ≈ 0.45, `lrm ≈ 0.74`. The remaining ~985 s of budget, the entire useful
  cooldown, and the Muon momentum cooldown were simply never executed on the runner.

### 1.3 No local run ever exercised the bare command

- `research/run_experiment.py:94-95` — `--num-steps` and `--max-train-seconds` are
  `required=True`; lines 111-112 always append them to the launch command.
- `research/experiment-log.jsonl` (242 rows): the only row whose `launch_command` lacks
  `--num-steps` is `mudd-compileonly` (a `--compile-only` smoke, no training). Every
  `*-fullbudget` row, including the four quoted in plan §12.4
  (`t04-stack-fullbudget`, `-v2`, `t04-stack-defaults-fullbudget`,
  `cold-cache-fullbudget`), was launched as
  `... train.py --num-steps 5000 --max-train-seconds 1800 --out-dir out/...`.
- Every one of those logs begins with `training target=5000 steps max_seconds=1800`.
  The official `train.log`, if the organizers share it, will begin with
  `training target=1000 steps max_seconds=1800 official_budget=1800s` and end with
  `num_steps: 1000` and `training_seconds:` around 815 to 830. That single line is the
  decisive confirmation.

### 1.4 The plan's "exact submitted command was tested" statement is not accurate

- `frontier-research-plan-2026-09-09.md` §12.1 says "no flags are needed" and §12.4 says
  the cold-cache diagnostic ran "the exact submitted command". The cold-cache row's
  logged command is `... --num-steps 5000 --max-train-seconds 1800 --out-dir
  out/cold-cache-fullbudget`. It refuted the cache hypothesis correctly, but it did not
  test the bare command. Same for the "defaults-flip verification" in `7a68267`.
- Consequence: §12.4's four "untested hypotheses" (per-step recompiles on the official
  SDK, `--optlevel=1`, `PACK_FACTOR=4`, runner crash/retry) were all ranked against a
  false premise. None of them needs to be true.

### 1.5 The trap was already known on 2026-09-05 and was worked around, not fixed

`trn2-experiment-results.md:84` (Sep 5): "`--num-steps 3000` deliberately set well above
the file's `NUM_STEPS=1000` default — at the 8-way throughput observed ... the *default*
1000-step cap would stop the run at ~1400s ... the current step-count default was sized
against GPU-proxy assumptions". The fix chosen was to pass the flag in every experiment
rather than to change the default in the file that gets submitted. From then on the
harness hid the default from every run and every reviewer.

### 1.6 Why the three earlier submissions did not show it

| submission | date (PT) | commit | config | local steps / bpb | board |
|---|---|---|---|---|---|
| 1 | Sep 5/6 | `bc28127`-era | LNC=1, 8 ranks, SEQ_LEN 1024, eager | 500 / 1.176151 | 1.176 |
| 2 | Sep 6 08:51 | `6c69869` | LNC=2, 4 ranks, SEQ_LEN 1024, eager, causal fastpath | 639 / 1.153610 | 1.1394 (per notebook line 1636) |
| 3 | Sep 6 22:12 | `28a9104` | LNC=2, SEQ_LEN 2048, eager (`--compile` default OFF at that commit) | 746 / 1.122271 | see §3.2 |
| 4 | Sep 10 | `bf1a3ae`/`ba8568d` | compile ON, tensor-lr scalars, NKI Muon, T0.1, T0.4 stack, PF=4, optlevel=1 | 2199-2205 / 1.034 | **1.1495** |

All three earlier submissions trained fewer than 1000 steps in 1800 s, so the cap never
bound and their board scores tracked local numbers to within ~0.015. Throughput crossed
1000 steps/1800 s at `a118ddf` (Sep 7, `USE_TENSOR_LR_SCALARS`: 847 → 1995 steps). From
that commit on, every submission of the file without `--num-steps` was guaranteed to be
capped. Submission 4 was the first one after that point.

### 1.7 Expected score of a 1000-step, no-cooldown run of this config

Direct calibration is not available (no local run has ever stopped at high LR with this
architecture), so two independent estimates:

- **Loss-to-bpb mapping.** Full run: final smoothed train loss ≈ 2.87 ↔ 1.034 bpb.
  900 s run with a complete cooldown (`t04-stack-900s.log`): 1097 steps, final smoothed
  loss ≈ 2.98 ↔ 1.0812. Slope ≈ 0.43 bpb per nat. The full run's smoothed loss over steps
  950 to 1040 (the no-cooldown regime at `lrm` 0.72 to 0.77) is 3.02 to 3.17, mean ≈ 3.11.
  Predicted bpb ≈ 1.0812 + 0.43 × (3.11 − 2.98) ≈ **1.135**, range 1.10 to 1.16 across the
  loss noise.
- **Analogues in the log.** `xf-fixed480` (480 steps stopped at `lrm 0.96`, no cooldown):
  1.2056, versus ~1.17 to 1.18 for a fully cooled run of similar step count, a
  ~0.03 no-cooldown penalty at 480 steps. `rwkv-fixed995` (995 steps stopped at
  `lrm 0.34`, partial cooldown): 1.144. Both point to a 0.03 to 0.06 penalty on top of the
  "1000 steps, cooled" level of ~1.09 to 1.10, i.e. 1.12 to 1.16.

1.1495 is consistent with both. It is also consistent with the notebook's own
observation that 1.1495 "corresponds to ~490-660 charged steps" — that estimate assumed a
completed schedule; a 1000-step run with no cooldown is worth about the same.

## 2. What changed since the 9/6 submissions, re-assessed against the official runner

All 35 `train.py` commits between `28a9104` and `ba8568d` were reviewed for anything that
behaves differently on the runner than locally. With the step cap identified, none of
them is needed to explain the gap, but two deserve a note because they change what the
runner sees and were never tested through the runner's own code path:

1. **`load_for_eval` widening (`12f7658`, Sep 9, `EVAL_MIN_SEQ_LEN = 8192`,
   `train.py:3973-4002`).** This is the only code the official scorer executes that the
   in-process eval does *not* execute (the in-process eval uses the live training model
   at `SEQ_LEN`). It only resizes non-persistent RoPE/mask buffers, it was verified on
   hardware via `prepare.py eval-public` on Day 1 (C1 table, notebook lines 1546-1588),
   and RoPE frequencies do not depend on `sequence_len` (`ROPE_BASE` is a constant,
   `train.py:313`). Judged safe; noted because it is eval-only.
2. **`os.environ.setdefault("NEURON_CC_FLAGS", "--optlevel=1")` (`783f7ef`) and
   `NEURON_COMPILE_CACHE_DIR=/tmp/neuron_cache` (`train.py:40, 70`).** `setdefault` means
   a runner that pre-sets either variable silently overrides them. Worst case is losing
   the +1.8% optlevel win, not a 0.115 regression.

Everything else (torch.compile default ON, tensor-lr scalars, foreach optimizer, NKI Muon,
GRAD_CLIP, local conv, PACK_FACTOR=4, T0.4 stack, Muon momentum cooldown as a device
tensor) is exercised identically by the local 5000-step runs and would show up locally
first.

## 3. Secondary findings that affect how the next score should be read

### 3.1 Local numbers are optimistic by a known, quantified amount

From the C1 calibration (notebook lines 1544-1650, checkpoint `nkimuon-fullbudget`):

| eval setting | bpb |
|---|---|
| in-process default: 2M tokens, seq_len 2048 | 1.041139 |
| 20M tokens, seq_len 2048 | 1.047077 |
| full 63.4M-token shard, seq_len 2048 | 1.047358 |
| 20M tokens, seq_len 1024 | 1.063264 |

So the quoted 1.034 is ~1.040 at 20M tokens and ~1.056 if the scorer uses
`prepare.py eval-public`'s default `--seq-len 1024` (`prepare.py:862`). The scorer's
context length is still unpublished. Historical board-vs-local offset for a SEQ_LEN 1024
model was −0.014 (board easier). A correctly capped resubmission of the current file should
therefore be expected in the **1.03 to 1.06** band, not exactly 1.034.

### 3.2 The identity of the 1.1394 score is ambiguous and worth resolving

The notebook (line 1636) attributes 1.1394 to the SEQ_LEN 1024 config (submission 2,
local 1.15361). The laptop session on 2026-09-10 dated the 1.1394 submission to Sep 6
22:12 PT, which is after `28a9104` (SEQ_LEN 2048, local 1.122271, submission 3). Both
cannot be right. The per-submission history on the leaderboard settles it, and the answer
calibrates the open eval-context question:

- If submission 3 (SEQ_LEN 2048, local 1.1223) scored ~1.13 to 1.14, the board is
  ~0.015 *worse* than local for a 2048-trained model — matching the +0.016 measured for
  evaluating at 1024 instead of 2048, i.e. evidence the scorer uses 1024.
- If submission 2 scored 1.1394 and submission 3 scored lower, the scorer likely
  evaluates at 2048 or longer.

### 3.3 Other protocol gaps that let this through

- The submission checklist never included a dress rehearsal with the *exact* launch
  string. `run_experiment.py` cannot even produce one (it requires `--num-steps`).
- `python prepare.py check --train-py train.py --launch-command "..."` was not part of
  the pre-submission routine. It would not have caught this (it does not inspect
  `--num-steps`), but it is the organizers' own gate and is free.
- Nobody read the first line of a local log (`training target=5000 steps`) against the
  submission command. The run header already prints the binding controls.
- Two agents (laptop and this box) both reasoned from plan §12.1's "no flags are
  needed" without checking the argparse defaults it summarized.

### 3.4 Forum items checked (nothing there explains the gap)

- `45164` "Submission Issue Resolution (Sep 2-3) & Runner Update Changelog" (organizer,
  1 day ago): a newer `prepare.py` enforcing `record_step_boundary` was deployed on Sep 2
  before PR #6 documented it; affected submissions do not count against quota. Organizers
  will now post dated announcements whenever runner artifacts change. No runner change is
  listed between Sep 6 and Sep 10.
- `45135` "Baseline compile hangs before neuronx-cc starts" (participant, 4 days ago):
  fresh-graph compiles hung 1200 s on a dev instance; fixed by clearing
  `/tmp/hlo_cache`, `/tmp/local_cache`, `/tmp/neff_cache`. Dev-instance cache
  corruption, not a runner property. Worth knowing because a 1200 s stall would exceed
  the 900 s startup allowance and be charged.
- `45108` "Submission platform": `USER_TRAINING_CRASHED (exit=1, status=no_run_meta)` for
  another team; unrelated (our run produced a score).
- Upstream `aws-neuron/trainium-frontier`: no commits after `c907da4` (Sep 3, PR #6).
  Local `prepare.py` is byte-identical to upstream (md5 `c4878383...`).

## 4. What to do (recommendations, not yet applied)

1. **Make the time budget the only binding control in the submitted file.** Change
   `NUM_STEPS` at `train.py:191` to a value that cannot bind (e.g. `10**9`), or drop the
   step cap from `stop_after_step` for the default path. A launch-command workaround
   (`... train.py --num-steps 100000`) is allowed by the checker but leaves the trap in
   the file for the next person.
2. **Make the schedule robust to being step-capped anyway.** Drive `progress` by
   `max(elapsed / max_train_seconds, (step + 1) / num_steps)` so a run that will stop on
   steps still cools down. Zero cost when the time budget binds.
3. **Add a startup guard.** After argparse, if `args.num_steps * 0.8 s < args.max_train_seconds`
   (or simply if `args.num_steps < 20000`), print a loud `WARNING: step cap may bind
   before the time budget` on rank 0. Cheap, and it would have appeared in every local
   log.
4. **Dress rehearsal rule.** Before any submission, run the exact launch string once at
   full budget (no harness, no flags), confirm the header reads the intended
   `training target` / `max_seconds`, confirm `num_steps` at the end is within a few of
   the expected ~2200, and run `python prepare.py check` on the file and command.
   `run_experiment.py` should grow a `--bare` mode that passes nothing so this is
   reproducible and logged.
5. **Do not spend submission slots on the optlevel=2 / PACK_FACTOR=2 hedges.** They test
   hypotheses that are not needed, and each would also be capped at 1000 steps unless
   item 1 or a `--num-steps` flag is included. One resubmission of the current file with
   item 1 applied is the whole fix.
6. **Ask the organizers for the official `train.log` of the 1.1495 submission.** The
   first line (`training target=1000 steps`) and `num_steps: 1000` confirm this analysis
   without burning a slot. Also ask for the per-submission scores of submissions 2 and 3
   (see §3.2) and, while asking, the scorer's `--seq-len`.
7. **Reset expectations.** With the cap fixed, the current file should score about
   1.03 to 1.06 depending on the scorer's context length (§3.1). If the scorer uses 1024,
   T0.7-style eval-side work (e.g. a model that degrades less at short context) becomes
   worth ~0.016 — larger than most single Day 2-4 wins.

## 5. Files and lines referenced

- `train.py:191` (`NUM_STEPS`), `:4071-4072` (argparse defaults), `:4008-4020`
  (`lr_multiplier`), `:4519` (`wd_progress`), `:4559-4568` (`stop_after_step`),
  `:4606` (break to checkpoint), `:4384` (`training target=` banner), `:40, 70`
  (`setdefault` env), `:3973-4002` (`load_for_eval`), `:4625-4634` (in-process eval at
  `SEQ_LEN`).
- `prepare.py:38-41` (budget constants), `:96-113` (`record_step_boundary`),
  `:857-894` (`eval-public` CLI, `--seq-len 1024`, `--eval-tokens 2M`).
- `research/run_experiment.py:94-95, 111-112` (required flags).
- `research/experiment-log.jsonl` (all `*-fullbudget` rows carry `--num-steps 5000`).
- `research/run-logs/t04-stack-fullbudget.log`, `t04-stack-900s.log`, `xf-fixed480.log`,
  `rwkv-fixed995.log`, `sol13-lnc2-fullbudget.log`, `sol-seqlen2048-fullbudget.log`,
  `bucket-e-compile-fullbudget.log`.
- `research/trn2-experiment-results.md:84` (Sep 5 note), `:1478-1650` (Day 1 C2/C1),
  `:2442-2530` (the refuted cold-cache section).
- `research/frontier-research-plan-2026-09-09.md` §12.1, §12.4.
- Commits: `73a6753`, `6c69869`, `28a9104`, `70821b0`, `a118ddf`, `12f7658`, `783f7ef`,
  `bf1a3ae`, `ba8568d`, `0df8e05`, `f4fd8be`.
