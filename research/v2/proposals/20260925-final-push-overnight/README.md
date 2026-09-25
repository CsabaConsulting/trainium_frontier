# FINAL PUSH overnight fallback proposals

This Researcher-only bundle supplies six unconditional, complete, existing-flag proposals per host, plus two conditional table replicates. It changes no training code, central queue, state, runner, host, or decision threshold. The newest human directive controls: flag-only arms on verified c5c need normal machine checks and independent receipt review after launch, not another prelaunch code review.

Source: `submissions/research_v2/train.py`, SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`. Source-definition commit: `8b9fc841ed817eac263c7087c3bce7c9f008bda1`. Each proposal contains its complete train and fresh public-20M evaluation argv. The exact source's argparse declarations were reconstructed from its AST and all 14 proposed commands parsed successfully without importing or executing training code; see `COMMAND_CHECKS.json`. This is static compatibility evidence, not a numerical PASS.

## Ready fallback order

| Slot | trn21 | trn22 |
|---|---|---|
| 1 | Matrix LR 0.0175 | Matrix LR 0.0125 |
| 2 | Warmdown 0.85 | Embedding LR 0.45 |
| 3 | Weight decay 0.35 | Final LR fraction 0 |
| 4 | Embedding LR 0.45 | Weight decay 0.35 |
| 5 | Final LR fraction 0 | Warmdown 0.85 |
| 6 | Matrix LR 0.0125 | Matrix LR 0.0175 |

These are **fallbacks**, and their two-host runs are deliberately predeclared. Priority table replicates, due controls, and independently verified moonshot stacks take the next compatible slot. Historical weak/negative studies are cited rather than hidden; FINAL PUSH explicitly reopens older parked axes. No claim is made that these small scalar arms can close the leaderboard gap.

`QUEUE_RECORDS.json` is a proposal bundle for Supervisor integration, not an active launcher. The twelve `READY_PENDING_HOST_PREFLIGHT` rows have complete source, science argv, unique label and same-host control. That status does not claim host staging or a passed live preflight. The Operator must bind the actual run directory and existing warm cache paths, verify the source/evaluator, label freshness, disk >= 6.5 GiB plus forecast footprint, STOP/breaker state, idle devices and one job per host. Never launch while the previous train or evaluator remains alive. The queue implementer was given the exact `queue_binding` schema and literal argv, including output/seed/budget flags; the runner must not append conflicting flags.

## Comparisons and conditions

trn21 control `fs-20260925-v2-c5c-a0-control-trn21-r1`: 3,675 steps, public-20M 0.9914739939934344. trn22 control `fs-20260925-v2-c5c-a0-control-trn22-r10`: 3,633 steps, public-20M 0.991513958330512. Their independent PASS receipts are bound in every proposal. Manual source-copy controls are identified by full source hash; the source-definition commit is not falsely presented as a recorded host Git HEAD.

Conditional priority work is A2 on trn21 and A1 on trn22 after the other host's reviewed valid delta is <= -0.0006. These two rows are **not counted** toward ready depth and are not armed before the condition. Each comparison is against its own host's A0, never the opposite host's raw score. A source/base adoption makes this fixed c5c bank stale: pause between jobs, rebind the newest verified base and controls with new proposals/labels, then resume. Refresh controls after eight intervening runs or twelve hours or drift.

Use unchanged `workflow/tools/decide.py` and the newest directive's provisional delta <= -0.005 policy. Keep the big-win cold replay/audit hold. A flag-arm step ratio outside [0.90, 1.10] is confounded, not a quality verdict. Every run gets a separate fresh process with unchanged prepare.py, context 1024, 20,971,520 tokens, and a 1,200-second evaluator timeout. Archive failures and receipts before using a result. This bundle does not authorize checkpoint deletion.

## Not counted as ready

- Batch 262,144: exact c5c has no total-batch CLI flag; it needs a separately gated code change.
- D1 shortened epoch-budget diagnostic: exact same-host epoch-boundary charged time still needs binding; never label it an 1,800-second scored run.
- S-A, S-B, S-C and B2: await their patches and new-code gates.

Research freeze remains Sep 29 18:00 IST (12:30 UTC). Nothing here authorizes submission or AWS/resource changes.

