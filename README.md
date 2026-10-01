# AWS Trainium Frontier: Phase 1 final package

**Best official val_bpb: 0.9647** (submission `sub-b6bcf387`, 30 Sep 2026, 21:03 IST), down from 1.0246 on 14 September. The team's first score below 1.0 was 0.9929 on 19 September.

Team: Csaba Toth (team lead) and Poojith Devan.

## What is here

| Path | What it is |
| --- | --- |
| `submission/train.py` | The exact file behind `sub-b6bcf387`. SHA256 `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`. |
| `submission/LAUNCH_COMMAND.txt` | The launch command used for every run and upload. |
| `writeup/WRITEUP.md` | Technical writeup: score history, what moved the score, the engineering, the scoring crash and its fix, where the wall is, and what we ruled out. |
| `results/official_submissions.csv` | Every official submission from the portal, with scores and notes. |
| `results/final_night_experiments.csv` | The 14 ideas tested on the final night and their measured results. |
| `results/seed_runs.csv` | Five seeds of the final file on our own hosts. |
| `evidence/final-submission-audit/` | Audit package for the final file: both full-run receipts, source diff against the previous file, cold-start and budget evidence, and single-core eval diagnostics. |
| `evidence/seed-alternates/` | Audit packages for seeds 43 to 46 of the final file (seed 44 was an outlier run; none were submitted). |
| `evidence/closed-arms/` | Audit packages for two arms that lost on full runs: CS21 (two-target weight 0.75) and CS22 (attention in 4 layers). |
| `evidence/history/` | Audit packages for earlier files on the way to the final one (A3, LaneB, e1bd and others). |
| `logs/` | The overnight run log (`STATE.md`), the round-by-round experiment instructions (`CLAUDE_ROUNDS.md`), research and verification notes, and an archive, with server hostnames and IP addresses removed. The evidence packages are byte-identical originals, so their manifests still verify. |

## Run it

Hardware: one `trn2.3xlarge` with the competition image (neuronx-cc 2.26, torch-neuronx 2.11, NKI 0.5), plus the organizers' `prepare.py`, data and tokenizer.

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
python prepare.py eval-public --train-py train.py --checkpoint out/final.pt
```

Leave `NEURON_CC_FLAGS` unset: `train.py` sets its own compiler flags, including `--optlevel=3`. Cold start (compile) is about 600 s, inside the 900 s cap. On our hosts the file scored 0.964447 and 0.964228 on the public dev set; the official platform scored it 0.9647.

## Key numbers

- 11 new team bests in the final 12 days (13 scored submissions from 19 to 30 September).
- About 342k tokens per second; about 3,020 steps and 600M tokens in the 30-minute budget.
- Top-10 cutoff at the close of Phase 1: 0.9562, so the final file was 0.0085 short, about 21% more effective training at our measured rate of 0.0004 bpb per 1% more steps.
