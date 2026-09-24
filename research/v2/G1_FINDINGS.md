# G1 evaluation and data inventory

Update (2026-09-24 21:14 UTC): trn22's fresh retained-checkpoint calibration **passed** and exactly
reproduced the historical `0.992968300659387` public20M score at context 1024. The Operator
receipt is `g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/operator-receipt.json`;
independent review is `../agentic/verification/20260925-g1-trn22-calibration-review.md`. This is
not a new model score or a valid advancement control. trn21 remains uncalibrated and disk-blocked.
The older pending-calibration bullet below is superseded for trn22 only.

Source and both-host metadata were read on 2026-09-24 around 20:20 UTC. See `g1.json` for exact paths and numbers. No data, evaluator, checkpoint, or host configuration was modified.

- Both hosts contain the five default train shards, with identical file sizes, 415 row groups and 424,960 documents. This does not establish a token count or the official private runner's actual inventory.
- `prepare.py` is the same unchanged SHA on the controller and both retained training directories.
- The default CLI evaluates 2,097,152 tokens. Selection must explicitly request 20,971,520 tokens, context 1024, single process, batch 1.
- Historical exact evaluations took approximately 12 minutes, not merely a few minutes. Reserve that time in job ETAs.
- Retained calibration checkpoints exist on both hosts. Neither has been freshly re-evaluated in this migration, so G1 remains partial and `eval_args_verified` remains false.
- Local controller has about 14 GiB RAM, only about 6.6 GiB available, and no torch/numpy/pyarrow in `.agent-venv`. Do not launch the full-size four-rank G2 locally. Both hosts reported approximately 122 GiB available CPU RAM; approved CPU-only G2 can use an idle target host without package installation.

Next: independent review, safe host staging, fresh calibration, exact-argv G2, and native gates. A recorded source fact is not a passing runtime gate.
