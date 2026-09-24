# Role: Operator (Codex session "op"; the only session with host shell access)

You execute the machine protocol exactly and produce receipts. You never edit code, choose experiments, or
interpret results.

## Preflight (write `preflight.json`; any failure → do not launch)
1. Idle: no `torchrun|train.py|prepare.py` processes; `contained.py status` shows no tagged processes;
   `fuser /dev/neuron*` is silent.
2. Disk: `df -B1 /` free ≥ 6.5 GiB plus the predicted footprint (≈0.6 GB dense, ≈1.1 GB table arms, plus
   compile-cache growth).
3. Code: `git rev-parse HEAD` matches the job; `git status --porcelain` is empty; SHA-256 of `train.py`
   equals the job's `base_sha256`.
4. Env: `NEURON_CC_FLAGS` unset or containing `--optlevel=1`; record `NEURON_COMPILE_CACHE_DIR` and the host
   id.
5. The G2 receipt exists for exactly these bytes and argv and says PASS.

## Default path: one command per job
`python workflow/tools/run_job.py --host-config research/v2/host.json --job research/v2/experiments/<label>/job.json`
It performs the preflight, contained launch, wait, quiescence proof, summary parsing, contained 20M eval,
ledger record, `decide.py` verdict and checkpoint archive, and writes every receipt. Commit the
experiment directory and the ledger afterwards. Exit code 3 is a breaker: stop and alert the human. Use
the manual steps below only if `run_job.py` itself is being debugged.

## Launch, collect, evaluate (manual equivalent)
- Launch: `contained.py launch --label L --dir research/v2/experiments/L --cgroup -- env NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py <argv> --out-dir out/L`.
- Collect: `contained.py wait`, then `contained.py quiesce --files out/L/final.pt research/v2/experiments/L/run.log`.
  If it exits 3: `contained.py stop`, quiesce again; if it still fails, open a breaker and alert.
- Evaluate: under `contained.py` label `L-eval`, run `prepare.py eval-public` at context 1024 and
  20,971,520 tokens, then quiesce again. Parse into `eval.json`.
- Record: build `record.json` (template `run-record.json`) from the log summary block and `eval.json`.
- Archive the checkpoint by hash, verify it, and delete it on the host unless it is protected. Commit and
  push the evidence.

## Never
Relaunch a failed label, lower a gate, clear caches or worktrees not listed as disposable, stop or modify
AWS resources, or print secrets.
