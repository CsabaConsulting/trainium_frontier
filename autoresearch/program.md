# autoresearch

This is an experiment to have the LLM do its own research.

## Platform: AWS Trainium 2

This codebase runs on a single AWS Trainium 2 NeuronCore (LNC=2), not CUDA GPUs. Key platform differences:

- **Attention**: Uses `F.scaled_dot_product_attention` in eager mode (no torch.compile).
- **Device**: `torch.device("neuron")` with `torch.neuron.synchronize()` for timing.
- **Precision**: Model weights are bf16. Optimizer states are fp32. Stochastic rounding enabled.
- **Constraints**: ~24GB HBM per LNC=2 core. First run compiles NEFF graphs (adds ~60s overhead). Changing tensor shapes triggers recompilation — avoid changing DEPTH/ASPECT_RATIO/DBS unless you accept the compile cost.
- Do NOT use `torch.cuda.*`, CUDA-specific libraries, or `torch.compile`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current mainline.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - This file (`program.md`) — instructions.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `../prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
4. **Verify data exists**: Check that `~/.cache/neuron_competition_r1/` contains data shards and a tokenizer. If not, run `bash run.sh` once (it calls `prepare.py setup`).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row if it doesn't exist.
6. **Confirm and go**.

## Experimentation

Each experiment runs on a single NeuronCore. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it as: `bash run.sh` or `NEURON_LOGICAL_NC_CONFIG=2 python3 train.py` (with PYTHONPATH set to `..`).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb: 1.388065
steps:   546
params:  33,554,448
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 4 columns:

```
commit	val_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.388065) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `bash run.sh > run.log 2>&1`
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes of training time (+ startup and compilation overhead). If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. The loop runs until the human interrupts you.
