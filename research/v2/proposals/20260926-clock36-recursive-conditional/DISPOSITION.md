# No executable flag-only stack on exact36

Research disposition only. No implementation, review approval, host reservation or launch.

Complete-source AST inspection of immutable commits found:

| Source | SHA256 | Startup-adjusted flag | Recursive flag/state |
|---|---|---|---|
| `7419927c7417b5666a984ec9c2b849feadabaaa9` | `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb` | Present, line3520 | Absent: zero flag constants and zero `RECURSIVE_SCHEDULE` Name nodes |
| `fe0969fd311d654e143dc6113614416c8970a424` | `75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f` | Absent | Present, line3666; seven state Name nodes |

Both refer to `workflow/proposals/research-v2/train.py`. Exact36 is324457 bytes; exact75 is325282 bytes. Text mentions of Recursive in unrelated feature comments are not a schedule switch. No source was imported or executed.

The currently running trn21 replication and completed trn22 score0.9883821509811542 use exact36 with:

```text
/home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 --max-train-seconds 1800 --num-steps 1000000 --out-dir out --unet-8x1024 --attention-layers 0,2,4,6,7 --initial-cache-refresh --startup-adjusted-schedule
```

This is an existing-run reference, **not a launch instruction for this proposal**. Appending `--recursive-schedule` is unsupported. Swapping to75b9 is not a solution: it loses the startup-clock factor.

## Why the combination is distinct but not ready

Exact36 lines3302–3312 reconstruct optimizer process age for steps0–4 and use `max(0, process_age +30 -900)`; at step>=5 they return the existing budget elapsed. Lines4083–4095 share that schedule elapsed across LR and progress-dependent optimizer settings. The original budget/reset/stop/checkpoint accounting remains separate. This is not a globally different900-second allowance or extra training time.

Exact75 instead adds separate Muon/Adam warmdowns0.95/0.65, floor0.05, quadratic Muon momentum and beta2 schedule, using the old elapsed clock. Its tensor-beta2 route at line2973 changes compiled optimizer inputs. Existing scalar tuning flags in36 are not the complete Recursive implementation.

The prior75b9 run scored0.9896621160468637 versus genuine same-host6ef7 control0.9899528600562257: delta−0.0002907440093620206, unchanged-selector **DISCARD**. Preserve that closed result. It did not test Recursive plus the startup-adjusted clock, so a future combined source is scientifically distinct, but cannot be called a flag-only arm or inherit source/graph qualification.

## Required boundary, not a new assignment

A separately selected implementation would have to define the combined progress semantics, preserve clock-only default/OFF behavior, undergo independent fresh review and exact-size CPU checks, and pass its own exact-stack true-cold five-step gate before full training: uncapped launch→fifth<=900s, target870s, MAX1800 unchanged. Exact36's845.6273377500474s cold result used no Recursive;75b9's cold result used no startup-clock adjustment. Neither proves the combination's graphs or startup.

Formal delta/replication evidence is still pending for exact36 at this freeze. The human architecture decision tree is conditional; an apparent comparison against an earlier `schedule_arm` is not a substitute for the genuine same-host control or unchanged selector. Future scientific OFF comparisons must retain historical row kinds rather than relabeling a candidate as a formal control. Any eventual quality run remains fresh-process public20M/context1024/B1 and official1800 through process exit, with sole-Operator common lifetime lock and fresh ps plus neuron-ls checks.

No source port, LR retune, invalid candidate argv, driver, host action, tensor test, ledger/state edit or follow-up reservation was made. The complete proposal records the failed prerequisite, null forecast/result/launch command, and prospective label only.

## Reproduction of the static finding

For each pinned commit, obtain the source with `git show COMMIT:workflow/proposals/research-v2/train.py`, compute SHA256 and parse with Python `ast.parse`. Count exact string Constants `--recursive-schedule` and `--startup-adjusted-schedule`, and Name nodes `RECURSIVE_SCHEDULE`; inspect the referenced function/CLI lines. Results are in `EXPERIMENT_PROPOSAL.json`. No Torch, model, data, network or remote host is needed.

Evidence snapshot: canonical `ce6ea8c9abc24452aad18e7cf265685700495583`; required directives/current state, exact36 replication proposal,75b9 formal-control crosscheck, and immutable source blobs. This is a Researcher observation, not independent source approval.
