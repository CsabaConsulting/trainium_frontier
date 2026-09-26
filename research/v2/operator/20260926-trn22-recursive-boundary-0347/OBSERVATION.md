# Protected Recursive-stack training to evaluation boundary

One authorized read-only observation at 2026-09-26 03:47:58.130986 UTC.
Training exited 0 at 03:44:52.417328 UTC with 3,625 steps, 475.1M tokens.
The rehearsal recorded process wall 2613.8114476309856 seconds, actual
startup allowance 824.5246403359342 seconds and charged-through-exit
1789.2868072950514 seconds, within the 1800-second budget.

Checkpoint remains on the host: 453048658 bytes, archived training SHA-256
1e1bf15aa6c2b95578f2d6317b8a400821cb4f66bea13312319e385236902b9e.
It was not downloaded or rehashed during the live evaluation.

Fresh-process public-20M/context-1024 evaluation was live with observed
timeout/harness/evaluator PIDs 3274486, 3274487 and 3274501. Existing
dispatch/flock/driver PIDs 3252200, 3252201 and 3252202 remained live.
The original device 66305/inode 832589 lock remained exclusively held by
3252201; 20237729792 bytes were free. No terminal BPB or fatal error was
present. All four closed training graph_stats CSVs are preserved in the raw
receipt. Full per-process identities and effective environment are recorded
there; retain the newly observed evaluator IDs in the final postflight union.

No job, source, cache, driver, lock, evaluator or host state was changed.
This is training-budget evidence only, not quality acceptance or a cold-full
claim. Evaluation must finish naturally and receive complete retrieval and
explicit descendant/device/lock postflight before any subsequent launch.
