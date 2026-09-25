# trn21 conditional queue bootstrap

Operator deployment at 2026-09-25 15:55:49 UTC / 21:25:49 IST.
Source review: `research/v2/verification/20260925-host-conditional-queue-independent-PASS.md`
at controller `aaf9b0aa`; reviewed runner SHA-256
`0ce40ad8aea85aa556b3b87e4753c25049751024a137c69947b8e688e482b50d`.

The **empty** queue is host-detached as PID **2926930**, PPID 1, waiting for
`/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/next-resolution.json`.
It contains zero jobs and has no flag-bank fallback. Distinct-SSH postflight at
15:56:18 UTC confirmed `waiting_resolution`, no device-lock file descriptor,
no inbox, empty error log and exact runner/manifest hashes. It does not acquire
the current A1 driver's lock or interrupt training/evaluation. A separately
authorized, hashed resolved manifest is needed before another job can start.

Bootstrap SHA-256:
`d9bedd841876be3a1a28f061420d56fa2d6bb07d1f625c73c42d25ed5696e7f8`.
The complete exact bootstrap is `bootstrap-manifest.json`; the retrieved host
receipt and distinct postflight are `deployment-and-postflight.json`.

The Operator verified local STOP/breakers clear, actual hostname, root free
52,102,086,656 bytes (above 6.5 GiB), no existing host-queue process, exact
c5c/evaluator SHA-256, executable paths, and all six directories in each of the
existing A1 train/eval cache roots. Only the new private queue directory and
its manifest, byte-identical runner and byte-identical c5c copy were staged.
No existing source, cache, model, evaluator, SDK or AWS resource was changed;
no cleanup occurred. The current A1 remains the sole Neuron training job.

At postflight, A1's summary reported 3,051 steps, training_seconds 1784.9,
charged_seconds_est 1785.5, startup_allowance_est 900.0, over_budget False.
Its final logged loss was 2.6718 at step 3050, epoch 2. Training exit and fresh
public-20M evaluation had not yet appeared, so this is not a BPB result.

The earlier deployment assignment was paused before any host write when the
human requested read-only status. This new bootstrap follows the later explicit
resume/conditional-queue authorization; no old scalar-arm queue was armed.
