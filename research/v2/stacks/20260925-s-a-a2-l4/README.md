# S-A A2 + L4 exploratory stack — Implementer handoff, HOLD

This is an **unrun, flags-only** candidate on exact research-v2 `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`. It changes no science code, evaluator, data or scoring path. It is not a claim of improvement or an authorized host job. The independent Verifier must check the exact argv and run CPU G2 on these bytes/argv, including the table optimizer path. The Supervisor must select a completed human-authored stack proposal and assign a separate Operator. A full same-host research-v2/matrix-LR-0.015 control is required for comparison.

## Explicit stack assumptions

This intentionally bundles causal factors under the human's Moonshot S-A request. It is a **stack-confirmation experiment**, so a good or bad result cannot attribute causality to any single component:

- Dense research-v2 base: width 1024 and inherited dense flags, matrix LR 0.015, seed 42, full chip, 1,800 s. Exact source is `workflow/proposals/research-v2/train.py`; no new `train.py` bytes are created here.
- A2 table profile from `research/v2/queue.json`: bigram+trigram value embeddings on layers `1,-1`, table multiplier 64, dimension 128, BF16 RMSProp, flat table LR and separate table clipping. This is the high-upside, high-resource component and has **no full native score yet**.
- Exploratory L4 schedule predeclared by the Supervisor from the human brief: warmdown 0.9, Muon momentum cooldown fraction 0.9 to final 0.8, embedding LR 0.45, final LR fraction 0.05 (already the default). These values are **hypotheses**, not measured optima. The earlier dense warmdown-0.9/embedding-0.45 single-factor results were near-null or weakly negative; stacking is not guaranteed to help.
- `--demon-beta1` is **not** in this exact run. Its default final/ref 0.55/0.8 is available, but enable it only as a distinct proposed arm after exact CPU G2 and optimizer-path review. Do not silently add it to this command.

## Run gates and result interpretation

1. Independent source/argv/G2 review, completed stack proposal, current authorization and fresh machine preflight. One job per host; at least 6.5 GiB free **plus projected table/cache/checkpoint footprint**; no other Neuron job. A resource or correctness failure is HOLD, not a negative quality result. Do not reduce tables or change the flags to fit.
2. Exact full 1,800-s launch from a clean, source-hash-verified run location. Do not use a 2M screen or a smoke result to decide quality. Use the G1-pinned fresh-process public-20M, context-1024 eval on the resulting checkpoint.
3. Compare to a **same-host, same-source, matrix-LR-0.015** control; report both final steps and BPB. A2+L4 may be slower, so >10% fewer steps is confounded for per-step quality. Use the unchanged `workflow/tools/decide.py` rules where applicable; no adoption or submission inference from this one stack run.
4. Preserve every run/failure in the v2 ledger and repository with exact hashes, command, logs, checkpoint status and evaluator output. This package never authorizes AWS lifecycle changes, cleanup or submission.
