# trn21 r10 r5 one-shot source-stage Operator receipt

Operator verdict: **HOLD, one-attempt marker consumed; no accepted source stage**. The reviewed source-stage entrypoint was invoked exactly once from the clean designated worktree. It returned `HOLD: RuntimeError: RuntimeError: SSH phase failed or emitted stderr: write; marker and all partials retained`. No native training, evaluation, AWS lifecycle change, cleanup, marker reset, or retry occurred.

## Authority and fresh preflight

- Designated execution checkout: `/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r5-enabled-final-pending` at clean HEAD `fc0e5953849d91c60f6b692a79b71e38f1784829`. `git status --porcelain=v1 --untracked-files=all` was empty. The 13-item package SHA manifest passed when invoked from its package directory. (An initial invocation from the repository root could not resolve its relative paths; it was rerun in the intended directory and passed before staging.)
- Science hashes: `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Exact r5 request SHA-256 `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a`, independent final review `d1170f0ac27285d4ca5670dda9828da2a7cc76a7527f2a6eb0b90239479b9ff3`, Supervisor one-shot assignment `f30803dc499966d58d52243b703ca2ec826ed93b182ded616bdfc70331ed6eb4`, operator source `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b`, remote source `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`, bundle `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` (109,800 bytes).
- Fresh local authority equality, active Desktop/former/package STOP/breaker, pinned known-host fingerprint/key metadata, marker-parent mode, and marker absence gates passed. There were no trn21-named entries in the local source-artifacts directory before invocation.
- At 2026-09-25 12:10:04 UTC, the reviewed read-only remote pre-marker phase passed on `ip-172-31-34-85`. Its remote GNU timeout identity probe passed. It found the clean private e65 source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and exact science hashes, target/bundle/label absent, both pinned checkpoint hashes, preserved e65-stage files and caches intact, host lock available, no STOP, training/eval process, or Neuron holder; 57,894,514,688 free root bytes and 31,563,294 free inodes after the check. Both exceed the stated floors of 13,177,973,911 bytes and 10,000 inodes.

## Exact one-attempt outcome

The single reviewed command was:

```text
/usr/bin/python3 -B /home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r5-enabled-final-pending/research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py
```

It exited 1 with the HOLD above. Local marker `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn21-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl` exists, owner-only mode 0600, 18,394 bytes, SHA-256 `54c65164271d895392c2e170b47dbdf07f62e0aab2cfa378e529afdbbeca3a4c`. Its complete byte-for-byte copy is `RAW_ONE_ATTEMPT.jsonl`.

The journal events are exactly `ONE_ATTEMPT_CONSUMED`, `FRESH_PREFLIGHT`, `STAGE_HOLD`, and `FULL_POSTFLIGHT_COMPLETED`. It has **no** `STAGE_WRITE` receipt or distinct-session readback because the write SSH phase failed. The phase helper intentionally exposes only `SSH phase failed or emitted stderr: write`, so the underlying remote refusal is **unknown**; do not infer a timing cause.

The operator's distinct-process completed full postflight at 2026-09-25 12:10:53 UTC found both bundle and target absent, the same boot ID, clean e65 source and science hashes, the same two checkpoint hashes and preserved assets, no STOP/process/Neuron holder, 57,894,498,304 free root bytes and 31,563,294 free inodes. This is a terminal read-only observation, not an accepted stage or a causal diagnosis. No cleanup/retry is authorized. A separate Verifier must review this result and decide the disposition before any further source-stage or native action.
