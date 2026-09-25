# trn22 r10 r5 one-shot source-stage Operator receipt

Operator verdict: **HOLD, one-attempt marker consumed; no accepted source stage**. The reviewed source-stage entrypoint was invoked exactly once from the clean designated worktree. It returned `HOLD: RuntimeError: RuntimeError: SSH phase failed or emitted stderr: write; marker and any empty/nonempty bundle retained`. No native training, evaluation, AWS lifecycle change, cleanup, marker reset, or retry occurred.

## Authority and fresh preflight

- Designated execution checkout: `/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-remote-timeout-repair-r1-disabled` at clean HEAD `ed2ca7fc43d29f6909c001ee97e2875ef8ae4ee7`. `git status --porcelain=v1 --untracked-files=all` was empty. The 14-item package SHA manifest passed.
- Science hashes: `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Exact r5 request SHA-256 `25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2`, independent r5 final review `da237ac25e8547c7cc9b0cc6dbf1fec9e2875af6be94b7e407d17616181a0994`, Supervisor r5 assignment `806c89ad2aa03cd706d5c8548c2862b0c8d0eb7e4389e5486b65f0e3859037bc`, operator source `6224763c0acdbba8bfadf4281498fe9801a8b8d4aaee3d37f8b58f2f97b412be`, remote source `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`, bundle `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` (109,800 bytes).
- Fresh local authority equality, active Desktop/former/package STOP/breaker, known-host fingerprint/key metadata, marker-parent mode and marker-absence gates passed. The r10 one-attempt marker and all four prior r9 markers were absent before invocation.
- At 2026-09-25 11:48:52 UTC, the reviewed read-only remote pre-marker phase passed on `ip-172-31-32-253`. Its remote GNU timeout identity probe passed. It found clean e65 source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and exact science hashes, target/bundle/label absent, checkpoint SHA-256 `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b` and protected caches intact, lock available, no STOP, training/eval process, or Neuron holder; 58,453,110,784 free root bytes and 31,660,014 free inodes after the check. Both exceed the stated floors of 13,177,973,911 bytes and 10,000 inodes.

## Exact one-attempt outcome

The single reviewed command was:

```text
/usr/bin/python3 -B /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-remote-timeout-repair-r1-disabled/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
```

It exited 1 with the HOLD above. Local marker `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r10-incremental-stage-r1.one-attempt.jsonl` exists, owner-only mode 0600, 8,793 bytes, SHA-256 `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`. Its complete byte-for-byte copy is `RAW_ONE_ATTEMPT.jsonl`.

The journal events are exactly `ONE_ATTEMPT_CONSUMED`, `FRESH_PREFLIGHT`, `STAGE_HOLD`, and `UNCONDITIONAL_FULL_POSTFLIGHT`. It has **no** `STAGE_WRITE` receipt or distinct-session readback because the write SSH phase failed. The phase helper intentionally exposes only `SSH phase failed or emitted stderr: write`, so the underlying remote refusal is **unknown**; do not infer a timing cause.

The operator's distinct-process full postflight at 2026-09-25 11:49:41 UTC found both bundle and target absent, the same boot ID, e65 source commit and science hashes, the same protected checkpoint hash/caches, no STOP/process/Neuron holder, 58,453,102,592 free root bytes and 31,660,014 free inodes. This is a terminal read-only observation, not an accepted stage or a causal diagnosis. No cleanup/retry is authorized. A separate Verifier must review this result and decide the disposition before any further source-stage or native action.
