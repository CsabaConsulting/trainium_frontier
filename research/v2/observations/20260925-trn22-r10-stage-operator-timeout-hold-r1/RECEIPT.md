# trn22 r10 source-stage operator receipt: local timeout gate HOLD

UTC: 2026-09-25T11:24:47Z. Operator verdict: **HOLD before one-attempt marker and before SSH**. No source stage, host write, native training, evaluation, AWS lifecycle action, cleanup, or breaker reset occurred.

## Exact subject and authority

- Designated clean execution checkout: `/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r4-stop-recheck-r1` at `e827fd2abecf813474f2d6c95518b4ac331e04b4`; `git status --porcelain=v1 --untracked-files=all` was empty.
- `train.py`: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py`: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Enabled request: `d19086a3f0c8aff9835f14f641d528980b12b21f45310070f6613ae261a33958`; independent final PASS: `c9be026c2609fbbdcfdc85005630675959a2b6661befb91268dd4cf73d59d615`; Supervisor assignment: `ed07f024e45b5a606c4c8ed3f0ad6bc335a6bcaead548bf89c1e0cbcd2ec5bb4`.
- Reviewed local operator: `af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f`; remote source: `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`; bundle: 109,800 bytes, `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.

## Gate sequence

1. The local authority equality check, active Desktop/former/package STOP and breaker checks, strict pinned SSH host identity check, exact bundle/source hash check, marker-parent metadata, current marker absence, and four prior r9 marker absences all passed. The local check printed `LOCAL_PREFLIGHT_PASS`.
2. For the required *pre-marker* complete host preflight, the Operator invoked only the existing reviewed read-only `operator_once._ssh_phase("preflight", None, b"", remote_source)` method. It failed locally at `_ssh_phase`'s absolute GNU `timeout` identity check, before constructing or running SSH. The complete host preflight therefore **did not occur**.
3. `/usr/bin/timeout --version` returned `timeout (uutils coreutils) 0.8.0`; the reviewed operator requires a prefix of `timeout (GNU coreutils) `. `/usr/bin/timeout` resolves to `/usr/lib/cargo/bin/coreutils/timeout`.
4. At 2026-09-25T11:24:47Z the exact trn22 r10 one-attempt marker remained absent, as did all prior r9 markers. The designated one-shot main command was **not** invoked. There is no stage result, source-readback, or postflight to review.

The exact refusal output is in `RAW_LOCAL_PREFLIGHT.txt`. The failure is a tooling-identity mismatch, not evidence about trn22 health. Do not bypass the guard, use an alternate timeout executable, or interpret this receipt as stage/native authority without a new independently reviewed decision. No retry, marker reset, provisional cleanup, or host mutation was attempted.
