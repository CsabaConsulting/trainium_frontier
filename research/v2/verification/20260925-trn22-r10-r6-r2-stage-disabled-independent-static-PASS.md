# Independent static PASS — disabled trn22 r10 r6/r2 source-stage draft

Checked 2026-09-25T12:15:49Z. Subject: Implementer commit
`7b250a74679e04a86f8dbf2211275a692f47fc49` on
`codex/20260925-trn22-r10-r6-stage-diagnosis-disabled`, with a clean subject
tree. Verdict: **PASS_DISABLED_SOURCE_ONLY; OPERATIONAL HOLD**. This approves
the static safety of this disabled draft, not a host attempt. I did not
contact a host, create or change an attempt marker, transfer source, stage,
train, evaluate, clean up, reset a breaker, or issue authority.

## Byte identity and authority

| Reviewed artifact | SHA-256 |
| --- | --- |
| `operator_once.py` | `1bc7a165919ffab3c30a2fccd5cbb40401b2356b72014a38afa74f75f4facab3` |
| `remote_incremental.py` | `f66cb6f9d4ac59cd3af2d4e1a87f92ef64809840463e9d60f6af105031a7987d` |
| `REQUEST.HOLD.json` | `6ff303b7fdb4f777ae49583d86ee95e4c99c856ea9603e93ac94452610521d16` |
| `e65-to-r10.bundle` | `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| `RUNBOOK.md` | `4581140f1035a1aef78e79f9362e8759b3204d5942026e156b33ea9541893d34` |

`sha256sum -c SHA256SUMS` passed for all ten listed files. `git bundle
verify` passed: the 109,800-byte bundle carries
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1` and requires the original
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`; its SHA-256 equals the r1
bundle. The target science identity remains `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and unchanged `prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The source/disk floors, full-chip training profile, data and evaluator were
not edited by this package.

The r2 label, remote target, remote bundle and local marker are each distinct
from the consumed r1 attempt. The r1 marker is bound by both request and
operator to 8,793 bytes and SHA-256
`229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`;
my read-only local check found those exact bytes, mode `0600`, one link,
and the new r2 marker absent at review time. The r5 result HOLD at
`b1387247faf1451dbfd04aadf72a821658b92063` is pinned by its exact
review SHA-256
`58461a36b0f6c0a9901be40139cf3da75d65f5c08727f55f16e2fd1ad20d9dc6`.
The checked-in request has all four dispatch/source-stage/native/eval
authorities false. Its future r2 enabled request, r6 final independent
review and r6 one-attempt Supervisor assignment are absent. Running the
entrypoint directly under normal and `-O` Python returned HOLD before marker
or SSH; optimized Python is also rejected inside `stage_once`.

## Retry and diagnostic review

The remote `perform('write')` holds the existing host flock across the
preflight-to-write probe and `_write_bundle`. Only a typed
`PreOpenFreshnessExpired` can enter the bounded loop, and only before the
first `O_EXCL`; `MAX_PREOPEN_ATTEMPTS` is three. Each pass repeats activity,
two broad label scans, post-label activity/Neuron, root byte/inode floor,
STOP and both new and prior target/bundle absence. Both scan duration and
activity/label/space sample ages must remain finite and at most one second
immediately before exclusive open. Non-timing exceptions escape without a
retry. The `O_EXCL` call is outside the retry exception handler and occurs
at most once. On its return, the unchanged one-second checks plus fresh
label/activity/space/STOP checks run before payload; any failure preserves
the provisional inode, even when empty. There is no unlink, rename, fallback,
sleep, widened window or lowered disk floor.

The remote wrapper emits a fixed phase/status/failure-code JSON on controlled
failure, suppressing exception text. The local wrapper logs only fixed
phase/category names, bounded return code, bounded stdout/stderr byte counts
and a whitelisted remote code; it does not journal raw remote streams.
Ambiguous transport exit values (`255`, `137`) are **categories, not proof of
root cause**. Unknown or malformed remote behavior still yields HOLD, not
success or an automatic retry.

## Independent offline evidence

- All **55/55 tests passed** under normal Python and **55/55 passed** under
  `python3 -O -B`; `git diff --check` passed. The checked-in tests cover
  late STOP/label/activity/disk, timed-out pre-open scans, delayed `O_EXCL`,
  empty-inode retention, old-r1 artifact refusal, authority refusal and
  sanitized phase errors.
- In separate disposable local witnesses using the real final-check and
  `_write_bundle` sequence, I expired the first label scan. A STOP injected
  at the second pass's first activity check, and independently an old r1
  bundle injected there, each blocked before any new `O_EXCL`. No target or
  bundle was created in either witness.
- I replayed the frozen r5 `_stable()` predicate for both trn22 and trn21
  against their archived preflight/postflight snapshots, with the synthetic
  clock fixed to each postflight: both stable same-identity pairs passed.
  I also sent a same-sized inert remote program, each full archived prior
  JSON and 109,800-byte synthetic stdin through the frozen local shell
  command construction for both hosts; both round-tripped. These tests do
  **not** establish the actual write-phase snapshot or SSH result.

## Blocking operational questions

The independent trn22 r5 result recorded only a generic write-phase SSH
failure; no remote return code, stdout or stderr was retained. The new
classification code would improve a *future* failed write, but it cannot
diagnose the consumed r1 attempt. Trn21's separate r5 stage has now failed
in the same generic phase, with no created bundle or target. Stable
pre/postflight evidence and local serialization tests do not identify the
common cause. **Before any new write/marker attempt, require a separately
designed and independently reviewed read-only diagnostic of the common
pre-`O_EXCL`/transport path, with its own authority and result review.** Do
not infer timing and do not spend the r2 one-shot merely to obtain a new
failure code.

Even after diagnosis, a distinct enabled r2 request must bind these exact
bytes and identities and receive independent final review, fresh STOP/
breaker/lock/source/label/disk/protected-asset and host checks, proof that
r1 artifacts and r2 names are absent remotely, and a new one-label
Supervisor assignment. A future post-open failure can leave an empty inode;
this static PASS authorizes no cleanup or retry. Native training and public
evaluation remain entirely separate, held gates.
