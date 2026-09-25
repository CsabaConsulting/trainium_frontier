# trn22 e65 → r10 r2 source stage — r6 diagnosis draft, DISABLED

**No authority is granted by this package.** `REQUEST.HOLD.json` has
`dispatch_authorized=false` and `source_stage_authorized=false`. There is
no r2 enabled request, independent final review, or Supervisor assignment.
The entrypoint therefore refuses before SSH or a one-attempt marker. This
Implementer has not contacted a host, staged source, trained, evaluated,
cleaned up, reset a breaker, or changed an AWS resource.

## Why this is a distinct attempt

The independently reviewed trn22 r10 r5 result at
`b1387247faf1451dbfd04aadf72a821658b92063` is
`HOLD_ONE_ATTEMPT_CONSUMED_NO_ACCEPTED_STAGE`. The archived journal shows
preflight succeeded and the write SSH phase failed with the generic message
`SSH phase failed or emitted stderr: write`. The exact cause is **unknown**:
the previous wrapper did not retain a safe failure class or remote exit
code. Its unconditional postflight saw no r1 bundle or target and protected
assets intact; that point-in-time observation is not permission to retry.
The old marker remains consumed and must remain byte-identical (8,793 bytes,
SHA-256 `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`).
Any old r1 bundle or target appearing on the host is a HOLD.

This r2 draft uses **all four new one-shot identities**:

| Identity | New r2 value |
| --- | --- |
| Attempt label | `fs-20260925-trn22-c5c-a0-r10-incremental-stage-r2` |
| Target checkout | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r2` |
| Remote bundle | `/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn22-stage-r2.bundle` |
| Local one-attempt marker | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl` |

The accepted e65 source path is unchanged. The checked-in direct Git bundle
is **byte-identical** to r1: 109,800 bytes, SHA-256
`b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
The target commit remains
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, with `train.py`
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and unchanged `prepare.py`
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
No model, evaluator, tokenizer, data, Git bundle, disk floor or native launch
profile changed.

## Narrow pre-open timing repair

Inside the existing held host flock, the remote writer may retry **only a
transient final-proof timing expiry before its first `O_EXCL`**. It allows at
most three total pre-open passes. On *each* pass it reruns the full initial
activity check, two broad label scans, post-label activity/Neuron check,
root byte/inode check, STOP check, and new/old target absence check. It then
requires all activity, label and space samples to be finite and at most
one second old immediately before `O_EXCL`. A slow scan does not skip the
remaining safety checks: STOP, holder, label, disk, identity, data or other
refusal fails immediately. Exhausted timing attempts fail without a file
open. No sleep, cold fallback, wider freshness window or lower disk threshold
has been added.

The code attempts `O_EXCL` **once at most**. If opening fails or a gate
fails after it returns, there is no retry, unlink, rename or cleanup.
Even an empty provisional inode is retained, the marker is consumed, and
human direction is needed before any future attempt. The unchanged
post-open gates recheck broad label, process/Neuron, STOP, root bytes/inodes,
and both actual open-return and post-open ages before payload. The source
still tries an unconditional full postflight on failure.

This repair does **not** establish that timing caused r1's failure. It
addresses one possible pre-open timing refusal while preserving unknown
causes as HOLD. There remains a userspace race after any last scan and
before a file operation; no finite sequence makes the host atomic against
a hostile same-UID/root actor. Timing could still fail after `O_EXCL`, in
which case the empty file stays. A static test is not live feasibility.

## Bounded, nonsecret failure receipt

The new phase wrapper emits only a whitelisted remote gate code on a
controlled remote refusal, never exception text. The local wrapper records
`STAGE_HOLD.classification`: fixed phase/category strings, bounded numeric
exit code, bounded stdout/stderr byte counts, and a whitelisted remote code
when available. It never journals raw stdout, stderr, credentials, or a
traceback. Distinct classes include remote pre-open timing, other remote
gate refusal, SSH transport failure, remote timeout, nonzero remote exit,
stderr present, malformed/oversized stdout and phase mismatch. A failure
whose cause is still not classifiable remains HOLD; do not infer success.

## Gates remaining

1. A different Verifier must independently inspect this exact disabled r6
   source and tests, its SHA manifest, bounded retry seam, structured failure
   codes, r1 marker preservation, host path separation, strict post-open
   no-cleanup policy, and normal/`-O` behavior. The r5 PASS and r5 result
   HOLD do not review or enable r6.
2. Before any host contact, a separate read-only r2 timing/diagnostic plan
   may be commissioned and reviewed. This disabled draft itself supplies no
   host authority. If a later stage is chosen, create a distinct **exact**
   r2 enabled request, independent final source-stage PASS, and one-label
   one-attempt Supervisor assignment. No r1 authority file may be reused.
3. A fresh Operator preflight must prove the old r1 journal intact and old
   bundle/target absent; new r2 label/bundle/target/marker absent; exact
   e65 source/Git/host/known-host identity, active Desktop STOP/breakers,
   one-job lock and Neuron quiescence, protected checkpoint/cache identities,
   13,177,973,911 free root bytes and 10,000 free inodes before material
   write, 10,737,418,240 bytes after reconstruction, and the immutable
   6.5-GiB floor. Unknown or changed facts HOLD.
4. A future one-shot result requires a full archived journal, distinct
   readback and completed postflight, followed by independent result review.
   Then separately review trn21/trn22 matching r10 HEAD, G2 source-identity
   carry-forward, native A0 request/monitors/preflight, training and fresh
   public-20M evaluation. This package does none of those tasks.

Offline Implementer checks only:

```text
sha256sum -c research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold/SHA256SUMS
git bundle verify research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold/e65-to-r10.bundle
python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold -p 'test_*.py' -q
python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold -p 'test_*.py' -q
python3 -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py
python3 -O -B research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py
```
