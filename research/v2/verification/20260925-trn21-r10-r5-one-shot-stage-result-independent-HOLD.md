# Independent result review — trn21 r10 r5 source-stage attempt

Checked 2026-09-25 12:14 UTC. Verdict: **HOLD_ONE_ATTEMPT_CONSUMED_NO_ACCEPTED_STAGE**.
This is result disposition only, not authority for another source-stage
attempt, marker reset, cleanup, native training, evaluation, AWS action,
or submission. I made no SSH or host contact.

## Exact evidence and authority

- Operator result `1b1f33bbd0933b6903bc55f5590b47b01f15ee1f` descends from
  reviewed/assigned source `fc0e5953849d91c60f6b692a79b71e38f1784829`
  and adds only `RAW_ONE_ATTEMPT.jsonl` and `RECEIPT.md` in
  `research/v2/observations/20260925-trn21-r10-stage-r5-operator-result-hold-r1/`.
- The archived raw journal is 18,394 bytes, SHA-256
  `54c65164271d895392c2e170b47dbdf07f62e0aab2cfa378e529afdbbeca3a4c`.
  It is byte-identical to the retained local one-attempt marker, observed
  owner-only mode `0600`, one link. The Operator receipt SHA-256 is
  `fdb69b8eea4b731f9f867c318ba17479770c6fdc71be1d16a704e0800cd8c664`.
- The consumed-marker event binds the exact trn21 label and source-stage-only
  scope to request SHA-256 `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a`,
  independent final review
  `d1170f0ac27285d4ca5670dda9828da2a7cc76a7527f2a6eb0b90239479b9ff3`,
  and Supervisor assignment
  `f30803dc499966d58d52243b703ca2ec826ed93b182ded616bdfc70331ed6eb4`.
  All three match canonical retained files. Both review and assignment limit
  authority to source staging and have `native_training_authorized=false`.

## Recomputed terminal facts

The journal has exactly four events: `ONE_ATTEMPT_CONSUMED`,
`FRESH_PREFLIGHT`, `STAGE_HOLD`, `FULL_POSTFLIGHT_COMPLETED`. There is
**no `STAGE_WRITE` receipt and no distinct-session readback**. The HOLD reason
is exactly `RuntimeError: SSH phase failed or emitted stderr: write`;
the Operator command exited 1/HOLD according to its receipt.

The raw fresh preflight records host `ip-172-31-34-85`, e65 source commit
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`, science hashes
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
for `train.py` and
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`
for `prepare.py`, no bundle or target, no matched process/Neuron holder/STOP,
57,894,506,496 root-free bytes, and 31,563,294 free inodes. It also pins
both protected checkpoint hashes, preserved e65-stage files, and cache
metadata.

The completed full postflight is a distinct remote process on the same
host, boot ID, root mount, and lock identity. It observes **bundle and
target still absent**, byte-identical e65 source identity and science
hashes, identical protected checkpoint/stage-file/cache metadata, no
matched process/Neuron holder/STOP, 57,894,498,304 root-free bytes, and
31,563,294 free inodes. Both space samples exceed the predeclared floors.
The raw source snapshot does not include a Git dirty-tree field; the
Operator's separate pre-marker check reports a clean source, while this
review independently establishes the archived identity and protected
asset continuity only.

## Cross-host diagnosis boundary

Trn22's independently held r5 result (`b1387247faf1451dbfd04aadf72a821658b92063`)
has the same **observable pattern**: consumed marker, successful preflight,
generic write-phase SSH HOLD, distinct-process postflight, no persisted
bundle/target, and intact source/protected assets. Both wrappers use the
same branch in `_ssh_phase()`: nonzero return code **or** nonempty stderr
**or** stdout over 2 MiB all raise the same generic error without archiving
the return code or bounded stderr/stdout. That shared diagnostic loss is
proven from source. It does **not** prove the underlying trn21 and trn22
write failures have one deterministic cause. Both remote write paths have
tight one-second pre-open freshness checks, but the journals contain no
write-phase timing or raw remote error; timing is a risk, not a diagnosis.
The specific remote cause is **unknown**.

## Safe disposition

Preserve the consumed marker and all host artifacts. Do not retry this
label, reset the marker, clean a possible provisional file, or launch native
A0 from an unstaged r10 target. The next useful diagnostic is a separately
reviewed **read-only** probe of the pre-open write-path checks on both hosts,
with bounded categorized failure/timing evidence; an offline repair should
also retain sanitized return-code/output-class evidence in any *future*
one-shot wrapper. Neither action recovers the discarded cause of this
attempt. Any new source-stage attempt needs a new unique label/target/marker
and a complete independent proposal/review/assignment under current human
authority. Protected deletion and breaker reset remain human-only.
