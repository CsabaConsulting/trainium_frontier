# Independent review: trn22 warm c5c A0 disk safety stop

Reviewed: 2026-09-24 21:37 UTC. This review is of the retrieved terminal evidence,
not an approval to relaunch or reset a breaker.

## Evidence and integrity

- Attempt label: `fs-20260925-v2-c5c-a0-control-trn22-r1`; one-shot authorization
  request SHA-256 `4f1495c86ac236789a085d5f235a3d2c3550653c9690f49ff66bb596e3d17c8e`.
- [Terminal receipt](../../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json):
  SHA-256 `65b88b3ba08dc5a896da0d59e1cc065821ebcd95d4b209e71c958fad2caa1719`.
- Archived [postflight](../../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/run-logs/fs-20260925-v2-c5c-a0-control-trn22-r1.postflight.json):
  SHA-256 `72518b1ca9522626d35b73ce81d77780b3d1d15d6e5a1838ef52ef0878471ade`.
- Archived [boundary observer](../../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/observer/fs-20260925-v2-c5c-a0-control-trn22-r1.observer.json):
  SHA-256 `603139c587a0adef9c68b2139a4f541a9c3d154e177cf42885c53d1bfd18349e`.
- These three hashes and the `tmp-nki-dirs.tsv` SHA-256
  `b95e190c7e9b7ac699b785949327d54a06cfab250df75a35e6e141a52c891ff7`
  were recomputed locally after the Operator's evidence commit was integrated.

## Findings

The first below-floor observation was 6,449,438,720 free bytes, versus the immutable
6,979,321,856-byte (6.5-GiB) floor. The Operator's receipt records that it verified
the guard PID, UID, label and exact argv before sending SIGTERM to **that guard only**.
The guard finished at 21:28:07 UTC with return code 1. Postflight reports no remaining
owned processes, cleanup actions, deleted artifacts or caches; its only clean-host
failure is disk space. The observer has an empty boundary list and explicitly says
`COMPLETE_NON_SCORED`; there is no checkpoint, fresh public20M evaluation or BPB.
This is a **failed safety gate before step 1**, not a model-quality result.

Preflight-to-postflight root free space dropped from 7,949,975,552 to 6,449,033,216
bytes: 1,500,942,336 bytes consumed during compilation. The read-only inventory
found 673,388,155 apparent bytes of new retained debug cache and 826,409,712
apparent bytes in 48 new `/tmp/nki_*` directories. Their sum differs from the `df`
delta by 1,144,469 bytes. Apparent directory sizes are not identical to allocated
blocks, so this supports attribution but does not establish exact physical ownership.
The postflight kernel log also records Neuron reset requests that completed; no
causal hardware diagnosis is made from that log alone.

## Decision

**HOLD.** The attempt label is consumed and cannot be relaunched. trn22 remains
below the floor (latest 6,448,951,296 free bytes), its disk breaker remains tripped,
and trn21 is also below the floor. No training, S0 cold replay, table job, cache
deletion, AWS operation or breaker reset is authorized by this review. Human root
volume expansion and subsequent host verification are required; a new-label A0
retry additionally needs an updated compile-space projection, independent gate
review and human acknowledgement/reset of the trn22 breaker.
