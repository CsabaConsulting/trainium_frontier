# trn21 r10 timing: separate read-only before/after snapshots — 2026-09-25

These are point-in-time observations around the separately assigned one-shot
read-only timing diagnostic. They do not authorize source staging, training,
evaluation, cleanup, breaker reset, or any AWS action. The timing diagnostic
returned HOLD at trial 1; its raw result and Operator terminal receipt are
on branch `codex/20260925-trn21-r10-timing-enabled-r3-r1` at `c238fe24`.

| Contact | UTC | Root free bytes | Free inodes | Process/Neuron/device/lock holder | STOP | e65 source |
| --- | --- | ---: | ---: | --- | --- | --- |
| Before | 11:09:46.815354 | 57,896,996,864 | 31,563,301 | none by bounded checks | none among 290 host paths | clean `e65a3097`, c5c train, pinned prepare |
| After | 11:10:35.726707 | 57,896,972,288 | 31,563,301 | none by bounded checks | none among 290 host paths | same clean source |

Both strict-SSH contacts returned 0, parsed the expected receipt schema,
were not timed out and had empty stderr. The pinned trn21 hostname, UID 1000,
boot ID and mount namespace matched across the pair. The active Desktop
controller-state SHA-256 was
`560daf07441133a14854bc108af484fb7a9c1f9f360379b830ca0bc85facb6d0`
both before and after each contact; its `launch_authorized` and both host
`launch_allowed` flags were false, all listed breakers untripped, and four
checked local STOP paths absent. The root-space difference was 24,576 bytes,
not attributed to the diagnostic; no host-write operation was run.

The reviewed snapshot does not explicitly stat the r10 bundle/target or A2
tools. The timing diagnostic's full remote preflight and trial record govern
its exact-path observations; this receipt cannot substitute for independent
raw-result review or a future fresh source-stage preflight. The consumed
timing result path and label must not be retried.

Raw redacted receipts: [before](OPERATOR_RAW_TRN21_READONLY_20260925T110941Z.json),
SHA-256 `694102330ebb1a41c9bc5d47589cb517593a50be9962bed2745d00921700dad4`;
[after](OPERATOR_RAW_TRN21_READONLY_20260925T111030Z.json), SHA-256
`38416da210cf9d3841033085dc98393279ea7bf779b2813108243dd09b74987f`.
