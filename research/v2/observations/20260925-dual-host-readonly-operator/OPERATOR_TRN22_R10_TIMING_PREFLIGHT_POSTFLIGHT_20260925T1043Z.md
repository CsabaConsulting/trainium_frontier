# trn22 r10 timing: separate read-only before/after snapshots — 2026-09-25

These are point-in-time observations around the separately assigned, one-shot
read-only timing diagnostic. They do not authorize source staging, training,
evaluation, cleanup, breaker reset, or an AWS action. The timing result itself
is on branch `codex/20260925-trn22-r10-timing-enabled-r4-r1` at `23d704cf`;
it awaits independent raw-result review.

| Contact | UTC | Root free bytes | Free inodes | Process/Neuron/device/lock holder | STOP | e65 source |
| --- | --- | ---: | ---: | --- | --- | --- |
| Before | 10:41:43.821363 | 58,455,560,192 | 31,660,021 | none by bounded checks | none among 262 host paths | clean `e65a3097`, c5c train, pinned prepare |
| After | 10:43:20.774412 | 58,455,519,232 | 31,660,021 | none by bounded checks | none among 262 host paths | same clean source |

Both strict-SSH contacts returned 0, parsed the expected receipt schema,
were not timed out, and had empty stderr. The pinned trn22 hostname, UID 1000,
boot ID and mount namespace matched across the pair. The local active
controller-state SHA-256 was
`e4fcbd7b6d1340d517f0d40b5de7092bdc184390a7466a5373375c122bc1d896`
both before and after each contact; its `launch_authorized` and both host
`launch_allowed` flags were false, all listed breakers untripped, and the
four checked local STOP paths absent. The root-space difference was 40,960
bytes, not attributed to the diagnostic; no host-write operation was run.

The reviewed `collect_snapshot.py`/`remote_snapshot.py` pair did not
explicitly stat the r10 bundle/target or A2 tools. The timing diagnostic's own
separate pre/post probes and trial records hold its exact-path observations;
this receipt does not substitute for their independent review or a later fresh
source-stage preflight.

Raw redacted receipts: [before](OPERATOR_RAW_TRN22_READONLY_20260925T104138Z.json),
SHA-256 `72423a9a529b9d316b72c67f596e01ab4d4ef25c82bd3c5cc40df4d180f1d789`;
[after](OPERATOR_RAW_TRN22_READONLY_20260925T104315Z.json), SHA-256
`bd4c5415f9c4cc9bac92509142278204b68a728c4a1e8ec9c1fd4cca4c71ff5b`.
