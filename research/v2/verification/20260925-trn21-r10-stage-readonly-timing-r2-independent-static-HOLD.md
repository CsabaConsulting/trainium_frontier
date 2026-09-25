# Independent trn21 r10 read-only timing r2 review — static HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
disabled subject: `35a12bec0705c937d2a051d7a3830ddcb3d8d522` on
`origin/codex/20260925-trn21-r10-stage-readonly-timing-r2`, following r1
`abb4ef4d0ee048aedc499fe5406190d65a2cfb62` and its independent HOLD
receipt at `11f10f35` (byte-identical receipt is present at `252d5610`).
**HOLD before read-only enablement**: a forged, physically inconsistent
timing record can still be promoted to a positive observation. No SSH/host/
EC2 contact, operational raw receipt, marker, transfer, checkout, cleanup,
training, evaluation, AWS action or submission occurred in this review.

The r2 collector now rejects nonempty stderr and the r1 forged 9,999-second
ages. Its parser requires exact start host/source/target/bundle/label/lock
identity, pinned commits, boot/mount fields, disk/inode floors, three ordered
complete trials for a positive terminal, finite nonnegative elapsed values,
and finite scan/activity/label/space ages in `[0, 1.0]`. The stage r3 source,
stage operator, independent r3 PASS receipt, disabled request and remote
timing source remain SHA-pinned and unchanged. Exact package SHA-256:

| Item | SHA-256 |
| --- | --- |
| `collect_readonly.py` | `5de8433d35300aa2239021bed0d9115ff441cd4a02a3b953c785161e0ef2ff51` |
| unchanged `remote_timing.py` | `5a1298febfb67f6a61eb1f6fa02f974c7cbe15c6eb0bc221338b85d892be0db7` |
| `REQUEST.HOLD.json` | `9f2ed81e1c04decc16e0747d6e71f52037da759241ea1c8774dbbf790604d897` |
| `SHA256SUMS` | `b9790833ee9746a11d5d358de77fe9bec377358570dfe559a503cee33eadccf2` |
| pinned stage r3 remote/operator | `47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b` / `6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f` |

`sha256sum -c SHA256SUMS` passed **5/5**; `--check` verified all five
external pins and returned `DISABLED_NO_HOST_CONTACT`. Offline tests passed
**9/9** normal and **9/9** under `-O`; `git diff --check` passed. Direct
default and `--capture` calls in both modes exited nonzero HOLD before SSH
or output, because the separate enabled request/review/assignment are absent.
The unchanged remote timing source is clock-only, opens no bundle/target for
creation, and consumes no one-shot stage marker.

## Blocking ordered-JSONL witness

In the pinned remote code, `preopen_seconds` is measured from *before* the
first helper to *after* the clock-only would-open. The two label scans run
inside that interval, and `label_age_seconds` runs from their end to the
would-open. Therefore a true trial must have
`preopen_seconds >= label_scan_duration_seconds + label_age_seconds`
(allowing negligible floating-point tolerance); its sample order also gives
`label_age_seconds >= activity_age_seconds >= space_age_seconds >= 0`.
`collect_readonly.py` validates each field separately but never checks
these relations.

I supplied exact full-identity, empty-stderr, canonical ordered start/three
trial/end JSONL, changing only trial 1 to `preopen_seconds=0.0` while its
scan duration was `0.4`, label age `0.3`, activity age `0.2` and space age
`0.1` seconds. `_parse_jsonl()` accepted all five records, and a fully
mocked no-host `capture()` archived
`READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY`. A zero-second trial cannot
contain 0.4 seconds of scans plus 0.3 seconds since those scans. This is a
false-positive timing-feasibility path despite the repaired individual
bounds and terminal count.

Minimum repair: enforce the interval and sample-order relations for each
positive trial (with an explicit tiny numeric tolerance), add this exact
adversarial test at parser and capture level, rehash the collector/manifest,
and seek a fresh independent static review. Until then no timing contact is
authorized. Even a later clean timing observation will not prove actual
`O_EXCL` latency or authorize the stage writer/native experiment; separate
live, final-review and Supervisor gates remain.
