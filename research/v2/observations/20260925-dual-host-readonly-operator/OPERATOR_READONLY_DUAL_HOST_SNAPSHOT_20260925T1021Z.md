# Operator dual-host read-only status snapshot — 2026-09-25 10:21 UTC

**Scope:** point-in-time observation only. This is not a launch or source-transfer
preflight, G2/native PASS, one-shot assignment, breaker reset, or authority to
train, evaluate, clean, stage, or submit. The reviewed `collect_snapshot.py`
streamed the reviewed `remote_snapshot.py` over pinned strict-host-key SSH to
each host. Both contacts returned 0, parsed the expected schema, had empty
stderr, and suppressed no raw stream for a sensitive marker. There were no
remote writes or signals.

| Host | Observed UTC | Root free bytes | Free inodes | Matched jobs / unreadable cmdlines | Neuron/device/lock holders | Host STOPs |
| --- | --- | ---: | ---: | --- | --- | --- |
| trn21 | 10:21:22.359703 | 57,897,144,320 | 31,563,301 | none / 0 | none; one Neuron device; both `fuser` checks rc 1/no PID | none among 290 checked paths |
| trn22 | 10:21:22.435877 | 58,455,601,152 | 31,660,021 | none / 0 | none; one Neuron device; both `fuser` checks rc 1/no PID | none among 262 checked paths |

Host identities matched the previously pinned hostnames and UID 1000. Root
filesystems are ext4, mount `/dev/root` with major:minor `259:1`; the
`/dev/root` and `/dev/nvme0n1p1` aliases both resolve to that device.
The existing host locks were present but had no observed holder; no lock was
acquired. The local controller state SHA-256 was
`af84c96ff0e3c46fe92e2accfbe9e398cef52e72762078ca94ec0551cf9c8a59`
both before and after contact: `launch_authorized=false`, both hosts'
`launch_allowed=false`, all three listed breaker `tripped` fields false,
and no local STOP among the four exact checked paths.

Both existing e65 c5c A0 source checkouts had clean Git status at
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`. Each had `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
Their launcher hash remained the older
`822d4317408a09ee0dd6ab3eaaa4f706f0a173741659ea5e47cc12e45e4f1bef`.
The exact old r4 target path was absent on each host. The bounded trn22
`/home/ubuntu` name inventory discovered no proposed r10 target, but this
probe did not explicitly stat r10. The trn21 r10 target was not in this
probe's name filter, so its presence is **unknown**, not absent.

The current local one-attempt marker paths named in the disabled r10 stage
requests were inspected without reading their contents and were absent:

- `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn21-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl`
- `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r10-incremental-stage-r1.one-attempt.jsonl`

The reviewed snapshot did **not** inspect the exact A2 G2 tool paths
`workflow/tools/preflight_dryrun.py` and `workflow/tools/contained.py` inside
the e65 checkout, nor exact r10 targets or provisional bundle paths. Those
remain unknown from this receipt and require a separately reviewed exact-path
read-only probe before any A2 or r10 stage decision. A clean e65 checkout is
not a native-launch source; no launch has been authorized by this observation.

Raw redacted receipts: [trn21](OPERATOR_RAW_TRN21_READONLY_20260925T102117Z.json),
SHA-256 `48941e2fc733f2d13d09594e21d34afa396cf7ad5a881750bc07568d112df474`;
[trn22](OPERATOR_RAW_TRN22_READONLY_20260925T102117Z.json), SHA-256
`4c12aa5370519c4d43107c02da87a525dd7bf217152e4129c0f3035a126c7773`.
The streamed remote source SHA-256 was
`4a8b8ed413240c229ce943dd9bd8d6781bbddf7c348aa76b5b496f5c5d14f70a`.
