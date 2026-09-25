# trn21 existing-lock read-only feasibility — 2026-09-25 05:30 UTC

**Disposition: HOLD for the disabled final-r3 transfer candidate.** The narrow
read-only test proved that the existing lock could be opened without following
symlinks and briefly acquired/released with `LOCK_EX|LOCK_NB` on an `O_RDONLY`
descriptor. That does **not** clear the candidate's stricter trust policy:
the live lock parent is `0775` and the lock file is `0664`; the exact r3
read-only probe returned rc 1 at `unsafe host lock directory:
trainium_freshstart`. The frozen r3 request independently has
`dispatch_authorized=false` and `source_transfer_authorized=false`, and its
fixed review/assignment files are absent. This receipt is point-in-time
feasibility, **not transfer or policy-change authority**. Prior r1/r2 HOLDs
remain controlling.

## Exact observation and limits

The pinned strict-host-key endpoint was
`ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`, ED25519
fingerprint `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`, with
known-hosts SHA-256 `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`.
The request SHA-256 was
`3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e`.
Local global/trn21 breakers and STOP checks were clear; the pinned
104154875-byte local capsule SHA-256 was
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The fixed final-r3 marker, transfer journal, final-r3 review and assignment
were absent. No marker, journal, host file, or permission was created or changed.

All rows below came from descriptor-relative no-follow name checks and
read-only opened descriptors on the same ext4 device `66305`. The raw record
includes the exact UID/GID, mode, device, inode, link count, symlink status,
size, and timestamps for both named and opened objects.

| Existing path | Type | UID:GID | Mode | Inode | Links | Symlink |
| --- | --- | --- | --- | ---: | ---: | --- |
| `/home/ubuntu/.cache` | directory | 1000:1000 | `0700` | 295736 | 6 | no |
| `/home/ubuntu/.cache/trainium_freshstart` | directory | 1000:1000 | `0775` | 596822 | 3 | no |
| `/home/ubuntu/.cache/trainium_freshstart/device0.lock` | regular, zero bytes | 1000:1000 | `0664` | 596823 | 1 | no |

The remote Ubuntu process had effective UID/GID `1000:1000`. Local
`/etc/group` has one GID-1000 record, `ubuntu`, with no explicit members;
local `/etc/passwd` has one primary-GID-1000 account, `ubuntu` (UID 1000).
Those are **local file facts only**, not a full NSS/directory-service
enumeration. POSIX ACLs on `.cache` and its descendants were not queried, so
any non-owner traversal grant via ACL is **unknown**; no alternate bind-mount
or other path to the same lock parent was investigated. These limits prevent
inferring an approved trust-policy exception from group membership alone.

The read-only `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` lock open succeeded;
nonblocking exclusive flock was acquired and released, held for 1,402 ns.
The lock's device/inode/owner/mode/links/size/atime/mtime/ctime were identical
before and after. This was an observation, not a persistent lock reservation.

## Bracketing host checks and raw evidence

The reviewed final-r2 **read-only probe** (used because r3 aborts before a
full snapshot on this mode) returned rc 0 before and after the narrow test;
both complete receipts passed its validator. Hostname
`ip-172-31-34-85`, UID 1000, boot ID
`1903f7f0-8e0d-4011-9566-346e4ba01a0f`, mount namespace, ext4
`/dev/root` `259:1` aliases, and parent identity were stable. Root free
bytes/inodes were `57,897,541,632`/`31,563,301` in both full probes, above
the request's bounds. Protected checkpoint/source/cache identities passed,
old paths were inventoried, STOPs clear, no matching processes or checkpoint
or shared-lock holders were seen, Neuron showed no non-diagnostic PIDs, the
science-label scans were empty, and the future archive target remained absent.
The exact r3 read-only probe was also attempted between them; its rc 1,
zero-byte stdout and 554-byte stderr are retained, including the exception
`RuntimeError: HOLD: existing host-wide device0.lock absent or unsafe`.

Raw [transport and host evidence](OPERATOR_RAW_LOCK_READONLY_20260925T053029Z.json)
SHA-256:
`687b4d7b57c4de041adad37352a385e15ead7cfba0b83c4d3a3df5bd12058d14`.
It records all four exact phase timestamps, return codes, redacted SSH argv,
argv and remote-command hashes, stdout/stderr lengths, SHA-256 and raw base64,
parsed full receipts, the lock diagnostic, and validation results. The
one-off [read-only diagnostic source](OPERATOR_EXISTING_LOCK_READONLY_DIAGNOSTIC_20260925.py)
SHA-256 is recorded in that raw file. The actual probe source hashes were
`6b1cfe6b0b0d178a48bc881c389a2b2c3949736d7e9457c2aa4fa840581803c7`
(r2) and `3a738cff02b26824ea0be1fb50ee16b6cacce96941a9e2162506c7cf608ead32`
(r3). No writer, transfer, checkout, CPU/native run, evaluation, cleanup,
breaker reset, instance action, or submission occurred. Any future proposal
requires a separate reviewed policy/implementation and an immediate fresh
preflight; this snapshot cannot authorize a write.
