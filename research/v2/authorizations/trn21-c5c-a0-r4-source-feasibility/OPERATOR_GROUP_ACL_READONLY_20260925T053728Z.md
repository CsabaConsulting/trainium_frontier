# trn21 group/ACL lock-ancestry follow-up — 2026-09-25 05:37 UTC

**Point-in-time read-only observation PASS; transfer dispatch HOLD.** This
separate follow-up resolves the POSIX-ACL unknown in the earlier
[trn21 lock receipt](OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T053029Z.md):
`/home/ubuntu/.cache` is owner UID:GID `1000:1000`, mode `0700`, with no
extended POSIX access or default ACL xattr (`ENODATA` on its opened
descriptor). Under the observed ordinary POSIX path permissions, a different
UID cannot traverse this `.cache` directory just by belonging to GID 1000,
so cannot reach its `0775` child through that pathname. This is a bounded
access-control inference, **not** a reviewed exception to the final-r3
writer's group-write rule. The r3 request remains disabled; its prior live
read-only probe returned HOLD on that `0775` child, and no transfer assignment
or writer was used.

The Operator used the existing trn21 isolated worktree, branching from
canonical `f852f771b2dcb06c4378238b20662747e60fac4e`. The pinned
strict-host-key endpoint was
`ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`, ED25519
fingerprint `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`,
known-hosts SHA-256
`e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`.
The frozen disabled request SHA-256 was
`3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e`.
Local global/trn21 STOP and breaker checks passed before and after; the
outside-Git 104154875-byte source capsule retained SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The fixed marker and journal remained absent. No private-key contents were
read or archived.

## No-follow path, ACL and membership facts

Each component was checked by no-follow name lookup and descriptor-relative
read-only open, with matching owner, device, inode, mode and link identity.
All listed paths were on ext4 device `66305`, not symlinks, and remained
stable through the diagnostic. Extended POSIX access ACL xattrs were absent
for all five; default ACL xattrs were absent for the four directories.

| Exact path | Type | UID:GID | Mode | Inode | Links |
| --- | --- | --- | --- | ---: | ---: |
| `/home` | directory | 0:0 | `0755` | 1684 | 3 |
| `/home/ubuntu` | directory | 1000:1000 | `0750` | 295711 | 167 |
| `/home/ubuntu/.cache` | directory | 1000:1000 | `0700` | 295736 | 6 |
| `/home/ubuntu/.cache/trainium_freshstart` | directory | 1000:1000 | `0775` | 596822 | 3 |
| `/home/ubuntu/.cache/trainium_freshstart/device0.lock` | regular, zero bytes | 1000:1000 | `0664` | 596823 | 1 |

Local `/etc/group` has one GID-1000 entry, `ubuntu`, with no explicit
members. Local `/etc/passwd` has only `ubuntu` (UID 1000) with primary GID
1000. Bounded NSS lookups `getent group 1000` and `getent passwd ubuntu`
returned the same facts with rc 0. The diagnostic process ran as effective
UID:GID `1000:1000`, supplementary GIDs `4,24,27,30,105,988,1000`.
This is not proof of exhaustive external NSS enumeration; it does not rule
out same-UID processes or privileged DAC override.

In the same mount namespace, `/proc/self/mountinfo` contained 30 entries;
none had a mount root or mountpoint text containing `/home/ubuntu`,
`/.cache`, or `trainium_freshstart`. Thus no explicit bind mount naming
these dirs was found, but this is **not** exhaustive proof against an
indirect alias, another mount namespace or other privileged access.
No ACL, permission, directory or lock metadata was changed. Unlike the
earlier receipt, this follow-up did not attempt flock.

## Bracketing safety and evidence

The exact final-r2 reviewed full **read-only** probe was used on both sides
because the r3 probe aborts on the existing group-write mode. Its remote
source SHA-256 was
`6b1cfe6b0b0d178a48bc881c389a2b2c3949736d7e9457c2aa4fa840581803c7`
and transport SHA-256 was
`d7d3b279929ae9cb764033178b5b48fd5aa8ded77e8e9111a6e1c148ddb03b58`.
Before/diagnostic/after were three separate pinned SSH sessions, all rc 0
with zero-byte stderr and no timeout. Full before/after receipts passed the
complete r2 host validator. Their stdout lengths/SHA-256 were 14,205/
`1cdc3993dc0c7eaf6cd689fd3582b46db255340f901054676de5df97fb23c958`
and 14,205/`7e9a3cdf35d19a267ec89a7667550935f30fdf68898df515bfe8cf59ee0731b1`;
the middle diagnostic was 6,090/
`74d6fd66f6e4050d5a4e9a979a9cb4c4518bfc50dda51255333b7588b41f3770`.

Host `ip-172-31-34-85`, UID 1000, boot ID
`1903f7f0-8e0d-4011-9566-346e4ba01a0f`, mount namespace and ext4
root aliases remained stable. Both full probes validated the protected
checkpoint/source/cache identities, old paths, absent STOPs and future
archive target, empty matching process and shared-lock/checkpoint holders,
Neuron idle, and zero science-label matches. Root free bytes moved from
`57,897,529,344` to `57,897,525,248` (4 KiB drift, cause not attributed),
while free inodes stayed at `31,563,301`; both exceed the request bound.

The [raw redacted transport and parsed receipts](OPERATOR_RAW_GROUP_ACL_READONLY_20260925T053728Z.json)
have SHA-256
`484ea4e76b6a59d5e5ec6757db6936b0bc3e7fc3ad89f69aa304a84da483443c`.
They preserve all three timestamps, return codes, redacted argv,
argv/remote-command hashes, stdout/stderr byte counts/digests/base64,
exact metadata/ACL errno, account facts and mountinfo filter. The exact
read-only diagnostic source is embedded there, SHA-256
`62665be616d0af2eeb8d93e0f1da27578e2993f278f9adb3cdec5478d1328a5e`;
it is the committed trn22 diagnostic template at SHA-256
`051e322a65f8dc86b746069943c4c14ab36319620eb5f1562a5424e6febcec31`
with only the recorded hostname and schema substitutions. No writer,
transfer, marker, journal, checkout, CPU G2, training, evaluation, cleanup,
breaker reset, AWS action or submission occurred. A future policy or writer
change requires separate implementation, independent review and fresh live
checks; this receipt grants none.
