# trn22 group/ACL lock-ancestry observation — 2026-09-25 05:34 UTC

**Point-in-time read-only observation PASS; transfer dispatch HOLD.** The
existing `/home/ubuntu/.cache` is owned by UID:GID `1000:1000`, mode `0700`,
and has no extended POSIX access or default ACL xattr (`ENODATA` on its opened
descriptor). Under the observed ordinary POSIX path permissions, a different
UID cannot traverse this `.cache` directory merely by belonging to GID 1000;
that blocks access through this pathname to its `0775` child. This is a
limited access-control inference, **not** a policy exception or source-transfer
approval. The earlier r3 writer remains under the independent late-label
HOLD, and the newer disabled lock repair has an independent static HOLD.

The Operator used the existing isolated worktree, a new evidence branch from
canonical `ee78ad6c0f980a6a99e9558651d99fc2998a6351`, and only the
pinned trn22 SSH endpoint
`ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com`.
The exact known-hosts SHA-256 was
`e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`
and ED25519 fingerprint
`SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`.
Local global/trn22 STOP and breaker gates were clear before and after; the
outside-Git 104154875-byte capsule retained SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
No transfer marker or journal was created. Private-key contents were neither
read nor archived.

## No-follow path, ACL and account facts

Every component was inspected by no-follow name lookup and a read-only,
descriptor-relative open, with matching owner, device, inode, mode and link
identity. All listed paths were on device `66305`, were not symlinks, and
remained stable through the diagnostic. The POSIX ACL access xattr was absent
for `/home`, `/home/ubuntu`, `.cache`, the child directory and `device0.lock`;
the directory default ACL xattr was absent for all inspected directories.

| Exact path | Type | UID:GID | Mode | Inode | Links |
| --- | --- | --- | --- | ---: | ---: |
| `/home` | directory | 0:0 | `0755` | 1684 | 3 |
| `/home/ubuntu` | directory | 1000:1000 | `0750` | 295711 | 145 |
| `/home/ubuntu/.cache` | directory | 1000:1000 | `0700` | 295736 | 6 |
| `/home/ubuntu/.cache/trainium_freshstart` | directory | 1000:1000 | `0775` | 832588 | 3 |
| `/home/ubuntu/.cache/trainium_freshstart/device0.lock` | regular, zero bytes | 1000:1000 | `0664` | 832589 | 1 |

Local `/etc/group` has one GID-1000 entry, `ubuntu`, with no explicit
members; local `/etc/passwd` has only `ubuntu` (UID 1000) as primary GID 1000.
Bounded NSS lookups, `getent group 1000` and `getent passwd ubuntu`, returned
the same facts with rc 0. The remote diagnostic process had effective
UID:GID `1000:1000` and supplementary GIDs `4,24,27,30,105,988,1000`.
These checks do not prove that every possible external NSS principal was
enumerated or exclude another process using UID 1000 or privileged DAC
override.

The same mount namespace exposed 30 `/proc/self/mountinfo` entries; none had
a mount root or mountpoint text containing `/home/ubuntu`, `/.cache`, or
`trainium_freshstart`. This finds no explicit bind mount naming the observed
path, but does not prove that no alternate namespace, indirect mount, or
other privileged path exists. No host permissions or ACLs were changed.

## Bracketing safety and evidence custody

The exact reviewed r3 **read-only** full probe source SHA-256 was
`e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086`;
transport SHA-256 was
`1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51`.
Its distinct before/after SSH sessions both returned rc 0, zero-byte stderr,
and passed the full host validator. The middle read-only ACL diagnostic also
returned rc 0 and zero-byte stderr. All three used strict pinned host-key
options, `/dev/null` SSH config, no stdin, `python3 -B`, and no writer path.
Raw stdout lengths/SHA-256 were 14,467/
`89a4259bbc616c848ec25ebaecf0ef72d4ecf04c2c9d3ec091c4cdedb7f781fd`,
6,092/`d4fc3c6c827c9d0909f2fb2aa6718704800c0879a1e138a669c51e21d8a595f5`,
and 14,467/`f84c63e461c288635179725a434ea9368eaa35d2eb01d75df508fca878b8b4f5`,
respectively.

Before and after, host `ip-172-31-32-253`, UID 1000, boot ID
`bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6`, mount namespace, root ext4
aliases, source/protected checkpoint/six-cache identity, old-path inventory,
STOP absence, absent future target, empty process/holder/Neuron activity and
zero science-label matches all validated. Root free bytes moved from
`58,453,839,872` to `58,453,831,680` (8 KiB drift, cause not attributed),
while free inodes stayed at `31,660,018`; both free-byte observations exceeded
the request's bound. The previously observed group-writable mode mismatch in
the disabled lock repair remains **HOLD** despite this access observation.

The [raw redacted transport and parsed receipts](OPERATOR_RAW_GROUP_ACL_READONLY_20260925T053454Z.json)
have SHA-256
`753cd90daee7ab595383cff8b1f02b79766986f452bb6a3521dde5065b88e112`.
They retain all three return codes, timestamps, argv/remote-command hashes,
stdout/stderr byte counts, digests and base64, exact metadata/ACL errno,
account facts and mountinfo filter results. The committed
[one-off read-only diagnostic source](OPERATOR_GROUP_ACL_READONLY_DIAGNOSTIC_20260925.py)
has SHA-256
`051e322a65f8dc86b746069943c4c14ab36319620eb5f1562a5424e6febcec31`.
No writer, flock, transfer, marker, journal, checkout, CPU G2, training,
evaluation, cleanup, breaker reset, AWS action or submission occurred.
