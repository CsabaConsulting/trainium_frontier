# trn22 r4 read-only refresh — point-in-time HOLD

Observed 2026-09-25 04:18:22–04:21:26 UTC. This Operator performed three
read-only SSH processes and local hash/authority checks only. No host path,
one-shot marker, transfer journal, checkpoint, cache, AWS resource, breaker,
job or leaderboard state was changed. **This snapshot is not a transfer
preflight or transfer authority.**

## Current gate and exact source

- The canonical controller was at `c1b590c31b9db765b1f5673aefcbc8d769162b9f`.
  Its new continuity final r2 request SHA-256 is
  `99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5`.
  Its `dispatch_authorized: true` field is request intent only. The fixed
  independent final PASS and Supervisor assignment files were absent before
  and after the probes; so were the fixed one-shot marker and raw transfer
  journal. Local controller/worktree STOPs were absent and global/trn22
  breakers untripped; controller state SHA-256 was
  `2c125f15b42ba1185185c2ce88a0bf15137612a4d1f467302ad7d13298921771`.
- The outside-Git source capsule remained a regular, one-link mode-0600 file
  of **104,154,875 bytes**, SHA-256
  `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
  This rehash did not transfer or extract it.
- Before each SSH process, the existing known-hosts file rehashed to
  `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`
  and its one matching ED25519 entry had fingerprint
  `SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`. The private key was
  checked by metadata only. The archived redacted argv shows `-F /dev/null`,
  strict host-key checking, pinned known-hosts path, batch mode and the exact
  endpoint; its SHA binds the full literal argv and source payload.

## Host observations

The unmodified earlier feasibility probe returned at 04:20:06 UTC, and a
separate read-only diagnostic returned at 04:21:26 UTC. Both observed
`ip-172-31-32-253`, UID 1000, boot ID
`bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6`, mount namespace
`mnt:[4026531832]`, and root ext4 mount ID 27, major:minor `259:1`, source
`/dev/root`. The diagnostic showed that both `/dev/root` and
`/dev/nvme0n1p1` are block nodes with kernel device `259:1`; the matching
`/sys/dev/block/259:1` link resolves under `/sys/devices` to `nvme0n1p1`.
The real, non-symlink `/home/ubuntu` parent remained UID 1000, device 66305,
inode 295711, mode 0750; `/` and `/home` were also safe directories.

Root free space was **58,453,884,928 bytes** and **31,660,018 inodes** at the
04:20 probe, above the planned pre-transfer floors of 13,177,973,911 bytes
and 10,000 inodes. The exact new r4 capsule target and checkout, monitor
target, scoped r4 output/log/step/observer paths, and exact science label
were absent. Twelve scoped host STOP paths were absent. The 41-entry old-path
diagnostic inventory (38 transfer-package paths plus the three `.git`
parents) found r1 and r2 source checkouts present, the r3 checkout absent,
and the older `/dev/shm` r3 G2 directory present. All were left untouched.

The protected checkpoint remained regular, single-link, UID 1000, device
66305, inode 2417091, **469,831,009 bytes**, with full-read SHA-256
`1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`.
All six protected warm-cache **directory identities** matched the frozen
device 66305/UID 1000/inode pins: `neff` 2430420, `local` 2430421,
`hlo` 2430422, `inductor` 2430423, `compiler` 2430424 and `debug` 2430425.
The contract pins these directory identities, not cache-content hashes.
There were zero matched or unreadable train/eval/G2/guard/launcher process
command lines, zero checkpoint/shared-lock/`/dev/neuron0` fuser holders,
and zero non-diagnostic Neuron process IDs. `neuron-ls -j -a` parsed and
returned rc 0; all three fuser checks returned rc 1 with empty stderr.

## Blocking reviewed-probe defect

The first SSH process invoked the **unchanged reviewed**
`stage_r4_remote.py` `probe` operation with Python bytecode writes disabled.
It exited 1 with no stdout. The sanitized raw traceback ends at
`_old_inventory` → `_safe_path`, line 281:

`NotADirectoryError: [Errno 20] Not a directory: '.git'`

The supplemental inventory pinpoints the path class:
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2/.git`
is a regular Git worktree link file, so the fixed
`r2/.git/HEAD` child cannot be opened as a directory. This is a blocker for
the current repaired transfer probe on the observed host state. The two
successful probes supply **limited planning observations only** and cannot
replace its reviewed complete preflight. A separate Implementer repair,
independent review, missing final PASS/assignment, and a fresh immediate
full host preflight are required before any future write.

## Raw redacted evidence

All three JSON files archive the redacted literal SSH argv, full-argv SHA,
source-payload SHA, exit code, UTC times, raw stdout/stderr byte counts and
SHA-256, credential-screened redacted stdout/stderr, local before/after
authority state, and any parsed host receipt. No key contents, environment
values or process command lines were archived.
The reviewed probe's full-argv SHA-256 is
`1020cea6f0a5f7c8a2e32c4f6a4aca4d04ae5d34f6be5bd6e1386117864e5f73`.
The two stdin-script probes share argv SHA-256
`e47416fce4490a021d4d410a369c4d4bc2990529f0374925601832b1654c8ed1`;
their distinct source SHA-256 values are bound separately in their records.

| Process | Raw evidence | SHA-256 | SSH result |
| --- | --- | --- | --- |
| Reviewed repaired probe, failed | `OPERATOR_RAW_READONLY_20260925T041822Z.json` | `2920278dd8d4f113fd9ae96204b6c702d9ae3b3784cfee8b033cf8970eadc960` | rc 1; 296-byte sanitized traceback |
| Unmodified earlier feasibility probe | `OPERATOR_RAW_READONLY_LEGACY_20260925T042001Z.json` | `a62f1992c3c5ed766560b8d8d877ba5e14cab174307b10960e26d153bf740cc1` | rc 0; empty stderr |
| Supplemental read-only aliases/old paths | `OPERATOR_RAW_READONLY_SUPPLEMENTAL_20260925T042121Z.json` | `cd1174a411fe7f43acfc23cf158967e413a7f26f2bb55bc915d003863acbe934` | rc 0; empty stderr |

The local evidence collector and supplemental diagnostic source are archived
alongside the records. No reviewed transfer or probe source file was edited.
