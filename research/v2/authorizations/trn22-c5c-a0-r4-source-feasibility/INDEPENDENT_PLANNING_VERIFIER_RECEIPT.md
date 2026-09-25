# Independent Verifier: trn22 r4 read-only feasibility snapshot

**Verdict: PASS for the internal consistency of this point-in-time, planning-only
snapshot. Exact transfer preflight and all host writes remain HOLD.** The
Operator's observation was at `2026-09-25T02:42:28.357805+00:00`; this
offline review was at `2026-09-25T02:47:10Z`. I did not SSH to trn22, inspect
secrets, transfer a file, check out source, run G2, train, evaluate, or touch
AWS. This review is neither a fresh pre-transfer preflight nor a Supervisor
one-shot assignment.

The reviewed Operator commit is
`83047a1411a9559b7894de0bd59095ee38d83b66` on
`origin/codex/trn22-r4-source-feasibility-20260925`, parent
`af50558bafea0830a7474acbeba5b7e8f0bb7c80`. It adds only the three
files below, and `git diff --check` passed. The separately reviewed r4 source
stage remains a HOLD-only package at
`4570f4e87d5e052a6d8e402326e146384759b8cf`; its
`GATE_CONTRACT.json` SHA-256 is
`79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`.
The Operator branch is not a descendant of that package, so the contract was
loaded by exact commit and hash, not inferred from the Operator checkout.

| Operator evidence file | SHA-256 |
| --- | --- |
| `remote_readonly_probe.py` | `fe6ace073432444cd2848fdcdf5f55a3e6652463446eacaba310251c61e7fc38` |
| `HOST_SNAPSHOT_20260925T024228Z.json` | `eb43d2100a933c2f4684c106d867d61014df057342168d4b9b861fd171051981` |
| `OPERATOR_READONLY_FEASIBILITY_20260925T024228Z.md` | `15d667846e0ec3c9bff442c4adc8a3cffd08d668cc4368d54605c57152d13ff1` |

## Offline checks

The committed probe uses metadata reads, read-only file hashing, bounded
`find`, process categorization without printing command lines, `fuser`
without a kill option, and `neuron-ls -j -a`. It has no host file-creation,
deletion, overwrite, signal, or AWS operation. The archived snapshot is
parseable JSON, and all 15 structural checks against the exact stage contract
and probe-derived target list passed:

- Hostname `ip-172-31-32-253`, effective UID `1000`, boot ID
  `bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6`, mount namespace
  `mnt:[4026531832]`. `/` and `/home/ubuntu` cover the same root ext4 mount
  (`/dev/root`, mount ID 27, major:minor `259:1`). `/`, `/home`, and
  `/home/ubuntu` are non-symlink safe directories on device 66305; the
  proposed parent is UID 1000, inode 295711, mode 0750.
- Root free `58,453,950,464` bytes and `31,660,018` inodes exceed the
  contract's pre-transfer `13,177,973,911`-byte and 10,000-inode thresholds.
  The contract's post-stage 10-GiB floor remains a later live check.
- The exact new archive, checkout and held monitor paths are absent, as are
  seven related output/log/observer targets. All 12 scoped host STOP paths
  are absent. The bounded same-filesystem label scan returned rc 0, empty
  stderr, zero matches and no truncation.
- The retained checkpoint's device/inode (`66305/2417091`), UID 1000,
  single-link regular type, `469,831,009` bytes and SHA-256
  `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`
  match the contract; all six protected cache child names, devices, UIDs and
  pinned inodes `2430420`–`2430425` match. The checkpoint, shared host lock,
  and `/dev/neuron0` each show `fuser` rc 1, empty stderr and zero PIDs.
- No matched training/evaluation/G2/guard/launcher process and no unreadable
  process command line were reported. `neuron-ls -j -a` returned rc 0,
  parseable JSON, empty stderr and zero non-diagnostic PIDs.

The receipt pins controller base `af50558b` and `research/v2/state.json`
SHA-256 `2edf7660514dd8b288c4cb8a1ec7cc96113fb74d998fe05f831c120effc28040`.
I recomputed that exact state blob's hash and confirmed
`launch_authorized=false`, trn22 `launch_allowed=false`, and the global
correctness and trn22 breakers untripped there. This verifies the archived
controller version, not its live state after the observation. The Operator's
statement that local STOP files were absent is not independently provable
from a committed Git tree alone.

The historical contract listed root source `/dev/nvme0n1p1`, whereas this
`mountinfo` snapshot reports `/dev/root` on major:minor `259:1`. Those names
may alias the same root device, but the archived snapshot does not prove the
alias. This is not a mismatch with the contract's *fresh* ext4/root-parent
gate, which is supported by the matching mount ID and device, but an eventual
pre-transfer preflight should reconcile the block-device identity rather than
copy either source string uncritically.

I checked the local known-hosts entry **without printing its key**: its
ED25519 fingerprint is
`SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`, matching the receipt.
The referenced private-key path and known-hosts file were each regular,
single-link, user-owned, mode-0600 files; only metadata was inspected. The
probe source and curated JSON are archived, but the exact SSH argv, raw
transport stdout/stderr and exit are not separately archived. Therefore I
cannot independently prove the Operator's claimed SSH flags or that this
JSON was the exact, unmodified transport stdout. The pin supports planning
confidence only; a later transfer must record a literal reviewed argv and
full sanitized transport evidence.

The stage RUNBOOK's Gate 1 also calls for an inventory of existing r1/r2/r3
source, cache and output paths. This snapshot authenticates the specifically
protected checkpoint and six warm-cache directories, but does **not** provide
a complete fresh inventory of those old path trees. That gap must be closed
in the future immediate full pre-transfer preflight; it cannot be silently
called done by this planning PASS. Host conditions can change at any time. Recheck exact
path absence, old assets, host/boot/mount/parent, locks/processes/STOP/breakers,
space/inodes, and the local capsule identity immediately before an authorized
one-attempt transfer. No transfer writer, assignment, checkout, G2, monitor,
native run, evaluation, or submission is authorized by this receipt.
