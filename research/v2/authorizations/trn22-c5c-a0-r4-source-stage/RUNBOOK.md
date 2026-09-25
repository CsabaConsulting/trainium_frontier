# trn22 c5c A0 r4 exact-source stage — Implementer HOLD plan

**No trn22 SSH, transfer, checkout, G2, monitor or native dispatch is
authorized by this package.** It is a local source-artifact build and a
non-dispatchable plan for independent review and later narrow assignments.
The approved r4 source is the frozen commit
`446b657695f086f9b484fa601c9a3b61c2495912`, not this packaging commit.
The r4 source-safety and disabled-package static PASS receipts are evidence of
their own narrow scopes; neither grants source transfer or a control run.

## Exact local artifact

| Item | Frozen value |
| --- | --- |
| Local, non-Git capsule | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz` |
| Capsule bytes / SHA-256 | `104154875` / `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| Git-tracked files at source commit | `6278`; no tracked symlink or submodule |
| Archive members | `6368`: `6305` regular files and `63` directories; four GNU long-name headers; `188917148` unpacked file bytes |
| Full member manifest | `MEMBERS.jsonl`, `1229617` bytes, SHA-256 `7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a` |
| Proposed **new** persistent trn22 target | `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz` on root ext4, mode `0600` |
| Proposed **new** checkout | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4` on root ext4, mode `0700` |
| Frozen scored source | `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |

The artifact is a private, detached, shallow **one-commit** Git checkout plus
its exact working tree. It was made from local ref
`codex/20260925-a0-trn22-r4-interlock` using `git clone --no-local --depth 1
--single-branch --no-tags`, then detached at the frozen commit and stripped of
the clone remote. The source clone passed `git fsck --full --strict`, HEAD and
all-ref reachable commit counts of one, no remote, empty porcelain status,
and an empty all-untracked check. All eleven source file hashes in the r4
disabled contract matched. The local builder used only Python's standard
library and local Git; it made no host/network request.

Every archive member is in the committed manifest with type, normalized
permission, byte count and per-file SHA-256. The builder checks path names,
tracked-file correspondence, regular-file types and single-link identity,
parses raw tar headers separately from Python's tar parser, rejects PAX,
links, devices, privileged modes, traversal, duplicates and missing parents,
and scans all payloads for five high-confidence credential-marker classes.
The build and a fresh-process `--verify-local` pass found **zero marker hits**.
That verifier rejects full capsule or manifest byte counts and SHA-256 values
other than the frozen values above; merely reporting a different hash is not
acceptance.
This is not a guarantee against every possible secret form: an independent
Verifier must repeat the scan and inspect the complete member list before
any transfer approval. No capsule bytes are committed to Git.

There was one **local builder** verification error, not a host action: the
first parser revision treated tar's root directory `./` as unsafe after the
capsule had been fully written. The parser was fixed to accept the sole root
directory spelling; `--finish-existing-local-capsule-from-private-clone`
revalidated the immutable clean clone, every raw and parsed member and every
payload hash **without reopening or overwriting the capsule**, then wrote
the previously absent manifest. A fresh `--verify-local` passed. This history
must be visible to the independent reviewer; do not silently relabel the
initial local test as a clean first attempt.

A disposable **local-only** replay through the separately reviewed trn21
safe-openat extractor unpacked all `6368` entries / `188917148` file bytes;
the extracted repository had exact HEAD `446b6576…`, shallow `true`, one
reachable commit, zero remotes, clean status including ignored untracked,
`git fsck` rc 0, and the expected train/prepare/launcher/guard/host hashes.
This Implementer test is not an independent review or proof about trn22.

## Gate 1: fresh read-only trn22 feasibility — **missing**

The last archived trn22 read-only prelaunch evidence (r2, checked
2026-09-24 23:39:25 UTC) reported 58,452,860,928 root-free bytes, but it is
**stale** and described source commit `e65a3097…`, not r4. An Operator, after
a separate read-only request, must newly record:

1. The pinned SSH host-key/identity, hostname `ip-172-31-32-253`, UID 1000,
   boot ID and mount namespace; real non-symlink `/`, `/home`, `/home/ubuntu`
   ancestry; the covering **root ext4** mount, parent device/inode and safe
   permissions. A transfer into host `/dev/shm` is forbidden: trn21's
   verified-in-process tmpfs copy vanished in the next SSH session without a
   reboot, and its original mount namespace was not recorded.
2. Both proposed new paths above are absent, as are the r4 output/log and
   science-label paths. Existing r1/r2/r3 source, cache and output paths are
   inventoried and left untouched. The r4 external monitor target remains a
   separate HOLD and must not be created in this stage.
3. Root free bytes and inodes. To leave the r4 **10-GiB prelaunch floor**
   after both a 104,154,875-byte capsule and 188,917,148-byte checkout, plus
   a conservative 2-GiB source-stage reserve, require at least
   **13,177,973,911 bytes free before transfer** and at least
   **10,737,418,240 bytes after each stage**. Require at least 10,000 free
   inodes before transfer. These are stricter source-stage planning gates,
   not a reduction of the 6.5-GiB operational floor or monitor threshold.
4. The retained trn22 checkpoint at
   `/home/ubuntu/trainium_freshstart_20260924_current_base_control_trn22_r1/out/fs-20260924-current-base-control-trn22-r1-work/train/out/final.pt`
   remains a regular file of 469,831,009 bytes, SHA-256
   `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`,
   device 66305/inode 2417091, unless a separately approved asset change is
   documented. The six warm-cache children `neff`, `local`, `hlo`,
   `inductor`, `compiler`, `debug` under the historical trn22 35c9 control's
   `caches/train` root retain device 66305 and respective inode values
   2430420–2430425. These are *historical expectations*, not fresh proof.
5. No matched train/eval/G2/guard/launcher process, Neuron holder, shared
   host-lock holder, STOP marker or open breaker. A failed/ambiguous check is
   HOLD; no deletion, cache cleanup, breaker reset or AWS action may make it
   pass.

## Gate 2: reviewed transfer-only request and assignment — **missing**

After independent artifact/manifest/plan review **and** fresh feasibility,
the Supervisor may separately issue a one-attempt **transfer-only** assignment
for exactly the local capsule SHA/bytes and the absent ext4 target above. A
different Operator must use an independently reviewed, hash-bound writer and
literal SSH argv with strict host-key checking. The writer must open only
the expected `/home/ubuntu` parent and target via no-follow descriptors and
`O_CREAT|O_EXCL`, stream exact bytes to a private `0600` single-link regular
file, fsync file and parent, then re-read SHA/bytes and record host/boot/mount,
parent and target identities. No shell redirection, in-place patch, old
checkout, `/dev/shm` host target, retry, cleanup, rename or overwrite.

**Same-process verification is insufficient.** A later distinct SSH process
must re-open the target read-only with `O_NOFOLLOW` and rehash it; a final
distinct read-only pass after all first verifications must repeat the same
check and a full postflight. All three `(PID,start_ticks)` pairs, exact SSH
argv hashes, transport stdout/stderr/exit, target device/inode/mode/size/SHA,
host boot/mount namespace and parent identity must be durably archived. If
any check fails, preserve the unique partial ext4 target and stop. A fresh
path and new reviewed authority would be needed; the failed path is not
reused. A distinct Verifier reviews the raw transport evidence before any
checkout is considered.

## Gate 3: separately assigned checkout — **missing**

Transfer PASS still does not authorize extraction. A new checkout-only
request must pin this capsule and manifest, the exact checked-out code and
full argv; get separate independent review and Supervisor one-shot authority;
and repeat the full read-only trn22 preflight immediately before attempting
the absent checkout path. Use a reviewed no-follow, exclusive extraction
procedure, not a host `tar -x` shortcut: scan the complete compressed SHA,
raw tar headers, all `6368` manifest members and payload SHA values before
creation; reject PAX/link/device/traversal/duplicate/secret-like paths;
create only new owner-private `0700` directories and `0600`/`0700` files
with descriptor-relative `O_EXCL|O_NOFOLLOW`; never apply archive ownership.
Rehash the same persistent capsule descriptor after extraction. A failure
leaves any partial checkout untouched, consumes that path, and emits no
success receipt.

The stage success receipt must prove exact detached HEAD
`446b657695f086f9b484fa601c9a3b61c2495912`, shallow one-commit Git,
no remote/alternates, `git fsck`, clean porcelain and all-untracked checks,
all eleven frozen source hashes, absent science output, intact retained
checkpoint/six caches, same host/boot/mount/target inode and ≥10 GiB root
free. The separate Operator wrapper must archive a **read-only full
postflight even on failure**, without cleanup or same-path retry. A distinct
Verifier then reviews the retrieved host checkout receipt.

## Later gates — all HOLD

The old trn22 four-rank CPU G2 was run against `e65a3097…`, not this r4
`446b6576…` source. It cannot be silently rebound: a new exact-source G2 or
an independently approved source-identity disposition is required. The
disabled r4 request/base/pair bindings remain false/static and cannot be
flipped in place. After clean staging/G2, a newly reviewed dispatchable
request and matching bindings, paired monitor plus boundary-observer live
readiness, fresh full host preflight, and exact one-label/one-attempt
Supervisor authority are still required before native A0. No experiment
score, adoption or submission follows from this source package.
