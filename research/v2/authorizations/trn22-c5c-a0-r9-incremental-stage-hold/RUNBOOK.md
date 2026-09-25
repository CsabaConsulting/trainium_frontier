# trn22 e65 → r9 incremental source stage: disabled Implementer package

**Disposition: HOLD; source staging only.** This package does not contain an
enabled request, independent final source-stage PASS, Supervisor one-attempt
assignment, native launcher approval, evaluation authority, or training
authority. `operator_once.py` exits HOLD before a marker or SSH today. No
trn22 host contact, transfer, training, cleanup, or AWS action was performed
while preparing it. The held full-capsule transfer writers are not reused.

## Exact local subject

| Item | Value |
| --- | --- |
| Accepted trn22 linked source | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2` at `e65a3097a5b00f6178188a935bb2acb9a5042c6c` |
| Distinct absent target | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r9_stage_r1` at `e046de16826bfc3b954136e3c60804aea0554ecc` |
| One-attempt label | `fs-20260925-trn22-c5c-a0-r9-incremental-stage-r1` |
| Incremental bundle | `e65-to-r9.bundle`, 101,237 bytes, SHA-256 `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |
| Remote stage engine | `remote_incremental.py`, SHA-256 `0de28ddea4e015cc32fdb0cd50433624c51c810a0b67c07a1cbe6f2d8f17f940` |
| Local one-shot entry | `operator_once.py`, SHA-256 `d4eaeb4abde4690017029628dab48130209b695f0a0b8ef2ed8404f5bc29674c` |
| Disabled contract | `REQUEST.HOLD.json`, SHA-256 `02d5f30416911c2bf52192563997733c98a1a4f51dbe4746913e468a202f1299` |
| Offline regression tests | `test_incremental_stage.py`, SHA-256 `eb3a9a7578433602b7b3c23eba869b93148286e0f8596490e652cddcfba6eff6` |

`git bundle verify` confirms only the target HEAD and e65 prerequisite. The
bundle was generated as the exact `e65..r9` Git delta, not a full repository
capsule. Local object inventory: 131 objects (14 commits, 57 trees, 60
blobs), 117 named paths; SHA-256 of sorted path names joined by newline is
`2e06bfa35779c4319755d1aa8392cc4e8077b9e680ef66a9e0d2f7306774756a`.
A bounded offline scan found zero `.env`/private-key/credential-like path hits
and zero PEM-private-key, AWS access-key-ID, GitHub, Hugging Face, or OpenAI
token-signature hits in the 60 new blobs. This is not a proof that arbitrary
natural-language text contains no sensitive value; independent review should
repeat or strengthen the inventory. Bundle SHA, prerequisite, and target HEAD
must be checked together, not inferred from its name.

The exact science files are unchanged between e65 and r9: `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The target checker pins those and all six changed production safety files by
SHA-256, exact r9 commit, 14-commit ancestry, clean status, no remotes or
alternates, and full strict Git fsck.

## Evidence and bounded threat model

- Archived trn22 four-rank CPU G2 receipt at
  `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/operator-receipt.json`,
  SHA-256 `c8d73cc447a4fb99c9aba1869f3df47accee550ea2e04d704b9a43cd1bd8e116`;
  independent numeric review at
  `research/agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md`,
  SHA-256 `e76654c7882c2ede9e382914aebb672b10841a083fc0eb2271e15b39965a1de4`.
  This is **e65 CPU G2 only**. G2 source-identity carry-forward to r9 remains a
  later independent gate.
- Independent r9 **source-only** PASS at
  `research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R9.md`,
  SHA-256 `5ba9550b2bc2cb3ebf16792f0f1a9be75582899be9106f61c70271014895472d`,
  commit `a669d4ebf8a8a94a31a0f8117b3ab39abe30e7c4`. It confirms the exact
  r9 launcher is trn22-A0-only; it is not stage or native dispatch authority.
- Current point-in-time Operator summary at
  `research/v2/observations/20260925-dual-host-readonly-operator/OPERATOR_READONLY_DUAL_HOST_SNAPSHOT_20260925T0733Z.md`,
  SHA-256 `3665847467803f05f1a61a040f0b38e7a88e486bc7af52b2fbfcf24472343777`,
  commit `47e64a32b37ec72f10bf387601bf3a69fb63e9ef`; trn22 raw SHA-256
  `c8a44adfea5b7be7f12603dc3f2ef97c32b1f1495d199e8ac42cc281591f94dd`,
  linked-Gitdir/ACL raw SHA-256
  `b878de92aa1be748599079ec1a1295a70dff3a4ee8856fc1d17208298801f3bc`.
  Independent paired read-only review at
  `research/v2/verification/20260925-dual-host-readonly-final-pair-independent-PASS.md`,
  SHA-256 `915724781f316ee82866605cf0b8290d2ce40a5f10b0524b477fbc1928dfd652`,
  commit `9868d9a6`. All these observations are planning evidence, not a future
  transfer preflight or authorization.

The accepted trn22 e65 r2 checkout is **not private 0700**: directory mode
`0775`, UID:GID `1000:1000`, device `66305`, inode `2432738`, link count 4;
its `.git` is a 148-byte regular linked-worktree pointer, mode `0664`, inode
`2432740`, SHA-256
`dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b`.
Its resolved Gitdir is the r1 `.git/worktrees/...r2` directory, inode
`2432736`, mode `0775`. The snapshot found no POSIX ACL on checked ancestry
and no other *local-file* account with GID 1000, but did not exclude external
NSS, same-UID, privileged processes or alternate namespaces. The stage
engine remeasures exact source pointer/dir, ancestry, owner/mode/inode,
absence of access/default ACL, NSS enumeration, clean e65 HEAD, source science
hashes, host mount, protected checkpoint/caches, and a cooperative existing
host lock under private `.cache` on every phase. Unknown ACL or drift is HOLD.
This is a bounded ordinary-DAC/cooperative-flock design, not an absolute
non-malleability claim against same-UID/root or noncooperating processes.

## Future guarded sequence (not authorized by this package)

1. Separate Verifier reviews the exact subject and tests. A **new** exact
   enabled source-stage request must be independently finalized, followed by
   an actual matching independent final source-stage PASS and Supervisor
   one-attempt assignment. The three future files are deliberately absent.
   Merely editing the checked-in HOLD JSON does not satisfy the entrypoint.
2. The future Operator must recheck local STOP/breaker state and strict pinned
   trn22 SSH identity. The one-attempt marker is then exclusively created;
   any failure consumes the attempt. Four distinct SSH processes run fresh
   read-only preflight, exclusive bundle/new-checkout creation, cross-session
   hash/status readback, and **unconditional full postflight**, respectively.
   No direct invocation of the remote engine is an authorized shortcut.
3. Each remote phase takes the pre-existing launcher-compatible `flock`
   without creating or mutating its lock file; FIFO/symlink/owner/mode/inode/
   ACL changes HOLD. Full probes check STOP paths, process and Neuron holders,
   label absence, source/host/lock/protected continuity, root ext4 mount,
   and root free bytes/inodes both before and after the slow scan. Fresh
   preflight and write require at least 13,177,973,911 free bytes; readback
   and postflight at least 10,737,418,240. **Every** endpoint also enforces
   the 6.5 GiB floor and 10,000 free-inode floor. Live host timing and
   noncooperating-actor exclusion remain unproven and require review.
4. A successful stage only leaves a **new** private `0700` detached r9 checkout
   and an exact `0600` bundle. The accepted e65 linked checkout is never
   patched. Failure leaves marker, uploaded bundle, and any partial target
   intact for inspection; there is no retry, overwrite, or cleanup. Even a
   successful local terminal result is evidence for an independent stage-result
   review, **not native A0 authority**. Native request, paired monitors,
   source-identity G2 carry-forward, fresh preflight and one-shot assignment
   are separate future gates.

## Implementer-local checks only

`python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-hold -p 'test_*.py' -q`

Both returned 12 tests OK on the Implementer checkout. These are regression
checks, **not independent verification**. They include synthetic linked
worktree reconstruction, missing/mismatched authority before contact,
group/ACL/pointer drift, FIFO lock, existing target, low post-scan inode count,
STOP during label scan, optimized-Python refusal, and postflight attempted
after a synthetic partial write. No remote host was used.
