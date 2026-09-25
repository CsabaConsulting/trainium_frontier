# Independent trn22 e65→r9 incremental source-stage review — HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact
subject: `1bbe2a3858682089eadca04f18541991ae255114` on
`origin/codex/20260925-trn22-r9-incremental-source-stage-hold`. Only the six
new files in
`research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-hold/`
are in the subject commit. This is a **static, offline source-stage review**:
no host/SSH contact, enabled request, marker, stage write, checkout, native
launch, evaluation, cleanup, breaker reset, or AWS action occurred.

**Verdict: HOLD, not independent final PASS or dispatch authority.** The
disabled entrypoint correctly refuses without its future exact request,
independent final review, and Supervisor assignment. Much of the identity
and one-shot design is sound within its documented bounded threat model, but
the write-phase final-scan boundary does not fail closed for a late STOP or
late attempt label. The existing one-attempt marker would be consumed before
that remote phase, so a partial write is also operationally costly.

## Blocking finding: late STOP/label reaches exclusive creation

`remote_incremental.py` lines 398–422 complete a slow attempt-label scan,
then a STOP/process/Neuron scan and free-space check in `_probe()`. The write
path at lines 521–527 then checks the prior snapshot and target absence,
calls another `_activity()`, and immediately calls `_write_bundle(data)`.
`_activity()` checks STOP paths **first**, before walking `/proc` and invoking
`fuser` and `neuron-ls`, each of which permits up to 20 seconds. It does not
check the attempt label. There is no second label scan, no final STOP sweep
after those slow observations, and no monotonic bound from the final scan to
the bundle's `O_EXCL` open at line 475. The STOP/process/Neuron checks after
`_reconstruct()` occur only after bundle and target creation.

The independent offline regression test
`research/v2/verification/test_trn22_r9_incremental_stage_adversarial.py`
keeps the real `_activity()` control flow and real `_write_bundle()` exclusive
creation, while replacing the host probe, `/proc`/Neuron responses and
filesystem root with a disposable synthetic fixture. It demonstrates:

1. A STOP created during `fuser`, after `_activity()`'s initial STOP sweep,
   does **not** prevent bundle and target creation. A later `_activity()`
   raises, too late to prevent the write.
2. A file named for the exact attempt label, created in the same interval
   after `_probe()`'s final label scan, still allows bundle and target
   creation **and a successful write-phase receipt**.
3. A 1.1-second delay in that final activity scan still reaches exclusive
   creation. No one-second scan-to-open timer exists in this subject.

All three observations reproduced under normal Python and `python3 -O`.
This is a control-flow counterexample, **not** a measurement of live trn22
latency and not a claim that any real STOP/label appeared on the host. A
finite final-check-to-open race always remains against noncooperating actors;
the repair must state and bound that residual instead of claiming atomic
exclusion. Before any final source-stage PASS, keep the held cooperative
flock, arrange a truly final attempt-label check and a fresh STOP/process/
Neuron/lock/disk continuity check so late changes fail before `O_EXCL`, and
prove a bounded scan-to-open window with synthetic late-arrival and delay
tests. Do not remove the expensive safety observations merely to meet a
timer. The explicit one-second numeric bound comes from the earlier trn22
r4 transfer package (`stage_r4_remote_lock_final_r3.py`, constant
`MAX_FINAL_SCAN_TO_OPEN_NS`) and its RUNBOOK/independent reviews. It is an
inherited trn22 transfer criterion, not a numeric rule located in the
top-level `AGENTS.md`, experiment protocol, or controller brief; this review
also applies it because the Supervisor explicitly requested it. The late
STOP/label failure is independently blocking even without that number.

## Exact package and evidence scope

| Subject file | SHA-256 |
| --- | --- |
| `REQUEST.HOLD.json` | `02d5f30416911c2bf52192563997733c98a1a4f51dbe4746913e468a202f1299` |
| `RUNBOOK.md` | `5a8d97eaa260104c548f454c3f0d6b3a50b524043bf7b2726a6996c0048f5692` |
| `e65-to-r9.bundle` | `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |
| `operator_once.py` | `d4eaeb4abde4690017029628dab48130209b695f0a0b8ef2ed8404f5bc29674c` |
| `remote_incremental.py` | `0de28ddea4e015cc32fdb0cd50433624c51c810a0b67c07a1cbe6f2d8f17f940` |
| `test_incremental_stage.py` | `eb3a9a7578433602b7b3c23eba869b93148286e0f8596490e652cddcfba6eff6` |
| Independent adversarial test (review branch only) | `6ded0c2ba4899e8160bce8be1b02d3651a6babff10123c980694bbf053983b20` |

`git bundle verify` reports only r9 `e046de16826bfc3b954136e3c60804aea0554ecc`
HEAD and e65 `e65a3097a5b00f6178188a935bb2acb9a5042c6c` prerequisite.
The 101,237-byte bundle hash above matches the subject. Independent
`git rev-list --objects e65..r9` counted 131 objects (14 commits, 57 trees,
60 blobs), 117 named-path entries, and the RUNBOOK's sorted-path SHA-256
`2e06bfa35779c4319755d1aa8392cc4e8077b9e680ef66a9e0d2f7306774756a`
when paths are joined by newline **without a trailing newline**. A bounded
scan of those new blobs and named paths found zero private-key, credential,
`.env`, AWS-key-ID, GitHub/Hugging Face/OpenAI-token signature hits. This
does not prove absence of an arbitrary secret embedded as ordinary prose.
The pinned `train.py` and `prepare.py` hashes are unchanged across e65→r9;
the target checker pins six safety files, exact ancestry, strict Git fsck,
clean detached target and no remotes/alternates.

The referenced point-in-time Operator snapshot is exactly SHA-256
`3665847467803f05f1a61a040f0b38e7a88e486bc7af52b2fbfcf24472343777`
at commit `47e64a32`, and its independent paired review is exactly SHA-256
`915724781f316ee82866605cf0b8290d2ce40a5f10b0524b477fbc1928dfd652`
at commit `9868d9a6`. The snapshot's trn22 e65 source is the linked worktree
at device `66305`, inode `2432738`, mode `0775`, with 148-byte `.git` pointer
inode `2432740`, mode `0664`, SHA-256
`dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b`,
resolved Gitdir inode `2432736`, mode `0775`. The package pins those exact
identities and the documented no-ACL/local-NSS exception; it does **not**
mistake the source for a private `0700` checkout. The source-target pair is
different, and the new r9 target must still be absent at a fresh live
preflight. Snapshot free space, idle state and target absence are historical
observations only, never a stage authorization.

`operator_once.py` pins trn22 SSH endpoint/fingerprint, source/target,
bundle and science identity, checks local STOP/breakers and strict one-shot
authority before any marker/contact, and uses distinct isolated remote
phases. The remote engine takes the pre-existing nonblocking flock without
creating or modifying the lock; checks source linked pointer/Gitdir,
mount/ACL/owner/clean status, protected checkpoint/caches, STOP/process/
Neuron activity and root bytes/inodes; and creates only an exact new bundle
and private target with exclusive no-follow operations. Failure retains
partials; local postflight is attempted unconditionally. These positive
properties are not sufficient to overcome the late-arrival defect. The
cooperative flock/ordinary-DAC proof does not exclude same-UID/root actors,
external NSS ambiguity or noncooperating activity.

## Local checks

- Checked-in source tests: 12/12 PASS under normal Python; 12/12 PASS under
  `python3 -O`.
- Independent race tests: 3/3 reproduced under normal Python; 3/3 under
  `python3 -O`. They use disposable local files only, no SSH or host contact.
- Exact subject `git diff --check`: PASS.

No independent final PASS, enabled request, Supervisor assignment or native
approval should be issued for these bytes. A repaired disabled package needs
fresh independent review and the separate authority chain before any stage.
