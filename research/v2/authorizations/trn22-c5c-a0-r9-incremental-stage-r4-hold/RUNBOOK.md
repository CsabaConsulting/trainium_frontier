# trn22 e65 → r9 incremental source-stage r4 — DISABLED / HOLD

This distinct Implementer repair responds to the [independent r3 HOLD](../../verification/20260925-trn22-r9-incremental-stage-r3-independent-HOLD.md). It removes the unauthorized provisional-bundle unlink. It has not contacted trn22 or AWS, transferred source, made a host checkout or marker, trained, evaluated, cleaned, reset a breaker, or submitted. The checked-in request disables dispatch and source staging. There is no enabled request, independent final PASS, or Supervisor assignment; `operator_once.py` exits before a marker or SSH.

## Exact disabled subject

| Item | SHA-256 / pin |
| --- | --- |
| `remote_incremental.py` | `871fd00aa61d0db2df8597949c8cad43c4c0a9a94ffda218a44851d60091c9b7` |
| `operator_once.py` | `64d0ed49261cd231b3193f315d114d43e39a94f3b3dea21833a37dd15c0c2d7e` |
| `REQUEST.HOLD.json` | `8cd3b2debb0c309ab72faef78b057e344fc0525b1269332d3d46b13337797f56` |
| `test_incremental_stage.py` | `ffde1ed789f2c54aaac171036f533263fd9e551a633ebad3a4de44c1487bb02f` |
| retained late-scan regression `test_late_scan_r2.py` | `36104c13dd52c2d7c81a79f6016a6360082be919822dce209bf92badefe5fe30` |
| no-cleanup regression `test_final_boundary_r4.py` | `c68fba09ad56495419a0f5583fcd33e8122cfee98f6dfb6779d7e8c3ee94f7da` |
| reused r1 `e65-to-r9.bundle` | 101,237 bytes; `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |

This branch is a direct child of disabled r3 `80c96851edbb6cbc707c82c68773a95ea1231f91`; r1–r3 packages and review evidence are untouched. The r3 HOLD is canonical at `524f761fb470db86460d851b8f3af8caa50bce95`, receipt SHA-256 `5a2b24a5b27661e61773004be35ea60308bb16385f199f5b2a24e719f43054ed`. It found that r3's empty-bundle deletion lacked human cleanup authority and could delete a same-UID replacement after a name swap. R4 has no explicit host `unlink` or rename path for the bundle. Earlier r2 HOLD remains historical evidence, not authority.

The frozen bundle still has r9 HEAD `e046de16826bfc3b954136e3c60804aea0554ecc` and sole prerequisite e65 `e65a3097a5b00f6178188a935bb2acb9a5042c6c`. The earlier bounded inventory counted 131 objects (14 commits, 57 trees, 60 blobs), 117 named paths, and zero credential-path/blob-scan hits; that scan is not proof against arbitrary prose secrets. Bundle bytes and length are checked before a future marker. No `.env`, key, token, or credential bytes are copied or printed.

## Exact boundary and terminal failure

The source is the observed group-writable `0775` linked e65 checkout, with a `0664` `.git` pointer—not an assumed private `0700` tree. Any future attempt must freshly recheck its owner/inode/mode, exact Gitdir ancestry, ordinary-DAC ACL/group condition, clean e65 commit and science hashes, ext4 mount, protected checkpoint/caches, private-parent cooperative lock, STOP/process/Neuron/label absence, and root byte/inode floors. The 07:33 read-only Operator snapshot is planning evidence only. Unknown or changed facts HOLD. The target is the same **new, absent, private** r9 checkout path, never an in-place edit of e65. The r4 local marker is distinct; existing r1, r2, or r3 markers block reuse of the shared remote label and target.

R4 retains r3's full fresh preflight and postflight, two broad final scans followed by full activity and root-space checks, the inherited ≤1-second combined scan and freshness limits, actual `O_EXCL` return-age/open-call checks, and a third broad scan plus full activity and disk check before payload. The local wrapper still requires finite ages, the unchanged 13,177,973,911-byte / 10,000-inode pre-write floor, the 6.5-GiB hard floor, cross-session hash readback, and a completed full postflight receipt for success. Postflight is **attempted** even on failure; a new local STOP can prevent remote completion, which must remain a HOLD.

If a gate fails **before** `O_EXCL`, no bundle or checkout is created, but a future attempt's consumed local one-shot marker remains. If a gate fails **after** `O_EXCL`, the empty or nonempty provisional bundle is retained for inspection; no checkout is created and no successful write receipt is returned. The code attempts to fsync its open bundle descriptor and parent directory on both failure and success. If durability itself fails, the state is uncertain and remains HOLD. The repair never invokes deletion, rename, retry, or cleanup of the provisional bundle or partial. A separate **human** decision is required before any cleanup or retry; neither an enabled request nor a Verifier can supply that authority. The operator's terminal error now says any empty/nonempty bundle and marker are retained.

## Bounded evidence and later gates

No finite userspace check makes broad scans, process/Neuron state, disk and `O_EXCL` atomic. R3's independent witness showed a persistent label introduced during the final post-open `fuser` after the last broad scan can reach write/checkout, and a 1.1-second pause inside the first payload `os.write` can still yield success because the numeric bound is on first exclusive open, **not payload completion**. Those are disclosed bounded-model warnings, not claims repaired by this narrow r4 no-cleanup change. Same-UID or privileged interference, alias paths and scheduler delays remain residual risks for independent review. The actual r4 sequence of three full activity checks, three broad scans and disk observations has not been timed live on trn22. The older r2 timing diagnostic's static PASS cannot establish r4 feasibility or dispatch authority.

The Implementer-only offline suite passed **36/36** under normal Python and **36/36** under `python3 -O`. It retains source/lock/ACL/inode/floor/late-STOP tests and adds no-unlink, empty-provisional retention, nonempty-partial retention, and same-UID replacement-preservation witnesses. These are synthetic regression results, not independent verification or host preflight.

Future progression needs independent exact r4 source review; separate read-only r4 timing evidence and its review; a distinct enabled source-stage request and independent final review; Supervisor one-attempt assignment; fresh host preflight; cross-session readback and completed postflight; then independent stage-result and G2 source-identity carry-forward reviews. A source stage is **not native A0 authority**. Moreover, experiment-protocol §1 requires the same Git HEAD on local and both hosts for scored runs. This trn22 stage targets r9 while the trn21 source plan targets descendant r10. R9 stage alone does not clear scored A0: a later exact trn22 r10 rebinding/stage or explicit human-approved protocol reconciliation is needed. Native launcher request, monitors and one-shot authority remain separate.

Offline commands:

`python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r4-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r4-hold -p 'test_*.py' -q`
