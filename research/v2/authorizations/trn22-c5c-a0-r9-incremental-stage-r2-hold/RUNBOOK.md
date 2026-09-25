# trn22 e65 → r9 incremental source-stage r2 repair — DISABLED / HOLD

This is a **new Implementer-only disabled** response to the independent r1
HOLD; it does not edit or supersede the frozen r1 subject or the Verifier's
evidence. No host/SSH/AWS contact, source transfer, new checkout, marker,
assignment, monitor, guard, native run, evaluation, cleanup, breaker change,
or submission occurred. `operator_once.py` exits HOLD before any marker or
SSH because its distinct enabled request/final review/assignment files are
absent. A source-stage PASS would still not be native A0 authority.

## Exact subject and inherited immutable bundle

| Item | SHA-256 / pin |
| --- | --- |
| r2 remote engine `remote_incremental.py` | `2b84ef670811b3f0015b30e619cb3ec5b23e914326cdd3e74523fdc5b68ee7a7` |
| r2 local entry `operator_once.py` | `6bc8df9e52fb97020779ac014d8d1a516ac83bfc94c4b24cc72fc32054166890` |
| r2 disabled contract `REQUEST.HOLD.json` | `f83bc93b1894dd21f813c624375cbf15e04a459e99578a21787ce33f6c83c01f` |
| r2 retained offline tests `test_incremental_stage.py` | `ad7228d2f83da9bac9c570607c9e93d6f22d86899e576838dae882a52e755c8c` |
| r2 adversarial regressions `test_late_scan_r2.py` | `f1586947e1c75597dc72ba859b20802a696476c63409fd7dfc0615ea65b939ff` |
| Reused exact r1 `e65-to-r9.bundle` | 101,237 B, `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |

The reused bundle remains in the frozen sibling r1 package and is hash- and
length-checked before any marker. `git bundle verify` reports r9
`e046de16826bfc3b954136e3c60804aea0554ecc` as HEAD with only e65
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` as prerequisite. The r1
independent inventory found 131 objects (14 commits, 57 trees, 60 blobs),
117 named paths and path-inventory SHA-256
`2e06bfa35779c4319755d1aa8392cc4e8077b9e680ef66a9e0d2f7306774756a`;
its bounded private-key/credential path and blob signature scan found zero
hits. These are bounded checks, not a proof about arbitrary prose secrets.
No `.env` or credential file is copied or printed.

The original r1 subject is `1bbe2a3858682089eadca04f18541991ae255114`;
its independent HOLD is
`research/v2/verification/20260925-trn22-r9-incremental-source-stage-independent-HOLD.md`
at `40b49c32`, SHA-256
`41e4c5b3837a3e5089352192d070c18711b5055ce259a3debec051d8cdb39e34`.
The adversarial witness is SHA-256
`6ded0c2ba4899e8160bce8be1b02d3651a6babff10123c980694bbf053983b20`.
It reproduced STOP or exact stage-label insertion during the last slow
`_activity()` reaching O_EXCL, plus a 1.1-second stale final proof. The r1
HOLD remains controlling for r1; this r2 code requires its own independent
review and later authority chain.

The r2 host/source/target/bundle paths, unique unconsumed stage label,
101,237-byte bundle, r9/science/safety hashes, protected checkpoint/caches,
trn22 linked-worktree pointer/Gitdir/DAC/ACL/NSS checks, private-cache lock
identity, ext4 mount, STOP/process/Neuron and root space/inode gates remain
as in the frozen r1 design. The distinct **local** one-attempt marker is
`trn22-c5c-a0-r9-incremental-stage-r2.one-attempt.jsonl`; r2 authority schema
and scope are distinct, so the held r1 request/review cannot be reused.
The wrapper also refuses if the original r1 one-attempt marker exists,
because r2 retains r1's remote label and target paths and cannot convert a
consumed r1 attempt into a retry. Fresh broad label and target absence checks
remain mandatory even when both local markers are absent.
The 07:33 Operator snapshot and its paired independent review are still
point-in-time planning evidence only, not current stage preflight or
authorization. The accepted e65 source is a group-writable linked worktree
under the documented private-parent ordinary-DAC condition, **not** a 0700
private checkout; the runtime remeasures the exact pointer, Gitdir, ACL,
group and lock ancestry before any write. Unknown or changed facts HOLD.

## Specific late-boundary repair

The slow activity check now rechecks STOP/process **after** `fuser` and
`neuron-ls`. Inside `_write_bundle()` under the existing nonblocking host
flock and just before the first O_EXCL, r2 repeats that full activity check,
performs **two** broad attempt-label scans, directly checks STOP and both
target names, then requires the final activity and label observations to be
at most **one second** old. The write receipt includes that proof, and the
local wrapper checks it before accepting readback. The one-second bound is
inherited from the earlier trn22 transfer design and explicitly requested
for this repair; it is **not** a top-level AGENTS/protocol numeric rule.
There is no live trn22 timing measurement for the two scans plus activity,
so this disabled design may systematically HOLD. It must not be enabled or
relaxed until the separate read-only timing diagnostic, independent review,
fresh preflight and one-attempt authority establish feasibility.

The adapted independent witness now asserts **no bundle and no checkout**
when STOP or label appears during `fuser`, when STOP or label appears after
the first final scan, and when a final scan takes 1.1 seconds. A fast, clean
synthetic sequence reaches a local-only new target. A further regression
refuses an existing r1 marker before r2 marker creation or contact. Normal
and optimized offline suites each passed **19/19**; these are Implementer regressions,
not independent verification. They use disposable local directories, no
SSH. No finite userspace check can absolutely exclude a noncooperating
same-UID/root actor after the last observation; the bound is explicit and
the cooperative lock/ordinary-DAC threat model remains limited.

If a separately authorized one-shot stage ever fails, its marker and any
partial uploaded bundle/new private checkout remain for inspection: no
retry, overwrite, or cleanup. Four separate SSH phases still require fresh
preflight, exclusive write, cross-session readback, and **unconditional
full postflight**, with the same 6.5-GiB/10,000-inode hard floors and
stronger phase-specific free-space gates. Even successful source staging
requires independent stage-result review, G2 source-identity carry-forward,
and separate native r9 request/paired monitors/one-shot authority.

Offline Implementer commands:

`python3 -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r2-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r2-hold -p 'test_*.py' -q`
