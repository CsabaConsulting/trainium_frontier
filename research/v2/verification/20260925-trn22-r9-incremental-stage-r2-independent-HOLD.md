# Independent trn22 e65→r9 incremental source-stage r2 review — HOLD

Reviewed 2026-09-25 by a Verifier separate from the Implementer. Exact subject:
`a7643f05c5468da6537c2d7b403a05ac048679f0`, direct child of the held
r1 subject `1bbe2a3858682089eadca04f18541991ae255114`, on
`codex/20260925-trn22-r9-incremental-source-stage-r2-hold`. Only the six new
files under `research/v2/authorizations/trn22-c5c-a0-r9-incremental-stage-r2-hold/`
were reviewed. This is a **static, offline source-stage review**: no host/SSH
contact, one-shot marker, stage write, checkout, native run, evaluation,
cleanup, breaker reset, AWS action, or leaderboard submission occurred.

**Verdict: HOLD.** The checked-in request remains disabled, and the local
entrypoint refuses before contact without a distinct enabled request, final
review and Supervisor assignment. R2 fixes the r1 witness for STOP/label
arriving during `fuser` or after the first final label scan; a 1.1-second
delay *inside* the final scans also now fails before the first write. However,
two separately reproduced boundary gaps still prevent a source-stage PASS.

## Blocking findings

1. **Fresh activity and disk status can be lost during the final label scans.**
   `remote_incremental.py:402-417` completes `_activity()` before two broad
   label scans. After the scans it directly rechecks STOP and target names,
   but not the process table, Neuron holder or root free bytes/inodes. The
   next `_activity()` and `_space()` are at lines 569-570, after the bundle
   and new checkout have been created by lines 511-523. The independent
   disposable-file fixture introduces a training process or Neuron holder
   during the first final label scan (well within the one-second age window):
   both first-write targets are created, and the persistent activity is
   detected only after reconstruction. It similarly introduces a root
   disk-floor loss during final `fuser`; the final `_space()` rejects only
   **after** both targets exist. These are synthetic races, not claims that
   activity or disk loss occurred on trn22. The prior r1 HOLD required fresh
   STOP/process/Neuron/lock/disk continuity; the older transfer design
   explicitly used a post-label host-safety check. A one-second-old activity
   observation cannot substitute for detecting persistent activity that
   begins *during* the final label traversal.

2. **The claimed one-second observation-to-`O_EXCL` bound ends before the
   actual open.** `_final_no_write_continuity()` computes ages and returns at
   lines 411-417; `os.open(... O_EXCL ...)` follows at line 511, with no
   monotonic stamp or check at the attempted/completed open. An independent
   1.1-second synthetic scheduling delay between those statements still
   creates the bundle and checkout and returns a successful write receipt
   whose `activity_age_seconds` is under one second. The implementation's
   1.1-second **in-scan** test does fail, but does not exercise this gap.
   A finite user-space check cannot eliminate all post-check scheduling
   races; at minimum the actual open boundary and any stale/partial result
   must be measured and reported honestly, rather than accepting it as a
   fresh successful write. The inherited numeric one-second criterion comes
   from the earlier trn22 transfer package and the Supervisor's review
   request, not top-level `AGENTS.md`.

The independently authored regression is
`research/v2/verification/test_trn22_r9_incremental_stage_r2_independent.py`,
SHA-256 `3be2b006052f7a5913cedb7d2ecf6ee90d09c71d425cd8cb52e738c4a1e88a4e`.
It retains real `_activity()`, `_label_scan()`, `_write_bundle()` and `perform()`
flow; only host-specific probes/subprocess outputs and the disposable local
root are synthetic. All nine cases pass under normal Python and `python3 -O`:
four repaired no-write cases, one clean local-only case, and four witnesses
of the remaining process/Neuron/disk/open-delay gaps. Thus passing means the
defects reproduced, **not** source-stage approval.

## Exact identity and bounded positive checks

| Subject item | SHA-256 |
| --- | --- |
| `REQUEST.HOLD.json` | `f83bc93b1894dd21f813c624375cbf15e04a459e99578a21787ce33f6c83c01f` |
| `RUNBOOK.md` | `5d1821caab1d0696e8f3eed38841b458899f13d68da723f07e5b9ef0bce4d43d` |
| `operator_once.py` | `6bc8df9e52fb97020779ac014d8d1a516ac83bfc94c4b24cc72fc32054166890` |
| `remote_incremental.py` | `2b84ef670811b3f0015b30e619cb3ec5b23e914326cdd3e74523fdc5b68ee7a7` |
| `test_incremental_stage.py` | `ad7228d2f83da9bac9c570607c9e93d6f22d86899e576838dae882a52e755c8c` |
| `test_late_scan_r2.py` | `f1586947e1c75597dc72ba859b20802a696476c63409fd7dfc0615ea65b939ff` |
| Reused r1 `e65-to-r9.bundle` | `f56ba94c4bc8d248583d356bf141b5523688d27c45c68a01765ceb70b5331ee7` |

The 101,237-byte bundle independently passes `git bundle verify`: r9
`e046de16826bfc3b954136e3c60804aea0554ecc` HEAD, e65
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` prerequisite. All eight
science/safety file hashes in `TARGET_FILES` match the exact r9 source
checkout, including unchanged `train.py` (`c5c410af…`) and `prepare.py`
(`5807c46e…`). Diff from r1 is narrow: final-continuity logic, distinct r2
authority/marker, and disabled tests/runbook. The prior r1 independent HOLD,
SHA-256 `41e4c5b3837a3e5089352192d070c18711b5055ce259a3debec051d8cdb39e34`,
already boundedly reviewed the linked e65 source's pinned pointer/Gitdir,
group-writable `0775` source, ACL/local-NSS condition, ext4/lock/protected
paths and immutable bundle inventory. R2 retains those pins and adds a
distinct r2 local one-attempt marker while refusing a consumed r1 marker.
These are source properties, **not a fresh host observation**. Same-UID/root,
external NSS and noncooperating activity remain outside that limited proof.

Checked-in suite: **19/19 PASS** normal and **19/19 PASS** `python3 -O`.
Independent suite: **9/9 PASS** normal and **9/9 PASS** `python3 -O`.
Exact subject `git diff --check`: PASS. Live trn22 latency of the required
full activity plus two broad label scans is **unknown**; there was no host
timing attempt. Even a read-only timing success would not fix either source
gap or authorize this disabled package.

Do not create an enabled request, final source-stage PASS, one-attempt
assignment, host transfer, or native launch for r2. A distinct Implementer
repair needs another independent review; its future host staging, if ever
approved, still requires fresh preflight, separate final authority,
cross-session readback, unconditional postflight, and later native gates.
