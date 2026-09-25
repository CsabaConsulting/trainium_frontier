# Independent trn21 final-r6 post-label review — static PASS, dispatch HOLD

**Verdict: PASS_DISABLED_STATIC_ONLY.** The exact r6 package closes the
reproduced r5 persistent-STOP-during-label-scan gap in local synthetic checks.
It is disabled and grants no transfer dispatch. Host timing feasibility is
unproved, so no enabled request, final dispatch PASS, Supervisor assignment,
source transfer, checkout, CPU G2, native run, evaluation, cleanup, AWS action,
breaker reset, or submission follows from this review.

## Subject and predecessor

I reviewed clean commit `71d6701f49f08ec626ae27827af684439460dff9`
on `codex/20260925-trn21-r6-post-label-stop-disabled`, directly above
`5a25a139df2128f9b9fde8c90b2db7775d60cb35`, in a separate clean sparse
checkout. The diff adds exactly six files in
`research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/` and changes no
older package file. `git diff --check` passes. Their SHA-256 values are:

| New file | SHA-256 |
| --- | --- |
| `FINAL_R6_DISABLED_POST_LABEL_RUNBOOK.md` | `532ace3812a74f484d20ef6710d052677d2166279a85bb94d43fb21a8359a67e` |
| `TRANSFER_REQUEST.FINAL_R6_HOLD.json` | `7cae6fba6cbd6e763ede016d4fec7dfa5aa8be78fa0fd7dfa0f7ae7b5b8f3b71` |
| `operator_transfer_final_r6_once.py` | `0520350e7aa18a04e70bfc54ee71dddd97b7e16c34c524cffc6bccbfca420787` |
| `stage_r4_remote_final_r6.py` | `bee6c660e18e084c0d55a66473b1418772ac58adcad294c0775aee0acaee71b6` |
| `stage_r4_transport_final_r6.py` | `c93f432132d3e23d6fc73ba541029149942acf6a41c2852cb83fa54b33c754bc` |
| `test_operator_transfer_final_r6_once.py` | `1055860408a8bf90ec751e05727ecaee75a2d2976fcb8ba05f510d73f0bfb85f` |

The r5 enabled independent HOLD remains pinned at SHA-256
`21a68fa2aa8e9ae3d0fa88d3f01366daee6e569c445ef3af4ea0a2337e1a40d3`.
The r6 request and wrapper bind that actual HOLD and its adversarial test,
along with the inherited r5 inode-floor and r4 private-lock evidence.

## Static findings

The writer holds the existing launcher-compatible flock while it performs a
clean first label scan, a full activity recheck, a second clean label scan,
named/opened lock identity rechecks, and then a STOP/process/Neuron/target
gate. It repeats the cheap checks after the potentially slow Neuron commands,
with STOP last. A persistent STOP introduced during either scan or during
postscan activity therefore blocks target `O_EXCL` in the tested flow. The
writer measures final-scan completion to target-open start with a monotonic
clock and refuses an interval above one second before `os.open`. It records
the ordered scans, gates, lock identities, and target-open times; the distinct
transport validator rejects missing, dirty, stale, or reordered receipts.
The inherited full-probe 10,000-free-inode floor and host/lock identity
checks remain active.

Both request authorization flags are `false`. `run_once()` fails at
`DISPATCH HOLD` before reading final authority, creating the one-shot marker
or journal, or spawning SSH. The fixed-path final r6 review and Supervisor
assignment files are absent. No fixed-path authority file was created here.

## Offline checks and limit

- Supplied package suite: **76/76** normal and **76/76** under `-O`.
- Archived r4/r5 independent negatives: **14/14** normal and **14/14** under `-O`.
- New independent negatives: **5/5** normal and **5/5** under `-O`, in
  `test_trn21_r6_post_label_independent.py` (SHA-256
  `5d3fa49b1497ac52995783cd5fb36a41a0832c910c9b05240d6505fe01446bde`).
  These execute the actual r6 writer against disposable local fixtures:
  persistent STOP in each scan and during postscan Neuron work leaves no
  target; a real delay above one second leaves no target; first/second-scan
  lock drift and boot identity drift leave no target. Receipt mutations and
  the disabled no-marker/no-SSH path also fail closed.

These tests contacted no Trn2 host or AWS endpoint and invoked no real SSH or
transfer. They do not establish that the final label scan, lock identity
proof, `fuser`, `neuron-ls`, and `O_EXCL` can consistently fit within one
second on trn21. The two broad scans and Neuron commands have much longer
individual timeouts. A separate authorized read-only host timing assessment
and independent review, or a simpler reviewed design, is required before an
enabled successor can be considered. The bounded final sample also cannot
atomically exclude an uncooperative actor arriving afterward. Fresh host
preflight, exact enabled authority, one-shot assignment, and independent
result review remain separate gates; the frozen native launcher retains its
own HOLD.
