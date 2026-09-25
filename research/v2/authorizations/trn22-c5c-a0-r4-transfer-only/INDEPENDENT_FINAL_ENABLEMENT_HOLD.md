# Independent final enablement review: trn22 r4 transfer-only

**Verdict: HOLD for final enablement.** Do not issue
`INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json` or dispatch this transfer. The
candidate's fixed request and code hashes match the bytes below, and its
missing-review gate correctly stops before marker creation or SSH. Two
additional safety invariants fail in offline adversarial tests. A prior
static PASS applies only to the disabled core at `7aa72f9c`; it cannot
authorize this enabled request or excuse these findings.

Reviewed 2026-09-25T03:46Z by an independent Verifier in branch
`codex/20260925-trn22-r4-transfer-final-independent-verifier` against exact
Implementer commit `b80c69c43158102d902ec8c24f02a17832309dc5` on
`origin/codex/20260925-trn22-r4-transfer-only`. I did not author or edit its
four added files. I read the repository contract in the required order,
the source-stage contract and independent receipt, the prior transfer
HOLD, the repaired disabled-core static PASS, and the full candidate diff.
No SSH, host/AWS contact, source transfer, checkout, G2, training,
evaluation, cleanup, breaker reset, key-content read, or submission
occurred. This receipt and the separate negative tests are review-only
additions on this branch.

## Exact bytes and checks that passed

| File or artifact | SHA-256 |
| --- | --- |
| Enabled transfer request | `3cd3c091d782dbaa535f67c399041900f9ceaf29b74ee046c6472bd3f44919f3` |
| Final wrapper | `ccdc7175e68f0e27eeff1dab71f17d8c49f4a9089ee88d75e73b9fcc29ffce7d` |
| Reused disabled core wrapper | `b81744bba97c1fd0c16ce90b9571a964f347b19908614fae16e617396624439d` |
| Reused transport | `58a2fc899470137fed940b0b96850d5ce304a432d24093bd6e49e38b1104e38b` |
| Reused remote payload | `2920dfc78b1bf3941e9eaf500e41a44f7f5e29f9799273c04eb7ddc7de3d756e` |
| Repaired disabled-core static receipt | `7dcfc65cde4dd29a409e8541ace24ada72465b15964d89af5852a5fd89f896db` |
| Source-stage contract | `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69` |
| Source-stage independent receipt | `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |
| Frozen outside-Git capsule | `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| Pinned local `known_hosts` file | `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3` |
| This review's negative tests | `1c59e4c15ad78efd7b94aa54ee86d9ebb033a94dd9d7b0b5c9db9998cc0b72e0` |

The capsule remains a regular, single-link, mode-`0600` file of
104,154,875 bytes. I independently rehashed it and read only the archive's
`.git/HEAD` and `.git/shallow`, both of which equal frozen **source** commit
`446b657695f086f9b484fa601c9a3b61c2495912`, rather than the package
HEAD. Its `train.py` and `prepare.py` member hashes match the source-stage
receipt (`c5c410af...` and `5807c46e...`). The existing PEM was checked by
metadata only; the local `known_hosts` hash and ED25519 fingerprint
`SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U` matched the
reviewed transport. The literal SSH argv isolates configuration with
`-F /dev/null` and fixes strict host checking, the user known-hosts file,
the ED25519 algorithm, and batch mode. These are local checks, not a host
handshake.

The final request binds one target on trn22's proposed ext4 root, the
exact capsule SHA/size, a fixed private `O_EXCL|O_NOFOLLOW` marker and a
second fixed exclusive journal. The source-stage contract/receipt and
reused code are rehashed before either authority file. The final review and
Supervisor assignment are absent at their fixed paths, as are the fixed
marker and journal. The supplied final-wrapper tests passed **6/6**; the
reused core suite passed **15/15** and the earlier independent suite
**9/9**. `git diff --check` passed. Local tests cover missing review,
request/hash drift, structured review-to-assignment binding, private
one-shot marker, local STOP/breakers, known-hosts drift, root-device alias
checks and failure postflight. No test or receipt here establishes current
trn22 mount, device aliases, STOP, process/Neuron, protected checkpoint,
caches, disk or target absence.

## Blocking findings reproduced against `b80c69c4`

1. **The four-session host identity chain is not joined.**
   `stage_r4_transport.compare_sessions()` compares only write to verify;
   `compare_postflight()` compares only preflight to postflight and checks
   the target stat. The final wrapper calls both, but never requires the
   write/verify boot, mount namespace, mount ID, aliases or parent identity
   to equal the accepted preflight identity. A synthetic pre/post probe on
   boot A and write/verify receipts on boot B passes both comparators and
   returns `READ_ONLY_POSTFLIGHT_PASS`. The first negative test states the
   required rejection and fails with `RuntimeError not raised`. In a
   possible A-to-B reboot between probe and write, the remote writer can
   create the target on B before postflight detects the changed boot; B
   has not passed the full protected-asset, STOP and quiescence probe.
   This breaks the runbook's promised identity continuity across all four
   sessions and the fresh prewrite gate.

2. **A host STOP after the separate probe does not gate the write.**
   `stage_r4_remote.write()` checks identity, free space and target absence,
   then opens the target exclusively; it does not inspect `STOP_PATHS`,
   process/Neuron/lock quiescence, or a preflight identity token. The
   controller's local STOP/breaker check in the final wrapper cannot detect
   a newly placed *host* STOP. An offline temporary-directory fixture put
   STOP in place at writer entry; the actual writer created and fully wrote
   its new target. The second negative test fails with
   `remote STOP was present but writer created target=True`. A later
   postflight may mark HOLD, but it cannot undo an unauthorized host write.

`PYTHONDONTWRITEBYTECODE=1 python3 -B
research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/independent_final_negative_tests.py
-v` produces **2 expected failures out of 2 tests** on the reviewed commit.
The fixtures are local, use temporary files, and never invoke SSH. The
positive suites passing does not close these gaps.

## Boundary and repair gate

Keep the fixed final PASS JSON absent. A separate Implementer should bind
the accepted preflight identity to the remote writer **before** its
`O_EXCL` open, recheck host STOP/quiescence at that boundary, and verify
preflight → write → separate readback → postflight continuity and ordering
in the final result. The changed exact code/request bytes then need a new
independent final review. A fresh read-only trn22 feasibility check and a
distinct exact Supervisor one-attempt assignment remain separate gates.
Even a later successful transfer may end only at
`TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW`; no checkout,
native run, evaluation, monitor action or submission is authorized here.
