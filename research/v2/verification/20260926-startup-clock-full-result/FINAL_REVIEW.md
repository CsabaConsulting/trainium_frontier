# Full36 final result: authenticated, not adopted

Supervisor/Analyst review 2026-09-26 00:28 UTC. Root did not implement this
scientific source or operate the host. This is result authentication, not a
fresh source/compliance review or an upload approval.

Root independently reran 65 training and 68 final checks, normal and `-O`,
with identical outcomes. Exact committed raw e5a29999 contains 26 files,
410,104 bytes; bundle SHA256
2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99.
Source/argv, all raw-file hashes, fresh context-1024/B1/public-20M totals,
process exit, unchanged post-eval checkpoint, original lifetime lock, and
15 observed process identities absent in 254 unfiltered rows authenticate.

Fresh public BPB is **0.9883821509811542**, independently recalculated from
57,179,468 nats / (83,462,186 bytes * ln(2)). Training completed 3,630 steps;
actual fifth 807.693891s, process wall 2597.199032s, charged through exit
1789.505141s. No over-budget result or eval timeout. The full run reused
research caches; the separate cold-five 845.627338s is not a cold full replay.

Comparator roles are preserved:

- Same-host confirmed 6ef7 Recursive6x schedule arm: 0.9902787975734985;
  delta -0.001896646592344231. This is a different-stack comparison, not an
  isolated test of startup clock or architecture.
- Same-host raw 3e79 old-clock parent: 0.9863403052402583;
  delta +0.0020418457408959423. That original result failed cross-host
  confirmation; it is neither adopted nor retroactively a formal control.

Unchanged `decide.py` returns **INSUFFICIENT**: no eligible same-host formal
control exists for base6ef7 on trn22. The historical schedule arm is not
relabeled. A genuine prospective control packet is separately prepared;
cross-host confirmation of this exact stack is warranted, not yet assigned.
No base adoption, prediction of official BPB, or submission-ready claim.
Canonical confirmed mean remains 0.9901674436550496.

Reproduce with `audit_final.py`; ledger-row.json is this observation only.
