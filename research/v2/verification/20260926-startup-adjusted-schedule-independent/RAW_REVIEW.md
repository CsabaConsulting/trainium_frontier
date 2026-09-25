# Independent reduced-receipt audit

**PASS_REDUCED_RECORDED_EVIDENCE_ONLY** for frozen225f27d13c25c9160f5fd2758cf016fb4c70bf0f,
candidate7419927c/SHA36a86274 and baseline b4d83b5d/SHA3e79a7c1.
`raw_audit.py` performs616 stdlib checks in both normal and optimized Python;
raw content hashes and exact scalar histories are in RAW-normal.json and
RAW-optimized.json. This is separate from the genuinely fresh source session;
none of these conclusions is fed into that reviewer.

All six corrected arms have four completed ranks, finite recorded losses,
65 FP32 parameter gradient/update digest records per step, consistent final
cross-rank updates, and rank0 checkpoint/reload records. Baseline/default/OFF
match all three losses, ownership, optimizer histories, full gradient/update
and cache observations, and rank0 state/checkpoint hashes exactly. The two
six-step ON runs with779/790-second injected delays match their complete
recorded histories and checkpoint bytes. In both, the first five Muon scalars
are LR0.015/WD0.2; step5 resumes elapsed-time WD at0.1999996666666667.
The1000-second ON run charges excess (step1 WD0.18555477777777782, step5
0.18555244444444452) and intentionally differs from the below-cap trajectory.
ON is not claimed equivalent to OFF or evidence of improved BPB.

The actual fixture is eight layers/width128, CPU BF16 with FP32 parameters,
synthetic forward slices2×128 per rank, and fake time. Its production-looking
token counter is not proof of131072 actual tokens per update. Only rank0 emits
checkpoint/reload/final-digest summaries; every rank emits the actual per-step
gradient/update histories compared here. The three-step default/OFF identity
does not cover the fifth boundary itself; the independent static fake-clock
checks and six-step ON histories provide distinct coverage.

Failed attempts remain failed, not hidden: the first observer accessed an
attribute missing on OwnedOptimizer, yielding aggregate/rank FAIL on all four
ranks although torchrun returned0. Exit0 alone is therefore explicitly not
accepted. The next six-arm attempt used unchanged stub-prepare startup count2,
and its explicit five-boundary WD assertion failed. Those raw receipts and
transcribed comparison failure remain preserved. The final observer explicitly
restores imported startup count5 to the real-prepare value, without changing
candidate/baseline/prepare bytes. Own two mistaken schema assumptions during
this audit are retained in RAW_FIXTURE_FAILURE.json and corrected only in the
audit, not in evidence or candidate.

No checkpoint tensor was independently reloaded here, no observer rerun or
model/native execution occurred, and the digest comparisons do not establish
native finiteness or compiler behavior. Full113M standard G2, cold/native,
quality and adoption gates remain unexecuted or separately controlled. The
fresh source verdict is pending at this receipt's creation.

Reproduce from this worktree:

```sh
python3 research/v2/verification/20260926-startup-adjusted-schedule-independent/raw_audit.py
python3 -O research/v2/verification/20260926-startup-adjusted-schedule-independent/raw_audit.py
```
