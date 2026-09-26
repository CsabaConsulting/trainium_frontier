# Failed first clean cold-five binding — no native result

The one scoped launch dispatched at 2026-09-26 04:07:04.877608 UTC.
Unchanged `submission_rehearsal.py` failed before spawning torchrun:

```
File "submission_rehearsal.py", line 38, in main
    assert sandbox.is_relative_to(root / "out")
AssertionError
```

The proposed sandbox `/home/ubuntu/trainium_confirmed36_clean_cold_20260926_trn22_r1/train`
is outside the protected harness's project `out` directory. Terminal `CT_HARNESS_FAILED`,
returncode 1 at 04:07:05.029980; outer returncode 1 at 04:07:05.035587.
No training, native model test, five-step boundary or checkpoint was produced.
All scientific bytes, harness, caches, failed files and existing artifacts are preserved.
No retry, cleanup or successor was attempted.

At 04:08:04 the complete 253-row process census shows no known PID/PPID/PGID/SID
or root-related descendant; all three known dispatcher/flock/driver IDs
3275899/3275900/3275901 are absent. Devices and fuser are idle and the original
66305:832589 lock is free. Short-lived timeout/harness identities were not observed;
they are not invented. No worker or compiler could start past the line-38 refusal.
The full census and job/compiler-name checks are archived explicitly.

Raw bundle SHA256: 4e30030f17bc58943d0dc76318e297ab7c3bd5f8f6c3211809c37936989b5326.
20 complete files, 136552 bytes, no checkpoint download. Free disk 20237492224 B.
trn21's independently launched anchored control remains untouched.
