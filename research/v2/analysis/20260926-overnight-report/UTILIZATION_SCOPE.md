# Utilization lower bound, not an exact busy percentage

Window: September26 01:00–07:30IST (September25 19:30–September26 02:00UTC).
Parsed canonical ledger snapshot03e676bb:6 completed timestamped native jobs
on trn21 and9 on trn22, including failed and startup diagnostics. This is not
the number of full quality runs. Four evaluated full runs per host are in
that interval; trn21 also had a live confirmation subsequently completed.

Unique archived native launch-to-terminal intervals, plus the live trn21
confirmation only through its authenticated01:52:59UTC observation, prove
at least16,533.074601s (70.6542%) for trn21 and14,691.568810s (62.7845%) for
trn22. These include compile, training and attached evaluation wall time,
not accelerator-compute utilization. CPU gates and missing observations
are excluded. The trn22 warm-start diagnostic lacks a matched launch key
in this extraction and is conservatively excluded, although its terminal
is counted. Unobserved gaps are UNKNOWN, not automatically classified idle.

This lower bound cannot substantiate a claim that both hosts were always busy.
The explicit trn22 idle incident after01:47:04UTC is separately reported.
