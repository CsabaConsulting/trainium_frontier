# Operator terminal receipt — trn22 r10 read-only timing, 2026-09-25 10:43 UTC

**Disposition: raw observation retrieved, independent result review pending.**
This is not source-stage, native-training, evaluation, or submission authority.
No host write, process signal, new checkout/bundle, one-attempt stage marker,
training, evaluation, cleanup, breaker reset, or AWS action was performed.

I invoked the exact reviewed `invoke_timing_readonly.py` once from the clean
stage execution root
`/home/poojith-devan/Desktop/Trainium-trn22-r10-stage-r3-active-root`
at HEAD `b3ec7406f0e7a3df6848698876e2106305dd283f`. The timing package was
clean at HEAD `440cad6f87021dee158c432778c5ef804cf8bc88`; the invoker SHA-256
was `753c61db6a8e416b18bad88f68db3aa19acb350f3e1aed760802459c46600de5`.
The pinned train, prepare, stage operator/remote/request, timing
collector/diagnostic/request/review/assignment hashes matched the independent
review and Supervisor assignment before contact. The fixed result was absent.
The active Desktop controller and all listed local STOP paths were clear;
`trn22.launch_allowed=false` and global/trn22 breaker `tripped=false`.

A separate, previously reviewed read-only trn22 snapshot at 10:41:43 UTC
confirmed UID/hostname/source identity, clean e65 checkout, 58,455,560,192
root free bytes, 31,660,021 free inodes, no matched train/eval process, no
Neuron/device/host-lock holder, and no observed host STOP. This was a
point-in-time feasibility snapshot, not a source-stage preflight. Its raw
receipt SHA-256 was
`72423a9a529b9d316b72c67f596e01ab4d4ef25c82bd3c5cc40df4d180f1d789`
on separate branch `codex/20260925-dual-host-readonly-1021`.

The exact one-shot invocation returned 0 and printed
`RAW_READ_ONLY_TIMING_RETRIEVED_PENDING_REVIEW`. The fixed
[raw result](OPERATOR_RAW_R10_TIMING_RESULT.json), SHA-256
`720d91ddf93cabdd11ccc718726bb673d470c4ac6ea2592c377b17c3e2b80b1a`,
contains three separate SSH phases (`probe`, `timing`, `probe`), all rc 0,
without timeout or stderr; the timing phase has one start, 20 trial, and
one end record. The collector found no parse/comparison error:

| Measure | Raw observation |
| --- | ---: |
| Clock-only modeled trials within the 1-s window | 20 / 20 |
| Timing misses | 0 |
| Double broad-label-scan duration, min / median / max | 963,413,623 / 976,377,573 / 980,357,752 ns |
| Longest double-scan margin below 1 s | 19,642,248 ns |
| Maximum pre-open label age | 69,574,401 ns |
| Maximum would-open label age | 69,580,937 ns |
| Maximum post-would-open age (all three fields) | 70,135,175 ns |

The margin is narrow. This clock-only model does not perform `O_EXCL`, create
a target or bundle, measure scheduler delay at a real open, or prove a future
source stage will pass. A separate Verifier must review the raw record before
any stage decision. The stage still needs a different enabled request,
independent final review, fresh preflight, and Supervisor one-shot assignment.

A second separate read-only trn22 snapshot at 10:43:20 UTC found the host
quiescent afterward: 58,455,519,232 root free bytes, 31,660,021 free
inodes, no matched process/Neuron/device/lock holder or observed STOP, and the
same clean e65 source. Its raw receipt SHA-256 was
`bd4c5415f9c4cc9bac92509142278204b68a728c4a1e8ec9c1fd4cca4c71ff5b`
on `codex/20260925-dual-host-readonly-1021`.
