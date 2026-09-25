# Independent static review — trn21 r10 r5 read-only diagnostic r1

Checked 2026-09-25 12:38 UTC. Verdict: **HOLD for any enabled diagnostic request**.
This is a Verifier-only, offline review of the disabled source commit
`2b1dbff6ca5d7d64cf6b0631fb1218dddddadf6e` on
`codex/20260925-trn21-r10-readonly-stage-diagnostic-r1-disabled`.
I did not implement the candidate, contact either host, run SSH, create an
enabled request or assignment, stage source, train, evaluate, clean, reset a
marker, or perform an AWS action. The checked-in entrypoint remains disabled.

## Evidence that passed

- `SHA256SUMS` verifies all ten listed files. The archived r5 local Operator,
  remote source, 109,800-byte bundle, 18,394-byte consumed journal and
  independent result HOLD have SHA-256 pins, respectively:
  `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b`,
  `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`,
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`,
  `54c65164271d895392c2e170b47dbdf07f62e0aab2cfa378e529afdbbeca3a4c`,
  and `786f3f26179db7a31adb7b4440460e662427494f7cbfe192c76878803cea14bf`.
  The r5 result HOLD says the one-shot marker was
  consumed, there was no accepted stage or persisted bundle/target, and the
  underlying write-phase cause remains unknown. The new source pins trn21
  `ip-172-31-34-85`, accepted e65
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, and unstaged r10
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`.
- The six-file commit changes only this diagnostic package. It does not change
  `train.py`, `prepare.py`, evaluator, tokenizer, or scoring code. The remote
  `preflight`/`postflight` call only the archived read-only `perform` branches;
  `preopen` calls the archived timeout identity, existing nonblocking lock,
  full probe, distinct-session binding, and original
  `_final_no_write_continuity`/`_freshness_evidence`. The reachable path has
  no `O_CREAT`, `O_EXCL`, `_write_bundle`, `_reconstruct`, Git fetch, or target
  creation. The original final check enforces a one-second scan duration and
  one-second activity/label/space ages. This is source evidence, not a live
  timing measurement.
- The archived full probe pins source/Git, mount, protected checkpoints and
  caches, STOP, process/Neuron, space, and lock conditions. Each phase uses a
  separate remote absolute GNU-timeout SSH probe, with local STOP/breaker
  checks before and after the identity/probe sequence. The fixed SSH host-key
  and key-file metadata checks are reused without reading private-key bytes.
- `python3 -B -m unittest discover` and the same command under `python3 -O -B`
  each ran 11 tests successfully. Direct local and remote entrypoints exit
  `HOLD` with status 1 under normal and optimized Python. The enabled request,
  independent final review, and Supervisor assignment files are absent.

## Blocking finding: malformed phase evidence can become diagnostic PASS

The local transport parser at `operator_readonly.py:224-232` checks only the
remote schema, phase, status and three false authorization/write flags. It
does not validate the phase-specific receipt. In particular, it does not
require the `preopen.would_open` freshness proof, its one-second age/scan
limits or root byte/inode proof, nor the full preflight/postflight snapshots.
The final `run_once()` gate at lines 297-316 checks only three `PASS_READONLY`
statuses and transport `PASS` values.

An offline adversarial test patched only the authority and SSH seams in
memory, with no actual enabled files or host contact. It supplied three
minimal `PASS_READONLY` phase receipts: preflight had `snapshot={}`, preopen
had no `would_open`, and postflight had no `snapshot`. The actual
`run_once()` returned
`READONLY_DIAGNOSTIC_PASS_NOT_STAGE_AUTHORITY` after three phase calls.
Consequently the local result can claim PASS without carrying evidence for
the very one-second pre-open gate this diagnostic is meant to observe. The
hash-pinned remote implementation normally emits these fields, but the
Operator's acceptance boundary must reject missing or malformed evidence
before producing a durable PASS. The archived r5 source-stage Operator had a
separate `_checked_final_gate()` for its analogous write receipt.

The same acceptance gap affects output sanitation. A synthetic
`HOLD_READONLY` SSH JSON response with the required marker fields plus an
unexpected `extra_raw_message` was accepted as transport `PASS`; the local
result retained that raw field. Failure/transport evidence is therefore not
restricted to the documented bounded enum and scalar fields, even though the
current pinned remote `_hold()` itself emits only sanitized values.

## Required repair and scope

Before an enabled request, validate exact phase-specific receipt keys and
types locally. Require the archived preflight and postflight full snapshot
identity/absence proofs; require `preopen.would_open` with finite ages and
scan duration in `[0, 1.0]`, the declared root byte/inode floors, and the
expected timed-check trace. Reject unknown remote fields or project only a
fixed safe set into the terminal result. Repeat independent adversarial
review on a distinct repaired source commit. No conclusion here authorizes
the read-only diagnostic, source staging, native launch, evaluation, cleanup,
marker reset, AWS action, or leaderboard submission.
