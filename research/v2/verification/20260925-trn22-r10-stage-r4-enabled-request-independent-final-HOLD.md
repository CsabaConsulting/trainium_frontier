# Independent final HOLD — trn22 r10 source-stage r4 invocation path

**Verdict: HOLD for final source-stage dispatch of exact Implementer subject
`1a152d208862c9f317743165c9e0fb11a1438910`** on
`origin/codex/20260925-trn22-r10-stage-r4-late-stop-recheck-r1`. This
supersedes the earlier r3 enabled request `16e1c982` but does not lift its
independent final HOLD `22b57441` until the whole r4 package, including
its single-attempt command, is internally consistent. I did not contact
trn22 or any host, write the operational marker, transfer a bundle, create
a target, assign a stage, run native training/evaluation, clean artifacts,
take an AWS action or submit. No final PASS JSON was created.

## Single narrow remaining defect

`research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/FINAL_PENDING_RUNBOOK.md:103`
still directs the Operator to execute:

```text
/usr/bin/python3 -B /home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r3-enabled-request-r1/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py
```

That is the old **held r3 worktree**. Its actual operator SHA-256 is
`a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda`,
not the repaired r4 operator
`af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f`
required by the new canonical enabled request. As checked now, the old
worktree lacks a final review and assignment, so it would fail closed;
nevertheless the published command does not execute the reviewed source
and cannot be a final one-attempt authority. This is an operational
integrity HOLD, not a newly discovered unsafe remote write path.

**Minimal repair:** change that command to the single designated clean r4
execution root
`/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-stage-r4-stop-recheck-r1/.../operator_once.py`,
update the runbook entry in `SHA256SUMS`, and add a test binding the command
path to the exact current package/operator bytes. Do not change
`operator_once.py`, the enabled request, remote source, bundle, science,
label, target or marker. Return the resulting exact commit for independent
final re-review. Integration of any later Verifier PASS and Supervisor
assignment into that exact execution worktree remains a separate gate.

## Checks that passed and carry forward only to unchanged bytes

- From `16e1c982` to `1a152d20`, the sole operational change is one
  `_local_stop_and_breakers()` call after `_ssh_identity()` and immediately
  before `_marker()` in `operator_once.py:320`. I independently injected a
  late STOP and a late breaker at that seam using mocked authority/SSH
  inputs, normal and `-O` bytecode: each produced **two local gate calls,
  zero marker calls and zero SSH/remote calls**. Direct normal and `-O`
  entrypoints still HOLD before host contact because the exact final review
  and Supervisor assignment are absent. The active state root remains
  `/home/poojith-devan/Desktop/Trainium`; former Documents/package STOP
  roots remain checked. This closes the specific r3 STOP-to-marker defect.
- The canonical enabled request SHA-256 is
  `d19086a3f0c8aff9835f14f641d528980b12b21f45310070f6613ae261a33958`;
  it pins r4 operator
  `af888c89dbed4360895e5cf93299095efe3280893dafd1e504cd88b6a9c4408f`.
  The remote source remains
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
  the 109,800-byte bundle remains
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  Root `train.py` and `prepare.py` remain
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Source/target/label/marker and one-attempt no-cleanup policy are unchanged.
- `sha256sum -c SHA256SUMS` passed **11/11**; `git bundle verify`
  confirms sole prerequisite `e65a3097a5b00f6178188a935bb2acb9a5042c6c`
  and exact HEAD `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`.
  The source worktree was clean and `git diff --check` passed. Full offline
  tests ran **45** normal with no failures and **45** with `-O` with no
  failures (one intentional optimized-mode skip). `SHA256SUMS` itself is
  `beb0575c3da48b1894ea5019709887dce1ce7794e157e891587f37463503dbce`.
- Independent timing review `02f02d7b` observed **20/20** read-only
  modeled passes of the **unchanged** remote pre-open sequence. The worst
  double label scan was 980,357,752 ns, leaving only **19.642 ms** to the
  one-second scan bound; the largest modeled would-open age was
  70,135,175 ns. The changed local pre-marker check does not alter that
  remote sequence, so this is bounded feasibility evidence for it, **not**
  a measurement or hash binding of the new r4 operator. The narrow margin
  risks a safe one-shot refusal, not an unsafe write by itself: a slow
  scan is rejected before `O_EXCL`, and a delayed actual open is checked
  again before payload. It does not guarantee actual `O_EXCL`, fsync,
  checkout or readback success.

After the runbook correction, final review must still authenticate the
exact command/worktree and canonical review schema. A different Supervisor
must then issue the one-attempt assignment and order fresh local/host
preflight. Stage-result, source-identity and native A0 gates remain
separate; there is no new BPB result.
