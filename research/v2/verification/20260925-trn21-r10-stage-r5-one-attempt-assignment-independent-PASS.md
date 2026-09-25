# Independent assignment-only PASS — trn21 r10 r5 source stage

**Verdict: PASS for the exact local one-attempt assignment binding only.**
Reviewed clean
`origin/codex/20260925-trn21-r10-stage-r5-enabled-final-pending@4a15c0ada4a5699bca7b08b8e669b11abdb0c1a1`,
whose sole change after the independent final request review is the canonical
`SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json`, SHA-256
`f30803dc499966d58d52243b703ca2ec826ed93b182ded616bdfc70331ed6eb4`.
I did **not** call `operator_once.py`'s main/stage entrypoint, make SSH/host
contact, create a marker, transfer a bundle, train, evaluate, clean anything,
change AWS/breakers, or submit. This receipt is not a fresh Operator preflight
or proof of a completed source stage.

## Exact binding and offline checks

- The assignment has exactly the frozen
  `trn21-r10-incremental-stage-supervisor-assignment-v2` dictionary: trn21,
  stage-only scope, unique r2 label/marker, one attempt, no native authority,
  exact enabled request SHA-256
  `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a`,
  and exact independent final PASS JSON SHA-256
  `d1170f0ac27285d4ca5670dda9828da2a7cc76a7527f2a6eb0b90239479b9ff3`.
  It is sorted, compact canonical JSON. I called only the **pure**, local
  `_authority` equality check on the actual request/review/assignment bytes:
  exact values passed; altered request hash, review hash, one-attempt bit,
  native-authority bit, or marker were each rejected.
- The assignment commit changes no source, evaluator, request, or review.
  Operator SHA-256 remains
  `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b`,
  remote source
  `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6`,
  and e65→r10 bundle
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `train.py` / `prepare.py` remain
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` /
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The existing 13-entry SHA-256 manifest passed. The 41 synthetic package
  tests passed normally and 41 under `python3 -O`; no real SSH or host
  process was run.
- At `2026-09-25T12:07:45Z`, read-only local checks found active Desktop
  `research/v2/state.json` with `launch_authorized=false`, trn21
  `launch_allowed=false`, and global/trn21 breakers untripped. None of the
  active Desktop, historical Documents, or designated package STOP paths
  existed, including broken symlinks. The exact local r2 one-attempt marker
  was absent, and the local source-artifacts directory had **no trn21-named
  marker or other trn21 entry**. These are point-in-time local observations,
  not host preflight or a promise that the state remains clear.

The independent disabled r5 static PASS is `739c5dac` (receipt SHA-256
`99e1fd9770171467c3679a396dff63d648f858513c0dcf1176eb0ad9257b7b2a`),
and the independent exact enabled **source-stage-only** final PASS is
`637966d6` (machine JSON SHA-256
`d1170f0ac27285d4ca5670dda9828da2a7cc76a7527f2a6eb0b90239479b9ff3`).
Neither grants native training. The previously held r4 request remains HOLD
at `06451f16`; its late local STOP-to-probe defect was repaired in r5, not
waived.

## Remaining gates and risk

The designated Operator must repeat fresh local/host preflight and exact
source/command/hash, STOP, process/Neuron, lock, disk, ACL and marker checks
before the one-shot command. The exact operator then checks the remote GNU
timeout through its pinned read-only probe before marker consumption and
again before each phase. The remote
one-final-scan **live** duration remains unmeasured; its strict finite
≤1-second actual-open gate can fail closed and consume the one-attempt
marker. A persistent duplicate behind the final pre-open scan can leave a
retained empty provisional bundle, and the same-UID/privileged post-open
race remains the documented bounded-model residual. There is no cleanup or
retry authority. Even a completed stage requires raw evidence retrieval,
independent result review and separate native A0 authorization before any
training or BPB claim.
