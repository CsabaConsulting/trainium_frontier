# Independent final PASS — trn22 r10 source stage r5 only

**Verdict: `PASS_SOURCE_STAGE_ONLY`** for the exact clean Implementer subject
`7f5e0bd3620d3a8ae0c64da3a132093542208219` on
`origin/codex/20260925-trn22-r10-remote-timeout-repair-r1-disabled`.
This supersedes the r4 Operator's local-preflight HOLD observation
`532ded624ffe04caa9af22f94e79ea792052b4f2` only for the identified
*local-uutils false veto*. The new collector-format independent review is
`research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW_R5.PASS.json`,
SHA-256 `da237ac25e8547c7cc9b0cc6dbf1fec9e2875af6be94b7e407d17616181a0994`.
This is **not** the distinct Supervisor assignment, Operator dispatch,
host/source staging, G2, native-training or evaluation approval, AWS action,
submission, or a BPB result. I made no host/SSH contact and no operational
marker, transfer, cleanup, training or evaluation.

## Exact r5 repair and authority

- The r4 attempted operator was stopped before SSH or marker because local
  `/usr/bin/timeout --version` reported uutils 0.8.0 even though the fixed
  320-second deadline executes remotely. The r5 operator
  `6224763c0acdbba8bfadf4281498fe9801a8b8d4aaee3d37f8b58f2f97b412be`
  removes that *local* test. It probes the remote absolute
  `/usr/bin/timeout --version` through the same pinned SSH host, known-hosts,
  key, options and endpoint as a phase. The probe is read-only and bounded
  locally to 25 seconds; only a zero-exit, empty-stderr, bounded-stdout,
  well-formed first-line GNU coreutils version passes. Non-GNU, malformed,
  nonzero, stderr and timeout replies fail closed. The actual remote phase
  still uses `/usr/bin/timeout --foreground --signal=TERM --kill-after=15s
  320s` and a separate local 350-second wait. **The live remote version is
  unknown** until the authorized Operator performs the probe.
- Stage entry rechecks active Desktop STOP/breakers after SSH identity and
  again after the read-only timeout probe, immediately before the
  `O_EXCL` one-attempt marker. Each phase rechecks after its probe and
  immediately before its phase SSH. Independently injected post-probe
  STOP prevented marker and host phase; a remote uutils reply was rejected.
- The canonical request is SHA-256
  `25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2`.
  New `INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW_R5.PASS.json` and
  `SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT_R5.json` paths/schemas
  cannot be satisfied by retained r4 review/assignment files. I tested
  both historical files against the r5 equality gate independently; each
  was rejected. There is **no r5 assignment**. The active controller root
  is `/home/poojith-devan/Desktop/Trainium`; former Documents and package
  STOP roots remain checked. The label, target, remote bundle and marker
  path are unchanged, the marker is O_EXCL and prior markers are refused,
  and a failure retains even an empty provisional bundle with no cleanup
  or retry.

## Source identity and offline evidence

- Unchanged remote `remote_incremental.py`:
  `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`.
  Unchanged 109,800-byte bundle:
  `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
  `git bundle verify` shows sole prerequisite
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c` and exact HEAD
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`. Clean science root
  `train.py` is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The sole enabled runbook command names the exact clean r5 worktree
  `/home/poojith-devan/Documents/ChatGPT/Trainium-trn22-r10-remote-timeout-repair-r1-disabled/research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py`.
- On exact clean subject **before** adding this review, its 14-entry
  `SHA256SUMS` verified, `git diff --check` passed, and 55/55 package tests
  passed under normal Python; 55 ran under `-O`, all passed with one
  expected skip. The real direct entrypoint returned HOLD under normal
  Python for absent r5 review and under `-O` for forbidden optimization;
  the named marker remained absent. Independent adversarial checks rejected
  the historical r4 review/assignment, rejected remote uutils, and proved
  post-probe STOP prevents marker and host phase. After adding this review,
  the 14-entry manifest still verifies, all 55 tests still pass under
  normal and `-O` (one optimized skip), and the direct entrypoint still
  HOLDs before SSH/marker for absent r5 assignment. I parsed this actual
  review as canonical JSON and called the exact `_authority` gate offline:
  a hypothetical matching assignment **in memory** was accepted, while
  mutated review/request and assignment/review hashes were rejected. No
  assignment file was created. No test contacted trn22. The unchanged
  runbook SHA-256 is
  `6300f980edd9988543b0de219749d806aae7d60fc2e7dd92d48b34d7b993ebee`;
  unchanged manifest SHA-256 is
  `8cb209324bcda649ea815dbd64889cf4e394afc25a93d6ff0b40aabf3aeaef8b`.
- The unchanged remote source checks source e65 clean identity, host lock,
  STOP, process/Neuron activity, label uniqueness, root free bytes/inodes,
  O_EXCL bundle creation, first-open and post-open one-second freshness,
  exact bundle/checkout identity, distinct-session readback, and
  unconditional postflight. The previous independent modeled timing
  observation `02f02d7bb1cc0a8d4b404c7373ba656bc6725562` passed 20/20
  read-only trials but had just **19.642 ms** worst double-label-scan
  margin. Its remote pre-open bytes are unchanged, so this is bounded
  feasibility evidence, **not** a live-open or r5-operator timing proof.
  A too-slow scan safely refuses before O_EXCL; a delayed actual open is
  checked before payload and any empty provisional inode is preserved.
  The narrow margin is a material *one-shot success risk*, not evidence
  of an unsafe write. No retry is authorized by this review.

## Remaining gates

This review file must be integrated into the designated clean r5 execution
worktree without changing operator/request/remote/bundle/science bytes.
The separate Supervisor must independently verify this exact review hash,
current active Desktop STOP/breakers, absent one-attempt marker, package
identity and the live risk; only then may it issue the **new r5** one-shot
assignment. The Operator must perform fresh local and host preflight;
the remote GNU probe must pass live. Any refusal remains HOLD, not
cleanup or retry authority. Raw marker journal, distinct-session
readback and attempted unconditional postflight still require retrieval
and independent result review. Native training/evaluation remain separate.
