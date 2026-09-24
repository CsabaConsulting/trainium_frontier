# Independent Verifier: repaired c5c A2 native-30 smoke package

Reviewed Implementer commit `4d69527648ffebea3c389123fe782987a6eb245a`
on 2026-09-25 IST, independently of its author and the Operator. The prior
STATIC FAIL receipt `research/agentic/verification/20260925-v2-c5c-a2-native30-static-review.md`
remains unchanged.

**Verdict: STATIC PASS FOR OPERATOR STAGING ONLY; NATIVE DISPATCH HOLD.** The
missing STEP_LOG-parent preflight is repaired without changing model or evaluator
bytes. This receipt permits the Supervisor to consider a distinct Operator
staging assignment; it is not a G2 pass, a smoke authorization, a full A2
authorization, a BPB result, a breaker reset, or a submission/adoption decision.

## Independent checks

- The repair changes only five A2-package files: the smoke contract, runbook,
  handoff, new read-only observer adapter, and its local regression test. The
  reviewed prior FAIL receipt is unchanged. No `train.py`, `prepare.py`, data,
  tokenizer, evaluator, scoring, queue, decision tool, or launcher bytes changed.
  `git diff --check 436d2e1e 4d695276` passed, the contract parsed as JSON,
  and the review worktree was clean before this receipt was written.
- Recomputed SHA-256 values match the repaired contract: c5c research-v2
  `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
  queue `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00`;
  original A2 contract `a40723151099e0fec912d670b702ef9f2774fe14853025660a9ffc116d2fc364`;
  previous A2 static PASS `c1f4665335c3af3201593ff3c1770b21c6430abf821be078422e890f4a16d85d`;
  prior native-package FAIL `a802f15a26fadc59e1ab0d5507b065213e14c3121e2cd763c554d3561dd4e3f1`;
  A0 observer core `a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`;
  new adapter `2eda64235b33ecd62cad193091bc4b434526b1c6ae2ccb5c7df4adaed1ff9022`;
  regression test `a7b9ba36a5a25e7a98a36b96c82846b4d3a410c4e06969c55b47d0c4f1b9bee8`.
  The repaired smoke contract itself is
  `498c088c60a9e54dc5091ca936d51f867e69dacb3eff4f3e70af9a99e6`.
- An independent ordered-array comparison returned `true` for repaired
  `smoke.train_argv == original.native_smoke.source_command_argv_prefix +
  original.a2_argv + original.native_smoke.argv_extension`. It retains
  LNC2/four ranks, exact c5c A2 bigram+trigram layers `1,-1`, multiplier 64,
  dimension 128, BF16/RMSProp, matrix LR 0.015, `--no-eval-public`, and only
  the diagnostic `--num-steps 30`/unique output extension. The clean environment
  pins seed 42, 1800-second parser budget and `NEURON_CC_FLAGS` optlevel 1;
  unresolved effective paths and final command hash correctly remain null.
- `prepare.py`'s unchanged `record_step_boundary` appends/fsyncs but does not
  create a parent. The adapter now requires a canonical, symlink-free,
  private label directory outside the pinned source checkout; exact run-UID
  ownership and writable/searchable access; stage-bound device/inode/UID/GID/mode;
  an absent STEP_LOG file; and a nonempty, single-link, same-UID sibling
  write probe with a matching SHA-256. The runbook requires that probe to be
  created exclusively and fsynced by the run UID before observer or guard,
  preserving it as stage evidence rather than precreating STEP_LOG.
- The adapter separately validates the guard-receipt directory inside the
  checkout, its stage-bound identity/ownership/mode and exact label paths;
  checks the guard script's path and staged hash; loads only the independently
  reviewed A0 observer bytes after their SHA-256 matches; and delegates to
  that unchanged `observe()` core. The core authenticates the live guard/child,
  process manifest's identical STEP_LOG assignment, append-only ordered
  boundary observations and guard postflight. The adapter additionally requires
  exactly boundaries `0..29` and rechecks both parent identities and the probe
  hash after the guard finishes. Any failed check returns nonzero/non-PASS.
- I re-ran the ten local-only adapter regressions: all passed. They cover
  positive parent/absent STEP_LOG, missing parent, precreated STEP_LOG, changed
  parent identity, missing write probe, missing/changed guard parent,
  unwritable parent, symlinked scratch component, and parent inside source.
  The adapter CLI accepted its complete declared argument surface with
  `python -B .../NATIVE_SMOKE_BOUNDARY_OBSERVER.py --help`. These are local
  static/CPU checks, not a host or Neuron result.

## Remaining HOLDs and narrow next proof

1. A distinct Operator may only **stage** the fixed-label package, after the
   Supervisor chooses it. The host must prove the unique label/output/probe
   paths unused; bind a reviewed scratch root and same-UID positive probe;
   recheck parent and guard-receipt device/inode/owner/mode at observer start
   and immediately before dispatch; bind identical canonical STEP_LOG bytes
   in `env -i`, adapter argv, guard manifest and the immutable request; and
   independently review that final request. The guard script and six warm
   cache path identities, checkout commit, disk/HBM projection, fresh host
   idle/Neuron/lock checks, STOP/breaker and one-shot authorization remain
   unresolved. The source checkout's `research/run-logs` is **not Git-ignored**
   by the reviewed `.gitignore`; stage/postflight must not assume it stays
   clean once guard receipts appear. Archive those receipts without deleting
   protected evidence, and retain the clean-before-launch proof.
2. The exact m64/d128 A2 four-rank/two-step G2 RMS trace has not passed
   independent review. Neither an m8/stub trace nor the A0/S0 G2 can substitute.
   No native smoke may launch before that, final stage/request review, a fresh
   preflight, a conservative A2-specific disk/HBM bound, containment and
   one-attempt authority. The 30-step result itself requires clean quiescence,
   checkpoint integrity, synthetic causality/batch-independence and four-rank
   finite/equality evidence; ordinary c5c logs may be insufficient.
3. The queue's older IMPL-1 criterion is **50 native steps with equal per-rank
   table hashes**. The exact 30-step smoke does not satisfy it. Before any
   full scored A2, commission a separately proposed, new-label, independently
   verified at-least-50-step exact-profile diagnostic that records table
   weights and RMSProp moments on every rank after updates, or obtain a
   durable explicit human supersession. Merely running more steps or showing
   rank-0 loss/checkpoint is not a rank-equality proof. A scored A2 also needs
   a valid archived same-host c5c A0 public20M control and its own one-shot
   authorization.

No SSH, host staging, CPU G2, Neuron smoke, training, evaluation, AWS action,
protected deletion, breaker change or leaderboard submission occurred in
this review.
