# Independent prelaunch review: 120-second compile-smoke proposals

**HOLD for both prepared assignments as written.** Reviewed the two new
Researcher proposals at `f51146bb` and their exact prepared-disabled
Supervisor assignments; this Verifier did not author them, contact either
host, or authorize training. The older 1800-second warm-S0 and 116188-cold
assignments are each `SUPERSEDED_UNLAUNCHED_BY_120_SECOND_COMPILE_SMOKE_INSTRUCTION`
with `launch_authorized:false`; they must remain inactive.

## Evidence that otherwise supports both smokes

- Exact source hashes independently match: S0
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`
  and 116188
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  Trusted `prepare.py` hashes to
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Both files parse `--max-train-seconds` as an integer (S0 at line 1250;
  116188 at line 1303); no model/source edit is proposed. The 116188
  science flags before the sole appended budget argument match its
  historical full launch exactly. Full-chip LNC2/four-rank topology is kept.
- Existing four-rank/three-step CPU G2 receipts using the *real* trusted
  `prepare.py` PASS for the source/default argv:
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-s0-bare-trn22-r1.json`
  and `.../fs-20260925-g2-s0-116188-trn22-r1.json`. They are **not**
  exact-120-second-argv G2 receipts. Under the latest human break-glass
  diagnostic instruction, base-G2 PASS plus this independently checked
  parser-only `--max-train-seconds 120` delta may substitute for exact G2
  if the assignment/Operator receipt says so explicitly; never mislabel the
  base receipt as an exact-argv PASS.
- The two new labels and run roots are distinct from the consumed S0 r1
  failure, the running/completing A0 controls, and the superseded full-run
  assignments. Each new assignment requires its same-host A0 **training and
  fresh public-20M evaluation** to be fully terminal before staging, plus
  fresh ≥6.5 GiB disk, no Neuron job/device/lock holder/STOP, source/data/
  tokenizer/prepare hashes and imported path, and unique-root preflight.
  Only `train.py` is copied; trusted `prepare.py` is imported via a pinned
  `PYTHONPATH`. These are appropriate prerequisites, not completed facts.
- trn22 cold smoke binds seven distinct, absolute, job-owned cache paths,
  including an explicit initially empty `NEURON_COMPILE_CACHE_URL`, and
  demands files under that URL after compilation. trn21 warm smoke names
  the populated persistent/native caches and requires hit/path evidence;
  without a proved hit it can show S0 starts natively but not cache causality.
  Neither smoke includes a BPB evaluation, `decide.py` verdict, submission
  clearance, or a valid full-budget quality comparison.

## Blocking contradiction: impossible direct per-rank boundary proof

Both proposals and assignments require `all four ranks` to show ≥20
step-boundary evidence, while both `train.py` files call the trusted
`prepare.record_step_boundary(step)`. In `prepare.py` lines 96–110 that
function **only writes for rank 0**, and only if
`NEURON_COMPETITION_R1_STEP_LOG` is set. Ordinary step printing is also
rank-0-only. The unmodified sources therefore cannot emit direct per-rank
step-boundary logs. Requiring those logs either makes a legitimate run
unpassable or tempts an unauthorized source/evaluator edit.

Narrow acceptable amendment, without changing either source or scientific
test: set `NEURON_COMPETITION_R1_STEP_LOG` to a unique, precreated,
job-owned append-only file path inside that run root (the sole permitted
`NEURON_COMPETITION_R1_*` instrumentation exception; if an OUT_DIR or data
cache override is truly needed, pin it separately without altering seed,
budget, shape or data). Define ≥20-step proof as **rank-0 ≥20 distinct
completed boundaries + all four worker exit codes zero**, with the existing
per-step synchronized distributed collectives and the source's final
`dist.barrier()` after checkpoint save. Archive raw step log, four worker
exit statuses/torchrun return, finite checkpoint metadata/hash, all output,
and quiescent postflight. A clean exit before 20 boundaries is
INCONCLUSIVE, never PASS; a true compiler exception is FAILED. Explicitly
record the no-exact-120s-G2 break-glass allowance and the statically
verified single CLI delta in the final assignments. This resolves the
contradiction without adding code or weakening the 20-step criterion.

## Separate dispositions and state transitions

- `research/v2/assignments/20260925-trn21-s0-warm-compile-smoke-r1.json`
  (SHA-256 `71cdcd013afbc138f6a1a3d26db3bec2acec3220b5a2c150788de94d5c9f970f`):
  **HOLD** until the boundary/step-log rule is repaired as above, and the
  final assignment retains its six warm-cache paths plus persistent URL.
  Their population and lack of concurrent writers are future Operator
  preflight checks after trn21 A0+eval, not an additional proposal defect.
  No state transition now.
- `research/v2/assignments/20260925-trn22-116188-cold-compile-smoke-r1.json`
  (SHA-256 `9dedf5a5b67e1f80ab973eabfe5ceb27e25e78b4153ad74ef200a61e21d50078`):
  **HOLD** until the boundary/step-log rule is repaired as above. Its current
  empty-cache and effective compiler-URL method is already specific enough;
  performing and recording that method is a future Operator preflight/result
  gate after trn22 A0+eval. No state transition now.

After a Researcher/assignment-author correction, a different Verifier must
review exact revised bytes. If it passes, the **only** assignment fields a
Supervisor may then change are `assignment_status` to `ASSIGNED` and
`launch_authorized` from `false` to `true`; fresh Operator preflight still
controls dispatch. A short smoke, even if clean, does **not** make S0
submittable. S0 still needs its own new-label true-cold, full-budget replay
and fresh-process public-20M evaluation.
