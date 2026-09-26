# Independent native packet review: original 54fd9a10

**REFUSE.** The frozen validator accepts four incomplete or contradictory forms
of evidence. This verdict applies to original packet `54fd9a10`; it does not
alter the earlier adapter source/CPU PASS, authenticate native results, or grant
stage/launch authority. No candidate or packet input file was changed.

Input manifest:
`4c2248de3a8dd8fb91954d34ab7655808e7b0452d82b0bf48b9f09c1cfb1c3b4`.
All 18 supplied files listed in the manifest authenticate. README.md and the
implementer's test source were intentionally omitted and were not required or
consulted. The three separately supplied old files match their explicitly
provided hashes. Work used only this restricted packet and this output directory.
There were no repository reads during review, host/device/AWS/network operations,
Tensor/native module loads, installs, training or native launches.

## Blocking findings

Each counterexample begins with an independently generated, complete synthetic
four-rank archive that the validator accepts. I applied the named mutation to
that baseline. Both ordinary Python and `-O` produce the same four false accepts.
All files are explicitly marked `SYNTHETIC_ONLY`, not native result evidence.

1. **Contradictory candidate forward output is accepted** —
   `validate_receipts.py:99` only checks each output against its own claimed
   expected record. In `counterexample-contradictory_candidate_forward.json`,
   registered compiled sentinel output/expected are both replaced with an
   all-positive-zero FP32 record on all ranks. The other required forwards
   remain nonzero for exactly the same seven input records. Completion still
   succeeds. Required forward records must agree across same-input registered
   eager, registered compiled and ordinary compiled routes, including the full
   shape/dtype/finite/nonzero/hash record. Otherwise the archive can contradict
   the embedding forward contract while being labeled exact.

2. **Contradictory fixture nonzero counts are accepted** —
   `fixture_contract` stores hashes for inputs, and `inputs` at lines 61–70 checks
   those hashes without reconstructing/checking the complete tensor records.
   In `counterexample-contradictory_upstream_count.json`, every required sentinel
   route reports upstream `nonzero: 0`, while upstream_roundtrip and the pinned
   fixture still report the original nonzero count and identical SHA256.
   Completion succeeds. Required fixture input records should match independently
   reconstructed complete records; route upstream records should also equal the
   upstream_roundtrip record. A SHA cannot consistently represent two different
   counts for the same shape/dtype.

3. **Runtime identity can be omitted** —
   `validate` at lines 123–167 never checks `torch` or `nki_version`.
   `counterexample-missing_runtime_versions.json` deletes both fields from all
   rank receipts and still completes. Complete native evidence needs typed,
   nonempty runtime identities, agreement across ranks, and binding to separately
   approved target versions. Local source review does not supply that runtime
   evidence. This is distinct from the already checked environment-variable map.

4. **Malformed old-route mismatch evidence is accepted** —
   line 120 accepts the presence of an `actual` key as sufficient mismatch
   evidence. `counterexample-malformed_old_mismatch.json` changes an observed
   old compiled gradient's actual record to `{}` on all ranks; completion still
   succeeds. Old failures must remain separate, but their retained observations
   still need a coherent schema. A reported mismatch needs a well-formed actual
   and expected record and consistent status/exact/error fields. This should
   preserve legitimate old exceptions, missing tensors and nonfinite results as
   explicit failures, not turn them into candidate success or discard them.

The four `-optimized.json` counterexamples reproduce the same findings. Exact
mutation code is in `review_validator.py`; complete results are in
`validator-normal-attempt1.log` and `validator-optimized-attempt1.log`. No oracle
or candidate source was modified to construct or discover these defects.

## Validator checks that passed

The independent positive archive uses the protected stdlib fixture definitions
and an independently implemented scalar, token-ordered FP32 recurrence. BF16
byte encoding and all fixture input records are constructed in the reviewer
test. The baseline includes sentinel, ordered all20 cases, two changed-tangent
replays, four ranks, source/argv/label/environment/exit records, and preserved
old-route failures.

Of 30 mutations in each Python mode, 25 invalid cases are rejected, one legitimate
separate old execution error is accepted, and the four defects above are falsely
accepted. Rejections cover missing rank/case/replay/sentinel, sentinel-only
results, case order, wrong pins/argv/label/environment, nonzero and boolean exits,
missing candidate gradient, substituting an old route for a candidate, forged
zero actual-and-expected dW, NaN and both infinity errors, nonfinite tensors,
changed metadata, missing or concealed old observations, cross-rank input
differences, stale replay identity and adoption claims.

## Checker and mechanics findings

Independent source and mocked checks of the frozen checker/mechanics passed in
ordinary Python and `-O`. These bounded passes do not override REFUSE above.

- The adapter/kernel/reference hashes match the protected versions. All bound
  support pins, driver hash, command text, checker arguments, environment and
  six unique cache directories agree. Driver and rank-entry executable bodies
  are AST-identical to the supplied old files after excluding only the documented
  rebound constants. The guard and dispatcher match their pinned old hashes.
- The checker defaults disabled without loading Torch/NKI. Its original Function,
  tensor-record, comparison and route-observer helper sources match the unchanged
  old portions of the supplied diff. Strict comparison requires finite records,
  shape/dtype/hash equality and retains signed-zero distinctions. NaN and both
  infinities serialize safely as a failed comparison with null max_abs plus a
  nonfinite descriptor. The route observer preserves execution errors and attempts
  post-observation input records.
- Seven mocked checker outcomes establish unique FP32 sentinel first, a failed
  sentinel preventing the suite, all original cases in BF16 then FP32 order,
  both negated-unique replays, exactly one registration per dtype, and the same
  compiled callable identity reused for every case/replay of that dtype.
  Required-case failure returns failure after observations; old execution failure
  remains separate and does not suppress candidate observations. Final receipt
  writes are attempted, and initialized process groups are destroyed even when
  the mocked receipt write fails. Initialization failure does not attempt to
  destroy an uninitialized group.
- The preserved original-comparison subset is 18 nonempty cases times eight
  comparisons plus two empty fixed-storage cases times three = 150 per rank.
  Ordinary compiled controls and old-route observations are additional. This is
  source/control-flow coverage, not an executed native all20 receipt.
- Twelve mocked driver outcomes verify exact child argv/env/cwd/inherited FD,
  fresh guard before the sole synchronous child, validation before success,
  exclusive output/cache creation and no automatic next child. Wrong host, STOP,
  disk headroom, source pin, existing/dirty cache and guard refusal prevent a
  child. Nonzero exit, timeout code, execution exception and invalid receipts
  remain incomplete.
- Nine mocked idle cases and seven mocked inherited-lock cases cover busy
  process/device, malformed or missing process evidence, wrong LNC, command
  timeout/stderr/exit failures, wrong owner/symlink, wrong parent command,
  ambiguous FD, missing exclusive lock and mismatched FD inode. The good path
  retains the same authenticated lock FD. Busy process evidence still triggers
  both requested fresh checks before refusal.

The bound command remains world4/LNC2 with 600-second TERM timeout and 60-second
KILL grace. The binding names original lock device 66305/inode 832589. The shared
guard authenticates the currently opened lock and parent/FD relationship;
independent verification of that original persistent identity remains the
Operator's explicit preflight duty. Likewise, the old driver does not prove all
descendants/compiler processes quiescent on every outcome. The binding correctly
requires observed process identities, unfiltered postflight snapshots, fresh
device-idle checks and original lock identity before any later job. None of that
real lifecycle evidence exists from this input-only review, and a released parent
lock or zero child exit cannot substitute for it.

## Evidence and limits

Runtime for these tests was `/home/poojith-devan/venv/bin/python`, Python 3.14.4,
using only the standard library. No real Torch or NKI module was imported.
Checker-main tests use unchanged AST with plain namespace imports, virtual
receipts and mocked native APIs. Driver/guard tests replace all process/device/
filesystem effects with virtual in-memory operations. They are not simulations
of native kernel arithmetic or actual operating-system process cleanup.

Two early reviewer checker-harness attempts failed because argparse read the
real review-process argv, leaving the checker on its disabled branch. The exact
sources and logs are retained as `review_checker_attempt1.py`,
`review_checker_attempt2.py` and corresponding logs. Correcting only the mock
argv binding produced the passing third run. These are reviewer test failures,
not candidate failures. Their history is detailed in `FAILED_ATTEMPTS.md`.

`COMMANDS.md` records executions and exits. `OUTPUT_SHA256.txt` binds the completed
review artifacts. Original `54fd9a10` remains REFUSE pending a separately frozen
repair and fresh reassessment. No staging, launch, native qualification, training,
performance/BPB conclusion, adoption or submission follows from this report.
