# Follow-up after the first NKI Library queues

Current checkpoint 21:53 UTC: both hosts are idle, all199/197r2 jobs and profile
backups are complete. See [the handoff](HANDOFF-CLAUDE-2026-09-20.md) and its
machine-readable audit. The chronological notes below retain the original
planning and failures; they are not instructions to restart old queues.

Direct SSH checks at 12:23 UTC found both hosts idle, with no training/compiler
processes or accelerator holders. Source196's finite queue completed at09:05 UTC;
source195's at09:28 UTC. Backup cron preserved the results but did not dispatch
new work. Future status messages must distinguish a finite queue from continuous
agent research. No new quality run or BPB result was produced during the gap.

## Verified results

Every original frozen requirement and source fingerprint was rechecked locally.
All jobs returned zero and clean postflights. Full proofs are in
`run-logs/fs-w195-partial-position-proof-r1.json` and
`run-logs/fs-w196-vector-square-result-proof-r1.json`.

- The source196 production saved forward takes4.878602ms with Vector square,
  versus4.776949ms with the matched Scalar square. The Vector substitution is
  about2.13% slower despite exact outputs. Retain the Scalar implementation.
  NativeBF16 is5.327761ms; the existing Scalar FP8 forward advantage replicates.
- Source195 first-layer-only replacement: mean496.616637ms,
  median487.668197ms, mean microstep433.155515ms.
- Last-layer-only: mean682.661014ms, median682.696181ms,
  mean microstep627.420897ms. This is a large position effect with the same
  number of library calls and the same native custom backward in every layer.
- All six layers: mean564.512086ms, median564.248650ms,
  mean microstep507.779427ms. Cost is not proportional to replacement count.
- Bracketing zero-library controls: means499.107528/498.149224ms,
  medians488.653169/489.301529ms. Identical final loss5.126208782196045.
  Both controls and the first-only arm have a final-sample optimizer spike;
  retain every sample. Repetition at the same step warrants investigating
  schedule-dependent optimizer dispatch before calling it random system noise.

## Next finite experiments

Source197 reuses the immutable source195 checkout on trn22. It repeats first-only
and last-only timings and records the training forward of rank0 while all four
ranks execute. Each capture is followed by a separate raw-preserving Explorer
ingestion. Every model, native evaluation and source-identity check remains.
The comparison holds the number and shape of custom calls fixed and tests where
the extra surrounding work appears. Instrumented latency is not speed evidence;
match runtime dispatches to NEFFs and do not infer exact DMA traffic if metadata
is incomplete. Maximum sequence duration11000seconds; independent cleanup and
host ownership for all four requests.

Source198 is planned to measure the unchanged source194 saved FP8 forward under
one and four concurrent ranks, with all preparation and state stores charged.
The prior BF16 component's advantage disappeared under four-rank load, so this
screen is useful before committing to a complete FP8 backward. Preserve the
original source194 numerical prerequisite, exact actual-stage outputs on each
rank, paired orders and maximum-rank timing with collectives outside the timer.
No new kernel arithmetic or backward is part of this screen.

Complete ROW backward remains required work. The contraction-axis scaling
constraint in `fresh-start-library-fusion-next-2026-09-20.md` still applies.
Neither new sequence can promote a model to a quality run automatically.

## Storage recovery and harness correction

Source197 first capture passed every frozen requirement and clean postflight.
Its63 files (2,491,558,829bytes) are copied locally and every remote/local SHA
matches. The subsequent ingestion returned0 with31 tables and unchanged raw
hashes, but free disk fell below the8GiB postflight reserve. Preserve the failed
postflight; do not call the original ingestion fully clean. The exact owned
local coordinator was held via pidfd before another capture could launch.

Maintenance r5 captures only old same-UID /tmp/nki_* files before12:20UTC;
1916 files totaling12,894,108,282bytes are independently verified in the local
archive. The archive's SHA256 is
`a9669e90981d9a966ffa32117643ba1336c305a08323683cecc37e8bf058dd37`.
GitHub payload and manifest digests were verified before authorizing retirement.
The retirement tool rechecks cloud assets, local files and the entire remote
inventory while holding the host lock. Current compiles, profiles, sources and
shared caches are retained. See the maintenance completion receipt for the
actual retirement result; archive verification alone does not mean deletion.

The new197 continuation first repeats ingestion into a new output directory
and requires a fully clean postflight. Only then can the original, byte-identical
last-profile and last-ingest requests run. Original reserve failure evidence is
retained. A finite local preservation sidecar will copy/hash/upload the four
accepted artifacts after the continuation, without selecting experiments or
changing cron. It must be committed and checked before being started.

Source198's native and FP8 repeat checks passed before the diagnostic reference
hit a TorchDynamo guard error: the extracted function globals claimed the old
probe's module, which has no `hidden` attribute. The run returned1 and cleaned
up successfully. Source199 registers the extracted globals in an isolated real
Python module. Reversing that binding change plus its imports/extra provenance
field recovers the complete source198 AST. No arithmetic, fixture, bound or
exact check was removed. Source199 is running the same one/four/one-rank screen.

## Completed199 and197 — handoff audit at21:53 UTC

Source199 completed at12:49:53UTC. Original frozen rules, request/source identity
and final log/postflight hashes were independently rechecked for all three jobs.
Every postflight is clean. Median maximum-rank forward latency:

| Concurrent ranks | Native BF16 ms | Saved ROW FP8 ms | FP8 reduction |
| --- | ---: | ---: | ---: |
| 1, first | 5.314404 | 4.811585 | 9.461% |
| 4 | 5.973897 | 4.830518 | 19.140% |
| 1, repeat | 5.320863 | 4.818947 | 9.433% |

Six alternating-order rounds,50calls per arm; collectives outside timing.
Master casts, online preparation and436,469,760bytes of saved state are charged.
This advances the premise for full ROW backward, which remains unimplemented.
There is no backward, complete-model or BPB claim from this component screen.

Source197 continuation completed at12:57:39UTC. Both captures and both accepted
ingestions pass all original rules and clean postflights. Four accepted raw and
derived artifacts are local and cloud-verified; finite preservation finished
13:20:42UTC. First/last model profile attribution remains pending. Original
reserve-failed ingestion and deliberate coordinator hold remain recorded; r1's
stale running state must not trigger a restart. See the handoff for exact paths.
