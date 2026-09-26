# Existing compile-cache evidence — Supervisor preservation

26 September 2026, 11:02 UTC. The analyst was stopped under the human's
concurrency limit and was not restarted. Supervisor read the existing script
and reproduced its saved result with a bounded stdlib-only local execution.
No host, compiler, model, source, evaluator or cache was changed.

## What is measured

Six authenticated completed full-run bundles each contain ten distinct graph
keys (forward, backward, eight inferred Muon specializations). All six compared
pairs have empty key intersections. Each graph has one nonzero recorded compile
counter; other ranks' zeros are not independent persistent-cache-hit evidence.

- Same-host trn22, same8c6 source and batch262144, LR.015 versus.021:
  the persistent cache namespace is the same, but every graph key differs.
  Startup853.7422744090437 versus811.3222134970129 seconds. This is not an
  exact-config repeat because LR differs. The approximately42.42s reduction
  does not show that the main graph or all optimizer graphs were reused.
- Accepted e1e2 cold versus same-host reused-directory control:
  startup falls by65.6084s, but all keys change and compilation is recorded.
  Compiler counter sums are not concurrent elapsed wall time.
- Historical3e79 fixed-source-path repeat ontrn22: identical source identity,
  command and cache paths; startup845.5936225 versus120.3379417s. All ten keys
  match, and the second run's compile counters sum to zero. This supports an
  operational stable-path/reuse hypothesis, not isolated proof that the path
  alone caused the speedup or explicit proof of each NEFF read.

The historical warm target was<=120s and was missed by0.33794s. That original
verdict stays unchanged. It was a five-step test on3e79, not a full run on the
accepted e1e2 or adopted8c6 source, and it establishes no BPB improvement or
cold-submission qualification.

## Next action and limits

Do not spend a new native slot or delay the active full runs merely to repeat
this analysis. The sole Operator is checking locally whether already-proven
transport supports a stable immutable per-source path without changing
prepare.py, official accounting or evaluation. No source-path switch has yet
been released. Existing warmdown preparation proceeds unchanged in parallel.
Current full-run terminal receipts will provide exact startup/charge; the
893.5s stack polling bound is not an exact timer, and900s is the actual cap.

Underlying frontend key differences remain unresolved without matched compiler
payloads. Do not strip locations, forge keys, suppress cold accounting or claim
a cache fix from directory reuse alone. No K2 reopening is implied.

Reproduction: `python -B research/v2/analysis/20260926-warm-cache-reuse-audit/audit.py`.
Supervisor parsed its output and compared it exactly with savedresult.json.
Script SHA256:68ce5fe33c92d396fb53ba89b16d8475013e9210b14f2e24d6062db23048bf14.
Result SHA256:5d500fbb93d81644f4e0a763987c74f7f8fdbc58ec5e2227916a33309c059fd2.
