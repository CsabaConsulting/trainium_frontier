# P2 runtime-select: local compiler, simulator and CPU checks PASS

Completed 26 September 2026, approximately10:06 UTC. Implementer evidence,
not a host launch or native/quality verdict.

Frozen train.py SHA-256:
`414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34`.

- Exact-production real NKI parser PASS; no factory/body rewrite.
- Exact-production authenticated neuronx-cc backend PASS, NEFF61386bytes,
  SHA`5a8540752529a7d52ffa0ffb28dd7458bb0d96f2a04463a97b4d27978d302951`.
  Local backward-only compile2.553s; this is NOT full-model cold startup.
- Official NKI simulator fixture1 PASS, maxerror0, outputFP32[524288,128],
  four collisions in first8192rows,4032 fully empty groups,516096 untouched
  rows.7.605s,1140200KiB peak RSS.
- Official NKI simulator fixture2 PASS, maxerror0, all4096groups active,
  two active rows per group (including the last table row), four collisions
  per active row,516096 untouched rows.130.769s,1183412KiB peak RSS.
- Int32 and int64 small CPU suites PASS on these exact bytes: FP32 collision
  reference, OFF3 bit identity, ON2 registered-vs-ordinary bit identity,
  hashes, FP32 RMS, separate clipping, mocked ownership/AVG and checkpoint.
  Common initialization and fixed-shape causality/batch independence exact.
  The existing cross-shape prefix diagnostic7.62939453125e-6 is retained.

Second simulator completed09:57:37UTC during the intentional task interruption.
On resume10:05:44UTC, no matching process remained and its durable PASS receipt
was present; the old unified terminal session was already closed. CPU checks
then ran sequentially. No overlapping heavy process was started.

## Change, failed parent and honest limits

Only the table runtime row-group HBM address views change from4079: replace
unsupported VirtualRegister Python arithmetic/slices with explicit grouped
reshape/select views. K2 source/helpers, FP32 arithmetic and all model/eval
logic remain unchanged. Parent4079's real parser failure is retained in
commit1899d6ea0. No blind4096-way unrolling or precision change.

CPU interpreter gained ordinary Torch reshape/select view methods; this
disclosed shim is not the native-compiler proof. Exact production parser,
backend and official simulator separately test the real VirtualRegister
indexing. Simulator reference equality is numerical array equality, not a
separate signed-zero assertion. Nonfinite gradients are not covered by the
official simulator fixtures. CPU mocked collectives are not native AVG.

No Trn2 table execution, full framework forward compilation, full-model cold
startup, native distributed optimizer test, throughput or BPB result exists
for this source. Per the latest human amendment, do not add a new local
framework-forward gate or repeat-review cycle; the one native timing run is
the native test. The Supervisor retains decision/Operator authority, actual
K2 timing dependency, source/argv bindings, original host lock/resource checks,
official budget and real-control/public20M selection. This delivery does not
authorize its own launch or bypass those unchanged checks.

## Evidence

- `SOURCE.diff`, `SPEC.md`: complete production delta/address equivalence.
- `cpu-result-int32.json`, `cpu-result-int64.json`: exact hashes and results.
- `parser-result-table.json`: exact production frontend PASS.
- `compile-production-result-1790416485199864469.json`: actual backend PASS.
- `simulate-result-1790416504816097348.json`: first production fixture.
- `simulate-result-1790416657288176259.json`: all-groups production fixture.
- `artifacts-manifest.json`, `artifacts/`: NEFF, actual compiler log and both
  nonempty logical-core instruction summaries.
- `REPRODUCE.md`: exact sequential commands and limits.

Local heavy-process slot is released after these completed checks. No host,
queue/state, proposal, AWS resource, evaluator or submission was changed.
