# Implementer component and test binding — not a new review

Completed 26 September 2026, 10:40:35 UTC. Candidate4b74 is frozen;
`composition-result.json` contains complete source, component, review, artifact
and fresh test pins. All tests/processes are terminal and the local-heavy slot
has been released. No host or current-job action occurred.

## Mechanically checked composition

`check_composition.py` rehashed candidate4b74, table parent414c and compact
donorf804 before checking the change with a source-AST boundary method separate
from the text-splice construction:

- Candidate bytes before and after `_k2_build_kernel` exactly equal414c.
- The complete replacement factory bytes exactly equalf804. Factory SHA256
  `a0f2c2111edb4f2896d4a20eac4686afb1bb78bd2d544a1747cc9556006a5754`.
- All119 other top-level AST statements/definitions exactly match414c.
- `_k2_table_build_kernel`, `_k2_table_build_registered_lookup`,
  `_k2_prepare_metadata` and `_k2_build_registered_lookup` additionally pass
  exact function-byte comparisons. Table factory SHA256
  `bf140c6c0960903e8c2e0a39a61b0598c93243f3d72a86770b03368ce994070e`.
- All ten inherited compiler artifacts match their original manifests, and
  all reused compiler/simulator receipts and existing review/composition files
  match their pinned hashes. These are old component results, not new combined
  model compilation or native evidence.

`SOURCE.diff` SHA256:
`fb676eb7c5184b8b2a53897b4f24f950c97cc8f3688350d21eec6f0a49c0efa8`.
Raw diff context whitespace is preserved for an exact ordinary unified diff.

## Fresh bounded CPU evidence

Both commands in REPRODUCE.md exited0, sequentially with OMP/OpenBLAS/MKL=1,
Torch threads1, 180s outer timeout and the existing small fixture geometry.
The actual table instruction interpreter retains its65536-element tile cap.
No full-size gradient allocation, compiler or official simulator ran here.

For INT32 and INT64 separately:

- FP32 master/collision/microstep reference passed; the rounding discriminator
  accumulated `[0.00390625,0.00390625,0.00390625,0.00390625]`.
- Six table order/subnormal/cancellation fixtures passed exactly, along with
  production signed-int32 hash oracles and three FP32 RMS updates.
- Mocked four-rank ownership and FP32 AVG/clipping routing passed. These are
  CPU routing tests, not native collective evidence.
- Three table-OFF steps matched compact f804 in losses, all gradients and
  updated parameters. Losses:
  `[4.161633014678955,4.1532745361328125,4.155467987060547]`.
- Two ON steps matched registered interpreted table backward against ordinary
  FP32-master/BF16-gather training, including nonzero tables, gates and actual
  table RMS updates. Common initial parameters matched.
- Strict ON and old-parent checkpoint roundtrips passed. Same-shape causality
  and batch independence were bit-exact. The inherited cross-shape prefix
  diagnostic remains7.62939453125e-6, not hidden as an exact result.

Model fixture:8layers/C128/T16/V64, table32x128. OFF uses testSGD; ON invokes
the source optimizer at reduced shape. Existing prepare/parquet import shims
are only test scaffolding. The ON test injects the CPU-interpreted table lookup
for both lookup hooks as before; it does not newly execute the compact K2 NKI
body. That exact body is independently bound to its prior f804 CPU/compiler/
official-simulator evidence above. No claim of combined native capture is made.

Fresh receipt SHA256 values:

- `cpu-result-int32.json`:
  `2b8a3f3decdbe75cb09962a3e5f9b1fd160b521c423ea707452d90b00553cb8b`.
- `cpu-result-int64.json`:
  `a66fa593da030e92ba49ae773fc2df2a5646b0c62f93a9e117fc762e33eb9a23`.

`test_fp32_reference.py` and `cpu_instruction_model.py` are byte-identical to
414c's tested harnesses. `test_tables.py` only adds candidate/donor hash guards
to the prior suite; its parent argument is the exact compactf804 donor.

## Authority and limits

Existing table review + root414c composition and freshf804 review + rootf804
composition retain their exact stated scopes. This implementation/composition
record is not an independently approved new review. Native eligibility is
not inferred from these local passes. No original source, old transport,
launcher, proposal, host, queue, model budget or evaluator was changed.
Tables remain conditional on actual compact K2 net gain>=3%; a missing or
failing result means no table dispatch. Root handles all later source/base/
command binding; the sole Operator owns any released native job.
