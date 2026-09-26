# K2 runtime row-group loop — isolated compile-size fallback

Implementer delivery, 26 September 2026. STATIC PREPARATION ONLY.
Selected by `research/v2/proposals/20260926-k2-runtime-groups/SPEC.md`
(commit `7096272ca`, SHA-256
`dbdc5d8ed71acb5466e6ba3f6f1c16bfc9fd7865991dfff7ab03c5c14c2d20c7`).
No native release, retry, source substitution, cancellation or result is implied.

Frozen source SHA-256:
`f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
Parent remains frozen and in its separately owned native test:
`research/v2/lanes/k2-compiler-local-constants/train.py`, SHA-256
`662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.

## Exact production change

`SOURCE.diff` is the complete delta. Only `_k2_build_kernel` changes:

- The outer 64 row groups use `nl.dynamic_range(groups)` instead of Python
  `range(groups)`. The inner ordered occurrence loop is unchanged.
- Unsupported Python arithmetic on a runtime register is avoided by reshaping
  counts/starts to `[64,128,1]`, maxima to `[64,1,1]`, and output to
  `[64,128,1024]`, then selecting group zero-axis through `.select(0, group)`.
- Output still selects the original program-owned 512-feature half.
- One comment describes the now-runtime outer loop.

No changes to geometry, masks, FP32 operations, sentinel handling, metadata,
registration, normal framework forward, dtype contracts, model, initialization,
optimizer, schedule, clock, command, data or evaluation. This does not include
tables or the pending batch/LR/schedule stack.

Exact geometry remains V=8192, T=32768, C=1024, P=128, F=512, G=64, LNC=2.
BF16 gradient `[32768,1024]`; INT32 counts/starts `[8192,1]`, positions
`[32768,1]`, maxima `[64,1]`; FP32 dense output `[8192,1024]`.

For group g, row p, owner o and feature f, the original and new flattened
addresses are counts/starts `g*128+p`, maximum `g`, output
`(g*128+p)*1024 + o*512 + f`. Their ranges are g=0..63, p=0..127,
o=0..1 and f=0..511. The view does not overlap the two owner halves or skip
rows. These formulas state the claim to verify; they are not executed evidence.

## Motivation and explicit limits

Supervisor observed the live parent whole-model compiler still working after
900 startup seconds with a large graph. That observation does not establish
that K2 caused all graph growth or that the running job failed. A runtime
outer loop may reduce repeated generated code, as in the locally compiled
414c table derivative, but K2 has the original much wider feature half.

This source has **not yet run CPU, parser, compiler, simulator or native tests**.
No startup or throughput improvement is measured. Even a smaller local backward
NEFF would not prove shorter full-model startup or higher whole-step throughput.
The running source and current table packets remain untouched.

## Prepared source-bound evidence, not new gates

`test_k2.py` preserves the parent's actual operation/model suite and adds exact
source, parent and accepted e1e2 base authentication. CPU fixtures include both
INT32/INT64 IDs, finite bitwise operation checks, separately labelled nonfinite
diagnostics, original full-production metadata, collision order, reused loader
storage, two accumulation microsteps, three OFF and three interpreted ON steps
of the reduced eight-layer model against accepted e1e2. The optimizer is small
test SGD, not full-size native training. Nothing is claimed before execution.

`cpu_instruction_model.py` adds only the same Torch-backed reshape/select and
integer-bound dynamic-range support used by the earlier table interpreter.
It preserves write-count views. CPU dynamic indices are Python integers, so
this model alone cannot prove real NKI runtime-register indexing legality.

`test_parser.py` is the parent's exact-factory parser check, rebound to this
source SHA. `check_backend.py` reuses the parent's actual success pipeline and
production-shape uniform/sparse official simulator fixtures. It extracts the
unchanged factory AST with original filename/lines and obtains options from
the exact kernel's `CompileKernel._compile_opts()`. The misleading raw
experimental allocator mode is not exposed in this harness. Compilation is
ParserFrontend -> BIR -> actual neuronx-cc -> NEFF, not parser-only proof.

The existing accepted framework-forward evidence applies; no new framework
forward test, OFF native run, review cycle or second native smoke is introduced.
One local heavy process at a time. `REPRODUCE.md` gives serial commands, all
pending the Supervisor's explicit slot release. No host/package changes.

After actual evidence, root owns narrow independent composition and prospective
proposal/control/command binding. Sole Operator owns any selected native run.
This delivery itself grants none of those authorities.
