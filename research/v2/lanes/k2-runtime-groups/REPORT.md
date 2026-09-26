# K2 runtime groups — completed local evidence

26 September 2026, 10:18:53 UTC. Source remains frozen at
`f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
Initial source/SPEC/harness delivery: commit `6f8bb4ca4`.
This report updates the execution status of the historical static SPEC; it
does not alter its source, scope, native scheduling or evidence boundaries.

## Actual result

All requested local checks passed after the Supervisor explicitly released
the one local-heavy slot. CPU, real compiler and simulator processes were
run strictly sequentially. No packages, host state or running source changed.

1. `test_k2.py --base <accepted e1e2 file>` exited 0. Each of INT32 and INT64
   passed 12 finite bitwise operation fixtures, original production metadata
   checks, collision order, reused input storage, two gradient accumulation
   microsteps and 21 input-contract rejection checks. Four historical
   NaN/Inf/overflow diagnostic cases per dtype retain their explicit
   `FAIL_NONFINITE_DIAGNOSTIC` label; these are not finite-training passes.
2. Both dtypes passed three OFF and three interpreted ON reduced-eight-layer
   steps against accepted e1e2. Every loss, parameter gradient and updated
   parameter matched bitwise; ordinary evaluation matched. OFF losses were
   `[4.156866073608398, 4.15728235244751, 4.162270545959473]`; ON losses were
   `[4.159947395324707, 4.159997940063477, 4.156461238861084]`.
3. Exact-source production backward passed the actual NKI parser, BIR emission,
   neuronx-cc backend and NEFF creation at 10:18:15 UTC. Total harness elapsed
   was 3.5214909439964686 s, Python peak RSS 160476 KiB. This is standalone
   backward compilation, **not full-model cold startup** or a combined
   descendant-memory peak. The independent parser-only harness was not run;
   the real backend check already includes the parser.
4. Official full-production uniform simulator fixture passed at 10:18:31 UTC,
   with four collisions per row, FP32 output `[8192,1024]`, exact equality and
   max absolute error 0. Elapsed 9.30493511199893 s; peak RSS 585564 KiB.
5. Official full-production sparse simulator fixture passed at 10:18:53 UTC,
   with eight collisions per active even row and 4096 untouched odd rows,
   same exact FP32 result and max error 0. Elapsed 14.45075403599185 s;
   peak RSS 590704 KiB. Inactive lanes exercise sentinel/skipped DMA behavior.

All process sessions reached exit 0. A final local process scan found no
remaining lane Python/compiler/simulator process; the local-heavy slot was
then explicitly released. No new local failure occurred in this derivative.

## Compiler authentication and structural observation

Exact host versions: NKI `0.5.0+28631259367.ga768afa6`, neuronx-cc
`2.26.6360.0+6f180f47`, Python 3.12.13. The unmodified source factory is
extracted by AST retaining its filename/lines and actual fixed geometry:
BF16 gradient `[32768,1024]`; INT32 counts/starts `[8192,1]`, positions
`[32768,1]`, maxima `[64,1]`; Trn2/LNC2. No payload is needed for compilation.

Production options are derived from the exact kernel's `CompileKernel`
subclass and `_compile_opts()`, the same authenticated pipeline as parent662c.
Experimental MLIR allocation/scheduling remain disabled. Actual backend log
records `--target trn2 --lnc 2 --optlevel=1 --auto-cast matmult --auto-cast-type
bf16` and the SDK's `--enable-nested-dynamic-loop`, then `Compiler status PASS`
and exit code 0. There was no raw-CompileOptions experiment or flag change.
The harness retained the 2 GiB address-space bound, 900-second CPU bound,
single library threads and external 900/300-second compiler/simulator limits.

`artifacts/k2.neff` is **61,338 bytes**, compared with parent662c's separately
archived 349,692-byte backward NEFF. Both logical-core instruction streams
are nonempty. The smaller standalone executable supports the compile-size
hypothesis only: native whole-model startup and step rate remain unknown.
No extrapolated speedup or startup saving is asserted.

## Source-bound receipts and artifacts

- `cpu-result.json`:
  `2cfb0410596c28b20adef0bf91b911283ff334263fbbf947b7ac838a9fcab7cd`.
- `compile-production-result-1790417895251858239.json`:
  `147a300be275cb74b01e0ce2321f82a7e35dd9edf84944b25e6ad4dec80cda4a`.
- `simulate-result-1790417911971128655.json` (uniform):
  `74d25e9302ba1ac64602fd33b191325f4372ac705857434026290a28b61d70cc`.
- `simulate-result-1790417933199676192.json` (sparse):
  `6baa893345aa6899e90f4672f1d8cba391eb17e80df2713c594d25bcbd3c99b5`.
- `artifacts-manifest.json`:
  `ccf4914bbe35206bf2ee87866204a639eab6dc05e33a4eebea6c432212ddde40`.
- `artifacts/k2.neff`:
  `e64314f25fbcbab62d6bcc41bdc1408e830821c296b8898e2c9a8569ed5441ac`.

The manifest binds the raw backend log, both logical-core instruction stats,
kernel metadata and memory-pressure output. Original log bytes are preserved,
including its trailing whitespace; do not reformat this evidence.

## Remaining limits and ownership

CPU interpretation uses reduced geometry and test SGD; it is not native
metadata/capture, production optimizer or full training-loop proof. Official
simulator references add exact BF16 inputs in token order using FP32. Their
numeric equality does not separately compare signed-zero bits and the two
official fixtures do not cover nonfinite gradients or highly skewed frequency.

Existing accepted framework-forward evidence applies; there is no new forward
gate, OFF native run, second native smoke or repeated review. No native call,
whole-step timing, cold whole-model compilation or BPB was measured here.
The running662c native201 and all disabled414c table packets remain untouched.
Root owns narrow independent composition and any prospective source/control/
command selection; the sole Operator owns a separately released native job.
