# Sequential local checks — do not run without the local-process slot

Use exactly one command at a time and wait for it to exit. These commands are
prepared, not claimed executed. Source SHA-256 must remain
`4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff`.
The parent for helper identity and OFF comparisons is K2 local-constants
`662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.

## 1. Cheap FP32 reference and existing small CPU suite

```bash
/home/poojith-devan/venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/lanes/tables-compiler-local-constants/test_fp32_reference.py
```

```bash
/home/poojith-devan/venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/lanes/tables-compiler-local-constants/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-compiler-local-constants/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int32
```

```bash
/home/poojith-devan/venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/lanes/tables-compiler-local-constants/test_tables.py --parent /home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-compiler-local-constants/train.py --repo /home/poojith-devan/Desktop/Trainium --index-dtype int64
```

The two suite invocations write distinct exact-source receipts in this lane.
They do not compile NKI or exercise native collectives and must not be named
native/simulator evidence. The unchanged interpreter reduces geometry.

## 2. Real NKI frontend at both exact production geometries

The existing isolated environment contains the exact host NKI version
`0.5.0+28631259367.ga768afa6`. The wrapper executes no subprocesses and checks
one factory per process. It allocates tensor metadata, not production tensors.

```bash
/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/lanes/tables-compiler-local-constants/test_parser.py --kernel embedding
```

```bash
/home/poojith-devan/Desktop/Trainium/research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python -B /home/poojith-devan/Desktop/Trainium/research/v2/lanes/tables-compiler-local-constants/test_parser.py --kernel table
```

These stop at verified MLIR, not backend compilation. Preserve each traceback
and `PARSER_FAIL`; never reinterpret parser PASS as BIR/NEFF or native PASS.

## 3. Full local compiler and simulator, after K2's serial compile lane

Exact backend/simulator commands remain pending the K2 local-compile lane's
verified CLI and availability finding. Do not guess the API or launch these
checks in parallel. Reuse its proven sequential harness with the source and
exact table geometry from `SPEC.md`, binding the selected factory to
`_k2_table_build_kernel`. Preserve parser, backend, output artifact and
simulator results separately. Simulator numerics compare the FP32 ordered
reference, including colliding rows and untouched rows; never downgrade to
BF16 accumulation or mask a failed numerical case.

The inspected K2 harness is
`research/v2/analysis/20260926-k2-full-local-compile/check.py`: modes `compile`
and `simulate`, exact NKI0.5.0+28631259367.ga768afa6 and
neuronx-cc2.26.6360.0+6f180f47, `ParserFrontend -> compile_to_bir ->
compile_bir_to_neff`, optlevel1/LNC2. It is currently hard-bound to K2 source
662c and `[8192,32768,1024]`; executing it unchanged would NOT test tables.
A separately source-bound table wrapper must be prepared before reuse.
Its first raw-options K2 BIR check failed in SBUF, but the compiler lane found
those options differed from actual JIT defaults. Do not treat that diagnostic
as a production-source failure or reuse those mismatched options. Await the
corrected exact-JIT-options wrapper before binding it to this table source.

Forward is ordinary Torch gather/cast and metadata, not an NKI forward kernel.
Compiling the backward through NKI alone does not satisfy a full forward-graph
compile claim. A genuine local framework compiler check, if supported by the
installed SDK, must cover the exact FP32 master/hashed rows/BF16 output and
sort/search metadata; otherwise record the unexecuted stage explicitly and
ask the Supervisor to resolve the gate before any native table launch.

No host launch command is authorized by this document. The sole Operator
uses the exact completed proposal, original lifetime lock and fresh checks.
