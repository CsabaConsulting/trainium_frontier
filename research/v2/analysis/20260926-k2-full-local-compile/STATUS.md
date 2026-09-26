# K2 exact-production local compiler check

Started 2026-09-26 09:43 UTC. Sole local compiler/simulator process owner:
`/root/k2_full_local_compile`. No source changes or host actions.

Source SHA-256: `662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.
Backward NKI production inputs: BF16 `[32768,1024]`, INT32 counts and starts
`[8192,1]`, positions `[32768,1]`, maxima `[64,1]`; LNC2; FP32 output
`[8192,1024]`. Forward is framework `F.embedding(ids, weight).to(bfloat16)`,
not an NKI kernel. Exact forward framework capture is not yet proved locally.

Preflight: disk 66 GiB free, MemAvailable 3.1 GiB, swap full. Compiler installation
and tests will run sequentially in the existing isolated parser environment,
with compiler threads and process address space bounded. No native timing is
authorized by a parser-only result.

09:47 UTC: Exact compiler installed, official simulator PASS at production
shape (max absolute error 0; 7.36 s, 571 MiB peak RSS). One non-production
pipeline test enabled the experimental MLIR allocator and reported SBUF
exhaustion at `nl.where`; this is NOT a demonstrated production failure.
The corrected kernel-derived default pipeline disables that experimental
allocator and passes BIR generation. The first actual neuronx-cc backend
invocation is blocked locally by missing `libxml2.so.2`, not candidate code.
Acquiring compatible Ubuntu libraries into scratch, with no system changes.
At 09:48:22 UTC the corrected production-policy backward compiled fully to
NEFF, compiler status PASS. Actual installed host and local compile-options
source are byte-identical. See REPORT.md and artifacts-manifest.json.
Exact framework forward compile remains UNMET. No native launch is authorized
by this report. No heavy local compiler/simulator process remains after the
completed diagnostic runs; ownership can be handed to tables packaging.
At 09:51:21 UTC the second official simulator fixture also PASSED exactly:
full production shape, eight collisions on even rows and 4,096 untouched odd
rows, exercising sentinel/OOB-skip lanes; 12.87 seconds, 577 MiB peak RSS.
