# Exact K2 local compilation: backward PASS, framework forward UNMET

Completed 26 September 2026, 09:51 UTC. No candidate edit or host launch.

## Result

The unmodified K2 candidate SHA-256
`662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`
passes the real version-matched NKI parser, BIR emission, **neuronx-cc backend**,
and NEFF generation for the exact production backward shapes on Trn2/LNC2.
`artifacts/k2.neff` is 349,692 bytes, SHA-256
`c3af17a145edf8af06e86d073cca6036d505e35039f6592c3079bbfd0ebcb48e`.
The actual compiler log ends with `Compiler status PASS` and exit code 0.
Both logical cores contain nonempty instruction streams, archived separately.

The official NKI CPU simulator also ran the exact kernel, with exact production
shapes, and matched an independently calculated ordered FP32 reference exactly:
maximum absolute error **0**, FP32 output `[8192,1024]`. The first uniform
fixture took 7.3565 seconds, peak process RSS 585,280 KiB. A second full-shape
fixture with 4,096 untouched rows and eight collisions on each other row also
matched exactly, in 12.8676 seconds with peak RSS 590,544 KiB. This is simulator
evidence, not device execution.

**The new user's complete forward-and-backward local requirement is NOT yet
satisfied.** The forward in this file is ordinary
`F.embedding(ids, weight).to(torch.bfloat16)`, registered through `nki_op`; it
does not contain a custom NKI forward kernel. No exact torch-neuronx framework
forward graph was compiled locally. The exact installed native-beta
torch-neuronx build is not in this isolated local environment. A bounded public
AWS wheel-index search found no `2.11.3` wheel. Operator's bounded read-only
scan found no identifiable forward HLO, but deeper hashed contents and their
semantic identity remain unknown. No fabricated NKI forward or hand-written
equivalent HLO was substituted. This report is not launch authorization.

## Source and compiler policy authentication

`check.py` extracts only the `_k2_build_kernel` function AST from the exact
hashed source, preserving the original filename, lines and body. It does not
import train.py, weights, a tokenizer, training data or the evaluator. Inputs
to compilation are genuine NKI tensor descriptors with no payload:

| Input | Shape | Dtype |
| --- | --- | --- |
| gradient | 32768 × 1024 | BF16 |
| counts | 8192 × 1 | INT32 |
| starts | 8192 × 1 | INT32 |
| positions | 32768 × 1 | INT32 |
| group_maximums | 64 × 1 | INT32 |
| output | 8192 × 1024 | FP32 |

The kernel's LNC grid is 2. Actual backend flags were `--optlevel=1 --auto-cast
matmult --auto-cast-type bf16`; compiler target is `trn2`. No cache was reused.
The metadata-only invocation proceeds through the real installed
`ParserFrontend`, `compile_to_bir` and `compile_bir_to_neff` implementations.
The latter invokes the actual `neuronx-cc` executable, not a stub or mock.

Production options are obtained by converting the exact `Kernel` to the SDK's
`CompileKernel` subclass and calling its `_compile_opts()`. The installed host
`TorchNeuronNKIKernelV3` inherits `CompileKernel` and does exactly the same in
`_compile_dump_config`; its separate lightweight `cache_opts` is only for the
cache key and is not the compilation policy. The archived host and local
`nki/framework/compiled.py` are byte-identical, SHA-256
`c416d73c01fd8810509c4496758a122df941e109221fd0ee9a47d30f710b378e`.
Host `nki_kernel.py` SHA is
`37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2`;
host `nki_hop.py` SHA is
`a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914`.
These are in Operator commit `7ff42226ef0144f1687020aacb3657bac2f02957`,
`research/v2/operator/20260926-nki-api-readonly-proposal/results/sources/`.

## Failed attempts retained, correctly classified

1. At 09:44:47 UTC, the initial diagnostic used raw `CompileOptions` defaults.
   This **enabled the experimental MLIR linear-scan allocator**, unlike the
   kernel's actual default production policy. It failed with:

   ```text
   error: Out of memory in sbuf while allocating 2048 bytes
   selected = nl.where(active.ap(pattern=[[1, partition], [0, features]]),
   ... nisa.alloc ... memref<128x512xf32, #nisa.mem<sbuf>>
   ```

   This was a real diagnostic-pipeline allocation error, not laptop RAM
   exhaustion, but **not a demonstrated production failure**. It was reported
   to Supervisor initially, then promptly corrected when policy mismatch was
   discovered. The failure is reproduced in
   `nonproduction-allocator-reproduction.log`; no source edit was made to
   accommodate the wrong policy. Process peak RSS was only 73,544 KiB.
2. A first harness correction attempted `dataclasses.replace` directly on the
   base `Kernel` with a `target` field and raised a local TypeError before
   compilation. It was corrected by the SDK's `_to_subclass(CompileKernel)`.
   `compile-production-result.json` preserves this harness failure.
3. With correct production options, BIR generation passed. Actual backend
   execution then failed because this laptop's Ubuntu uses libxml2.so.16:

   ```text
   walrus_driver: error while loading shared libraries: libxml2.so.2:
   cannot open shared object file: No such file or directory
   ```

   Compatibility libraries were extracted into this diagnostic's scratch
   directory only. No system package or service was changed. The next backend
   invocation passed at 09:48:22 UTC in 4.7199 seconds. Python process peak RSS
   was 163,044 KiB; this does not claim a measured combined descendant peak.

## Simulator fixture and limits

The real public API `nki.simulate(exact_kernel)` is selected explicitly for
Trn2/LNC2. Inputs are NumPy arrays with ml-dtypes BF16, not the team's custom
Python instruction model. All 32,768 positions are distributed cyclically over
8,192 rows: four collisions per row, with stable occurrence ordering. The
gradient varies by token and channel with both signs and BF16-exact fractions.
The independent reference starts from FP32 zeros and adds each token's BF16
gradient, widened to FP32, in token order. This exact full-production fixture
exercises both LNC feature partitions, all row groups and repeated accumulation.

The second fixture uses even-numbered vocabulary rows only. Each active row
has eight occurrences; 4,096 odd rows are untouched, interleaved within all
128-row groups. Their inactive lanes take the sentinel and `oob_mode.skip`
paths while adjacent active lanes accumulate. This exact full-shape official
simulation also passes, including zero output for untouched rows; the equality
test does not separately compare the sign bit of floating-point zero.
Its reference is independently calculated token-by-token in FP32 as above.

Neither fixture covers nonfinite inputs, highly skewed nonzero frequencies,
metadata construction through the actual Torch backend, registration/capture,
distributed AVG or optimizer updates. Those remain distinct from earlier CPU
tests. Simulator success is not runtime DMA/OOB behavior proof or native timing.

## Reproduction

Interpreter:
`research/v2/analysis/20260926-k2-parser-failure/.venv/bin/python`.

Run from the repository root, with the existing diagnostic venv's `bin`
prepended to PATH. Set `LD_LIBRARY_PATH` to this directory's
`scratch/compat/root/usr/lib/x86_64-linux-gnu` for the local backend.
No device or SDK framework import is needed for the backward checks:

```text
<venv>/bin/python -B research/v2/analysis/20260926-k2-full-local-compile/check.py compile-production
<venv>/bin/python -B research/v2/analysis/20260926-k2-full-local-compile/check.py simulate
<venv>/bin/python -B research/v2/analysis/20260926-k2-full-local-compile/check.py simulate --fixture sparse
```

The actual commands were wrapped with a 900-second timeout for compilation,
300 seconds for simulation, and set OMP/OPENBLAS/MKL threads to 1. The script
also limits address space to 2 GiB and CPU time to 900 seconds. Each process
was allowed to finish before the next started. Full successful command/output:
`compile-production-r3.log`. Successful structured receipt:
`compile-production-result-1790416102882879459.json`, SHA-256
`9acca1236547f7f49dcebfe60b0a683f8ad9420167a26cfaf10d16f97d923b34`.
Simulator receipt: `simulate-result.json`, SHA-256
`d5c0414bbea38eb37f410fcc90f32a7a961e5dfbcaaa069fdea9e08d4e1c0019`.
Sparse simulator receipt: `simulate-result-1790416281677768577.json`, SHA-256
`47dcbc578e419d07bf65c9c7063983883bfcc2d54edec94d925fef034d25e506`.
All preserved compiled artifacts are bound by `artifacts-manifest.json`.

## Environment, downloads and resource boundary

Python 3.12.13; exact host NKI `0.5.0+28631259367.ga768afa6`; exact host
neuronx-cc `2.26.6360.0+6f180f47`. Compiler wheel from the official AWS index:
`https://pip.repos.neuron.amazonaws.com/neuronx-cc/neuronx_cc-2.26.6360.0%2B6f180f47-cp312-cp312-linux_x86_64.whl`,
SHA-256 `dec7ace1a340e23cacc972af19b9a0f9365c24036d9263dbe2695ab9ecf39dbe`.
`compiler-install-report.json` and `compiler-dependencies-report.json` contain
exact package URLs, hashes and versions. Dependencies are not all host-matched.

Official Ubuntu compatibility packages, extracted without installation:

- `https://archive.ubuntu.com/ubuntu/pool/main/libx/libxml2/libxml2_2.9.14+dfsg-1.3ubuntu3.9_amd64.deb`,
  SHA `6e578bc383096718c9eea8a76a3edfacfea06e525e1aa7a187ee41906436d94e`.
- `https://archive.ubuntu.com/ubuntu/pool/main/i/icu/libicu74_74.2-1ubuntu3.1_amd64.deb`,
  SHA `c9a70989678660eed9a1e904c74fa043da8bec8e2036856fc16e31ced79b04f8`.

Venv footprint about 1.7 GiB; scratch about 65 MiB before final artifacts,
below the authorized 5 GiB cap. No user app was stopped. No source, evaluator,
data, AWS resource, host job, submission or system environment was changed.
Only diagnostic-process environment and isolated packages were used.

Primary documentation consulted: [AWS compiler overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/about-neuronx-cc.html)
and [versioned NKI setup](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/setup-env.html).
Actual installed code and executed receipts, not documentation alone, establish
the result above. The required ML tooling skill was read; no Hugging Face
service, dataset, model, or remote job was used.
