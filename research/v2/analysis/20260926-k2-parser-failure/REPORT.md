# K2 parser failure: reproduced on CPU with the exact NKI version

Completed 2026-09-26, approximately 09:35 UTC / 15:05 IST. Diagnosis only.
The native-failed variant remains parked. No candidate edit, host access,
training launch, new review gate, source adoption, or native qualification.

## Finding

The NKI parser fails to bind the six Python closure cells used by the nested
`resident_consumer`. Moving their values into the kernel body is the minimal
prospective repair. This is now demonstrated using the exact host NKI version,
not merely inferred from Python scope rules or documentation.

The exact original `_k2_build_kernel` AST, extracted without importing the
training module, fails on this CPU at the same original `train.py:93:35`:

```text
assert gradient.shape == (tokens, channels)
                         ^
error: unbound variable 'tokens'
```

The parser reports `[x2]`, matching the two LNC programs. The archived native
log contains this diagnostic on all four training ranks and terminates with
`TRAIN_EXIT_1`, before a completed training step. This is a compile-time name
resolution failure, not evidence about scatter numerics or performance.

## Direct compiler experiment

All cases use the unmodified installed `nki.compiler.frontend.source_to_mlir`
with `FrontendKind.PARSER`, target `trn2`, and `CompilationResult.verify()`.
The full consumer uses LNC2 and metadata-only `NkiTensor` arguments with exact
production shapes and dtypes. No tensor payload, weights, data, framework
execution, device runtime, BIR lowering, or neuronx-cc backend is used.

| Case | Captured names remaining | Actual parser result |
| --- | --- | --- |
| Exact original factory | all six | `tokens` unbound |
| Minimal nested copy kernel, outer `tokens=128` | `tokens` | `tokens` unbound |
| Same minimal nested kernel, local `tokens=128` | none | Verified MLIR PASS, 1,094 bytes |
| Actual body + local `tokens` | five | `channels` unbound |
| Actual body + local `tokens, channels` | four | `vocab` unbound |
| Actual body + local `tokens, channels, vocab` | three | `groups` unbound |
| Actual body + local `tokens, channels, vocab, groups` | two | `features` unbound |
| Actual body + local `tokens, channels, vocab, groups, features` | one | `partition` unbound |
| Actual body + all six local bindings | none | Verified MLIR PASS, 1,062,965 bytes |

The minimal local case is still nested in its factory. Therefore nesting
alone is not the failing condition demonstrated here; the unsupported capture
of outer locals is. The final actual-body case has no Python closure cells.
`nl` and `nisa` remain normal globals as in the candidate and are resolved.

The in-memory diagnostic variants add local scalar assignments and retain the
now-unused outer assignments. They never create or modify a model candidate.
The prospective production edit would place the two existing geometry tuple
assignments at the start of `resident_consumer` and remove them from the outer
factory; this exact textual edit was not applied. Binding only `tokens` is
insufficient: every subsequent name in the table was actually observed failing.

Reproduce from this directory:

```text
.venv/bin/python -B diagnose.py
.venv/bin/python -B diagnose.py --case original
.venv/bin/python -B diagnose.py --case minimal_closure
.venv/bin/python -B diagnose.py --case minimal_local
.venv/bin/python -B diagnose.py --case production_local_6
```

`parser-results.json` preserves the second execution's complete per-case
stdout/stderr; the first execution produced the same status/error sequence.
The process exit code records successful diagnostic capture, not successful
compilation: use each JSON `status` and compiler diagnostic. Virtual source
filenames are populated only in the diagnostic process's `linecache` to give
the real parser source text for the in-memory variants. The exact-original
case uses the actual source filename and unmodified AST instead.

## Why the CPU exactness tests passed

The candidate's `cpu_instruction_model.py` replaces the NKI package with
test objects. Its `Jit.__getitem__` returns `self.function`, and
`interpreted_kernel` executes the factory under the ordinary Python
interpreter. Python then resolves all six values through `__closure__`.
The harness changes two outer tuple assignments for reduced geometry but
does not compile NKI. Its own docstring explicitly disclaims SDK/compiler
legality. `test_k2.py` injects that Python consumer into registered autograd.
Thus the CPU numerical and OFF-identity results never tested this parser
binding boundary; their scope must remain as originally disclosed.

AWS's version-matched [NKI simulator limitations](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/guides/nki_simulator.html#nki-meta-programming-support)
also explain that Python simulation can accept closures which the restricted
NKI compiler rejects. Even the official simulator is not a parser test.
The [NKI compiler overview](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/deep-dives/nki-compiler.html)
describes the separate AST/semantic-analysis path. These documents corroborate
the executed comparison; they do not replace its exact-version evidence.

## Environment and acquisition receipts

No NKI, neuronxcc, or torch_neuronx package was installed in the pre-existing
local development venv. Targeted searches found no reusable parser/wheel in
the checked repository, Desktop/Documents, package caches, or ordinary local
Python locations. This is a bounded search, not a claim about every filesystem.

Root then authorized one isolated local diagnostic venv, a 5 GiB total cap,
and the existing 30-minute limit. The host archive
`operator/20260926-final-sprint-successors/trn22-indexrepair-live.json` records:

- NKI `0.5.0+28631259367.ga768afa6`.
- neuronx-cc `2.26.6360.0+6f180f47`.
- torch-neuronx `2.11.3.0.0+unknown.dev`.

Only NKI was required for the parser. Neither neuronx-cc nor torch-neuronx was
installed locally. This does not reproduce the complete host environment.
Local Python is CPython 3.12.13, Linux x86_64, from an already installed runtime;
pip is 25.0.1. Dependencies were locally pinned to numpy 2.2.6, ml-dtypes 0.5.1,
and typing-extensions 4.15.0; their host versions were not archived/matched.
The kernel inputs are `NkiTensor` metadata, so these tests do not use NumPy
numerical calculations as a substitute for compilation.

AWS's [versioned setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/setup-env.html)
identifies the official package index. The exact official wheel fetched was:

```text
https://pip.repos.neuron.amazonaws.com/nki/nki-0.5.0%2B28631259367.ga768afa6-cp312-cp312-linux_x86_64.whl
SHA256 66255c4f4211a1b9d688d2cb432cb1472bfd607afd2a7ab9fc41fb48c33d755a
```

`nki-install-report.json` and `dependency-install-report.json` preserve pip's
URLs, wheel hashes, versions, and environment. NKI is an exact version match
to archived host metadata; no separate comparison of the host wheel binary
hash was available. Local installed parser binary SHA256:
`cc512d0fc25723323f290497a18eb1479a739efe416199422f950441bcea8287`.
Local `nki/compiler/frontend.py` SHA256:
`595eaeb5c0638e883e8420df0f378db0c5d20475e5034605e6c3a5ae6950a65d`.
Its `ParserFrontend._compile` calls `parser_kernel.compile` at line 300,
matching the native traceback.

Installation commands used an isolated `.venv` in this directory:

```text
/home/poojith-devan/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -m venv .venv
TMPDIR=<this-directory>/pip-tmp .venv/bin/python -m pip install --no-deps --no-cache-dir --report nki-install-report.json --index-url https://pip.repos.neuron.amazonaws.com nki==0.5.0+28631259367.ga768afa6
TMPDIR=<this-directory>/pip-tmp .venv/bin/python -m pip install --no-cache-dir --no-deps --report dependency-install-report.json --index-url https://pypi.org/simple numpy==2.2.6 ml-dtypes==0.5.1 typing-extensions==4.15.0
```

The initial NKI download using default `/tmp` failed with `Errno 122: Disk
quota exceeded` at 10 MB. Redirecting temporary files into the authorized
diagnostic directory succeeded; no cleanup or system quota changes occurred.
The diagnostic venv is 163,670,515 apparent bytes (~156 MiB), safely below the
5 GiB cap. Venv, temp directory, wheels, and future compiler scratch are ignored
by Git. No persistent wheel copy was retained and no device artifact was made.

## Evidence identities and limits

- Candidate, unchanged: `a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`.
- Diagnostic `diagnose.py`: `3812ebb533af43e2b4251688e1b9b1c9fc5dc613fe4e8f315836aae08f765197`.
- Native results bundle: `98d21e753ca955e67d3d78418ca7972b1804af0fede83c2826c946f0c8118164`.
- Bundle member `training.log`: `27767f3773dcc037fbfd29ee087651e1f3ad52a61f2cf40fb341b07634855222`, 77,666 bytes.
- Full local-binding MLIR text hash: `4e3e0f1c579e624ca11f184a24cbfe81297e01ebb590368b904c417f72e24310`.

Proven: the exact observed parser failure is reproducible off-host; the
captured constants cause it; all six bindings must be addressed; the complete
consumer with local bindings passes this parser's MLIR verification.

Not proven: BIR/backend legality, SBUF/liveness/allocation, executable generation,
native DMA/rounding behavior, full training capture, cold startup, throughput,
fixed-budget BPB, or native qualification. No additional parser error was
observed after all six bindings, but later compiler/runtime failures remain
possible. This report supports a separately scoped source-fix proposal; it
does not authorize a third native retry or alter the parked lane's verdict.
