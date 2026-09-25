# K2 first native failure: name resolution, no numerical result

**HOLD/REFUSE remains.** Independent59-check stdlib audit (normal/-O) authenticates
all27 raw files/113871 bytes from94fb52c4, canonical ce819b26. Bundle SHA256
`129b9e69352a63d49922dc386c849836ad829dcbb85ef3e33b50b28b64cf9e04`.
Exact source18b9365a: consumer57b5b118, checker89189024, reference14572563.
The consumer is byte-identical to original077481d0; the reporting repair did not
introduce this kernel behavior.

All four ranks on torch2.11.0+cpu / NKI0.5.0+28631259367.ga768afa6 fail at the
first `cross_tile_large` BF16 case. Each has exactly one successful check,
`upstream_roundtrip`, with matching hashes and zero absolute error. The case
remains STARTED, timings are empty, and each final status is
REFUSE_OR_UNSUPPORTED. Driver terminal22:45:12.709986 UTC is
DIAGNOSTIC_CHILD_FAILED/returncode1, completeFalse. No direct/repeated consumer,
autograd comparison, FP32 case or latency sample completed. This is a real
native frontend gate failure, not a numerical mismatch, tolerance issue,
component PASS or speed result.

The reported checker location, `check_native.py:133`, only dispatches
`kernel(device_upstream, *device_meta)`. The offending source expression is
`consumer_nki.py:22`, the first NKI-language API access:

```python
assert gradient.dtype == nl.bfloat16 or gradient.dtype == nl.float32
```

The concrete source scope difference is that `build_kernel()` imports `nl`
and `nisa` into factory-local variables and returns a decorated nested function,
without declaring globals or module-level aliases. Own Python source/bytecode
inspection (compile only, no imports/execution) confirms both aliases are
closure cells; the line22 access is LOAD_DEREF, and there is no module-global
binding. In contrast the existing b4d83b5d/3e79 `build_kernel()` explicitly
declares `global nl, nisa` before its local import statements.

**Inference:** a native frontend resolving API names through globals rather
than these captured local aliases is the strongest source-level explanation of
`failed to resolve name 'nl.bfloat16'`. The exact installed frontend namespace
implementation and dtype inventory were not archived; therefore this audit
does not claim to have proved that compiler's closure-resolution internals or
that adding a global binding is a demonstrated complete fix. Later dynamic
loop/gather/SBUF compilation and exact arithmetic remain entirely untested.

The public version-matched
[AWS2.31.1 BF16 API documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/api/generated/nki.language.bfloat16.html)
documents `nki.language.bfloat16`; the
[NKI0.5 release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/release-notes/components/nki.html)
identify that release family. These are reference evidence, not a substitute
for the installed build's missing namespace inventory. The failure alone does
not justify saying BF16 was removed or changing dtype/thresholds/SDK.

The narrow unresolved evidence is the installed frontend's captured-namespace
handling (and actual `nki.language.bfloat16` attribute), not another model run.
Any repair and subsequent native qualification require separate Implementer,
Verifier/binding and Operator decisions. This audit performs no patch, source
mutation, retry, host call, NKI/Torch import or tensor allocation. Original
REFUSE and exact-byte tolerance remain unchanged. Its own initial -O bytecode
inspection mistake is preserved in FIXTURE_FAILURE.json, not attributed to the
candidate.

```sh
python3 research/v2/verification/20260926-k2-native-name-resolution/audit.py
python3 -O research/v2/verification/20260926-k2-native-name-resolution/audit.py
```
