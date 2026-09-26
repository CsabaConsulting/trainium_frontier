# Installed source narrows the next integration experiment

Researcher follow-up, 2026-09-26. Original proposal `60070dee` and its design
remain preserved; this addendum recommends a smaller, separately identified
localization factor before the two-operator experiment. No candidate, checker,
kernel, oracle, host or SDK was changed. No CPU tensor test or native retry.

## Newly observed facts

Read the actual `0770f22a` / canonical `5d91a0d2` observation at
`research/v2/operator/20260926-nki-api-readonly-proposal/results/trn21.json`.
Its local archive SHA256 is
`a8f4b8f721511112d95326e2b2b0a932475a633143d5aa2d86b21b924b72137a`;
the recorded stdout string SHA256 is
`0d8344a8664025b70bfbb8490045a7a7358005792ff42eb1457874948a3a214c`.
This was a stable, no-SDK-import read, not runtime API execution.

| Exact installed evidence | What it establishes |
|---|---|
| `torch_neuronx/__init__.py:186–187`, SHA `91f3fd461b084f1426fbf1a2df9a910950d37c662e52270c1822a6338f31d99c` | Exports `nki_op` and `wrap_nki` from `.nki_hop`; import path is no longer guessed. |
| `nki_hop.py:545–561,564–566`, whole-file SHA `a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914` | `wrap_nki(nki_kernel, **kwargs)` accesses `.func` and dataclass fields, registers kernel metadata, then constructs `NKIHOPCaller` with an empty grid. Its docstring explains dispatcher tracing cannot see raw NKI calls. |
| Same file `271–287` | The compiled route forwards explicit `grid`, tensor arguments, return types and aliases to the torch-MLIR NKI operation builder. |
| Same file `569–603` | `nki_op(name, fn=None, mutates_args={})` creates `custom_op` with an inferred schema. It requires an FX-traceable body and begins describing fake registration. The implementation after603 is absent from the excerpt. |
| `nki_kernel.py`, SHA `37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2` | File exists/stable at8598bytes, but no body was emitted. Its contract is not yet reviewed. |

Static definition hashes are `911baf40120649b1ac42006fb57fc97336a66919b20dff1ca6167ce1d27eb5c3`
for `wrap_nki` and `742146581f6008fd8c8ade1f9730cd9d2cec1a0891c290822044aac1c3589902`
for `nki_op`. Those authenticate the observer's complete definitions but do not
make the unprinted lines visible to this reviewer.

The independently authenticated FP32-first result `696730ff` / `ecd541b1`
now supersedes the earlier report-only qualifier: the audit reconstructs all
four zero gradients, confirms direct/eager and ordinary-compiled exactness,
and retains the failed gate. I read that report; I did not independently rerun
its461checks here. Raw bundle SHA is
`db4fd0b148fd29930bc4925ba898a8f80108d1bd84c6ad3efc33e8694e15e2c6`.

## Recommended smallest next factor

Prefer **dispatcher-visible NKI bridging only** as the next localization
candidate, subject to the exact grid contract below. Retain the frozen custom
Function forward, saved context, backward formula and enclosing compile
settings; change only the kernel callable supplied to it to the documented
dispatcher-visible bridge. Preserve kernel bytes `2c8295a4`, reference
`14572563`, all metadata/math/dtypes, original failures and strict byte checks.

This factor is now source-motivated: the failed wrapper directly calls an NKI
callable inside backward, while this installed implementation supplies a
specific dispatcher-capture bridge. That is not proof the call was omitted or
the reason gradients became zero. It also does not imply that wrapping alone
will succeed. Direct/eager and ordinary-compiled controls narrow the symptom,
not the responsible compiler stage.

The two-operator plus `register_autograd` route remains a distinct fallback
design, not silently combined with this factor. It changes both the autograd
integration and the consumer registration boundary, making a successful result
less diagnostic. Its ATen-forward functionalization/decomposition and complete
fake-registration behavior still require the unprinted `nki_op` remainder.
No reason has appeared to modify kernel arithmetic or restrict the FP32 gate.

## Exact remaining blocker: preserving the existing grid

Frozen `consumer_nki.py:75` returns `resident_consumer[2]`. The kernel asserts
two programs, derives disjoint64-feature halves from program ID, and has no
alternative grid. `wrap_nki` constructs a caller with `[]` at line566, while
copying all dataclass fields except `func` earlier. The excerpt does **not**
establish the type returned by `[2]`, whether its grid is a copied field, or
whether indexing must occur on the HOP caller instead. Empty caller grid alone
does not prove grid loss; hidden fields/defaults may preserve it.

Do not guess either `wrap_nki(kernel)` or `wrap_nki(kernel)[2]`, unwrap `.func`,
rebuild the factory or change the kernel file. First obtain the single bounded
read-only request in `READONLY_DEFINITION_REQUEST.json`. Its complete class and
method definitions make the requested mapping decidable without a device run.
If the indexed callable cannot be bridged without changing protected source,
report that precise obstacle before selecting a different factor.

## Implementation contract once source semantics are known

- Put the bridge outside the unchanged custom Function in a new disabled
  diagnostic-only adapter. Explicit opt-in remains required before native import.
- Own CPU/static tests must exercise the actual extracted grid-mapping helper
  semantics against a minimal fake kernel: `[2]` survives exactly, all tensor
  arguments and order survive, no metadata-value cache or unintended mutation.
  A fake helper test is not actual native availability proof.
- Reuse the already-reviewed FP32-first fixture and route/input-hash controls
  for localization, changing only the declared bridge factor. Report every
  route then fail nonzero on mismatch. Any diagnostic exact result is still not
  the original20-case qualification; full20/all4 with strict unchanged byte
  oracle must subsequently pass before component qualification.
- No timing, BPB, production integration, SDK upgrade or automatic retry. New
  frozen source, own executed tiny tests, fresh independent review and separate
  binding/Operator release remain required. Original failure receipts stay intact.

## Archived path search — an important qualification

A bounded local decode/length/hash check of80 archived small files across the
reporting-r2, namespace-r3 and symmetric native bundles found the exact path
`/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/nki/framework/kernel.py`.
The earlier reporting-r2 error stack contains `__call__` at line239 followed
by `return torch_hop(*args, **kwargs)`, then `nki_hop.py:515` entering eager
dispatch. This is actual historical call-path evidence, not current source.
It shows that raw NKI call syntax already reached a HOP in eager execution.
Therefore explicit wrapping is **not** proved to introduce a previously absent
bridge; the difference in capture/functionalization must be established by
reading that underlying `__call__` and `__getitem__` implementation. If the
existing compile route is identical, wrapper-only may be a no-op and the
registered-autograd route becomes the distinct useful factor instead.

Exact provenance: `94fb52c4`,
`research/v2/operator/20260926-trn22-k2-reporting-native/completed/results-bundle.json`,
SHA `129b9e69352a63d49922dc386c849836ad829dcbb85ef3e33b50b28b64cf9e04`;
decoded `diagnostic.log`44857bytes/SHA
`e9a645d695ee5ed965f7b0280e7bebc30493c1282cc7f1ac4e8a608640f72dd6`.
The same stack names `nki/framework/compiled.py:42,197` and
`torch_neuronx/nki_kernel.py:202,211`. No `nki/__init__.py` origin or jit-factory
definition was found in this bounded search, so those remain unknown rather
than guessed. The single request now uses the observed kernel.py path and
asks explicitly whether automatic and explicit bridging differ.
