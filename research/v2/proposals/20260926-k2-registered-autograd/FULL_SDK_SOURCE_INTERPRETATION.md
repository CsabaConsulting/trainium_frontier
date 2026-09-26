# Full installed wrapper source: conditional minimal contract

Researcher-only follow-up to `4b6350b3`/`36e6bec1`. The full two-file observation
`7ff42226` (canonical `3f897610`) closes the previously missing HOP and
registration definitions. The automatic NKI callable remains pending the
separately prepared read of its already-observed source; no guessed init path
or further host request is made here.

I read both complete files and independently rehashed them from
`research/v2/operator/20260926-nki-api-readonly-proposal/results/sources/`:

- `nki_hop.py`:22473bytes, SHA256
  `a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914`.
- `nki_kernel.py`:8598bytes, SHA256
  `37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2`.

Only stdlib hash/AST line-range extraction was executed. These SDK files were
not imported or executed. No tensor, candidate edit, host access or retry.

## Grid behavior now established from source

`NKIHOPCaller` spans433–534. Its `__getitem__`453–463 normalizes integer2 to
tuple `(2,)` and returns a new caller with that grid. Its call465–534 preserves
`self.grid` both for real-Neuron eager dispatch and the HOP path. The eager
branch requires not-compiling, actual plain Neuron Tensor arguments and the
enabled eager-dispatch environment setting; other recognized trace tensor
types take the HOP path. Argument names/defaults are preserved in ordered form.

`wrap_nki`545–566 reads the NKI object's function and dataclass fields, creates
a registered TorchNeuron kernel, then returns a HOP caller with an **empty**
grid. `get_dumped_config`175–199 selects `grid[0]` or1 and explicitly invokes
`kernel[lnc].dump_config`. Thus one cannot assume that wrapping an already
indexed kernel automatically conveys its LNC through this caller. The copied
field behavior is not the same as setting the HOP grid. Whether that last
indexing replaces an existing LNC field must still be read in CompileKernel.

The wrapper's source documentation and `nki_kernel.py:112–145` identify the
grid/LNC relationship. `TorchNeuronNKIKernelV3` inherits `CompileKernel` imported
from `nki.framework.compiled`; it does not implement `__getitem__` itself.
Fake-tensor conversion and return metadata are explicitly handled by the HOP;
their existence does not establish correctness of this failed backward graph.

## Registration behavior now established from source

`nki_op`569–634 uses `custom_op` with the inferred function schema, registers
the supplied function as fake implementation, then installs a
`FunctionalTensorMode` decomposition that calls that same function for the
recognized tensor types. It returns the custom-op object. There is no explicit
NKI-only-body check; an ordinary traceable ATen forward is compatible with the
visible implementation contract, although native tracing is not thereby proved.
It does not register a derivative automatically. Explicit `register_autograd`
is still needed to replace the existing custom Function's backward semantics.

## Minimal next factor — decide after automatic-call source arrives

**If automatic NKI `__call__` already constructs the same HOP with its LNC:**
explicit wrapping is not a justified distinct factor merely because its name
appears in documentation. Prefer **one registered forward operator plus an
explicit registered autograd formula**, preserving the existing automatic
backward kernel callable. Do not additionally register a consumer operator or
add an explicit bridge in that candidate.

The implementation contract for that factor is:

1. New disabled diagnostic adapter only; unchanged consumer/reference bytes.
   Define one typed six-tensor forward schema per output dtype, with distinct
   diagnostic operator names. Forward remains exactly ordinary embedding then
   the same output cast; metadata inputs are not recomputed.
2. Register this traceable forward using the installed `nki_op`; register a
   setup-context function saving the same four metadata tensors and a backward
   formula calling the **same** prebuilt `resident_consumer[2]` callable on
   `gradient.contiguous()`. Return FP32 dW and five None entries. The saved
   metadata and supplied tangent remain runtime inputs, not cached values.
3. Remove custom Function use only in the new route, retaining the old route as
   a named failed control. No math, grid, dtype policy, eager-dispatch flag,
   cache policy, mutable alias, initializer or tolerance change is bundled.

**If automatic calling follows a materially different capture path:** document
the precise branch/lines, then consider a bridge-only adapter with the original
custom Function unchanged. Preserve the exact required grid explicitly using
the now-visible caller contract, but only after the NKI object's dataclass and
indexing behavior are known. Do not choose syntax by trial native execution.

The first branch is a smaller registered design than the original two-op
proposal; the second is a different localization factor. The Supervisor may
select one after the remaining source evidence, not both simultaneously.
Neither is a demonstrated repair or a diagnosis of the zero-gradient cause.

## Remaining evidence and gates

Read the pending `nki/framework/kernel.py` full `__call__`/`__getitem__` and
dataclass fields, plus `nki/framework/compiled.py` inherited indexing, to
establish automatic bridge construction, LNC assignment and compile-path
selection. Exact paths already came from the archived traceback; no guessed
`nki/__init__.py` is required for this decision.

After selection, independent tiny CPU/static tests must verify the declared
single-factor mapping, runtime tangent/metadata replay and strict negative
controls. CPU success cannot prove native dispatch. A fresh no-context source
review and separately bound native localization are still required. Preserve
all original failed receipts and byte thresholds; any localization PASS is
not full qualification. All original20 cases on all4 ranks must pass before
component correctness, with no timing, production or BPB acceptance implied.
