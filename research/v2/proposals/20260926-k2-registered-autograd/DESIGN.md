# K2 registered-operator follow-up — design only

Proposal-first commit `60070dee`; Researcher-only, 26 September 2026. No source,
kernel, oracle, checker, SDK or host was changed. No CPU tensor test was run.
The original failures remain failures. This is a new integration hypothesis,
not a working fix, performance result, source acceptance or launch binding.

## What the evidence supports

The Supervisor reports that the fresh FP32-first attempt still returns zero
custom-compiled dW, despite exact direct/eager results, exact custom forward,
exact ordinary-compiled forward/dW and unchanged input hashes on all four
ranks. That raw result is still subject to its independent outcome audit; this
task did not retrieve or authenticate it. Taken as reported, earlier BF16 or
zero-fixture history is not necessary for failure. It does not distinguish
capture, tangent/metadata plumbing, registration, lowering or code generation.

The authenticated prior archive still proves a real failed byte gate, not a
tolerance problem. Direct/eager success does not validate compiled-backward
consumption, and ordinary-compiled success does not validate the custom call.
No concrete Python arithmetic defect is identified by this source reading.

## Version-specific evidence and limits

| Evidence | Supported fact | Unresolved boundary |
|---|---|---|
| [AWS native guide v2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/frameworks/torch/pytorch-native-overview.html#neuron-kernel-interface-nki-integration) | Describes `nki.jit`, `nki_op`, explicit autograd registration; appendix discourages the old Function pattern. | No exact import/signature, grid dispatcher or our installed build contract. |
| [PyTorch 2.11 library API](https://docs.pytorch.org/docs/2.11/library.html#torch.library.register_autograd) | Backward/context must be traceable; non-traceable work can be a separate custom op. Fake implementation supplies metadata without tensor data. | Correct registration is not Neuron lowering proof; opcheck is not mathematical gradient proof. |
| [AWS-owned lab](https://awslabs.github.io/accelerated-compute-tutorials/en/aws-ai-chip/education/ndd-day2-hands-on-lab/) | Uses `nki_op`/`wrap_nki`, NKI0.5.0+28631259367.ga768afa6 and compiler2.26.6360.0+6f180f47. | Lab torch-neuronx2.11.3.0.1417+1431f083.dev differs from archived host2.11.3.0.0+unknown.dev. Unversioned page is not package source. |
| [Current PyTorch tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html#adding-training-support) | Registration is preferred for compile composability. | Tutorial2.14/Triton is not Torch2.11/NKI implementation evidence. |

The older archived host metadata confirms Torch2.11.0+cpu, the NKI/compiler
builds above and torch-neuronx2.11.3.0.0+unknown.dev. The original symmetric
rank0 receipt independently confirms Torch2.11.0+cpu and the same NKI build.
Local CPU Torch2.13 is different. Package versions alone cannot establish
the presence or exact semantics of a registration function.

Narrow searches for `nki_op|wrap_nki|register_autograd` in archived Operator
JSON/text/Markdown and the relevant analysis found no installed registration
source inventory. The analysis contains candidate FP8 use of `wrap_nki`, not
an SDK definition; successful FP8 forward is limited operational evidence for
that used path, not for wrapping K2's already-grid-indexed callable. The
AWS guide's linked public `aws-neuron/torch-neuronx` repository returned HTTP404
through the read tool, so no code was inferred from its name. Third-party
search hits were excluded from the evidence.

## Smallest design, with one explicit API-observation dependency

The proposed component has two separately named registered operators:

1. Forward keeps exactly `F.embedding(ids, weight).to(output_dtype)` with FP32
   master weight. It remains the old diagnostic boundary for each output dtype;
   the FP32 case is mandatory. No production external-cast boundary is changed.
2. Registered consumer backward calls the unchanged kernel built by
   `build_kernel(enabled=True)`, including its `[2]` launch grid, on
   `gradient.contiguous()` and the four unchanged metadata inputs. Attach a
   registered autograd formula to the forward; save only those four tensors,
   return FP32 dW and None for non-differentiable inputs. No autograd.Function,
   cached tensor values, dtype bypass, alias trick or zero-filled fallback.

This is a contract design, deliberately not speculative decorator code.
`torch.library.custom_op` being present does not imply that an opaque ATen
forward will lower through Neuron. Before an implementation is selected, one
bounded, separately assigned **read-only installed-source observation** must
record:

- exact distribution/file hashes, exports, signatures and source of `nki_op`
  and `wrap_nki`, plus registration/FakeTensor/decomposition helper definitions;
- whether `nki_op` supports an ordinary ATen forward composition and exposes
  `register_autograd`, or which documented dispatcher registration achieves it;
- whether the bridge accepts the already-indexed NKI callable and preserves
  `[2]`/LNC2, without rebuilding or modifying the protected kernel;
- whether backward custom calls and FP32 output metadata are supported in this
  build, and how fake shape/dtype/stride/device behavior is supplied.

Use installed-file metadata and bounded source reads first, not an SDK upgrade,
tensor allocation or trial native kernel. If these semantics are absent, stop
with the specific missing contract; do not silently choose a guessed API.
No host inspection was requested or performed by this task.

Wrapping only the old backward call through `wrap_nki` could be a smaller
localization arm, but it retains Function and tests a different factor. It is
not bundled into this two-operator proposal or asserted to work.

## Cheapest falsifier and unchanged gate

Future tiny CPU prototype: V8/C4/T8, one thread, below64MiB tensors, one fresh
namespace/process. Independently reconstruct ordered FP32 accumulation and
ordinary embedding results. Compare eager and aot_eager on nonzero FP32 unique
first, duplicate indices, cancellation, signed zero, empty metadata, then two
different nonzero tangents and changed valid metadata through the same compiled
call. Test strict byte/error reporting by injecting zero or stale gradients.
Inspect that the captured backward contains the registered consumer call with
the current tangent/metadata inputs; record an unavailable capture as unknown.
Use API contract tests separately from numerical comparisons. Never treat
opcheck's dtype-dependent default tolerance as the byte oracle. No CPU result
would establish installed native success.

After API evidence, frozen implementation, fresh independent review and a
separate guarded binding, retain **all original20 BF16/FP32 fixtures on all4
ranks** and every original direct/repeated/oracle/forward/backward check with
identical bytes. Add FP32-first nonzero as a sentinel and preserve all failures;
it cannot replace the suite. Missing cases, nonfinite values, zero-wrong dW or
changed hashes fail. No runtime timing or model-quality gate is passed here.
Only later independently reviewed production integration may pursue the
predeclared >=5% matched whole-step threshold. No full training is proposed.

## Local provenance

- Original bundle: `research/v2/operator/20260926-trn21-k2-symmetric-native/completed/results-bundle.json`, SHA256 `16dc0f7d84eed33eaaa821cb6afd21d00fc5b6d60b4ad1b9f93e55f3e07b9c7b`.
- Its rank0 receipt redecoded and length/SHA checked here: `6a8fa48e399a39486101b4662585e875de6a7458542660abac21bf7f0f247fbd`; rank0 environment likewise `5cef919d275eb4b693c503c681c898841868b84ec7a01e86bd1def267101ff3e`.
- Archived metadata: `research/v2/operator/20260925-trn21-l1-fp8-native-r1/runtime-metadata-preparation.json`, SHA256 `a807a703b0b6a239b208051a5335ca61005e287063e8bdcbbf75e3ba6b6888d9` (older observation, not refreshed).
- Kernel rehashed `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`; reference `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.
- Primary snippets/versions/retrieval time are in `PRIMARY_EVIDENCE.json`. Hashes cover these exact bounded snippets, not full web pages. No installed SDK implementation was available in the scoped archive search.
