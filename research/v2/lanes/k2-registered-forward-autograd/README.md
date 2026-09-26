# K2 registered-forward adapter — own CPU evidence only

Status: implemented in isolation under the selected CPU-only proposal. The
adapter remains disabled by default. Fresh independent review, native binding,
native exactness, production integration and performance gates are **pending**.
No hosts, real data, model, production source or protected kernel were touched.

The runtime change is the 44-line `adapter.py`: one typed forward registration
per output dtype and its registered derivative. A native caller would pass the
existing `build_kernel(enabled=True)` result without another wrapper. No native
caller or launcher is provided here. Default execution only reports disabled.

## Own executed checks

Local runtime: Python 3.14.4, Torch 2.13.0+cpu, one intra/inter-op thread.
Native evidence is Torch 2.11.0+cpu / NKI 0.5.0+28631259367.ga768afa6;
local results do not prove that version's lowering or automatic NKI bridge.

The CPU checker executes the unchanged `nki_op` AST from the authenticated
installed SDK with real local `custom_op` and `infer_schema`, not a hand-written
decorator imitation. The original Function AST is also unchanged; its test-only
namespace redirects the unavailable `neuron` compile dependency to `aot_eager`.
The original source and native failure receipts remain untouched.

Six fresh-process runs (normal and `-O`, each ordered/zero/stale consumer):

- Ordered: 70 route observations each; all 28 registered eager/AOT forward and
  dW comparisons and all 42 original/ordinary control observations exact.
- Zero consumer: 20/28 registered dW observations mismatched, including both
  dtypes' compiled changed-tangent replay; the eight zero-result fixtures match
  legitimately. All original/ordinary controls stayed exact.
- Stale consumer: 24/28 registered dW observations mismatched; only the first
  unique fixture per dtype/route matched. Compiled changed-tangent replay was
  refused, demonstrating dependence on the registered consumer in local AOT.
- Eight validator negatives per run rejected; NaN/Inf are serialized safely.
- Seven stdlib scope/default tests passed in normal and optimized Python.

There are 8x4 fixture tensors (largest fixture tensor 128 bytes). CPU unrolling,
autograd intermediates and test fixtures are tiny, far below the selected
64 MiB tensor budget; the Torch process/runtime itself is not claimed to fit
in 64 MiB. No full model was imported or allocated. The CPU tensor slot was
released after the six runs.

`raw/commands.json` records commands and exit codes. The six JSON receipts
retain every route, exact comparisons, all seven input pre/post hashes,
runtime/dependency identities, SDK/wrapper AST hashes and validator negatives.
Wrong-consumer selftest exit 0 means the wrong path was **refused**, not that
the wrong gradients passed. Input hash order is weight, ids, counts, starts,
positions, maximums, upstream.

## Retained failed attempt and limitations

The first stale test double returned a closed-over real `initial.clone()`
Tensor. Local AOT fake tracing rejected that test double before completing
coverage. Exact stderr/traceback is `raw/cpu-stale-attempt1.txt`, exit 1;
checker SHA at that attempt was
`ae783da33ef41c3bf674ebc27f9e47b821a157175ee38dd3f79f0c5004f104cc`.
Only the stale test double was corrected to construct the same fixed first
fixture dW inside the traced function with `arange`; no captured real Tensor,
adapter edit, schema change or fake-mode override was used. All six final runs
then used the final checker bytes. An earlier exploratory ordered run succeeded
but emitted a scalar-conversion warning; its comparison helper was changed to
detach before diagnostic error calculation. That early truncated console output
is not offered as the final receipt; the final complete reruns supersede it.

The wrong-consumer experiment is a negative test, not another candidate or an
additional registered consumer operator. Each process builds exactly two
forward operators (one per dtype); the actual adapter source is identical.
Fake and functional handler presence, saved-input identities and CPU AOT
dependence are observed; native compiler graph contents are not observed.
Opcheck was not run. Ordinary embedding control is eager; the original Function
and candidate are each tested eager and AOT. Full original 20-case native
coverage, device rounding, LNC2 runtime capture and higher-order gradients are
not inferred from seven tiny CPU fixture types.

## Reproduce locally (no native action)

From this worktree:

```sh
python3 -I -B research/v2/lanes/k2-registered-forward-autograd/test_static.py
python3 -I -B -O research/v2/lanes/k2-registered-forward-autograd/test_static.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -I -B research/v2/lanes/k2-registered-forward-autograd/check_cpu.py --enable-cpu-test --substitute ordered
```

Use the same explicit command with `--substitute zero` or `stale` for wrong-
consumer checks, and `-O` for optimized runs. These are sequential tiny CPU
jobs only. No native launch command is authorized or supplied.

Reviewer input can be the adapter diff, neutral `SPEC.md`, protected source
context and raw receipts. Implementation notes/tests are not an independent
verdict. Leave this component disabled to undo the reversible implementation.
