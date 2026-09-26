# Independent restricted source review

Verdict: ACCEPT_BOUNDED_SOURCE_DIAGNOSTIC_ONLY. Default REFUSE was lifted only for the exact supplied checker and the restricted FP32-first diagnostic design, after independent CPU checks. This is not native qualification, launch authorization, kernel adoption, production approval, timing evidence, or submission approval. No native backend, device, network, full model, or package installation was used.

Input was only the isolated packet `/tmp/trainium-k2-fp32-independent-7QD8G2`; no repository/history or implementer contact was used. Installed-runtime discovery located `/home/poojith-devan/venv/bin/python`, PyTorch `2.13.0+cpu`. The system Python lacked torch; this environment probe is not a candidate failure. Supplied author CPU receipts are corroboration only; the attached tests were independently written and executed.

## Exact source binding

- New checker SHA256: `bd2bca7c29615df599094d3c618f5cdf0e4f4381e6d19213dc022691214ce788`.
- Original checker SHA256: `89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850`.
- Kernel SHA256: `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`.
- Reference SHA256: `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.
- `make_wrapper` has identical Python AST in old/new checkers, excluding location attributes. Its math, contiguous gradient operation, custom Function, saved tensors, return arity, and compile settings are preserved. Kernel/reference hashes match the stated immutable pins. No author source was edited.

## Findings

The enabled path seeds 42 and calls only `fixture('unique', 'fp32')`, V256/C128/T256. Route ordering is direct consumer, eager custom autograd, compiled custom autograd, then a separately named and separately compiled ordinary embedding function. There is no prior BF16 or cancellation fixture in this path. The ordinary function bypasses the custom Function and kernel. CPU control compilation confirmed distinct `enclosing` and `ordinary` calls with fullgraph=True, dynamic=False.

All four numerical route results survive a custom-gradient mismatch. Each route takes host pre/post observations of all seven input tensors, records shape/dtype/finite/nonzero/SHA256, and fails if the observations differ. Output/gradient comparisons require matching shape, dtype, finite values, and SHA256 over raw contiguous bytes. Missing results fail explicitly; nonfinite maximum errors are serialized without NaN JSON. Exceptions in a route body are retained with post observations. Classification requires the exact complete route list, exact dense oracle, and exact unchanged-input status for every route; any tested mismatch/mutation/exception classifies to exit 1. `main` returns that code and its script entrypoint raises SystemExit with it.

## Independent executed evidence

`independent_tests.py` uses ordinary CPU torch for an independently handwritten 3x2 unique-permutation embedding backward oracle, then independently verifies that the packet's 256-row unique dense gradient equals the upstream rows. It executes the exact new `run_routes` with explicitly substituted CPU `aot_eager` compilation and an identity-clone unique-gradient stand-in for NKI.

Both normal and `-O` executions passed 33 explicit checks. Positive control: all four routes EXACT, classification 0. Negative control: a test-only weight-gradient hook on the compiled custom path deliberately delivered all zeros; compiled custom became MISMATCH with nonzero count 0, all four routes were retained, the subsequently compiled ordinary embedding was EXACT, and classification was 1. The test also independently rejects missing, wrong shape, wrong dtype, NaN, infinity, changed bytes, and signed-zero byte differences. A metadata mutation with exact output fails, and a deliberate body exception retains its traceback/post observations; both map to classification 1. Raw complete cases including deliberate failures are preserved in `normal.json` and `optimized.json`. Empty stderr files show no hidden execution errors.

These tests replace the backend and kernel explicitly; they cannot establish Neuron compilation, NKI correctness, four-rank behavior, or the original failure's cause. The negative hook tests checker observability of a wrong delivered gradient; it does not reproduce the native compiler defect.

## Limits retained

The first compiled custom fixture is unique/FP32, but direct and eager custom calls intentionally precede it. This does not isolate every possible cache/warmup effect. Pre/post hashes observe host copies, not values inside compiled VJP, and can perturb execution. Mutation is detected after the fact; later routes may see mutated shared inputs, so their results require contextual interpretation. Infrastructure failures before `run_routes` or between route setup steps can produce INCOMPLETE rather than four route records; the nonzero result remains a failed diagnostic. Hardware/process hangs require the separately bound timeout and Operator protocol. Those mechanisms and the proposed host packet were not supplied as executable review targets and are not approved here. The original failed twenty-case gate remains failed.

An initial Python import created ordinary `__pycache__` files in the isolated packet directory; no source bytes changed. The executable test now disables further bytecode writes. No cleanup was performed.

Reproduction: `/home/poojith-devan/venv/bin/python independent_tests.py` and `/home/poojith-devan/venv/bin/python -O independent_tests.py`, from this output directory. Only the exact isolated packet is imported.
