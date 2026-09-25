# Independent CPU-only checkpoint diagnostic review

**PASS for the frozen CPU diagnostic source/binding gate only.** Exact packet `d524de5cb974aa77883037cdf858fcfddba3113d` (canonical `135b19e8`) is suitable for a separately assigned, bounded CPU-only inference diagnostic against the pinned already-trained checkpoint. This is not evidence that the full model will pass, and is not native acceptance or permission to launch. The original fresh clean-source `REFUSE` remains unchanged, as do full training and upload HOLD.

Read the independent plan `b27239d4` (original0537291f) and global artifact clarification `e6484d84` (original8848863b), exact probe, CPU driver/binding, validator, both unchanged loaders, manifest and proposal. No candidate/model/prepare edit, live host call, checkpoint download/load, full-size tensor allocation, training, public/private data or evaluator execution occurred in this review. No provider call or reviewer re-prompt occurred. The only local tensors were tiny standalone helper/toy-reference tests.

## Independent executed evidence

`audit.py`: **94 checks**, both normal and `-O`, with Torch2.13.0+cpu, one intra/inter-op thread, largest independent tensor8 elements. This independently authenticates12 manifest entries plus canonical byte identity; CPU/source/support/checkpoint/command pins; immutable no-patching inference scope; and exercises exact AST-extracted probe helpers. It uses an independently written Python scalar/`struct.pack` reference for actual FP32 byte equality, byte mismatch count, element count and maximum absolute error. Full raw helper error records are in `RESULT.json`.

- Exact values pass; a binary0.125 perturbation fails with exact max-error0.125.
- Positive versus negative zero has equal elements but differing bytes and correctly HOLDs.
- The same0.004999999888241291 FP32 difference fails strict batch/source comparison but passes only the separately labelled causal1e-2 criterion.0.019999999552965164 fails causality.
- NaN/Inf remain failure records with finite=false and JSON-safe null metrics, not a serialization crash or PASS.
- Own row-local cumulative-sum reference passes batch/causal checks; own batch-mean dependence and future-token-leak controls fail.
- Shape/dtype mismatch, an empty case set, state mutation, training mode and attached forward hook fail the intended checks.
- Exact scalar validator rejects missing/failed cases, wrong rank, changed source pin/order and attempted `source_review=PASS` relabelling.

`driver_refusals.py`: **57 checks**, normal and `-O`, stdlib only. It executes the exact CPU driver's entire `main` with all host paths/files/processes replaced by synthetic observations, not a rewritten launcher. The successful fixture verifies command/FD/session, authenticated guard-before-child, before/after checkpoint rehash, raw-receipt validation and diagnostic-only terminal. Negative fixtures cover wrong root/host, STOP, RAM/disk floors, prepare/checkpoint size/SHA/support/guard drift, pre-existing receipts, nonempty caches, compiler URL mismatch, guard refusal, child timeout/nonzero, post-read checkpoint change, missing/failed cases and artifact cap group termination. The authentic shared guard parser also rejects another torchrun, unknown device schema and failed ps; both ps and neuron-ls execute when ps detects busy work.

No failed test attempts occurred in this packet's independent normal/optimized test runs. The earlier Implementer tests are not substituted for this evidence. Exact production assertions in extracted functions are compiled with optimize=0; these drivers instead use explicit `require` functions for refusal and remain fail-closed under optimized Python as checked.

## What is actually bound

The CPU command is one single-thread, real-loader inference process under600s timeout plus15s termination grace. There is no torchrun, training, backward/update, public evaluation or automatic native stage in that CPU command. The probe loads reference then candidate sequentially from the same retained453,048,594-byte checkpoint, authenticates its65 state tensors/113,256,724 parameters, strict loaded bytes, eval/frozen status and absence of hooks. It does not replace compute dtype, EVAL_MIN_SEQ_LEN, forward, globals or loader classes. The candidate must remain ordinary `GPT`; research retains its own actual per-row wrapper. No behavioural result can be predicted from the tiny helper tests.

For each source it makes exactly A,B,AB,BA,AC,AA,suffix_AB,AB_repeat,A_repeat atT1024 with deterministic synthetic in-vocabulary tokens.25 checks per rank cover row/batch/permutation/partner/duplicate/repeat invariance, nine same-backend source comparisons, separate suffix causal prefix and before/after registered/direct tensor state. Same-backend noncausal outputs require identical dtype/shape/finite bytes, not a relaxed tolerance. Causality alone follows unchanged prepare's suffix reversal, `cut=max(2,T//2)`, float32 `:cut-1` maximum absolute error<=1e-2. This is explicitly not the official real-data causality invocation or universal purity proof. Arbitrarily nested Python state is not asserted.

The same common lifetime flock path is used; pinned guard authenticates the exact parent/FD and runs fresh ps plus neuron-ls before the immediate child. The authenticated FD is passed into the timeout child. Root/source/support/checkpoint hashes, host, STOP,64GiB available RAM,6.5GiB free disk and absent receipt/cache/metric paths are enforced. No existing SDK cache is deleted or reused: six private empty cache roots are created and compiler URL equals DIR. This is a disclosed implementation choice stricter than the plan's permission to reuse existing roots; it does not create performance/cold evidence. CPU work does not acquire Neuron tensors; actual host/import/runtime availability remains unmeasured and errors are UNRESOLVED rather than permission to patch/retry.

The before/after retained-checkpoint hash is recorded after ordinary child exit, including nonzero exits. Artifact-cap interruption records incomplete status instead of falsely claiming post-read authentication. At most16MiB per rank receipt, zero full mismatch-output tensor pairs, and a global256MiB emitted packet/log/metric/receipt monitor with250ms polling plus final validation are bound. SDK caches are explicitly excluded from that artifact cap and actual compile/RSS/disk cost is unmeasured. Polling may overshoot; incomplete/capped evidence cannot PASS. Timeout/group termination does not prove descendants are gone, so final fresh ps/neuron-ls quiescence remains mandatory for the sole Operator before any subsequent job.

Successful CPU evidence would mean only `BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE`; any difference remains HOLD and execution/timeout/incomplete evidence UNRESOLVED. The proposal has `operator_assignment=NOT_ASSIGNED` and both stage/launch authorization false. Independent raw-result acceptance and a new exact Operator assignment are still required. No gate is silently promoted by this receipt.

## Native scope and minor note

Native driver functions were statically compared with CPU: the shared control flow is unchanged, with distinct label/root, backendneuron, four-rank torchrun and900s+15s constants. **Native mechanics and native source/runtime behaviour are not accepted by this CPU receipt**; its proposal still requires independently accepted CPU evidence and a separate scoped assignment/review. No native execution occurred.

The proposal's final `prior_result_refs` contains the documentation-only transposed path `20260926-fresh-astra-clean-architecture`; the actual immutable fresh REFUSE is `research/v2/verification/20260926-fresh-astra-architecture-clean/` (canonical527db181). This does not change runtime pins or conceal that refusal. Record/correct the citation in a separately frozen documentation change; it is not a CPU runtime blocker.

## Pins and reproduction

- Probe `d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1`.
- CPU binding `0e5fd285080e57d0b03de80c927d89b87d1f35d094e82baf16af7bb3a8964e3e`.
- CPU driver `a2620e94474aeda9463baa173233ac43ac7074ce03c04ba48fdf641518faeb9d`.
- Validator `9565ad654d75a897bc2d8c3f39872c4af54763996af827beca87b0b8e34a5c19`.
- Manifest `251ef57349d3449c49f7c81980c0e974f5c42eb0d9d81534a45fc3ac88a1d10a`.
- Candidate `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`; research `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
- Retained checkpoint `068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25`.

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -O -B research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/audit.py
python3 research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/driver_refusals.py
python3 -O research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/driver_refusals.py
```
