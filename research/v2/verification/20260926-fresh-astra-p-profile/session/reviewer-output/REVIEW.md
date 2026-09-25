**PASS — exact Recursive6 source binding and bounded CPU diagnostic scope only.**

This is the user-authorized fresh **SAME-family substitute review**, not a cross-family review. The decision remained REFUSE until reconstruction, independent source review, raw-receipt inspection, and my own executed CPU tests were complete. No blocking defect was found within this limited scope. This does not complete or authorize the native profiling task.

Only the supplied packet, reconstructed files, my own fixtures, and standard/local CPU Python/PyTorch were used. No extra agents, repository/history access, external services, native launch, model allocation at native shape, upload, or candidate fixes were used. The dependency diff is whole-file context, not a production change.

**Content binding.** Both complete addition diffs were reconstructed without modifying their contents. All nine files listed in the supplied manifest match their byte counts and SHA256 values; details are in [hash_verification.json](/work/hash_verification.json).

| Object | SHA256 |
| --- | --- |
| Reconstructed Recursive6 source | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |
| Reconstructed candidate wrapper | `eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d` |
| Dependency context diff | `6107544c8450a2580749ceeb6da09adc4e3e04ecd54d7705261cabead0c8df13` |
| Wrapper diff | `eecc55d15cb202238a93579f06ed46cc0cd34677c914a9d7619b8dadb028fe88` |
| Supplied profiling API source | `ac8a9a43e5d8326cf532dfd2916cf6b7d7833d3a1d67e8ebe0e075aa1dfea3d6` |

The packet identifies dependency commit `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` and wrapper commit `d5e185ed429145d451f602086691a36127714b3b`, at `research/v2/lanes/p-profile/profile_actual_loop.py`. Those commit associations are supplied metadata; content hashes, rather than repository ancestry, were independently verified. The wrapper's other architecture hashes are not approved.

**Source findings.** The exact source resolves to `recursive6`; the wrapper's training argv matches the specified arguments, including seed42, width1024, context1024, pack32, recursive schedule, budget1800 and 22 steps. Argument verification was performed without launching that argv. Default capture is OFF. `run_exact` executes through `runpy` and restores argv, path contents and trace hook on return or failure. The wrapper introduces no model, optimizer or training-function replacement, frame-local writes, or explicit synchronization calls.

The resolved while-loop is dependency line 3953. Its statement annotations are existing synchronization at 3965/4041, forward3977, backward3980, data3982, unscale4007, clip/reduce4008, optimizer4027, cache4039 and boundary4042. The trace prepares before step0, starts at step20 after 20 completed warmups, and stops at the next loop boundary before step21. The source's final step, checkpoint and conditional evaluator paths remain in place. My test does not execute the full model, training adapters, actual checkpoint/evaluator, or real data path.

No candidate bug was reproduced in the tested source/CPU behavior. The following limits affect interpretation:

- Timing is observational and perturbed by profiling. The source schedules learning rate using elapsed wallclock time; fixed-clock CPU equality cannot establish native training identity. The source retains its startup-cap/excess calculation, but its imported `prepare.py` constants and Operator containment are absent. Native fixed1800/cap900 accounting and the existing 2700s hard wall remain unverified.
- With the prescribed `--device-ranks 0`, the source configures CPU+PrivateUse1 and runtime mode on every rank, adding device mode only on rank0. The CLI also accepts other nonempty subsets of ranks0–3 (wrapper lines323–324); this review does not endorse those alternate invocations or treat the wrapper as enforcing rank0-only for every possible argv. Rank0 device collection cannot establish all-rank device coverage.
- Profiling API names, keyword arguments and session-directory call order agree with the supplied Python API text. That is static compatibility evidence only. Native extension behavior, profiler entry/export, event limits, precaching, nonempty runtime/device files, and completeness remain unproved.
- CPU ranges are inclusive/nested dispatch observations. They cannot be summed as exclusive device phase times. Missing embedding/attention/Newton-Schulz events cannot establish absent device work. The wrapper appropriately keeps device embedding-backward time and MFU null, and all-operation BF16 coverage UNKNOWN. No native dtype or MFU result is accepted here.

**My independent exactness test.** Complete source is [test_reference_exactness.py](/work/test_reference_exactness.py). It uses the unchanged extracted real while-loop AST, retaining original line numbers, and extracted source `MuonAdamW`, Muon/Newton-Schulz math, clipping, scaling and schedule definitions. Both comparison arms use a newly written tiny embedding/attention/model fixture, synthetic seeded data, an identity-collective fixture, CPU synchronization/GC stand-ins and a separate deterministic training clock. Candidate `LoopTrace` is unchanged; a CPU-only profiling engine replaces `NativeEngine` solely inside this test. This is an instrumentation exactness test, not a recreation of native training or a four-rank correctness test.

Bounds: width16, context8, vocabulary32, microbatch2, accumulation2, 22 updates, seed42. Python3.14.4, PyTorch2.13.0+cpu; one intra-op and one inter-op thread, with thread environment limits. No full training source import or full-shape execution occurred.

Actual commands, all exit0:

```sh
/opt/cpu/bin/python /work/reconstruct_packet.py > /work/reconstruction.stdout.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 timeout 120s /opt/cpu/bin/python /work/test_reference_exactness.py > /work/own_test.stdout.txt 2> /work/own_test.stderr.txt
/opt/cpu/bin/python /work/audit_receipts.py > /work/receipt_audit.stdout.txt
```

Six of six test groups passed:

| Test | Observed result |
| --- | --- |
| Source binding, exact argv, landmarks, default OFF | Hashes/landmarks match; five altered-argv cases rejected; OFF creates no engine |
| FP32 real-loop OFF/ON | Bitwise equality across 133,170 compared tensor elements |
| BF16 real-loop OFF/ON | Bitwise equality across 133,170 compared tensor elements |
| Early exit and injected forward failure | Incomplete window refused; original exception preserved; scopes balanced on abort |
| `run_exact` | Tiny direct-runpy parity; argv/path/trace restored; preexisting trace refused |
| Startup excess and clock sensitivity | Fixture cap900/excess retained exactly; LR at elapsed0 versus900 differs (1.0 versus0.55) |

Each numerical arm compared 44 losses, 22 per-update snapshots of parameters/gradients/optimizer state, final weight cache, Torch/Python/data RNG states and ordered data/control calls. Each had 45 fetches and exactly 44 existing synchronization calls. Both capture runs recorded prepare(step0,0 updates), start(step20,20 updates), stop(step21,21 updates), then completed update22 without capture. Expected forward/backward/data, optimizer/Muon scalar/math, clip/reduce, cache and boundary labels were observed. The identity-collective labels test tracing, not communication. CPU embedding-backward events were present; they are not native embedding timing evidence.

Raw artifacts: [own_test_results.json](/work/own_test_results.json), [stdout](/work/own_test.stdout.txt), [stderr](/work/own_test.stderr.txt), `float32.raw.pt`, `bfloat16.raw.pt`, matching `.raw.json` tensor hashes and event records, `float32_on.trace.json`, `bfloat16_on.trace.json`, `exception.raw.json`, `exception.trace.json`, and `startup_on.trace.json`. The `.pt` files contain complete comparison tensors. CPU profiler stderr reports unavailable processor/process-name metadata; tests and trace exports succeeded. These host warnings were retained verbatim.

**Supplied receipt audit.** [receipt_audit.json](/work/receipt_audit.json) records actual fields and comparisons. `tiny-off.json` fails on missing pyarrow and `tiny-off-r2.json` on missing requests on every rank, despite both reporting torchrun_rc0. They are failed attempts, not successes. The later OFF-r3/ON pair has exactly equal three-loss sequences on all four ranks and matching plan hashes/summaries. Only rank0 supplies final parameter digests: 22 matching 16-hex-character digests, not full tensor evidence for every rank. All ranks report 22/22 changed parameters and empty mismatch lists; rank0 reports a deterministic checkpoint check.

That pair uses width128/depth2, three CPU steps, effective 1x8 input slicing, no compilation, BF16 compute override, NKI CE no-op, synthetic tokenizer/data, fake clock and stub prepare. ON captures step1→2, not the required native step20. Its named CPU events support limited annotation execution, not native exactness, timing or coverage. My independent test supplies the additional 22-step/step20 CPU evidence without inheriting its implementation.

The API text hash matches the candidate pin. Capability metadata describes read-only help/source inspection, explicitly no profiling import/collection/native launch; all three help commands returned1. Referenced help outputs and `metrics.py.txt` are absent. None constitutes native capture evidence.

**Pending scope.** Native API execution, actual 22-step full-chip LNC2/world4 capture and artifact coverage, real data/checkpoint/evaluation, Operator containment/accounting, device phase correlation, embedding-backward/Newton-Schulz device attribution, actual matmul/attention operand dtype, worst-rank step timing, reviewed FLOP estimate/MFU and profiling overhead, quality/performance adoption, submission/upload, and every other architecture binding remain separate and unapproved. No numeric native MFU or performance claim is made.
