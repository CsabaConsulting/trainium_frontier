**Verdict: REFUSE — bounded source/CPU exactness review.**

The executed candidate CPU backward passes every tested exact comparison. The proposed native algorithm's tile grouping does not preserve dense CPU FP32 addition order: my independent CPU semantic emulation fails six exact comparisons, including a finite, exactly representable BF16 counterexample with dense result **0** and grouped result **1**. This blocks the requested exactness claim. It is a source/CPU finding, not a report of a native execution failure.

The candidate was not patched. No native kernel, compiler, full model, production distributed collective, external data, download, browser/network call, or other agent/model session was used. This is the fresh independent SAME-family substitute lane described by the supplied specification; it is not a cross-family claim. Commit associations and requested model identity are packet metadata, not independently authenticated history/provider evidence.

**Blocking finding R1: per-tile coalescing changes FP32 rounding.**

Source: [candidate.py:1945](/work/candidate.py:1945), [candidate.py:1970](/work/candidate.py:1970), and [candidate.py:1977](/work/candidate.py:1977), corresponding to `workflow/proposals/research-v2/train.py` in the reconstructed candidate.

The kernel starts a fresh FP32 `grouped` sum for each 128-token tile, then adds that sum to an existing output row. FP32 accumulation alone does not make this equivalent to the dense CPU embedding backward's token-order addition. The actual CPU branch uses `index_add_` at [candidate.py:2008](/work/candidate.py:2008), so testing that branch alone misses this difference.

Reproducer: vocabulary 256, width 8, IDs shape `[2,128]`, gradient shape `[2,128,8]`. All IDs are 7 except flattened positions 0, 128, 129, whose ID is 3. Every gradient is zero except those three positions, which contain `2**24`, `1`, and `-2**24` respectively in every channel. These inputs are exactly representable in BF16 and FP32.

| Accumulation | Row 3, every channel |
|---|---:|
| Actual dense CPU backward | 0 |
| Actual candidate CPU backward | 0 |
| Reviewer emulation: FP32 CPU matrix multiplication per tile | 1 |
| Reviewer emulation: sequential FP32 summation within each tile | 1 |

Dense ordering rounds `2**24 + 1` to `2**24`, then subtracts `2**24`. Tile grouping instead forms the exactly representable second-tile sum `1 - 2**24 = -16777215`, then adds `16777216`, giving 1. Replacing those three values with `1`, `2**-24`, `-1` gives dense 0 and grouped `5.960464477539063e-08`. The sequential emulation reproduces both failures, isolating the cross-tile grouping issue from CPU matrix-multiply reduction choices. A seeded all-same-ID case also differs by up to `4.76837158203125e-07` in both emulations.

The emulations reproduce the source's intended finite-input grouping, address selection, and FP32 read/add/write. They do not emulate NKI instruction lowering, native accumulation details, DMA legality, synchronization, or compiler behavior. No claim about measured native outputs is made. Nevertheless, the source algorithm's exact-equivalence premise already fails on a bounded CPU reference.

Required correction/evidence: preserve the agreed dense accumulation semantics across tile boundaries and duplicate IDs, then independently recheck these retained counterexamples and the accumulation/optimizer cases with zero tolerance. If a numerically approximate algorithm is intended, an explicit user-approved change to the exactness requirement and a separately defined error contract are needed; the reviewer has not made that change. A later native run cannot retroactively turn the current CPU semantic counterexample into an exact PASS.

**Input integrity and source scope.**

I reconstructed both entire source files by interpreting the single full-context unified-diff hunk, checked its line counts, parsed both files as Python ASTs, and recomputed SHA256 independently. All ten files listed in the packet manifest match their recorded byte counts and hashes. The manifest itself was also hashed; it has no independent expected hash.

| Source | Lines / bytes | Independently computed SHA256 |
|---|---|---|
| Base, associated with `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` | 4890 / 318875 | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |
| Candidate, associated with `25b731f7638672aa8d03a7cd15b6a02b8f5776ea` | 5014 / 326383 | `e4d93babb4cec8ce53ee1ef9f9d4cc4238c40b93dff0245b66fbcf6bae158b03` |

The complete change is exposed in [changed_context.diff](/work/changed_context.diff). It adds the default-OFF flag, lazy installer, backward kernel/custom Function, token/value embedding training dispatch, CLI parsing, and installation after device setup. There is no changed architecture, Recursive schedule, batch-size calculation, data path, optimizer algorithm, evaluation implementation, or timer logic in the supplied diff. No repository/history is available to verify ancestry or anything outside that supplied diff.

| Reviewed area | Evidence and conclusion |
|---|---|
| FP32 masters / gather | Embeddings are ordinary `nn.Embedding` parameters at lines 2024–2027; main moves the model by device at 3974 without a whole-model BF16 cast. The custom Function rejects non-FP32 weights (1993), performs `F.embedding` in FP32, then casts the output (1998). Existing embedding defaults have no padding, sparse mode, or max-norm behavior to preserve separately. |
| Output-cast backward | The upstream gradient arrives in the output dtype; CPU backward promotes it to FP32 before row accumulation. A separate test supplying FP32 `1.003` gradients to a BF16 output confirms BF16 quantization occurs before accumulation. It differs from accumulating the unquantized FP32 inputs. No BF16 accumulation tolerance was used. |
| Duplicates / unused rows | The CPU branch allocates a fresh zero FP32 dense gradient each call and adds all occurrences. Independent tests cover IDs within/across 128-token boundaries, all-same IDs, cancellation, unused rows, empty input, boundary IDs 0/255/256/511, int32/int64 IDs, and noncontiguous IDs/upstream gradients. All CPU comparisons pass exactly. |
| Repeated backward / optimizer | Returning FP32 weight gradients lets autograd accumulate into `.grad`. Two backward calls on the same graph, two separate micro-backwards per step, and existing flat-buffer gradient views pass. Token and value parameters remain in their unchanged AdamW groups (2294–2302). The test executes the unchanged `_device_scalar`, `_adamw_state`, and `_adamw_step_foreach` methods on a tiny holder, comparing parameters and both moment tensors for three steps. It does not execute the ownership/distributed adapter. |
| Default / explicit OFF | Module default at 200 and BooleanOptionalAction default at 3721 are both false; main assigns the parsed flag at 3781. Token/value dispatch at 2343 and 2355 requires both the flag and training mode. Installation at 3953 is conditional. OFF follows the original gather/cast statements and does not install the new kernel. Parser and three-step extracted-path tests pass for default/OFF; ON CPU also matches. |
| Native duplicate addressing | Static inspection: equality matrix groups identical IDs; the last occurrence survives, other positions select distinct dummy rows `vocabulary + position`, and two programs own disjoint channel halves. Sequential tile iteration carries state across tiles. Reviewer emulation checks 128 distinct addresses per tile and zero dummy rows for finite inputs. This supports collision-free intended algebra, not native execution legality. It does not fix R1. |
| Native restrictions | Kernel requires BF16 gradient, token count and vocabulary divisible by 128, even width at most 1024, flattened IDs, and `vocabulary + 128 < 2**24` (1923–1930). The wrapper checks most shape restrictions and BF16/installation, launches two programs, and slices away dummy rows (2009–2014). IDs are converted to FP32 for comparisons, not BF16. Valid IDs must be in `[0,vocabulary)`; the kernel has no explicit value-range or index-dtype check. Normal forward uses embedding's index contract, but native invalid-index behavior remains untested. Model vocabulary padding is only to 64, so some otherwise valid configurations will be explicitly rejected when native backward starts. CPU acceptance of shapes does not establish native support. |
| Checkpoint / evaluation | `save_checkpoint`, `current_toggles`, `restore_eval_toggles`, `load_for_eval`, and `_SubmissionEvalGPT` have identical ASTs across sources. The new training-backward flag is absent from `ARCH_TOGGLE_NAMES` (3277) and final checkpoint metadata (4218), so checkpoints do not record/restore whether training used scatter. This is a provenance limitation, not an evaluation correctness defect: `load_for_eval` calls `.eval()` and disables gradients at 3360–3361, and both changed branches bypass scatter in eval. Extracted eval lookup tests prohibit calling the custom Function and pass even with ON. Full `load_for_eval` was not executed: its unchanged context expansion to 8192 exceeds this lane's limits. |

**Independent executed test and raw failures.**

Test source: [reference_exactness_test.py](/work/reference_exactness_test.py). It was authored for this review, without supplied test source. It AST-extracts the exact `_EmbeddingScatter`, CPU installer, gradient-buffer helper, parser statements, token/value lookup statements, and three AdamW methods. No full module import, GPT construction, training main, or native import occurs. Extracted locations are recorded in `reference_results.json`; generated wrapper functions only provide tiny call-site context. The manual oracle uses explicit token-order FP32 row additions, independent of `index_add_` and embedding backward. The two tile emulations are reviewer-written.

Actual command, exit **1** (retained exactness failures; no harness exception):

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /opt/cpu/bin/python /work/reference_exactness_test.py > /work/reference_test.stdout 2> /work/reference_test.stderr
```

Python `3.14.4`, PyTorch `2.13.0+cpu`; seed 42 for Python and PyTorch; PyTorch intra/inter-op threads both 1. Thread environment variables were set explicitly, without enumerating the process environment. Build details are in `torch_build.txt` (MKL 2024.2, MKL-DNN 3.12.0, GCC 13.3). PyTorch emitted `Error in cpuinfo: failed to parse processor information from /proc/cpuinfo`; this raw diagnostic is retained. Tensor operations completed and there was no harness exception.

Component bounds: vocabulary at most 512, width at most 128, 256 tokens, context 128, microbatch 2. Three-step integration uses three `[64,8]` embedding tables, IDs `[2,16]`, two backward calls per step, and both FP32 and BF16 outputs. All computation is CPU eager; there is no compilation or timing comparison. Equality requires identical shapes, dtypes, and tensor bytes (`atol=rtol=0`); no failed test was weakened or retried.

| Checks | Passed / total |
|---|---:|
| Component forward, dense/manual backward and unused rows | 280 / 280 |
| Three-step output, gradient-buffer, parameter and moment comparisons | 306 / 306 |
| CLI, gradient aliasing, absent value table, evaluation, casts, input rejection and static checks | 110 / 110 |
| Native-algorithm CPU emulation exactness | 12 / 18 |
| Total | **708 / 714** |

All 696 non-emulation checks pass. Failures are preserved in `reference_test.stdout`, `reference_results.json`, and `reference_tensors.pt`. The tensor file includes actual/expected outputs for all six failures plus IDs/upstream gradients for the two explicit cross-tile counterexamples. The seeded test source reproduces the other input. It is not necessary to trust a PASS label to see the differing values.

**Audit of supplied receipts.**

I parsed every raw JSON record and compared actual values; [receipt_audit.json](/work/receipt_audit.json) preserves the extracted fields and comparisons.

The four G2 records name the reconstructed base/candidate hashes and report process exit 0, four CPU ranks, three steps, depth 2, width 128, and an effective training slice of `[1,8]`. Although argv says context 1024/pack 32, the declared shim slices training input. Logits shape in the checkpoint check is `[1,8,8192]`. These receipt sizes are not shapes executed by my test. All four modes have exactly the following recorded losses:

| Rank | Step 1 | Step 2 | Step 3 |
|---|---:|---:|---:|
| 0 | 9.011563301086426 | 9.02148723602295 | 8.995948791503906 |
| 1 | 9.009450912475586 | 9.009400367736816 | 8.911516189575195 |
| 2 | 9.015210151672363 | 9.00666618347168 | 9.009931564331055 |
| 3 | 9.011550903320312 | 8.9981050491333 | 9.014042854309082 |

All 22 rank-0 final parameter digest strings agree across base/default/OFF/ON, including WTE `36b1643c6ec3531c` and value embedding `ab16f0000c27eb47`. All plan hashes agree at `190ff1c2328be1226db47a5cad55e262745dd4e401524cfb26aedac7128f5935`; owner element counts are `[1048576,1048576,1048576,393764]`. Each rank records 22/22 changed parameters and empty cross-rank mismatch lists. Rank-0 checkpoint records say present/ok/deterministic; the checkpoint bytes are not supplied. Other ranks have no final digest maps. The digests are 16 hexadecimal characters, and there are no per-step full parameter/optimizer tensors; receipt equality has that evidentiary limit.

G2 shims force BF16 compute on CPU, skip NKI cross-entropy installation, use a deterministic fake clock (1 ms/call), provide synthetic tokenizer/data, and slice training inputs. Public evaluation and compilation are disabled. Recorded elapsed times 4.7/4.2/3.9/3.9 seconds cannot establish a native whole-step speedup or cold startup. These are supplied reduced CPU receipts, not a rerun of the full training loop in this review.

All four component records claim the candidate hash, CPU PyTorch `2.13.0+cpu`, shape `[256,128,256]`, and six cases. Initial/final normal/optimized records report the same numerical errors and tolerances:

| Case | Reported maximum absolute error | Reported maximum allowed |
|---|---:|---:|
| unique | 0 | 0.000093875 |
| repeated | 5.960464477539063e-08 | 0.000347625 |
| all_same | 9.5367431640625e-07 | 0.004599760009765625 |
| boundary | 0 | 0.00015684375 |
| zeros | 0 | 0.000002 |
| cancellation | 0 | 0.0050455791015625 |

These values appear under algebra and both `eager_native_control`/`eager_candidate`; equal errors against an unspecified oracle do not prove pairwise byte equality. All forward flags are exact, gradient dtypes FP32, unused-row flags true, and empty-gradient flags true. Only the two final records add `cpu_two_backward_fp32_accumulation_exact: true`. The control's name does not make it an NKI execution: the device is CPU. No supplied test source allows verification of oracle construction, tolerance formula, case construction, or whether the `optimized` filenames correspond to Python optimization flags. The records explicitly leave whole-model correctness PENDING and performance UNMEASURED. First-call timings differ substantially and are not a comparable whole-step benchmark. My own CPU tests independently reproduce the relevant CPU exactness/accumulation claims, not the unspecified component oracle or native behavior.

**Pending scope and disposition.**

R1 prevents a scoped source/CPU exactness PASS. The default/OFF source path, bounded actual CPU Function, and tiny optimizer comparisons have positive evidence as described above. No merge permission or hardware launch authority follows from those results.

Native NKI compilation/API/DMA legality, numeric outputs, compiled autograd, production shapes, two-program ownership, distributed world4 behavior, full-model correctness, cold-start cost, quality gates, and performance are all unestablished. Native nonfinite-input behavior is also outside these finite-input emulations. Later adoption requires the separately assigned native gates and at least 5% faster whole step on the base. Official accounting remains fixed 1800 seconds with startup allowance capped at 900 seconds and excess charged, fullchip LNC2/world4, seed42 comparisons. Only the sole Operator may assign those later gates; none was run or invented here.

Artifacts include this report, `result.json`, the independent test and packet-audit sources, reconstructed sources, input hashes, command/exit records, parsed receipt audit, raw stdout/stderr, numerical results/tensors, PyTorch build details, and `ARTIFACT_MANIFEST.json` with SHA256/size entries. The manifest excludes itself to avoid a self-hash cycle.
