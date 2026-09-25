**PASS — limited to source and reduced-CPU suitability for the next exact full-size CPU qualification.**

Candidate `fe0969fd311d654e143dc6113614416c8970a424` meets that narrow review scope. I began at REFUSE, reconstructed and authenticated the supplied sources, inspected the whole change and relevant execution context, compared receipt fields rather than accepting their PASS labels, and wrote and ran an independent bounded numerical test. No blocking candidate defect was found in this scope. This decision does **not** complete or authorize host/native/cold/quality/adoption/submission gates, and no full-size qualification was run.

The evidence supports default/explicit-OFF arithmetic identity at identical step/time inputs, faithful ON schedule arithmetic relative to the frozen reference, and the reviewed toggle/load plumbing. It does not prove identical production trajectories under different real wall clocks, compiler behavior, or full-model quality.

**Source and input identity.** Full-context diffs reconstructed 4,948 baseline lines, 4,980 candidate lines and 4,890 schedule-reference lines. All reconstructed source hashes and all seven manifest-listed file hashes/byte counts matched. The manifest itself was also hashed. Commit IDs below are supplied labels; no repository/history was available to authenticate commit reachability.

| Source | Supplied commit | Reconstructed SHA256 |
|---|---|---|
| baseline | `b4d83b5db66971f3c348837f04f53cba65aa6d33` | `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` |
| candidate | `fe0969fd311d654e143dc6113614416c8970a424` | `75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f` |
| schedule_reference | `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |

| Packet input | Bytes | SHA256 |
|---|---:|---|
| `INPUT_MANIFEST.json` | 1889 | `75a0dbd970b2397c8ad7e75ad564cf1deb28f1a1eed5e4a51e0a288d952895dd` |
| `baseline-to-candidate.diff` | 330617 | `fb62105a077e275d6fe3c8a3e4ef9c0cbeeee3bb07a8c0ed776dc78a33c58afb` |
| `lane-spec.md` | 6408 | `8e84529b9b048da3eb529cde4a2b5985c1593caae076196e190a1896a5cf0b6b` |
| `raw/tiny-g2-default.json` | 17729 | `db8f03e81d0ecd8db096e66dddfe7c09928465ac0a41322483a446f406803b7c` |
| `raw/tiny-g2-off.json` | 17760 | `47a53a74dbb868dbb2beccccc1bd08dacea9abb58b21e44cdf9e53c3df5b0b57` |
| `raw/tiny-g2-on.json` | 17757 | `fac24e14937223420f431b06ac2a55555c31dcf8363c3a3ab1861cd971a76db2` |
| `raw/tiny-g2-reference.json` | 17735 | `ab20ac34f011c476cee5913e812730f16c78bf273d9ca5a71281b166b3cdc357` |
| `schedule-reference.diff` | 323830 | `5a4de9f24603db1d78f4b0d2f3fcf9d131a400a65d9ccd92cd8a67343dbcb651` |

The full compact candidate diff is [candidate.diff](/work/artifacts/candidate.diff); the detailed authentication record is [input_hashes.json](/work/artifacts/input_hashes.json). Original packet files were not changed. Reconstructed source is under `/work/source`.

**Findings and reviewed locations.**

- Default and CLI handling are correct: [module default](/work/source/candidate.py:199), [BooleanOptionalAction](/work/source/candidate.py:3666), [assignment](/work/source/candidate.py:3742). Default, explicit OFF, explicit ON, and ON followed by OFF were parsed using the actual extracted action and assignment.
- [LR](/work/source/candidate.py:3308), [momentum](/work/source/candidate.py:3376) and [beta2](/work/source/candidate.py:3435) functions are structurally identical to the supplied schedule reference and numerically agree in the tested cases. ON retains step warmup, clamps schedule progress, uses Muon/Adam warmdowns 0.95/0.65 and floor 0.05, applies the reference momentum cooldown over the step ramp, and raises beta2 from 0.95 to 0.97. The actual momentum comparison is `p > 1.0 - 0.95`, where the binary64 threshold is `0.050000000000000044`. At step 1, the exact threshold retains `0.8503333333333333`; its next representable successor produces `0.95`. This inherited precedence/discontinuity was preserved and tested, not smoothed into a different schedule.
- [Group updates](/work/source/candidate.py:4100) preserve initial LR and batch-scale multiplication order. Muon gets its schedule; non-Muon groups, including RMSProp tables, get the reference's Adam schedule. `no_lr_decay` takes precedence with the existing step-warmup multiplier. OFF uses the same multiplier for all kinds. [Group construction](/work/source/candidate.py:2196) and [ownership copying of live hyperparameters](/work/source/candidate.py:4714) are unchanged.
- [Compiled beta2 selection](/work/source/candidate.py:2973) adds only the authorized schedule condition. The existing [scalar staging](/work/source/candidate.py:2644) forms `1.0 - beta2` in host binary64 then rounds to FP32. ON passes a changing tensor; ordinary OFF keeps the Python float; the old explicit beta2 ramp still stages a tensor. [Eager beta2 use](/work/source/candidate.py:3014) reads the current group value. Tiny CPU trajectories passed byte comparisons for FP32 and BF16 computation, both eager and the compiled-body dispatch route. The latter executed `_muon_math` directly on CPU; no compiler was invoked.
- [Toggle registration/default capture](/work/source/candidate.py:3221), [restoration](/work/source/candidate.py:3228), [serialization](/work/source/candidate.py:3242) and [evaluation loading](/work/source/candidate.py:3271) preserve the model/load semantics. Actual save/load and restoration were tested with six test-only tensor elements: saved OFF/ON survives, missing old metadata resets to False after ON, unknown keys are ignored, and malformed toggle type raises the existing error. Full-model eval loading was inspected, not executed. Added checkpoint metadata means identical model digests do not imply identical checkpoint files.
- The entire diff contains 11 changed regions. Model construction, forward/loss/evaluator code, optimizer mathematics, base LRs, architecture, data paths, [initial cache refresh](/work/source/candidate.py:4062), [timer initialization](/work/source/candidate.py:4023), [startup reset](/work/source/candidate.py:4137), and [token accounting](/work/source/candidate.py:4158) are unchanged. The complete main-function AST equals baseline after removing only the approved schedule additions and restoring the LR variable name. This corroborates the manual diff review; it is not the numerical test.
- Nonblocking documentation issue: beta2-path docstrings at [line 2658](/work/source/candidate.py:2658), [line 2387](/work/source/candidate.py:2387) and [line 3438](/work/source/candidate.py:3438) still describe the old `MUON_BETA2_FINAL > 0` condition without the new toggle. Executable routing is correct. No source was edited.

The static parameter census matches 113,256,724 parameters / 65 tensors for the specified 8x1024 configuration, and 4,523,316 / 65 for 8x128. It assumes the packet configuration and padded vocabulary 8192; this is arithmetic consistency, not full-model instantiation evidence. Production tokens/update calculate as `1 × 32 × 1024 × 4 = 131072`. All 12 source-audit checks passed; details and source locations are in [source_audit.json](/work/artifacts/source_audit.json).

**Independent bounded test.** [exactness_test.py](/work/exactness_test.py) was written for this review. It extracts exact real functions, optimizer methods and the contiguous schedule/group-update statements without importing the full training module. A test-only class shell provides optimizer state/cache dictionaries; `print0` is suppressed and native NKI is disabled. No mathematical body is replaced. A separate Decimal oracle, precision 60, computes piecewise analytical expectations without calling candidate/reference schedule functions. Frozen baseline/reference execution supplies strict exactness comparisons; the independent oracle supplies a separate numerical correctness check.

The successful run completed **6,263 checks, zero failures**, in **1.950247551 seconds inside the script** (3.961109821 seconds process wall time). Python **3.14.4**, PyTorch **2.13.0+cpu**, CPU only, one intra-op and one inter-op thread; BLAS/OpenMP thread limits were all one. No training model or worker was instantiated. Only one tiny optimizer fixture was live at a time; largest parameter was 3x2. Dispatcher tracking covered fixture tensors and temporary tensor results, conservatively counting shared-storage views repeatedly: **522 bytes peak**, maximum single storage 24 bytes, far below 64 MiB. This is tensor-storage tracking, not total process RSS or internal library workspace measurement.

Inputs cover warmup 0/7, steps 0/1/6/7/150/299/300/777, negative/zero/late progress, adjacent representable values around both warmdown starts, progress above one, nonpositive/tiny/1800-second budgets, legacy OFF knobs, all three group kinds, flat/nonflat groups, and batch scales 1 and sqrt(2). Tiny six-update optimizer fixtures cover tall/wide/square and transposed matrices, FP32 and BF16 Newton-Schulz math, eager and CPU compiled-body dispatch, ordinary OFF and the old beta2 ramp. Initial parameter/gradient formulas are recorded in `trajectory()`; raw records preserve all compared parameter bytes, moment bytes, scalar operands and outputs. Thirty-one negative controls detect wrong floor/kind/linear momentum/ramp precedence/literal threshold/early beta2 ramp/flat precedence/stale beta2 or unintended ON=OFF identity.

Tolerance is **zero** for frozen-source binary64 comparisons and parameter/state byte equality. The independent high-precision schedule/group oracle uses **absolute 5e-16, relative 0**, reflecting differences in algebraic evaluation/rounding; this latter check is not labeled bit exact. All inputs, expected/actual outputs, contexts and tolerances are in [exactness_cases.json](/work/artifacts/exactness_cases.json); extraction line ranges/hashes are in [extractions.json](/work/artifacts/extractions.json). The final failure list is empty in [exactness_failures.json](/work/artifacts/exactness_failures.json).

One failed harness attempt is preserved in [run01](/work/artifacts/run01/exactness.stdout): the extractor searched for a `for` loop and raised `StopIteration` before numerical execution; the source uses a `while` loop. I corrected only the extractor to accept both. No candidate defect, tolerance relaxation, expected-value change or discarded numerical failure was involved. Final code/outputs are also archived in `artifacts/run02/`. Its stderr contains the library diagnostic `Error in cpuinfo: failed to parse processor information from /proc/cpuinfo`; execution completed with exit 0. Both logs and exit codes are retained.

**Supplied raw receipts.** The file named `tiny-g2-reference.json` identifies the baseline hash, not the separate schedule-reference source. All four receipts have the expected source hash, four rank IDs, three finite losses/rank, zero recorded process exit code and no timeout. All ranks report 65/65 changed parameters, no unchanged parameters and empty cross-rank mismatch lists. The 65 rank-0 digest keys are identical across cases.

| Comparison with baseline receipt | Exact recorded losses | Rank-0 parameter digests | Plans |
|---|---|---|---|
| Candidate default | 12/12 equal | 65/65 equal | Equal hash and summaries |
| Candidate explicit OFF | 12/12 equal | 65/65 equal | Equal hash and summaries |
| Candidate ON | First two updates equal on each rank; third differs | 65/65 differ, expected | Equal hash and summaries |

Common plan hash: `83b06c658067902fe9e255779ce792e5117006563c932b16bc1f4bba6f6595d5`. Owner element counts are `[1097728, 1097728, 1081376, 1246484]`, totaling 4,523,316. Each rank-0 receipt records checkpoint existence, successful deterministic reload and logits shape `[1,16,8192]`. All **116 field-audit checks** passed; complete loss arrays, changed-digest names, flags, shims and comparisons are in [receipt_audit.json](/work/artifacts/receipt_audit.json). Copies of supplied raw JSON are in `/work/artifacts/supplied_raw`.

Limits matter: receipt-byte authentication is not independent proof of execution provenance. Only rank 0 supplies final digests, each 16 hex characters; no raw parameter tensors, complete ownership plan, checkpoint bytes or logits were supplied. Thus I independently verified finite losses and consistency/equality of recorded fields, but cannot independently verify all parameter finiteness, the recorded cross-rank mismatch computation, reload determinism, or recompute the plan hash. These are reduced diagnostics, not completed full-size G2 evidence. Synthetic 1x16 microbatches across four ranks mean **64 actual tokens/update** despite retained production-style counters. Fake timing, test prepare/data, CPU execution and no native compile invalidate speed/cold/real-data conclusions. The log phrase “COMPILED body” means the CPU dispatch route here, not a native compiler result.

**Pending gates and authorization limit.**

- Exact full-size 8x1024, 113256724-parameter / 65-tensor CPU qualification of candidate default, explicit OFF and ON, including production group/ownership layout, actual tensor finiteness and reload/logit evidence.
- Production sequence 1024 / pack 32 / total batch 131072 / four LNC2 ranks and full gradient-history validation; supplied receipts use only 64 actual synthetic tokens per update.
- Native Neuron/NKI compilation, dynamic beta2 transport/recompile behavior, device arithmetic, host runtime and cold startup.
- Real timing/startup allowance/checkpoint accounting and 1800-second budget behavior.
- Real-data training, evaluator/causality/quality evidence, adoption and submission review.

The passing result supports taking this frozen candidate to the next separately bounded exact full-size CPU qualification. None of the pending gates is waived or claimed complete. The next full-size run was not launched in this restricted session.

Exact executed test commands, exit statuses, runtime and replay instructions are in [COMMANDS.txt](/work/COMMANDS.txt); `bash /work/run_tests.sh` reproduces the checks sequentially. [decision.json](/work/decision.json) records the machine-readable verdict, scope, hashes, findings and pending gates. [artifact_manifest.json](/work/artifact_manifest.json) hashes the final scripts, report and raw evidence.
