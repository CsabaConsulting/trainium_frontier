**Decision: REFUSE.** The independently written CPU subtest passes, and the source has the intended symmetric feature partition under the addressing semantics described below. The packet does not establish documented NKI 0.5 support for those semantics. No definite new source defect was demonstrated; the blocking issue is an evidence gap, not a demonstrated candidate compiler failure.

This is an independent review of only `/packet/lane-spec.md`, its full-context source diff, manifest, and original raw receipts. No repository, prior reviewer conclusion, network, external documentation, SDK/model import, installation, remote host, native compiler, or training was used. The reconstructed candidate has not been edited. Nothing is authorized for adoption or native execution by this report.

**Identity and scope**

All five manifest-listed packet files match their stated byte lengths and SHA-256 hashes. The single full-context hunk reconstructs both complete source files, including final newlines:

| Source | Lines / bytes | SHA-256 |
| --- | --- | --- |
| Baseline | 72 / 4,396 | `4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8` |
| Candidate | 75 / 4,650 | `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae` |

The manifest labels the baseline commit `d0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca` and candidate commit `9ce2e6a037212aaa76bd85b9b4ab7d311a481757`. With no repository, these commit associations are packet assertions; the file hashes are independently verified. Hash consistency does not establish external authenticity.

The actual source change adds a two-program assertion and `feature_first = nl.program_id(0) * 64`, narrows the accumulator/update/value tiles and predicate broadcast from 128 to 64 features, narrows the gradient access pattern with a feature base offset, slices the output stores by feature half, and returns `resident_consumer[2]`. The group traversal, metadata loads, dynamic bound, active predicate, pointer/count updates, and token traversal do not change. There is no routing, metadata construction, reduction, tolerance, or data change in this diff.

**Source reasoning**

Line references below refer to the exact reconstructed [candidate](candidate_consumer_nki.py).

| Property | Finding |
| --- | --- |
| Explicit two-program launch | Lines 25–26 and 75 assert two programs and return the kernel indexed by `[2]`. Mapping that grid onto LNC2 hardware remains an API/lowering question. |
| Output ownership | Lines 27–28 and 72 declare shared output and store rows `128*g : 128*g+128`, columns `64*p : 64*p+64`. Given one invocation for each `p=0,1` and a shared output, each of the 32,768 elements has exactly one writer. Each program writes 16,384 elements, including unused rows. |
| Runtime-flow symmetry | Lines 28–37 load both row groups and their group-specific maxima identically in both programs. Line 42 uses the loaded bound. The only program-ID computation is line 26; `feature_first` is subsequently read only at the gradient access and the output feature slice. It does not feed a branch, row group, bound, metadata address, or token order. Unequal maxima are allowed: both programs execute the ordered schedule `(group 0, M0), (group 1, M1)`, rather than each taking a different group. |
| Ordered accumulation | For row `r`, iteration `i` has `remaining=counts[r]-i` and `pointer=starts[r]+i`. Valid stable positions therefore supply that row's original token order. The accumulator starts at FP32 positive zero (line 38); lines 61–65 commit one vector addition per active occurrence. There is no cross-tile sum or reassociation in the source. |
| Inactive rows | Lines 45 and 53 replace inactive indirect addresses with sentinel 256; token and value buffers are initialized before skipped copies. Line 63 selects the previous accumulator for inactive rows. The source computes an unused update for inactive lanes, but does not commit it. Initially unused rows are still stored as positive zero. Preservation by the compiled select is unproved. |
| Input and metadata | Shapes and dtypes are asserted in lines 20–24. Stable positions, valid counts/starts/positions, and correct maxima are preconditions supplied by fixtures; this kernel does not validate their contents. The CPU tests construct valid fixtures. |
| Disabled state | The factory defaults to `enabled=False` and raises before NKI imports (lines 8–14). The external checker's disabled state and four-rank/LNC2 checks cannot be inspected because its source is absent. |

The ownership proof is an index bijection: `(p,g,r,c)` maps to `(128*g+r,64*p+c)` for `p,g in {0,1}`, `0 <= r < 128`, `0 <= c < 64`. Quotient and remainder by 128 for the row and 64 for the column recover the unique owner and tile index. This establishes the source-level partition, conditional on the shared allocation/grid interpretation; it does not establish native allocation or placement.

**Offset semantics and documented support**

The CPU model explicitly assumes that `.ap`'s scalar `offset` is an element offset, and that `vector_offset` supplies the index of `indirect_dim` while retaining that dimension's declared stride. Under those assumptions, the positions access reads `positions[address]`, and the gradient access at lines 56–57 reads:

```text
gradient_flat[128 * token_address + 64 * program_id + local_feature]
```

The row stride correctly remains 128 when the feature extent becomes 64. For example, token 1 / program 1 / local feature 0 reads flat element 192. Token 255 / program 1 / local feature 63 reads element 32,767, the final valid element. Sentinel token 256 is out of range for both feature halves. This is the intended addressing calculation, not a proof that NKI 0.5 implements it with this exact API combination. Treating the scalar offset as bytes, treating the vector offset as an already flattened element address, or ignoring the feature offset would change the result.

No NKI documentation, API definitions, or candidate compilation receipt is supplied. The original receipt reports `nki_version` `0.5.0+28631259367.ga768afa6`; that version string does not document support. Specifically, the packet cannot establish:

- JIT grid indexing `[2]`, the no-argument `nl.num_programs()` call, `nl.program_id(0)`, and their LNC2 placement semantics.
- A program-ID-derived element offset combined with this indirect `.ap` pattern, `indirect_dim=0`, and `swdge` DMA.
- Skipped out-of-bounds DMA preserving initialized destination lanes; BF16-to-FP32 widening through this copy; shared-output allocation and feature-slice stores.
- Register-loaded `dynamic_range`, vector arithmetic rounding/subnormal behavior, and bit-preserving selection on this NKI/compiler version.

The first two items directly concern the feature-ownership change. Other items are inherited preconditions whose native support was not demonstrated by the baseline's failed run. They are not newly identified defects. **The precise blocker to an unconditional source/support acceptance is the missing version-specific evidence for the addressing/grid semantics used by the changed source.** No claim is made that these calls are unsupported; support is unestablished from this packet. CPU assumptions cannot serve as their documentation.

**Independent bounded CPU test: PASS_CPU_MODEL_ONLY**

The reviewer's [test](cpu_exactness_test.py) parses the exact candidate with `ast`, checks the relevant launch, assignment, metadata, arithmetic, access-pattern, and store expressions, and extracts the grid/tile/pattern constants. It does not import or execute the NKI function. A scalar model implements the source's grouped metadata traversal and two feature halves using the stated addressing assumptions. Every scalar addition is stored as FP32 using `struct` before the next occurrence.

The separate reference walks original tokens directly, without grouped metadata or feature splitting. Its adder represents finite FP32 numbers as signed integer multiples of `2^-149`, sums them exactly, and rounds to binary32 nearest/even with integer operations. This makes the reference independent of the model's host floating-point addition. Ten known bit-level arithmetic vectors and 4,096 deterministic bounded operand pairs cross-check the two adders. BF16 input bytes are widened exactly by placing their bits in the upper 16 bits of FP32; no SDK is used.

The final run completed **16 cases**, eight fixtures each for FP32 and BF16 input, all with shape V256/C128/T256 and FP32 output:

| Fixture | Group maxima | Purpose |
| --- | --- | --- |
| Empty | `[0,0]` | Zero-trip loops, complete output initialization |
| Unequal bounds | `[3,5]`, `[5,3]` | Both orientations, inactive lanes after short rows finish |
| Single populated group | `[253,0]`, `[0,253]` | Long dynamic loops, zero-trip other group, row-group boundaries |
| All tokens in final row | `[0,256]` | Maximum valid count and final positions entry |
| Interleaved original tokens | `[129,127]` | Stable noncontiguous token order across row groups |
| One occurrence per permuted row | `[1,1]` | Every output row populated, including rows 127/128/255 |

All **2,097,152 output bytes** match the reference; comparisons use complete byte strings, without tolerance. The model performs **330,240 committed FP32 scalar additions** across the fixtures. All 32,768 output elements in each case have exactly one writer with the expected program owner. Both programs' complete modeled control/metadata schedules match, including group maxima and inactive iterations. Per-row observed token sequences match their original order. Unused rows contain positive zero.

Inputs include signed zeros, positive and negative FP32/BF16 subnormals, normal/subnormal boundaries, halfway rounding, cancellation, and distinct values at columns 63/64/127. A separate inactive-select witness preserves negative zero, subnormal words, and an opaque NaN payload as raw bits; the payload is not treated as successful nonfinite arithmetic exactness. Finite arithmetic fixtures reject nonfinite results, and all emitted JSON uses `allow_nan=False`.

Four model mutations are detected: overlapping feature halves, omitted feature offset, a program-dependent loop bound, and unstable positions. Three additional arithmetic/byte witnesses detect reassociation, flushing an FP32 subnormal, and replacing positive zero with negative zero. In particular, ordered `[2^24, 1, -2^24]` produces `0x00000000`, while reordering to `[2^24, -2^24, 1]` produces `0x3f800000`.

The [bounded runner](run_bounded_tests.py) imposes 15 CPU seconds, 30 wall seconds, and 256 MiB address space per subprocess. Both reconstruction and the final CPU test exited 0, with no timeout and empty stderr. The CPU subprocess took approximately 1.12 seconds. Each gradient has only 32,768 elements, and dynamic bounds are at most 256. Exact commands, limits, elapsed times, and exit codes are preserved in [run_receipts.json](run_receipts.json).

**Original native evidence**

The raw rank-0 receipt identifies the **baseline** kernel hash `4939265b…`, not this candidate. Its single case, `cross_tile_large` / BF16, remained `STARTED` with metadata group bounds `[253,0]`. The only successful comparison shown is `upstream_roundtrip`, checking input transfer; there is no completed consumer-output exactness comparison and no timing result. It is not a native exactness pass.

The rank-0 error reports `[INTERNAL_ERROR] [NCC_IXGM002]`: function `sg0000` was expected to have 9 basic blocks in subgraph 0, but core 1 had 1. The actual cache key contains **`_[1]_`**. The accompanying log records a `trn2` / `--lnc 2` invocation and NeuronX Compiler `2.26.6360.0+6f180f47`; its supplied tail ends at the start of tensorization. The error is present in the rank-0 receipt, not a completed candidate compile log. The receipt also notes asynchronous error reporting, so the synchronization traceback does not identify the dispatch origin.

The terminal receipt says `complete: false`, `returncode: 1`, and `DIAGNOSTIC_CHILD_FAILED`. Ranks 1–3 terminal receipts are absent. Their outcome is unknown; no substitute receipts or inferred passes have been created. The cause of the baseline compiler failure remains undetermined. Neither an implicit-grid explanation nor a claim that this candidate fixes the compiler failure follows from these records. Source symmetry cannot prove lowered basic-block symmetry.

The supplied contract states that reference `14572563…` and checker `89189024…` are unchanged, and that native acceptance requires 20 cases / 150 strict full-output comparisons **per rank**, with all four receipts complete. Their full files are absent; rank-0 hash fields alone cannot independently establish the checker's implementation, continued disablement, metadata validation, signed-zero/subnormal checks, or handling of nonfinite metrics. The CPU tests here are reviewer fixtures, not substitutes for those native cases. The missing candidate native results prohibit native acceptance, but were not demanded or attempted as part of this bounded CPU task.

**Disposition and artifacts**

The feature partition and modeled schedules satisfy the contract conditionally, and the independent finite CPU reference checks pass. Keep the overall decision **REFUSE** because the supplied packet cannot complete the documented-support assessment. There is no demonstrated new source-level blocking defect to repair from these tests. Separate unresolved native lowering, device roundoff/subnormals, allocation, performance, model/autograd, and training behavior remain outside what CPU emulation can establish. The old route remains parked; no production route or candidate adoption was performed.

Files left in `/work`:

- `REPORT.md`: this review and decision.
- `reconstruct_packet.py`, `baseline_consumer_nki.py`, `candidate_consumer_nki.py`: reproducible reconstruction and byte-exact source copies.
- `reconstruction.json`, `reconstruction.stdout.txt`, `reconstruction.stderr.txt`: raw integrity results.
- `cpu_exactness_test.py`, `run_bounded_tests.py`: independently written test and resource-limited runner. Reproduce with `python3 /work/run_bounded_tests.py` while the supplied `/packet` remains available.
- `cpu_results.json`, `cpu_test.stdout.jsonl`, `cpu_test.stderr.txt`: structured results and raw CPU output.
- `run_receipts.json`, `run_bounded.stdout.jsonl`, `run_bounded.stderr.txt`: actual subprocess commands, exit statuses, limits, and raw runner output.
- `cpu_artifacts/`: original input gradient bytes; per-case fixtures, both full output byte arrays, ownership and writer-count byte arrays, and both programs' schedule traces. FP32 words and BF16 input words are little-endian, row-major. Writer/owner maps contain one unsigned byte per output element.
- `artifact_manifest.json`: byte lengths and SHA-256 hashes for the final review artifacts, excluding the manifest itself.
