# Independent NKI dpre gatefixed source/CPU review

Verdict: `PASS_SOURCE_CPU_ONLY` for exact source `0bc973832a707893c73304449e219759f5431c672a919103e1247ffabc91c869`, parent `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`. No source blocker found. Native dispatch, enclosing-graph compile/correctness and performance remain **NOT_ESTABLISHED**. This receipt is not launch/adoption/submission authority and adds no extra host or repeat compiler gates.

## Exact composition and policy

Removing the five gate-policy lines reproduces archived dpre source3267 byte-for-byte. Independently removing the inserted component block and new native MLP dispatch reproduces500f byte-for-byte. The standalone kernel/helper files match their archived bytes; integrated ASTs match after mechanical normalization of the four function names. Every other source byte—including model/state shape, loss, schedules, optimizer, evaluation and defaults—is inherited500f.

Actual production-meta source execution verifies65 unique optimizer tensors/113,256,724 elements, complete four-owner coverage and exactly the same policies/owner plan as500f. Matrix VE gate remains AdamW LR.021, betas(.9,.95), epsilon1e-10, WD0; scalar U-Net gates retain LR.2. No optimizer-policy leakage from historicalc239 remains.

Ordinary MLP forward math and saved ReLU are unchanged. Native eligible training dispatch uses the registered operator; missing registration raises. CPU calls retain the parent diagnostic path, so those calls alone would not verify the new wrapper. The independent test therefore calls the actual integrated wrapper directly. Targetless/eval forward remains ordinary linear→ReLU²→linear; no evaluator, tokenizer, data or launch command changes.

## Independent small reference

The verifier wrote a bounded CPU interpreter for only the operations used by the exact kernel AST. This is neither AWS's simulator nor a native execution. It executes the real loop/access-pattern code, applies explicit BF16 destination casts, and counts each output write. An independent float64-on-BF16-input matmul/rounding oracle supplies expected values.

- Six small shape/upstream cases cover LNC1/2, tile128/512, multiple token iterations/contraction blocks, zero ReLU and changing upstream. Every output is written exactly once with no cross-core overlap; worst relative RMS2.4215741e-5 and max absolute.001953125, below the existing.01 RMS tolerance.
- BF16 Q-rounding sentinel: a=1.328125, b=.63671875, ReLU=2.046875 gives required3.453125 versus3.46875 if FP32 Q bypasses its BF16 boundary. Actual kernel-body execution gives the required result.
- The exact integrated registration factory was exercised with the **actual archived nki_op decorator body**, extracted without importing the native module. Three upstream layouts (ordinary, transposed, expanded), two accumulated microsteps each, match500f forward and all gradients bit-for-bit in these fixtures. A separate own FP64/BF16 dpre oracle—not the author's reference—was injected.
- Two changing-upstream local AOT cases pass; worst gradient relative RMS6.6142531e-6. Zero-dpre poison makes dX/dU exactly zero while dD remains live in both eager and AOT paths, proving local backward dispatch was not bypassed.
- Actual tile512 production fake metadata checks `[32768,1024]` output and contiguous BF16 `[4096,32768]` saved/dpre. Three invalid input cases are refused. Production token ownership was checked exhaustively without allocating production tensors.

One small CPU process; local Torch2.13 differs from host Torch2.11. Test body elapsed1.89s. Exact command, hash and full results are in the adjacent JSON. CPU numerical checks do not establish native TensorCore accumulation order, compiler lowering, registration or performance.

## API and inherited artifact scope

Read the installed frozen `NkiTensor.ap` and `nc_matmul` definitions directly: scalar offsets use the selected original stride, and matmul computes stationary-transpose times moving. Kernel grad reads therefore offset by token×C; saved/output reads by token; first contraction initializes PSUM and later contractions accumulate. Production LNC2/tile512 has32 token iterations×32 hidden tiles×8 contractions=8192 matmuls per core. Estimated SBUF guard is13MiB including4MiB reserve, not an allocator/spill measurement.

The new integrated registration fixes tile512 and requires N divisible by1024. Production N32768 qualifies. Small N256 wrapper cases deliberately use tile128 and must not be mistaken for valid production-native dispatch cases. Incoming upstream contiguity copies remain inside any eventual component/whole-step measurement.

Original author simulator receipt5f9ea40a and offline NEFF receipt735d8e8d still bind byte-identical standalone kernelbe1c1e3a; their source/status hashes and the104,810-byte NEFF7dd21137 were checked. They remain **author simulator/offline evidence** for that kernel, not independent native proof for source0bc or its enclosing training graph. No compiler or NKI simulator was rerun.

Existing native dispatch/changing-upstream/master-gradient correctness, cold<=900s, whole-step timing and budget/fresh20M rules remain pending under the sole Operator. Compare against fresh500f, not historicalc239; any changed scientific parent needs rebase/review. No speed/BPB gain is claimed. Source, host, queue, state, SDK and evaluator were untouched.
