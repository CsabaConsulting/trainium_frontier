# Independent L1 FP8 source and bounded CPU review

Verdict: **PASS_SOURCE_AND_REDUCED_CPU_ONLY**. No concrete new L1 defect was
found in the reviewed diff. This receipt does not claim exact-size G2,
native compilation/numerics, native full-model correctness, throughput, BPB,
or submission readiness.

Reviewed 25 September 2026 by `/root/fp8_independent_verifier`, a distinct
Verifier that did not propose or implement the candidate. A separate bounded
receipt/benchmark audit was delegated to `fp8_receipt_audit`; this reviewer
also reran the final candidate's small ON/OFF G2, supplementary CPU checks,
benchmark synthetic checks, and the independent audit below. No host, external
API, Neuron device, training data or credential was accessed. Production source
was not edited.

## Exact scope and authority

- Implementation commit: `8b6d6cd9cf3079ab098794abc44864bf6d513b21`.
- File: `workflow/proposals/research-v2/train.py`.
- Candidate SHA-256: `dea47910547572c704d8d5cc0e3dd058282fc8d862c913939bfc5e40830713b4`.
- Exact parent-file SHA-256: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- Protected `prepare.py`: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`, unchanged.
- G2 harness: `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`, unchanged.

Canonical Desktop AGENTS, README, current handoff, experiment protocol,
decision log, workflow role contract, exact L1 lane requirements, FINAL PUSH,
resume/table correction and compile-constraint correction were consulted.
The current instruction is a local implementation/review lane. Older parked-FP8
and provider-routing text is superseded. The pre-existing c5c startup predictor
is unchanged by L1 and is under separate review; this receipt neither reports
it as a new L1 defect nor clears it for future integration.

## Source findings

1. `USE_FP8_PROJ=False` and the default-false CLI preserve the base path.
   Construction adds only a non-parameter boolean. OFF still installs the
   existing fused BF16 MLP implementation. Parameter names and shapes remain
   unchanged; no additional RNG operation is introduced. ON bypasses that
   adapter, so the FP8 MLP projections cannot be silently replaced by BF16.
2. The marked modules are MLP `c_fc/c_proj` and attention Q/K/V/O, including
   fused QKV when that existing option is selected. Embeddings, LM head and
   small gates retain their base behavior. Current FP32 masters are cast and
   prepared on each forward. The FP8 modules clear/skip the BF16 cache, avoiding
   stale prepared weights after an optimizer update.
3. Quantization scales depend only on one token row or one output-channel
   weight row. Native byte packing maps `[output,K]` to `[K%128,K//128,output]`;
   the input transpose and two-block contraction return `[token,output]`.
   Output-channel scaling followed by token scaling has the correct axes.
   The independent coordinate witness confirms this index algebra, not NKI
   execution. FP8 operands reach `nc_matmul` directly; FP32 accumulation and
   rescaling precede the BF16 output conversion.
4. Saved BF16 effective operands are reconstructed from the quantized values
   used by the forward. Backward computes `dX=dY@W_effective` and
   `dW=dY.T@X_effective` with BF16 operands, then returns the original input
   dtype and FP32 master-weight gradient. Token-dependent scales remain inside
   dW's token sum. This matches the declared straight-through approximation.
   Non-unit/dynamic loss scaling and transposed/coalesced layouts are rejected.
5. Native shapes require K divisible by 256, N by 128, K<=8192 and a source-level
   SBUF bound. Padding is on token rows only; row-local scales prevent padding
   or companion examples from changing valid rows. The source bound is not a
   compiler-allocation or spill proof.
6. The new toggle is captured before `_EVAL_TOGGLE_DEFAULTS`; checkpoints restore
   it before model construction, while older checkpoints restore OFF. The
   load/save functions are unchanged and use the existing strict state load.
   No new train/eval math branch, collective, mutable quantization history,
   data access, network access or evaluator change was introduced.
7. AST comparisons confirm unchanged load/save/restore, dataloader, LR schedule,
   gradient clip/synchronization and entire `main` while-loop code. The timer
   correction, any schedule stack and any future table stack require their
   own combined-byte checks when integrated.

The native implementation follows repository FP8 layout precedents cited by
the author. Its README cites AWS's legacy E4M3/double-row support. This review
did not query external documentation or claim that the installed competition
SDK supports every new kernel operation; native compilation is still pending.

## Bounded numeric evidence

All reruns used `/home/poojith-devan/venv/bin/python` (Torch `2.13.0+cpu`),
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, seed 42,
and the lane's explicit test-only PyArrow stub. The G2 runs used the existing
synthetic prepare stub and recorded BF16, CE, deterministic-clock and 1x16-token
microbatch shims. No production stubs were added.

- Independent OFF and ON G2 reruns each PASS: four ranks, depth 2, width 256,
  three optimizer steps, 22 parameter tensors and deterministic checkpoint
  load. The runs took 10.6 and 10.2 seconds respectively; these CPU timings
  are not performance evidence. All ranks report finite losses and no parameter
  disagreement. ON differs in argv only by `--fp8-proj`.
- Independently rechecked the corrected exact-c5c base receipt against both
  supplied OFF and independently rerun OFF: ownership plan, every rank's three
  losses, and all 22 final parameter digests are identical. The base itself
  was not rerun in this review. The incorrect historical `g2-base-small.json`
  is not used as the control.
- Independently rerun ON matches supplied final ON losses, plan and digests.
- The supplementary tiny CPU test rerun PASS covers projection output/VJP on
  ordinary, zero, tiny and outlier rows; per-token scale placement; FP32 masters;
  three finite SGD steps; causal prefix invariance; changed-input sensitivity;
  deterministic and batch-independent evaluation; checkpoint toggle restoration;
  and missing-toggle OFF compatibility. Its forward oracle uses the production
  quantizer, so this is not independent proof of legacy hardware rounding.
- `audit.py` additionally checks the protected ASTs, exact source/harness bytes,
  raw receipt contents and distinct-coordinate native index mapping algebra.
  `independent-audit.json` records PASS.
- `check_benchmark.py` rerun PASS_SYNTHETIC_ONLY: injected 4% improvement passes;
  2% and an unadjusted 3% fail after polling uncertainty; host mismatch rejects.
  There is no measured device speedup.

Exact G2 rerun command, with `<arm>` empty for OFF and `--fp8-proj` for ON:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/v2/lanes/fp8-projections/test_support /home/poojith-devan/venv/bin/python workflow/tools/preflight_dryrun.py --train-py workflow/proposals/research-v2/train.py --stub-prepare --steps 3 --micro-rows 1 --micro-tokens 16 --nproc 4 --out research/v2/verification/20260925-l1-fp8-independent/g2-<off-or-on>-rerun-small.json -- --depth 2 --n-embd 256 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 <arm>
```

Supplementary rerun:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/v2/lanes/fp8-projections/test_support:workflow/tools/stub_prepare /home/poojith-devan/venv/bin/python research/v2/lanes/fp8-projections/check_cpu.py --train-py workflow/proposals/research-v2/train.py --out research/v2/verification/20260925-l1-fp8-independent/supplied-cpu-rerun.json
```

Independent static/receipt witness and benchmark synthetic rerun:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python research/v2/verification/20260925-l1-fp8-independent/audit.py
/home/poojith-devan/venv/bin/python research/v2/lanes/fp8-projections/check_benchmark.py
git diff --check
```

## Benchmark and remaining limits

The observer uses 80 complete production boundary intervals after 20 warmup
steps, with OFF/ON/ON/OFF on one host. Boundaries follow optimizer, cache refresh
and device synchronization; consecutive intervals also include control
reductions and subsequent data acquisition. Both matched pairs must beat 3%
after conservative endpoint polling uncertainty. It is a whole-step timing
criterion, not a projection-only microbenchmark.

The comparator does not independently bind argv, compiler environment, date,
exit status, checkpoint or native correctness. The lane README explicitly
requires those separate checks, so a standalone timing PASS is insufficient.
This is an evidence boundary, not a new source defect.

Still pending: exact 6x1024 four-rank G2 and exact OFF comparison; native kernel
compile and forward/backward numerical checks; native full-model causality,
batch independence and checkpoint load; same-host whole-step >=3% timing;
measured cold startup; and any fixed-budget quality experiment. The current
receipt clears only the bounded independent source/CPU review. It does not
add a launcher, change a breaker, or authorize result/submission claims.
