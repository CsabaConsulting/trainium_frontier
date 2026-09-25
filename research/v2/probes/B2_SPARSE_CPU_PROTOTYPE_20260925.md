# B2 sparse-table CPU prototype — native HOLD

This branch carries a **default-off, CPU-only** `--b2-sparse-ngram` prototype in
`submissions/research_v2/train.py`. It is an implementation exercise, not a
scored experiment or a claim of speed/BPB improvement. The normal launch has
the flag off. When the flag is on, `main()` refuses non-CPU devices before
runtime initialization, so this file is **not** a Trainium B2 candidate.

## What the local code demonstrates

- `nn.Embedding(..., sparse=True)` produces a row-sparse CPU gradient.
- Each rank coalesces local rows; a fixed-capacity all-gather exchanges row
  indices and values; duplicate rows are summed in FP32 and divided by world
  size. A rank with zero touched rows still takes the same collective path.
- The optimizer writes only touched parameter/moment rows. An FP32 moment's
  `beta2 ** gap` catch-up matches the logical dense zero-gradient recurrence
  on a small CPU test. It deliberately does **not** claim bit identity to the
  dense BF16-moment optimizer.
- Invalid gradients fail before the tested parameter/step state changes, and
  a rank-local NaN causes all four CPU ranks to fail together before row
  all-gathers.

Local checks passed using the workstation's preinstalled PyTorch 2.6 CPU/Gloo
capability (no package installation or host contact):

```
python -m py_compile submissions/research_v2/train.py research/v2/probes/test_b2_sparse_cpu.py
python research/v2/probes/test_b2_sparse_cpu.py
torchrun --standalone --nproc_per_node=4 research/v2/probes/test_b2_sparse_cpu.py
git diff --check
```

The local test stubs the unused `prepare`/`pyarrow` imports and does not load
competition data or a checkpoint. It tests small FP32 and BF16 tables; it is
not a full-model G2 or a throughput test.

## Why native dispatch remains HOLD

The earlier B2 candidate `b2e61508` was rejected before hardware, documented
in `research/agentic/TRACK_B2_SPARSE_CANDIDATE_REJECTION_2026-09-24.md`.
The CPU tests here do not settle any of its native issues: sparse COO backward
on Neuron; dynamic `coalesce`/boolean compaction; fixed-row collective lowering;
indexed BF16 writes; native memory/startup; and a bounded exact BF16 lazy
recurrence. The full-table FP32 moment is still allocated, so even a native
port would save gradient synchronization/update traffic, **not** moment
storage. The fixed exchange's capacity is the entire local token count and
could be slower than dense all-reduce. Those are measured questions.

Also, `--b2-sparse-ngram` currently requires `--ngram-rms-fp32`. That changes
the table optimizer's state precision relative to a BF16-moment A1/A2 arm.
A future scientific comparison must use a **same-host FP32-moment dense table
control**, not attribute a combined precision/sparsity delta to B2.

Before a native run, a distinct Verifier must review these source bytes and
the exact per-rank outputs; an authorized Operator must use a unique
non-scored native component-probe label for the four unresolved primitives;
then a separate G2/proposal/control/full-run gate is still required. No
protected file, evaluator, data, active job, or AWS resource was touched here.
