# K2 final sprint: complete default-OFF candidate

Implementer delivery, not independent approval or native qualification.
Base: immutable audited `workflow/candidates/confirmed-clock36-116188/train.py`
(e1e2f686; official 0.9878 reported by the human). Candidate changes only the
ordinary WTE and VE7 backward when `--k2-embedding` is explicitly enabled.
The original base and earlier numerical failure artifacts are untouched.

## Actual implementation

`train.py` adds a default-false flag and a training/grad-enabled branch. It
imports K2 only when enabled, rejects non-Neuron or non-pack32/context1024/
width1024 launches, and constructs metadata once per forward shared by WTE/VE7.
Parameter objects, initialization, optimizer, collectives, checkpoint format,
training clocks and ordinary evaluation remain unchanged. No data/evaluator edit.

`k2.py` clones int64 IDs on their existing device and uses actual
`torch.argsort(stable=True)` plus left/right `torch.searchsorted` to produce
counts, starts and a strictly token-ordered position permutation. Group maxima
are device reductions of integer counts. All tensors remain on device; there
is no `.cpu()`, Python ID extraction, host routing or global metadata cache.
The snapshot also feeds forward gather, so loader-storage reuse cannot change
the association between forward and saved backward metadata. Each forward and
accumulation microstep gets fresh tensors. No composite row*T keys are used.
Only bounded positions/counts/starts are narrowed to int32; token IDs stay int64.
As in ordinary embedding, valid in-vocabulary IDs are the supported domain.

The registered op keeps `F.embedding(ids, weight).to(bfloat16)` forward. Actual
BF16 output tangents enter the NKI backward; FP32 tangents are rejected, never
silently narrowed. The saved metadata is immutable and internal to this call.
The consumer is the previous real-shape ordered consumer, copied without
algorithm changes: two 512-channel owners, 64 groups of 128 rows, runtime
occurrence loops, sequential FP32 addition, inactive-lane selection. No
duplicate scatter writes, atomics, regrouped sums or tree gradient reductions.
Higher-order differentiation is not qualified; ordinary first-order training
is the intended use.

## Tests and concrete remaining uncertainty

Run the self-contained patch test with the existing laptop venv:

```
/home/poojith-devan/venv/bin/python -B test_k2.py --base /home/poojith-devan/Desktop/Trainium/workflow/candidates/confirmed-clock36-116188/train.py
```

`cpu-result.json` records 16 actual registered-autograd changed-op cases versus
plain Torch embedding/cast backward. The producer is executed directly; the
actual NKI body is CPU-interpreted with only geometry constants changed. It is
not an SDK simulator. Twelve finite fixtures are raw-bit exact; four NaN/Inf/
overflow fixtures remain explicitly FAIL_NONFINITE_DIAGNOSTIC, never training
passes. Both BF16 subnormal signs, normal-to-subnormal cancellation, the
2^24/1/-2^24 and 1/2^-24/-1 ordering witnesses, and zero signs are retained.
Additional checks cover full production metadata shapes, storage reuse, two
tables/two microsteps, unused rows, interleaved duplicates and all-one-row skew.

The eight-layer width64/context8 model passes three OFF steps versus exact base
and three ON steps using the same interpreted consumer; losses, every gradient
and every updated parameter match by bytes. Optimizer is test SGD .01. There
is one metadata generation per ON forward; eval follows ordinary embedding.
The laptop lacks pyarrow, so unused parquet import and existing test-only
prepare import are disclosed shims. This does not claim full-size optimizer,
full training-loop, installed NKI or native behavior.

Native unknowns are specific: installed backend lowering for stable argsort
and searchsorted; registered-autograd capture of the consumer; 512-channel
indirect DMA legality/performance; runtime-loop allocation/liveness; actual
BF16 widening and FP32 subnormal/rounding semantics. No claim of an unsupported
SDK has been made, since no native compile was attempted. The enabled path
does not catch and hide unsupported-operator or compile errors.

## Operator timing use

Stage `train.py`, `k2.py`, `consumer_nki.py` together. Existing official prepare
must be importable as usual. Use the ordinary checked full-chip four-rank/LNC2
launch machinery and one unique proposal/label per arm; Implementer does not
launch or hold a host. Compare the exact same candidate with `--no-k2-embedding`
and `--k2-embedding`, same seed42, pack32/context1024/width1024 and batch131072.
A short 45-step run (`--num-steps 45 --no-eval-public`) covers five startup,
five settling and 35 steady complete optimizer steps. Preserve normal budget
charging and report cold fifth boundary plus per-step tok/s. Do not use a
consumer-only measurement as the required whole-step result: the enabled
training source includes clone, sort, searches, casts, both lookups, both
backwards, optimizer, collectives and cache refresh. Stop/report a native
compile or numerical failure; no fallback is installed. Timing by16:00IST is
still pending. Full run requires the current human >=3% whole-step speed bar.

## Useful cost estimate (not measured timing)

For T32768/V8192/C1024, stable sort orders 32768 int64 keys (roughly T log2 T =
491520 comparisons for a comparison-sort model; backend algorithm unknown).
The two binary searches do roughly 245760 probes. Snapshot is256KiB; returned
count/start/position/max metadata is196864B; int64 sorted positions and IDs add
256KiB each while live, excluding opaque sort workspace. This replaces the
old 8MiB prefix producer and its268M comparisons; it is not a measured win.

Two backwards read128MiB useful BF16 tangents and write64MiB FP32 gradients.
With Q=sum of64 group maximum counts, each table performs128*1024*Q padded
FP32 additions; Q ranges256..32768, hence33.6M..4.29B adds/table. Strong skew
can make this slow. Four obvious128x512 FP32 tiles use1MiB/owner, excluding
compiler staging. No SBUF/spill guarantee follows from this estimate.
On an illustrative0.50s control step, 3% faster token rate permits at most
0.48544s/candidate step, requiring a net saving of14.56ms after all producer
costs. Neither cold startup nor a net saving is known before native timing.

Primary semantics: PyTorch argsort documents stable order for equal values:
https://docs.pytorch.org/docs/stable/generated/torch.argsort.html
and searchsorted documents left/right insertion boundaries:
https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html
Native support is not implied by these CPU/API semantics. Consumer NKI sources
are the SDK2.31 dma_copy, dynamic_range and tensor_tensor references in the
previous consumer specification. No new package was installed.
