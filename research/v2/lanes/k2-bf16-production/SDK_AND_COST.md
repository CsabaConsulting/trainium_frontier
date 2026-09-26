# Production cost and SDK scope

Selected production: V8192/T32768/C1024, two FP32 tables WTE/VE7, BF16 output
tangents, world4/LNC2. Only tiny synthetic inputs were executed locally.

Per call, immutable int64 ID snapshot adds262,144bytes and a device copy;
it must not alias the reused loader buffer. Counts/starts/positions/maxima add
196,864bytes. Stable row/tile prefixes add8,388,608bytes of HBM scratch. The
straight producer performs268,435,456 row/token comparisons plus4,194,304
within-tile comparisons and16,384 row-group/tile bodies. These must use runtime
loops rather than being blindly unrolled into compile-time Python bodies.
Additional device validation, layout/cast and launch/dependency costs are not
yet implemented or quantified; estimates are not complete timing accounting.

Two backward tables read128MiB of useful BF16 tangent payload and store64MiB
of dense FP32 output. Each128x512 FP32 accumulator is256KiB; four obvious live
tiles total1MiB per owner, plus metadata/BF16 staging/compiler allocations.
No allocation/spill proof follows from capacity arithmetic.

Q=sum of64 group maxima ranges256..32768. Padded additions per table range
33,554,432..4,294,967,296. Without effective inactive skip, requested BF16
payload can reach8GiB per table. These are work bounds, not measured times or
claims about training-data frequencies. Metadata amortization is across two
tables only. Dense optimizer/all-reduce costs remain. No whole-step embedding
budget or defensible5% gain forecast is available.

## Official references inspected, 26 September2026

- [DMA copy, SDK2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html):
  supports HBM indirection with per-partition indices, including bounds errors.
  Vector and scalar offsets cannot be combined. An in-range wrong index is
  not a bounds error; skip mode cannot validate real IDs or semantic metadata.
- [Dynamic range, SDK2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.dynamic_range.html):
  runtime loop bounds can use virtual registers. This avoids T-sized unrolling.
- [Tensor reduction, SDK2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.tensor_reduce.html):
  free-axis arithmetic reductions compute in FP32. Counts bounded by32768
  are exactly representable. Gradient reductions remain prohibited here.
- [Partition reduction, SDK2.31](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.tensor_partition_reduce.html):
  describes partition reductions that could produce a shared group maximum.
- [NKI language guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/get-started/nki-language-guide.html):
  separates specialization-time Python from runtime operations, and describes
  register load/store. Python assertions are not proof of runtime ID validation.

These references make a producer plausible. They do **not** prove the installed
SDK accepts the intended full composition, that a validation dependency cannot
be removed/reordered, or that actual BF16 recurrence/underflow is preserved.
Protected tiny native evidence does not cover real512-channel owners or a
production metadata builder. No unsupported-SDK verdict is asserted.

The hf-cli skill was inspected because of its explicit AI/ML trigger. No
Hugging Face operation was relevant to this scoped local kernel task; no
download, external model/data, install or service was used.
