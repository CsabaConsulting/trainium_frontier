# K2 resident-row consumer falsifier — disabled

This is a tiny separate experimental module, not a train.py candidate. The
Supervisor selected only V256/C128/T256 consumer feasibility. No host action,
model import, source adoption, device metadata producer, actual-data access,
routing speed claim or whole-step claim occurred. The rejected e4d93 source
remains REFUSE. Exactness is unchanged: same dtype/shape and identical bytes.

## Implemented boundary

The consumer takes supplied stable counts/row starts/token positions and two
group maximums. Those small synthetic fixtures are made on CPU and uploaded
before calls. **This is not a legal production routing implementation or a
claim that CPU routing is free.** A later train.py-only device producer remains
unimplemented and must charge all of its work.

Two fixed groups each own 128 rows. Each loads its count-bound into a device
register, retains a 128x128 FP32 accumulator in SBUF, and loops through each
row's occurrences in increasing token order. Each active lane indirectly
gathers a token position then its gradient, adds exactly once, and stores the
final row only after its loop. Unused rows are written as positive zero.
Inactive addresses use the explicit out-of-range sentinel 256 and both gathers
skip them. Inactive accumulator bits are selected from previous state, not
multiplied by zero. Address arithmetic stays within small exact integers.

The plain Python factory refuses by default before importing NKI. The native
checker also exits DISABLED before importing Torch/NKI unless explicitly enabled.
No production module imports these files. This gives a literal no-production-
change/default-disabled property, not a fabricated three-step full-model gate.

## Primary documentation and open implementation risks

AWS documents [runtime dynamic ranges](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.dynamic_range.html)
and [register-driven loops](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html#dynamic-control-flow).
The consumer uses a 1x1 int32 bound and a loop-carried accumulator. Only two
group bodies are statically expanded; counts 0/1/3/253/256 stay runtime data.
Installed SDK/compiler acceptance, scheduling and allocation are UNPROVEN.

[DMA copy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_copy.html)
documents per-partition SBUF index vectors, HBM row indirection and skipped
out-of-bounds transfers. This probe explicitly requests software descriptor
generation, not Trn3-only tensor indirection or duplicate scatter-add.
The intended stride units, skip behavior and two successive gathers need
actual device evidence; no archived older kernel is substituted for that proof.

[Tensor arithmetic](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html)
documents FP32 float arithmetic. [where](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.where.html)
is documented but experimental; exact inactive selection is therefore a live
falsifier, not a promise. [Scalar arithmetic](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_scalar.html)
is used only for bounded integer address/count bookkeeping and the active mask.
Native subnormal flushing, signed-zero behavior, reassociation, lost loop state,
unsupported helpers or spills must be exposed rather than hidden.

These public API checks were made September 25, 2026 UTC. They do not
authenticate the installed version. No SDK installation or compile was attempted.

## Local evidence and retained counterexamples

The stdlib test covers ten fixture kinds in both BF16-quantized and FP32 form:
the two retained cross-tile cancellation cases, unique IDs, a single hot row,
counts 253/1/1/1, boundary IDs, signed zero, BF16/FP32 subnormals, and empty
metadata. It validates routes/order/store counts, rejects malformed metadata
and IDs, checks absence of gradient reduction/hidden host reads in the kernel,
and tests both default-disabled entry points. Normal and -O receipts are retained.
Peak Python traced memory is below 8 MiB; Torch is not imported.

For the original values at token positions 0/128/129, ordered FP32 yields 0.
Rejected grouped accumulation yields 1 for [2**24, 1, -2**24], or 2**-24 for
[1, 2**-24, -1]. These differences remain test failures under the old arithmetic;
no tolerance, reference, or previous refusal was changed.
The CPU model is mathematical emulation, **not NKI simulation or actual dense
Torch backward parity**. The future checker verifies actual dense CPU backward
as another exact reference before interpreting native gradients.

## Resource bound, not a speed estimate

One named group accumulator/update/gather/select set totals 256 KiB. Named
small tiles add under 8 KiB. Even conservatively retaining both groups is under
528 KiB before compiler-created temporaries: not a spill/liveness proof.
Input payload is 64 KiB BF16 or 128 KiB FP32; dense output is 128 KiB; metadata
is 3,080 bytes. Logical accumulators have no intermediate HBM read/store.
The two groups need Q=2..256 rounds, or 32,768..4,194,304 padded additions.
A hot row still serializes 256 additions. Inactive DMA skips reduce useful
payload, not padded arithmetic or descriptor overhead. Compile cost, actual
runtime, full-model throughput and quality are UNKNOWN.

## Disabled native procedure

EXPERIMENT_PROPOSAL.disabled.json pins every support hash and exact proposed
four-rank workload argv. This is **not** a host-driver binding or direct launch
permission. Only the existing sole-Operator manual route may stage it after
independent review and separate assignment, using the original shared flock,
fresh in-lock ps AND neuron-ls checks, the current STOP/disk/hash checks,
fresh private caches and inherited lifetime FD. No launcher is added here.
Architecture/CT follow-up retains priority. No active host slot is reserved.

The checker compares direct/repeated consumer results, tiny native eager
autograd and compiled enclosing autograd, plus forward values, upstream
roundtrips and actual dense CPU gradients. Empty metadata tests only the
zero-trip direct consumer; empty autograd/production model are not claimed.
Every result must be byte-exact on all four ranks; errors or incomplete
coverage refuse the form. No eager-production fallback or threshold relaxation.

If all checks finish, optional timing is two warm calls plus three synchronized
consumer samples for unique and hot-row fixtures. These exclude metadata
construction/upload, and cannot establish a production or whole-step win.
Record raw timing, first-call wall, installed version, per-rank compile/graph
inventory and cold-cache proof separately. The proposed timeout is 600 seconds
plus 60 seconds TERM grace, not a predicted compile cost or new competition gate.
No automatic retry/full training/eval/quality-ledger row/adoption follows.

Night-mode decision at 20:50 UTC: isolate a reversible micro-falsifier before
device routing. Undo by declining this packet; no remote or production state
was changed. Supervisor owns the canonical decision log.
