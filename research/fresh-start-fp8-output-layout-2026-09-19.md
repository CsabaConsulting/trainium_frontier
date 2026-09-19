# FP8 caller output-layout control

Design only,2026-09-19. Finish the compact-save and streamed-weight controls
before implementing another caller change. No speed, correctness or BPB result
exists for the designs below.

The source140 native/FP8 caller instruction data now gives an observable layout
lead. At ordinary Linear source line2853 on physical core0, native forward has
54,855 BF16 MATMUL records using128x128 array geometry/512 free elements. FP8
forward has16,384 records using128x32 and49,152 using32x128 with512 free elements,
plus43,671 ordinary full-size records. Regular shared-Linear MATMUL activity
unions rise14.688 ->45.144ms. The small-array work occurs in multiple regions,
so this is not evidence that a single vocabulary head is responsible. EVENT/
DMA notification loss and missing layer metadata remain explicit limitations.
See the opcode and operand JSON records named in RESUME.md. Do not interpret
these counts as a complete graph inventory or sum overlapping engine intervals.

One possible cause is memory pressure from the resident weight consumer; the
source147 streaming experiment tests that. Another is layout propagation across
the opaque NKI result and its reshape into the original batch/sequence rank.
Both the native/custom-backward MLP and FP8 MLP flatten inputs and reshape the
result, but only the latter imposes an opaque forward-output boundary. This is
a source-level distinction, not proof of how the compiler chooses its layout.
Earlier BF16 streaming/fixed-SBUF and separately compiled FP8-forward trials
lost in their exact configurations; do not repeat them without a new control.

The installed native wrapper was inspected read-only. Its Meta/Fake output
paths create torch.empty tensors from the NKI return shape/dtype. The local
snapshot of nki_hop.py has SHA256
`a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914`
and comes from
`/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_hop.py`.
This motivates testing a different explicit output shape, without modifying
the SDK or assuming a compiler-layout override exists.

A first narrow control could let NKI return the final result directly in the
original BxTxC shape, with the same token-row order and linear-address stores.
ReLU, raw activation bytes and metadata can retain their existing shapes. The
autograd interface must preserve the input-gradient shape and separate FP32
master gradients. Avoid immediately flattening the result again in the adapter,
which would remove the intended boundary change. Padding of short MLP inputs
must remain row independent; no attention or prediction-time state changes.

A distinct later control could store the result channel-major, paying explicit
transpose/consumer-layout costs. It may lose standalone and still improve the
full caller. Do not combine both output-shape controls with a new quantizer or
optimizer change, and do not assert a smaller buffer fixes the compiler.

For either control, keep the current independent quantizer/four-product gates,
exact retained values/gradients against the current FP8 algorithm, real32x1024
rows, changed masters/caches, accumulated microbatches, causality/row/padding
checks and native evaluation/checkpoint restore. Then run a same-host/source
whole-model comparison with all layout/refresh/dispatch costs included. Only
an observed whole-step improvement can justify profiling a claimed tiling
repair and subsequently a full-budget quality trial.
