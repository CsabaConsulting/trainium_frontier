"""Device-only stable routing, shared by WTE and VE7 for one forward call.

No host IDs, cached metadata, gradient regrouping, or low precision accumulation.
CPU execution of prepare_metadata is useful for tests; the enabled train.py path
requires Neuron. Native argsort(stable=True)/searchsorted support is unmeasured.
"""
import torch
import torch.nn.functional as F


def prepare_metadata(ids, vocab=8192, group=128):
    if ids.dtype != torch.int64 or ids.ndim != 2:
        raise ValueError('K2 requires two-dimensional int64 token IDs')
    if vocab % group or ids.numel() >= 2**24:
        raise ValueError('K2 metadata geometry is unsupported')
    # This snapshot also supplies the gather, so all consumers have the same
    # values even when the loader reuses its input storage before backward.
    snapshot = ids.clone()
    flat = snapshot.reshape(-1)
    positions64 = torch.argsort(flat, stable=True)
    ordered_ids = flat[positions64]
    rows = torch.arange(vocab, dtype=ids.dtype, device=ids.device)
    starts64 = torch.searchsorted(ordered_ids, rows, right=False)
    ends64 = torch.searchsorted(ordered_ids, rows, right=True)
    counts = (ends64 - starts64).to(torch.int32).reshape(vocab, 1)
    starts = starts64.to(torch.int32).reshape(vocab, 1)
    positions = positions64.to(torch.int32).reshape(-1, 1)
    maximums = counts.reshape(-1, group).amax(dim=1).reshape(-1, 1)
    return snapshot, counts, starts, positions, maximums


def build_registered_lookup(*, kernel=None, nki_op_factory=None):
    # Optional explicit injections exist solely for CPU tests. Production calls
    # this with no arguments, and never substitutes a CPU backward.
    if kernel is None:
        from consumer_nki import build_kernel
        kernel = build_kernel(enabled=True)
    if nki_op_factory is None:
        from torch_neuronx import nki_op as nki_op_factory

    @nki_op_factory('trainium_k2_final_sprint::embedding_bf16', mutates_args=())
    def lookup(weight: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor,
               starts: torch.Tensor, positions: torch.Tensor,
               maximums: torch.Tensor) -> torch.Tensor:
        # Keep the ordinary gather and exact existing output cast.
        return F.embedding(ids, weight).to(torch.bfloat16)

    def setup_context(ctx, inputs, output):
        weight, ids, counts, starts, positions, maximums = inputs
        if weight.dtype != torch.float32 or output.dtype != torch.bfloat16:
            raise ValueError('K2 requires FP32 weights and BF16 output')
        ctx.save_for_backward(counts, starts, positions, maximums)

    def backward(ctx, gradient):
        if gradient.dtype != torch.bfloat16:
            raise ValueError('K2 refuses a non-BF16 output tangent')
        counts, starts, positions, maximums = ctx.saved_tensors
        dense = kernel(gradient.reshape(-1, gradient.shape[-1]).contiguous(),
                       counts, starts, positions, maximums)
        return dense, None, None, None, None, None

    lookup.register_autograd(backward, setup_context=setup_context)
    return lookup
