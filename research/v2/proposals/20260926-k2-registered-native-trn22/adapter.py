"""Disabled diagnostic registration only; no production import or activation.

The caller supplies the unchanged consumer_nki.build_kernel(enabled=True)
result. Tests may supply a tiny CPU callable and the hash-pinned, AST-extracted
installed nki_op. Neither substitution is native evidence.
"""


def build_registered_forward(kernel, output_dtype, *, enabled=False, nki_op_factory=None):
    if not enabled:
        raise RuntimeError('EXPERIMENT_DISABLED: no Torch/NKI module imported')
    import torch
    import torch.nn.functional as F
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Only the original FP32/BF16 output contracts are supported')
    if nki_op_factory is None:
        from torch_neuronx import nki_op as nki_op_factory
    suffix = 'fp32' if output_dtype == torch.float32 else 'bf16'

    @nki_op_factory('trainium_k2_registered_v1::embedding_' + suffix, mutates_args=())
    def forward(weight: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor,
                starts: torch.Tensor, positions: torch.Tensor,
                maximums: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, weight).to(output_dtype)

    def setup_context(ctx, inputs, output):
        _, _, counts, starts, positions, maximums = inputs
        ctx.save_for_backward(counts, starts, positions, maximums)

    def backward(ctx, gradient):
        counts, starts, positions, maximums = ctx.saved_tensors
        result = kernel(gradient.contiguous(), counts, starts, positions, maximums)
        return result, None, None, None, None, None

    forward.register_autograd(backward, setup_context=setup_context)
    return forward


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(dict(status='EXPERIMENT_DISABLED', launch_authorized=False,
                          torch_imported='torch' in sys.modules,
                          nki_imported='nki' in sys.modules)))
