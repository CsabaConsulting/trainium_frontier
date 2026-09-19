"""Experimental cached-weight head/CE with the original BF16 gradient boundary.

No model installation or persistent cache. The BF16 weight cast retains the
original Linear master's autograd connection. A real transpose is charged on
every forward. This preserves compiled-native FP32 projection accumulation;
it does not claim equality to eager BF16-rounded Linear logits.
"""


def loss_factory(save_softcap=False, fp32_carrier=False):
    import torch
    from torch_neuronx import wrap_nki
    from head_loss_fused_nki import build_kernel
    from softcap_ce_parallel_nki import build_kernels

    forward = wrap_nki(build_kernel())
    _, backward = (wrap_nki(kernel) for kernel in build_kernels())

    class CachedHeadLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, target):
            activation = x.bfloat16() if fp32_carrier else x
            z, loss, lse = forward[2](activation, weight.T.contiguous(), target,
                                     2, True, True, save_softcap)
            ctx.save_for_backward(activation, weight, z, target, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, gradient):
            x, weight, z, target, lse = ctx.saved_tensors
            dz = backward[2](z, target, lse,
                             gradient.reshape(-1, 1).float().contiguous(), 2, True, save_softcap)
            # Original BF16 Linear boundary; the weight-cache cast returns
            # this gradient to its FP32 master through normal autograd.
            # A split compiled graph otherwise materializes a BF16 input
            # gradient even when the monolithic compiler retains its FP32
            # accumulator into the normalization backward. The optional FP32
            # interface carries exact BF16 forward values, and asks for an FP32
            # gradient buffer from BF16 matmul operands. This needs separate
            # component and unchanged whole-model numerical gates.
            dx = dz.float() @ weight.float() if fp32_carrier else dz @ weight
            return dx, dz.T @ x, None

    def loss(x, weight, target, reduction="mean"):
        assert x.ndim == weight.ndim == 2 and weight.dtype == torch.bfloat16
        assert x.dtype == (torch.float32 if fp32_carrier else torch.bfloat16)
        assert reduction in ("none", "mean")
        value = CachedHeadLoss.apply(x, weight, target.reshape(-1, 1).float())
        return value.mean() if reduction == "mean" else value.view_as(target)

    return loss
