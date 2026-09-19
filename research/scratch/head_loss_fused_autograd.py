"""Experimental cached-weight head/CE with the original BF16 gradient boundary.

No model installation or persistent cache. The BF16 weight cast retains the
original Linear master's autograd connection. A real transpose is charged on
every forward. This preserves compiled-native FP32 projection accumulation;
it does not claim equality to eager BF16-rounded Linear logits.
"""


def loss_factory():
    import torch
    from torch_neuronx import wrap_nki
    from head_loss_fused_nki import build_kernel
    from softcap_ce_parallel_nki import build_kernels

    forward = wrap_nki(build_kernel())
    _, backward = (wrap_nki(kernel) for kernel in build_kernels())

    class CachedHeadLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, target):
            z, loss, lse = forward[2](x, weight.T.contiguous(), target, 2, True, True)
            ctx.save_for_backward(x, weight, z, target, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, gradient):
            x, weight, z, target, lse = ctx.saved_tensors
            dz = backward[2](z, target, lse,
                             gradient.reshape(-1, 1).float().contiguous(), 2, True)
            # Original BF16 Linear boundary; the weight-cache cast returns
            # this gradient to its FP32 master through normal autograd.
            return dz @ weight, dz.T @ x, None

    def loss(x, weight, target, reduction="mean"):
        assert x.ndim == weight.ndim == 2 and x.dtype == weight.dtype == torch.bfloat16
        assert reduction in ("none", "mean")
        value = CachedHeadLoss.apply(x, weight, target.reshape(-1, 1).float())
        return value.mean() if reduction == "mean" else value.view_as(target)

    return loss
