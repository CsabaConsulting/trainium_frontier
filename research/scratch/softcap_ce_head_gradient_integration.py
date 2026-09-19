"""Keep original FP32 logits and compress only the BF16 head's CE derivative.

This is a restricted consumer contract, not a generic FP32-logit loss gradient:
the model's final normalized input and cached head weights must be BF16, and
no output-bigram branch may consume the unrounded FP32 derivative. GPT.forward,
prediction, loss arithmetic, parameters and checkpoint format are unchanged.
"""
import torch


def install(train_module):
    if getattr(train_module, '_fresh_start_ce_head_gradient_installed', False):
        raise RuntimeError('Head-gradient CE adapter is already installed')
    if (not train_module.USE_BF16_NORM_OUTPUT or train_module.USE_OUT_BIGRAM
            or train_module.LOSS_BYTE_WEIGHT or train_module.LOGIT_SOFTCAP != 15.0):
        raise ValueError('Head-gradient CE requires BF16 final norm, no output bigram, unweighted softcap15')
    if getattr(train_module, '_fresh_start_bf16_ce_installed', False):
        raise ValueError('Do not combine the failed BF16 forward boundary with this separate candidate')
    if not getattr(train_module, '_fresh_start_adaptive_ce_installed', False):
        from softcap_ce_parallel_integration import install as install_adaptive
        install_adaptive(train_module)
    from softcap_ce_parallel_integration import selected_lanes
    from softcap_ce_parallel_nki import build_kernels
    from torch_neuronx import wrap_nki
    _, backward = (wrap_nki(kernel) for kernel in build_kernels())
    original_loss = train_module.softcap_cross_entropy
    forward = train_module._softcap_ce_fwd

    class HeadLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target):
            loss, lse = forward(logits, target)
            ctx.save_for_backward(logits, target, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, gradient):
            logits, target, lse = ctx.saved_tensors
            lanes = selected_lanes(logits.shape[0])
            upstream = gradient.reshape(-1, 1).float().contiguous()
            derivative = backward[lanes](logits, target, lse, upstream, lanes, True)
            return derivative, None

    def loss(logits, targets, reduction):
        rows = logits.numel() // logits.shape[-1]
        if (logits.device.type == 'neuron' and logits.dtype == torch.float32
                and logits.is_contiguous() and rows % 128 == 0
                and logits.shape[-1] % 512 == 0 and reduction in ('mean', 'none')):
            value = HeadLoss.apply(logits.reshape(rows, logits.shape[-1]),
                                   targets.reshape(-1, 1).float())
            return value.mean() if reduction == 'mean' else value.view_as(targets)
        return original_loss(logits, targets, reduction)

    train_module.softcap_cross_entropy = loss
    train_module._fresh_start_ce_head_gradient_installed = True
    return dict(original_loss=original_loss,
                source_edit='none; original FP32 head logits and GPT forward retained',
                contract='BF16 head consumer gradient; not arbitrary FP32-logit leaf equivalence')
