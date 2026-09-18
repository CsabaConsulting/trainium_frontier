"""Install the gated CE scheduling change without editing the protected trainer.

Only independent row scheduling and the target gather change. Softcap, exp/log,
the reduction tree, saved logsumexp and the original autograd interface remain.
The single-iteration two-core forward discrepancy is avoided by shape dispatch.
"""


def selected_lanes(rows):
    if rows <= 0 or rows % 128:
        raise ValueError("the NKI CE adapter requires positive multiples of128 rows")
    return 2 if rows >= 512 and rows % 256 == 0 else 1


def install(train_module):
    """Replace the two primitive calls used by the existing CE autograd class.

    Returns the original pair so an isolated diagnostic can restore it. Model
    parameters, loss reductions, scoring and checkpoint formats are unchanged.
    Installation belongs before compilation, once per process.
    """
    if not train_module._HAS_NKI or train_module.LOGIT_SOFTCAP != 15.0:
        raise ValueError("adaptive CE is gated only for NKI with softcap15")
    if not train_module.USE_NKI_SOFTCAP_CE:
        raise ValueError("adaptive CE requires the existing NKI loss path")
    if getattr(train_module, "_fresh_start_adaptive_ce_installed", False):
        raise RuntimeError("adaptive CE is already installed in this process")
    from torch_neuronx import wrap_nki
    from softcap_ce_parallel_nki import build_kernels

    forward, backward = (wrap_nki(kernel) for kernel in build_kernels())
    originals = train_module._softcap_ce_fwd, train_module._softcap_ce_bwd

    def adaptive_forward(z, target):
        lanes = selected_lanes(z.shape[0])
        return forward[lanes](z, target, lanes, True)

    def adaptive_backward(z, target, lse, gradient):
        lanes = selected_lanes(z.shape[0])
        return backward[lanes](z, target, lse, gradient, lanes)

    train_module._softcap_ce_fwd = adaptive_forward
    train_module._softcap_ce_bwd = adaptive_backward
    train_module._fresh_start_adaptive_ce_installed = True
    return originals
