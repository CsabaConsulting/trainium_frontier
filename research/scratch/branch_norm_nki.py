"""Scratch FP32-statistic RMS normalization with BF16 branch outputs.

The precision policy and epsilon match train.USE_BF16_NORM_OUTPUT. A row is
one token; no statistic mixes tokens or batches. The backward returns FP32
input gradients. Evaluation and unsupported shapes keep the native function.
"""
import torch
import torch.nn.functional as F

EPS = 1.1920928955078125e-7


def reference(x):
    return F.rms_norm(x.float(), (x.shape[-1],), eps=EPS).bfloat16()


def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def forward(x, lanes: int = 2):
        rows, channels = x.shape
        assert x.dtype == nl.float32 and channels % 128 == 0
        assert lanes in (1, 2) and rows % (128 * lanes) == 0
        y = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        inverse = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        for i in nl.affine_range(rows // (128 * lanes)):
            start = (i * lanes + nl.program_id(0)) * 128
            values = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            squared = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            summed = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            mean_eps = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            inv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            output = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=values, src=x[start:start + 128, :])
            nisa.tensor_tensor(dst=squared, data1=values, data2=values, op=nl.multiply)
            nisa.tensor_reduce(dst=summed, op=nl.add, data=squared, axis=(1,))
            nisa.tensor_scalar(dst=mean_eps, data=summed, op0=nl.multiply,
                               operand0=1.0 / channels, op1=nl.add, operand1=EPS)
            nisa.activation(dst=inv, op=nl.rsqrt, data=mean_eps)
            nisa.tensor_scalar(dst=output, data=values, op0=nl.multiply, operand0=inv)
            nisa.dma_copy(dst=y[start:start + 128, :], src=output)
            nisa.dma_copy(dst=inverse[start:start + 128, :], src=inv)
        return y, inverse

    @nki.jit
    def backward(x, gradient, inverse, lanes: int = 2):
        rows, channels = x.shape
        assert x.dtype == nl.float32 and gradient.dtype == nl.bfloat16
        assert gradient.shape == x.shape and inverse.shape == (rows, 1)
        assert lanes in (1, 2) and rows % (128 * lanes) == 0
        dx = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        for i in nl.affine_range(rows // (128 * lanes)):
            start = (i * lanes + nl.program_id(0)) * 128
            values = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            gb = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            grad = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            inv = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            products = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            dot = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            r2 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            correction = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            centered = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            result = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=values, src=x[start:start + 128, :])
            nisa.dma_copy(dst=gb, src=gradient[start:start + 128, :])
            nisa.tensor_copy(dst=grad, src=gb)
            nisa.dma_copy(dst=inv, src=inverse[start:start + 128, :])
            nisa.tensor_tensor(dst=products, data1=grad, data2=values, op=nl.multiply)
            nisa.tensor_reduce(dst=dot, op=nl.add, data=products, axis=(1,))
            nisa.tensor_tensor(dst=r2, data1=inv, data2=inv, op=nl.multiply)
            nisa.tensor_scalar(dst=correction, data=dot, op0=nl.multiply, operand0=1.0 / channels)
            nisa.tensor_tensor(dst=correction, data1=correction, data2=r2, op=nl.multiply)
            nisa.tensor_scalar(dst=centered, data=values, op0=nl.multiply, operand0=correction)
            nisa.tensor_tensor(dst=centered, data1=grad, data2=centered, op=nl.subtract)
            nisa.tensor_scalar(dst=result, data=centered, op0=nl.multiply, operand0=inv)
            nisa.dma_copy(dst=dx[start:start + 128, :], src=result)
        return dx

    return forward, backward


def make_function():
    from torch_neuronx import wrap_nki
    forward, backward = (wrap_nki(kernel) for kernel in build_kernels())

    class BranchNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            y, inv = forward[2](flat, 2)
            ctx.save_for_backward(flat, inv)
            ctx.shape = shape
            return y.reshape(shape)

        @staticmethod
        def backward(ctx, grad):
            values, inv = ctx.saved_tensors
            gradient = grad.reshape(values.shape).contiguous()
            return backward[2](values, gradient, inv, 2).reshape(ctx.shape)

    return BranchNorm.apply


def install(train_module):
    assert train_module.USE_BF16_NORM_OUTPUT
    native = train_module.norm
    apply = make_function()

    def norm(x):
        if (x.device.type == "neuron" and torch.is_grad_enabled()
                and x.dtype == torch.float32 and x.shape[-1] % 128 == 0
                and x.numel() // x.shape[-1] % 256 == 0):
            return apply(x)
        return native(x)

    train_module.norm = norm
