"""Scratch AdamW fusion for contiguous FP32 parameters and moment states.

This does not change Muon or install an optimizer. Scalars are runtime inputs,
including each parameter's bias corrections. The functional form is the safe
default; explicit in-place behavior is tested separately before any integration.
"""
import torch


def scalar_values(lr, step, beta1, beta2, weight_decay, eps=1e-10):
    assert step >= 1 and 0 <= beta1 < 1 and 0 <= beta2 < 1
    assert eps == 1e-10, "only the model's existing epsilon is gated"
    lr32 = torch.tensor(lr, dtype=torch.float32)
    decay = 1.0 - lr32 * weight_decay
    return torch.tensor([lr, 1.0 - beta1 ** step, 1.0 - beta2 ** step,
                         1.0 - beta1, 1.0 - beta2, float(decay), eps], dtype=torch.float32)


def reference(p, g, m, v, scalars):
    lr, b1, b2, om1, om2, decay, eps = scalars[0].unbind()
    nm = torch.lerp(m, g, om1)
    nv = torch.lerp(v, g.square(), om2)
    update = (nm / b1) / ((nv / b2).sqrt() + eps)
    return p * decay - update * lr, nm, nv


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def adamw(p, g, m, v, scalars, lanes: int = 2, inplace: bool = False):
        rows, cols = p.shape
        assert p.dtype == nl.float32 and g.dtype == nl.float32
        assert m.dtype == nl.float32 and v.dtype == nl.float32
        assert p.shape == g.shape and p.shape == m.shape and p.shape == v.shape
        assert lanes in (1, 2) and rows % (128 * lanes) == 0 and cols <= 2048
        assert scalars.shape == (128, 7) and scalars.dtype == nl.float32
        if inplace:
            out_p, out_m, out_v = p, m, v
        else:
            out_p = nl.ndarray(p.shape, dtype=nl.float32, buffer=nl.shared_hbm)
            out_m = nl.ndarray(p.shape, dtype=nl.float32, buffer=nl.shared_hbm)
            out_v = nl.ndarray(p.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        meta = nl.ndarray((128, 7), dtype=nl.float32, buffer=nl.sbuf)
        ib1 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        ib2 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=meta, src=scalars)
        nisa.reciprocal(dst=ib1, data=meta[:, 1:2])
        nisa.reciprocal(dst=ib2, data=meta[:, 2:3])
        for i in nl.affine_range(rows // (128 * lanes)):
            start = (i * lanes + nl.program_id(0)) * 128
            ps = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            gs = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            vs = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            work = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            reciprocal = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ps, src=p[start:start + 128, :])
            nisa.dma_copy(dst=gs, src=g[start:start + 128, :])
            nisa.dma_copy(dst=ms, src=m[start:start + 128, :])
            nisa.dma_copy(dst=vs, src=v[start:start + 128, :])
            nisa.tensor_tensor(dst=work, data1=gs, data2=ms, op=nl.subtract)
            nisa.scalar_tensor_tensor(dst=ms, data=work, op0=nl.multiply,
                                      operand0=meta[:, 3:4], op1=nl.add, operand1=ms)
            nisa.tensor_tensor(dst=work, data1=gs, data2=gs, op=nl.multiply)
            nisa.tensor_tensor(dst=work, data1=work, data2=vs, op=nl.subtract)
            nisa.scalar_tensor_tensor(dst=vs, data=work, op0=nl.multiply,
                                      operand0=meta[:, 4:5], op1=nl.add, operand1=vs)
            # Store the updated FP32 states before reusing their on-chip tiles.
            nisa.dma_copy(dst=out_m[start:start + 128, :], src=ms)
            nisa.dma_copy(dst=out_v[start:start + 128, :], src=vs)
            nisa.tensor_scalar(dst=work, data=vs, op0=nl.multiply, operand0=ib2)
            # Avoid 0*inf at an untouched row. Below the smallest normal FP32
            # variance, sqrt(var) is <1.1e-19, below one ULP of the model's
            # 1e-10 Adam epsilon. The numerator remains the actual variance.
            nisa.tensor_scalar(dst=reciprocal, data=work, op0=nl.maximum,
                               operand0=1.1754943508222875e-38)
            nisa.tensor_scalar(dst=reciprocal, data=reciprocal,
                               op0=nl.rsqrt, operand0=0.0)
            nisa.tensor_tensor(dst=work, data1=work, data2=reciprocal, op=nl.multiply)
            nisa.tensor_scalar(dst=work, data=work, op0=nl.add, operand0=meta[:, 6:7])
            nisa.reciprocal(dst=reciprocal, data=work)
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=ib1)
            nisa.tensor_tensor(dst=ms, data1=ms, data2=reciprocal, op=nl.multiply)
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=meta[:, 0:1])
            nisa.tensor_scalar(dst=ps, data=ps, op0=nl.multiply, operand0=meta[:, 5:6])
            nisa.tensor_tensor(dst=ps, data1=ps, data2=ms, op=nl.subtract)
            nisa.dma_copy(dst=out_p[start:start + 128, :], src=ps)
        return out_p, out_m, out_v

    return adamw
