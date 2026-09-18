"""Two-tap depthwise causal convolution with explicit contiguous NKI tiles.

Experimental, not enabled in train.py. Supports [B,T,C], T/C multiples of 128,
FP32 or BF16 activations, and the existing [C,1,2] parameter. Two programs split
batch rows when B is even. No tile crosses a sequence boundary. Weight-gradient
partials reduce 128 time rows on the TensorEngine, then a small Torch reduction
combines tiles; that reduction and the weight layout conversion are real costs.
"""
import torch
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx import wrap_nki


def _taps(wt, channels):
    prev = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
    curr = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
    pattern = [[0, 128], [1, channels]]
    nisa.dma_copy(dst=prev, src=wt.ap(pattern=pattern, offset=0))
    nisa.dma_copy(dst=curr, src=wt.ap(pattern=pattern, offset=channels))
    return prev, curr


def _fwd_tile(x, out, prev_w, curr_w, batch, start, channels, first):
    prev = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    curr = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=curr, src=x[batch, start:start + 128, :])
    if first:
        nisa.memset(dst=prev[0:1, :], value=0.0)
        nisa.dma_copy(dst=prev[1:128, :], src=x[batch, 0:127, :])
    else:
        nisa.dma_copy(dst=prev, src=x[batch, start - 1:start + 127, :])
    acc = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=acc, data1=prev, data2=prev_w, op=nl.multiply)
    nisa.tensor_tensor(dst=prod, data1=curr, data2=curr_w, op=nl.multiply)
    nisa.tensor_tensor(dst=result, data1=acc, data2=prod, op=nl.add)
    nisa.dma_copy(dst=out[batch, start:start + 128, :], src=result)


@nki.jit
def conv2_forward(x, wt, lanes: int = 1):
    batch, length, channels = x.shape
    assert length % 128 == 0 and channels % 128 == 0
    assert wt.shape == (2, channels)
    assert lanes in (1, 2) and batch % lanes == 0
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    prev_w, curr_w = _taps(wt, channels)
    lane = nl.program_id(0)
    for b in nl.affine_range(batch // lanes):
        bi = b * lanes + lane
        _fwd_tile(x, out, prev_w, curr_w, bi, 0, channels, True)
        if length > 128:
            for t in nl.affine_range(length // 128 - 1):
                _fwd_tile(x, out, prev_w, curr_w, bi, (t + 1) * 128, channels, False)
    return out


def _dw_partial(xpart, grad, ones, partials, batch, tile, tap, channels):
    prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=xpart, data2=grad, op=nl.multiply)
    for c in nl.affine_range(channels // 128):
        summed = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        result = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_matmul(dst=summed, stationary=prod[:, c * 128:(c + 1) * 128],
                       moving=ones, accumulate=False)
        nisa.tensor_copy(dst=result, src=summed)
        nisa.dma_copy(dst=partials[batch, tile, tap, c * 128:(c + 1) * 128, :], src=result)


def _bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
              batch, tile, length, channels, first, last):
    start = tile * 128
    xp = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    xc = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    gc = nl.ndarray((128, channels), dtype=grad.dtype, buffer=nl.sbuf)
    gn = nl.ndarray((128, channels), dtype=grad.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xc, src=x[batch, start:start + 128, :])
    nisa.dma_copy(dst=gc, src=grad[batch, start:start + 128, :])
    if first:
        nisa.memset(dst=xp[0:1, :], value=0.0)
        nisa.dma_copy(dst=xp[1:128, :], src=x[batch, 0:127, :])
    else:
        nisa.dma_copy(dst=xp, src=x[batch, start - 1:start + 127, :])
    if last:
        nisa.dma_copy(dst=gn[0:127, :], src=grad[batch, start + 1:length, :])
        nisa.memset(dst=gn[127:128, :], value=0.0)
    else:
        nisa.dma_copy(dst=gn, src=grad[batch, start + 1:start + 129, :])
    acc = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=acc, data1=gc, data2=curr_w, op=nl.multiply)
    nisa.tensor_tensor(dst=prod, data1=gn, data2=prev_w, op=nl.multiply)
    nisa.tensor_tensor(dst=result, data1=acc, data2=prod, op=nl.add)
    nisa.dma_copy(dst=dx[batch, start:start + 128, :], src=result)
    _dw_partial(xp, gc, ones, partials, batch, tile, 0, channels)
    _dw_partial(xc, gc, ones, partials, batch, tile, 1, channels)


@nki.jit
def conv2_backward(x, grad, wt, lanes: int = 1):
    batch, length, channels = x.shape
    assert length % 128 == 0 and channels % 128 == 0
    assert grad.shape == x.shape and wt.shape == (2, channels)
    assert lanes in (1, 2) and batch % lanes == 0
    tiles = length // 128
    dx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    partials = nl.ndarray((batch, tiles, 2, channels, 1), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    prev_w, curr_w = _taps(wt, channels)
    ones = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    lane = nl.program_id(0)
    for b in nl.affine_range(batch // lanes):
        bi = b * lanes + lane
        if tiles == 1:
            _bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                      bi, 0, length, channels, True, True)
        else:
            _bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                      bi, 0, length, channels, True, False)
            if tiles > 2:
                for t in nl.affine_range(tiles - 2):
                    _bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                              bi, t + 1, length, channels, False, False)
            _bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                      bi, tiles - 1, length, channels, False, True)
    return dx, partials


# Resolve the native higher-order operator before Dynamo enters autograd.forward.
# Constructing a raw nki.Kernel grid inside the traced function hits the installed
# NKI 0.5 / Torch 2.11 dataclass-hash path before that kernel has its func field.
# wrap_nki is the installed backend's explicit, graph-capturable interface.
_forward = wrap_nki(conv2_forward)
_backward = wrap_nki(conv2_backward)


class Conv2NKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        assert weight.shape == (x.shape[-1], 1, 2)
        lanes = 2 if x.shape[0] % 2 == 0 else 1
        wt = weight[:, 0, :].T.contiguous().to(x.dtype)
        xx = x.contiguous()
        y = _forward[lanes](xx, wt, lanes)
        ctx.save_for_backward(xx, wt)
        ctx.lanes = lanes
        ctx.weight_dtype = weight.dtype
        return y

    @staticmethod
    def backward(ctx, grad):
        x, wt = ctx.saved_tensors
        dx, partials = _backward[ctx.lanes](x, grad.contiguous(), wt, ctx.lanes)
        dw = partials.sum(dim=(0, 1)).squeeze(-1).to(wt.dtype)
        return dx, dw.T.unsqueeze(1).to(ctx.weight_dtype)


def nki_causal_conv(self, x):
    assert self.kernel_size == 2
    return Conv2NKI.apply(x, self.weight)
