"""Causal pair selection/expansion with bounded, contiguous-width NKI DMA tiles."""
import torch
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    from torch_neuronx import wrap_nki as _pair_wrap_nki
    _PAIR_HAS_NKI = True
except ImportError:
    _PAIR_HAS_NKI = False


def _pair_kernel_assert(condition, text):
    assert condition, '[NCC_INKI016] Kernel validation exception: ' + text


def _pair_div_ceil(n, d):
    return (n + d - 1) // d


if _PAIR_HAS_NKI:
    @nki.jit
    def _pair_select_fwd(x, parity: int, lanes: int):
        """Select x[:, parity::2]. Args: contiguous [B,T,C]. Returns [B,T/2,C]."""
        batch, length, width = x.shape
        half = length // 2
        _pair_kernel_assert(length % 256 == 0 and batch % lanes == 0, 'aligned T and batch required')
        _pair_kernel_assert(width <= nl.tile_size.sbuf_fmax, 'channel tile exceeds SBUF')
        output = nl.ndarray((batch, half, width), dtype=x.dtype, buffer=nl.shared_hbm)
        for bi in nl.affine_range(batch // lanes):
            b = bi * lanes + nl.program_id(0)
            for ti in nl.affine_range(_pair_div_ceil(half, 128)):
                row = ti * 128
                tile = nl.ndarray((128, width), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=x.ap(pattern=[[2 * width, 128], [1, width]],
                                               offset=(b * length + 2 * row + parity) * width))
                nisa.dma_copy(dst=output.ap(pattern=[[width, 128], [1, width]],
                                            offset=(b * half + row) * width), src=tile)
        return output

    @nki.jit
    def _pair_select_bwd(gradient, parity: int, lanes: int):
        """Scatter selected gradients and explicitly zero the unselected positions."""
        batch, half, width = gradient.shape
        length = 2 * half
        _pair_kernel_assert(half % 128 == 0 and batch % lanes == 0, 'aligned half and batch required')
        output = nl.ndarray((batch, length, width), dtype=gradient.dtype, buffer=nl.shared_hbm)
        zero = nl.ndarray((128, width), dtype=gradient.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zero, value=0.0)
        for bi in nl.affine_range(batch // lanes):
            b = bi * lanes + nl.program_id(0)
            for ti in nl.affine_range(_pair_div_ceil(half, 128)):
                row = ti * 128
                tile = nl.ndarray((128, width), dtype=gradient.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=gradient.ap(pattern=[[width, 128], [1, width]],
                                                      offset=(b * half + row) * width))
                nisa.dma_copy(dst=output.ap(pattern=[[2 * width, 128], [1, width]],
                                           offset=(b * length + 2 * row + parity) * width), src=tile)
                nisa.dma_copy(dst=output.ap(pattern=[[2 * width, 128], [1, width]],
                                           offset=(b * length + 2 * row + 1 - parity) * width), src=zero)
        return output

    @nki.jit
    def _pair_expand_fwd(residual, parity: int, lanes: int):
        """Repeat each residual twice, left-pad odd parity, never cross a row."""
        batch, half, width = residual.shape
        length = 2 * half
        _pair_kernel_assert(half % 128 == 0 and batch % lanes == 0, 'aligned half and batch required')
        output = nl.ndarray((batch, length, width), dtype=residual.dtype, buffer=nl.shared_hbm)
        tiles = _pair_div_ceil(half, 128)
        for bi in nl.affine_range(batch // lanes):
            b = bi * lanes + nl.program_id(0)
            if parity == 1:
                zero = nl.ndarray((1, width), dtype=residual.dtype, buffer=nl.sbuf)
                nisa.memset(dst=zero, value=0.0)
                nisa.dma_copy(dst=output.ap(pattern=[[width, 1], [1, width]], offset=b * length * width), src=zero)
            for ti in nl.static_range(tiles):
                row = ti * 128
                tile = nl.ndarray((128, width), dtype=residual.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=residual.ap(pattern=[[width, 128], [1, width]],
                                                       offset=(b * half + row) * width))
                nisa.dma_copy(dst=output.ap(pattern=[[2 * width, 128], [1, width]],
                                            offset=(b * length + 2 * row + parity) * width), src=tile)
                valid = 127 if parity == 1 and ti == tiles - 1 else 128
                nisa.dma_copy(dst=output.ap(pattern=[[2 * width, valid], [1, width]],
                                           offset=(b * length + 2 * row + parity + 1) * width),
                              src=tile[0:valid, :])
        return output

    @nki.jit
    def _pair_expand_bwd(gradient, parity: int, lanes: int):
        """Sum the two causal consumers; odd-parity final residual has one consumer."""
        batch, length, width = gradient.shape
        half = length // 2
        _pair_kernel_assert(length % 256 == 0 and batch % lanes == 0, 'aligned T and batch required')
        output = nl.ndarray((batch, half, width), dtype=gradient.dtype, buffer=nl.shared_hbm)
        tiles = _pair_div_ceil(half, 128)
        for bi in nl.affine_range(batch // lanes):
            b = bi * lanes + nl.program_id(0)
            for ti in nl.static_range(tiles):
                row = ti * 128
                left = nl.ndarray((128, width), dtype=gradient.dtype, buffer=nl.sbuf)
                right = nl.ndarray((128, width), dtype=gradient.dtype, buffer=nl.sbuf)
                result = nl.ndarray((128, width), dtype=gradient.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=left, src=gradient.ap(pattern=[[2 * width, 128], [1, width]],
                                                       offset=(b * length + 2 * row + parity) * width))
                valid = 127 if parity == 1 and ti == tiles - 1 else 128
                if valid == 127:
                    nisa.memset(dst=right, value=0.0)
                nisa.dma_copy(dst=right[0:valid, :],
                              src=gradient.ap(pattern=[[2 * width, valid], [1, width]],
                                              offset=(b * length + 2 * row + parity + 1) * width))
                nisa.tensor_tensor(dst=result, data1=left, data2=right, op=nl.add)
                nisa.dma_copy(dst=output.ap(pattern=[[width, 128], [1, width]],
                                            offset=(b * half + row) * width), src=result)
        return output

    _pair_select_fwd_native = _pair_wrap_nki(_pair_select_fwd)
    _pair_select_bwd_native = _pair_wrap_nki(_pair_select_bwd)
    _pair_expand_fwd_native = _pair_wrap_nki(_pair_expand_fwd)
    _pair_expand_bwd_native = _pair_wrap_nki(_pair_expand_bwd)


class _PairSelectNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, parity):
        ctx.shape = tuple(x.shape)
        ctx.parity = parity
        ctx.native = x.device.type == 'neuron'
        ctx.lanes = 2 if x.shape[0] % 2 == 0 and _NKI_LANES == 2 else 1
        if ctx.native:
            if not _PAIR_HAS_NKI:
                raise RuntimeError('pair selection requires installed NKI')
            return _pair_select_fwd_native[ctx.lanes](x.contiguous(), parity, ctx.lanes)
        return x[:, parity::2, :].contiguous()

    @staticmethod
    def backward(ctx, gradient):
        if ctx.native:
            return _pair_select_bwd_native[ctx.lanes](gradient.contiguous(), ctx.parity, ctx.lanes), None
        result = torch.zeros(ctx.shape, dtype=gradient.dtype, device=gradient.device)
        result[:, ctx.parity::2, :] = gradient
        return result, None


class _PairExpandNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, parity):
        ctx.parity = parity
        ctx.native = residual.device.type == 'neuron'
        ctx.lanes = 2 if residual.shape[0] % 2 == 0 and _NKI_LANES == 2 else 1
        if ctx.native:
            if not _PAIR_HAS_NKI:
                raise RuntimeError('pair expansion requires installed NKI')
            return _pair_expand_fwd_native[ctx.lanes](residual.contiguous(), parity, ctx.lanes)
        expanded = residual.repeat_interleave(2, dim=1)
        if parity:
            expanded = torch.cat((torch.zeros_like(expanded[:, :1]), expanded[:, :-1]), dim=1)
        return expanded

    @staticmethod
    def backward(ctx, gradient):
        if ctx.native:
            return _pair_expand_bwd_native[ctx.lanes](gradient.contiguous(), ctx.parity, ctx.lanes), None
        left = gradient[:, ctx.parity::2, :]
        right = gradient[:, ctx.parity + 1::2, :]
        if ctx.parity:
            right = torch.cat((right, torch.zeros_like(left[:, :1])), dim=1)
        return (left.float() + right.float()).to(gradient.dtype), None
