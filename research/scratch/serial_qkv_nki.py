"""Three BF16 projections with independent inputs and contiguous outputs.

This changes execution only. Weight tensors retain their separate square
shapes; normalization, RoPE, attention, master parameters and optimizer states
are outside this kernel. Both layouts charge ordinary weight reads, on-chip
transposes and output stores. Neither requires an externally packed cache.
"""
import torch
import torch.nn.functional as F


def kernel_assert(condition: bool, message: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {message}"


def build_kernel():
    # The installed NKI parser resolves helper functions through module globals,
    # as in fp8_mlp_token_rows_nki.py; closure-only helpers are unbound there.
    global nl, nisa, load_weight, project
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def load_weight(weight, channels, width):
        blocks = channels // 128
        cache = nl.ndarray((128, channels // width, blocks, width),
                           dtype=nl.bfloat16, buffer=nl.sbuf)
        for group in nl.affine_range(channels // width):
            for part in nl.affine_range(width // 128):
                row = group * width + part * 128
                raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=raw, src=weight[row:row + 128, :])
                for contraction in nl.affine_range(blocks):
                    transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed,
                                      data=raw[:, contraction * 128:(contraction + 1) * 128])
                    nisa.tensor_copy(dst=cache[:, group, contraction,
                                               part * 128:(part + 1) * 128], src=transposed)
        return cache

    def project(inputs, weight, output, first, tile, token_major):
        channels = inputs.shape[1]
        blocks = channels // 128
        values = nl.ndarray((128, blocks, tile), dtype=nl.bfloat16, buffer=nl.sbuf)
        for part in nl.affine_range(tile // 128):
            raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=inputs[first + part * 128:first + (part + 1) * 128, :])
            for contraction in nl.affine_range(blocks):
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed,
                                  data=raw[:, contraction * 128:(contraction + 1) * 128])
                nisa.tensor_copy(dst=values[:, contraction, part * 128:(part + 1) * 128],
                                 src=transposed)
        if token_major:
            # X is stationary. The result already has token rows, so no
            # result transpose is needed before the contiguous output DMA.
            width = min(512, channels)
            rows = nl.ndarray((128, channels), dtype=output.dtype, buffer=nl.sbuf)
            for group in nl.affine_range(channels // width):
                accumulator = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.psum)
                for contraction in nl.affine_range(blocks):
                    nisa.nc_matmul(dst=accumulator, stationary=values[:, contraction, :],
                                   moving=weight[:, group, contraction, :],
                                   accumulate=(contraction > 0))
                nisa.tensor_copy(dst=rows[:, group * width:(group + 1) * width], src=accumulator)
            nisa.dma_copy(dst=output[first:first + 128, :], src=rows)
        else:
            # W is stationary. Form wide token rows on chip after each
            # channel-major accumulator, then write contiguous row blocks.
            rows = nl.ndarray((128, tile // 128, channels), dtype=output.dtype, buffer=nl.sbuf)
            for group in nl.affine_range(blocks):
                accumulator = nl.ndarray((128, tile), dtype=nl.float32, buffer=nl.psum)
                for contraction in nl.affine_range(blocks):
                    nisa.nc_matmul(dst=accumulator, stationary=weight[:, group, contraction, :],
                                   moving=values[:, contraction, :],
                                   accumulate=(contraction > 0))
                rounded = nl.ndarray((128, tile), dtype=output.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rounded, src=accumulator)
                for part in nl.affine_range(tile // 128):
                    transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed, data=rounded[:, part * 128:(part + 1) * 128])
                    nisa.tensor_copy(dst=rows[:, part, group * 128:(group + 1) * 128], src=transposed)
            for part in nl.affine_range(tile // 128):
                nisa.dma_copy(dst=output[first + part * 128:first + (part + 1) * 128, :],
                              src=rows[:, part, :])

    @nki.jit
    def serial_qkv(xq, xk, xv, wq, wk, wv, lanes: int = 2,
                   token_tile: int = 512, token_major: bool = False,
                   value_fp32: bool = False):
        """Return three independent contiguous [tokens, channels] projections.

        Args: BF16 input streams and ordinary BF16 [out, in] square weights.
        Notes: physical cores own disjoint token tiles; no cross-token or
        cross-stream reductions. The backward is independently gated native
        BF16 matmul, preserving six separate gradients and FP32 masters.
        """
        tokens, channels = xq.shape
        kernel_assert(xk.shape == xv.shape == xq.shape, 'Equal input shapes required')
        kernel_assert(wq.shape == wk.shape == wv.shape == (channels, channels), 'Square weights required')
        kernel_assert(xq.dtype == xk.dtype == xv.dtype == wq.dtype == wk.dtype == wv.dtype == nl.bfloat16,
                      'Only explicitly rounded BF16 operands are supported')
        kernel_assert(channels % 128 == 0 and lanes in (1, 2), 'Unsupported channel/LNC geometry')
        kernel_assert(token_tile in (128, 256, 512) and tokens % (lanes * token_tile) == 0,
                      'Token rows must divide the physical-core tiling')
        kernel_assert(not token_major or token_tile == 128, 'Token-stationary layout requires tile128')
        kernel_assert(6 * channels * channels + 14 * channels * token_tile + 2 * 1024**2
                      <= nl.tile_size.sbuf_size_bytes, 'Conservative SBUF budget exceeded')
        width = min(512, channels) if token_major else 128
        kernel_assert(channels % width == 0, 'Output width must divide channels')
        qcache = load_weight(wq, channels, width)
        kcache = load_weight(wk, channels, width)
        vcache = load_weight(wv, channels, width)
        q = nl.ndarray(xq.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        k = nl.ndarray(xq.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        # Native attention retains the V accumulator before adding value
        # embeddings. An explicit BF16 store here changes that rounding boundary.
        # This opt-in output preserves FP32 accumulation, not FP32 input GEMMs.
        v = nl.ndarray(xq.shape, dtype=nl.float32 if value_fp32 else nl.bfloat16,
                       buffer=nl.shared_hbm)
        for block in nl.affine_range(tokens // (lanes * token_tile)):
            first = (block * lanes + nl.program_id(0)) * token_tile
            project(xq, qcache, q, first, token_tile, token_major)
            project(xk, kcache, k, first, token_tile, token_major)
            project(xv, vcache, v, first, token_tile, token_major)
        return q, k, v

    return serial_qkv


def native(xq, xk, xv, wq, wk, wv):
    return F.linear(xq, wq.bfloat16()), F.linear(xk, wk.bfloat16()), F.linear(xv, wv.bfloat16())


def native_value_fp32(xq, xk, xv, wq, wk, wv):
    # Exact BF16 operands represented as FP32. The installed compiler's
    # declared matmult autocast remains BF16, with an explicit FP32 result.
    return (F.linear(xq, wq.bfloat16()), F.linear(xk, wk.bfloat16()),
            F.linear(xv.float(), wv.bfloat16().float()))


def make_autograd(kernel_call):
    class Projections(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xq, xk, xv, wq, wk, wv):
            ctx.save_for_backward(xq, xk, xv, wq, wk, wv)
            return kernel_call(xq, xk, xv, wq, wk, wv)

        @staticmethod
        def backward(ctx, gq, gk, gv):
            xq, xk, xv, wq, wk, wv = ctx.saved_tensors
            # BF16 GEMM operands match the original native matmult policy.
            # With FP32 V, this conversion is explicit and independently gated.
            gv = gv.to(wv.dtype)
            return (gq @ wq, gk @ wk, gv @ wv,
                    gq.T @ xq, gk.T @ xk, gv.T @ xv)

    def apply(xq, xk, xv, wq, wk, wv):
        # The usual cast graph connects three separate cache gradients to
        # their original FP32 Parameter objects. No combined Parameter exists.
        return Projections.apply(xq, xk, xv, wq.bfloat16(), wk.bfloat16(), wv.bfloat16())

    return apply
