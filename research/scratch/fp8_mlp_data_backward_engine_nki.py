"""Fuse sequential FP32 rescalings in the existing FP8 MLP data backward.

Retain dPre in BF16 for the separate weight-gradient product. The intermediate
G@Wdown result stays in SBUF and is not written to HBM. Gradient quantization
changes the declared VJP. This variant preserves its original scale order,
FP32 intermediate math and final BF16 rounding. Never precombine scale factors.
Source161 validated this two-multiply Vector instruction in square dX; the MLP
requires its own unchanged independent stage and edge-case checks before use.
"""


def build_kernel():
    global nl, nisa, calibrate_token_rows
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def calibrate_token_rows(values, nonnegative: bool = False, fused_abs: bool = False):
        high = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        if fused_abs:
            # The signed-gradient contract is unchanged: one magnitude/max
            # instruction replaces the max/min/negate/max sequence. Keep the
            # original exponent policy and independently check both boundaries.
            magnitude = nl.ndarray(values.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=magnitude, data=values,
                                       op0=nl.abs, operand0=0.0,
                                       reduce_op=nl.maximum, reduce_res=high)
        else:
            nisa.tensor_reduce(dst=high, data=values, op=nl.maximum, axis=(1,))
        if not nonnegative and not fused_abs:
            low = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=low, data=values, op=nl.minimum, axis=(1,))
            nisa.tensor_scalar(dst=low, data=low, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=high, data1=high, data2=low, op=nl.maximum)
        nisa.tensor_scalar(dst=clamped, data=high, op0=nl.maximum, operand0=7.888609052210118e-31)
        nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                           op0=nl.bitwise_and, operand0=2139095040)
        nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.multiply,
                           operand0=-1.0, op1=nl.add, operand1=2181038080.0)
        nisa.tensor_scalar(dst=inverse, data=exponent, op0=nl.subtract, operand0=50331648.0)
        nisa.tensor_copy(dst=meta[:, 0:1], src=scale.view(nl.float32))
        nisa.tensor_copy(dst=meta[:, 1:2], src=inverse.view(nl.float32))
        nisa.tensor_copy(dst=meta[:, 2:3], src=high)
        return meta

    @nki.jit
    def backward_data_engine(x, relu, packed_up, packed_down, up_scales, down_scales,
                      lanes: int = 2, token_tile: int = 128, diagnostic: bool = False,
                      fused_abs: bool = False):
        """Return dX[T,C] and dPre[T,H] for the two data-gradient products.

        x is incoming gradient G. The first packed operand (packed_up/up_scales)
        is Wdown.T with Wdown's existing metadata. The second operand is Wup.T
        with Wup's metadata. Parameter names track projection order within this
        kernel, not the original model's forward weight names.
        Both gradient inputs are quantized per row; this is a new approximate VJP.
        """
        tokens, channels = x.shape
        assert relu.shape[0] == tokens
        hidden = relu.shape[1]
        assert x.dtype == relu.dtype == nl.bfloat16
        assert channels % 256 == 0 and hidden % 256 == 0
        assert lanes in (1, 2) and token_tile == 128
        assert tokens % (128 * lanes) == 0
        assert up_scales.shape == down_scales.shape == (128, 3)
        assert up_scales.dtype == down_scales.dtype == nl.float32
        up_width, down_width = min(512, hidden), min(512, channels)
        assert hidden % up_width == 0 and channels % down_width == 0
        nc, nh = channels // 128, hidden // 128
        assert 2 * channels * hidden + 8 * (channels + hidden) * 128 + 4 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        result = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_pre = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if diagnostic:
            saved_x = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
            saved_hidden = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
            up_stage = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
            down_stage = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
            x_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
            h_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        up_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        down_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=up_meta, src=up_scales)
        nisa.dma_copy(dst=down_meta, src=down_scales)
        up_cache = nl.ndarray((128, hidden // up_width, nc, up_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_cache = nl.ndarray((128, channels // down_width, nh, down_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        assert packed_up.dtype == packed_down.dtype == nl.uint8
        assert packed_up.shape == (hidden // up_width, 128, nc, up_width)
        assert packed_down.shape == (channels // down_width, 128, nh, down_width)
        packed_up_values = packed_up.view(nl.float8_e4m3)
        packed_down_values = packed_down.view(nl.float8_e4m3)
        for group in nl.affine_range(hidden // up_width):
            nisa.dma_copy(dst=up_cache[:, group, :, :], src=packed_up_values[group, :, :, :])
        for group in nl.affine_range(channels // down_width):
            nisa.dma_copy(dst=down_cache[:, group, :, :], src=packed_down_values[group, :, :, :])

        for tile in nl.affine_range(tokens // (128 * lanes)):
            first = (tile * lanes + nl.program_id(0)) * 128
            rows = nl.ds(first, 128)
            raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=x[rows, :])
            x_meta = calibrate_token_rows(raw, fused_abs=fused_abs)
            x_fp8 = nl.ndarray((128, channels), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_fp8, data=raw, op0=nl.multiply, operand0=x_meta[:, 0:1])
            if diagnostic:
                x_restored = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=x_restored, data=x_fp8, op0=nl.multiply, operand0=x_meta[:, 1:2])
                nisa.dma_copy(dst=saved_x[rows, :], src=x_restored)
            inputs = nl.ndarray((128, nc, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for ci in nl.affine_range(nc):
                transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                  data=x_fp8[:, ci * 128:(ci + 1) * 128])
                nisa.tensor_copy(dst=inputs[:, ci, :], src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            twice_relu = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            raw_relu = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            hidden_rows = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw_relu, src=relu[rows, :])
            nisa.tensor_scalar(dst=twice_relu, data=raw_relu, op0=nl.multiply, operand0=2.0)
            for group in nl.affine_range(hidden // up_width):
                columns = nl.ds(group * up_width, up_width)
                accumulator = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(nc // 2):
                    nisa.nc_matmul(dst=accumulator, stationary=inputs[:, pair * 2:pair * 2 + 2, :],
                                   moving=up_cache[:, group, pair * 2:pair * 2 + 2, :],
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                scaled = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.sbuf)
                rounded = nl.ndarray((128, up_width), dtype=nl.bfloat16, buffer=nl.sbuf)
                # Two sequential FP32 multiplies; one Vector pass and final cast.
                # A precomputed scale product would have a different range.
                if diagnostic:
                    nisa.tensor_scalar(dst=scaled, data=accumulator,
                                       op0=nl.multiply, operand0=up_meta[:, 1:2],
                                       op1=nl.multiply, operand1=x_meta[:, 1:2],
                                       engine=nisa.engine.vector)
                    nisa.tensor_copy(dst=rounded, src=scaled)
                else:
                    nisa.tensor_scalar(dst=rounded, data=accumulator,
                                       op0=nl.multiply, operand0=up_meta[:, 1:2],
                                       op1=nl.multiply, operand1=x_meta[:, 1:2],
                                       engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=hidden_rows[:, columns], data1=rounded,
                                   data2=twice_relu[:, columns], op=nl.multiply)
                if diagnostic:
                    nisa.dma_copy(dst=up_stage[rows, columns], src=scaled)
            nisa.dma_copy(dst=saved_pre[rows, :], src=hidden_rows)
            # dPre has both signs; the nonnegative forward shortcut is invalid.
            h_meta = calibrate_token_rows(hidden_rows, fused_abs=fused_abs)
            h_fp8 = nl.ndarray((128, hidden), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h_fp8, data=hidden_rows, op0=nl.multiply, operand0=h_meta[:, 0:1])
            if diagnostic:
                h_restored = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=h_restored, data=h_fp8, op0=nl.multiply, operand0=h_meta[:, 1:2])
                nisa.dma_copy(dst=saved_hidden[rows, :], src=h_restored)
            hidden_cache = nl.ndarray((128, nh, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for hi in nl.affine_range(nh):
                transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                  data=h_fp8[:, hi * 128:(hi + 1) * 128])
                nisa.tensor_copy(dst=hidden_cache[:, hi, :], src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            output_rows = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            for group in nl.affine_range(channels // down_width):
                columns = nl.ds(group * down_width, down_width)
                accumulator = nl.ndarray((128, down_width), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(nh // 2):
                    nisa.nc_matmul(dst=accumulator, stationary=hidden_cache[:, pair * 2:pair * 2 + 2, :],
                                   moving=down_cache[:, group, pair * 2:pair * 2 + 2, :],
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                scaled = nl.ndarray((128, down_width), dtype=nl.float32, buffer=nl.sbuf)
                if diagnostic:
                    nisa.tensor_scalar(dst=scaled, data=accumulator,
                                       op0=nl.multiply, operand0=down_meta[:, 1:2],
                                       op1=nl.multiply, operand1=h_meta[:, 1:2],
                                       engine=nisa.engine.vector)
                    nisa.tensor_copy(dst=output_rows[:, columns], src=scaled)
                else:
                    nisa.tensor_scalar(dst=output_rows[:, columns], data=accumulator,
                                       op0=nl.multiply, operand0=down_meta[:, 1:2],
                                       op1=nl.multiply, operand1=h_meta[:, 1:2],
                                       engine=nisa.engine.vector)
                if diagnostic:
                    nisa.dma_copy(dst=down_stage[rows, columns], src=scaled)
            nisa.dma_copy(dst=result[rows, :], src=output_rows)
            if diagnostic:
                nisa.dma_copy(dst=x_metadata[rows, :], src=x_meta)
                nisa.dma_copy(dst=h_metadata[rows, :], src=h_meta)
        if diagnostic:
            return result, saved_pre, saved_x, saved_hidden, up_stage, down_stage, x_metadata, h_metadata
        return result, saved_pre

    return backward_data_engine
