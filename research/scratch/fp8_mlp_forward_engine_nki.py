"""Sequential FP32 Vector scale fusion in the original FP8 MLP forward.

Preserve both multiplies, their order, BF16 rounding, ReLU-square, quantizers,
packed layouts and retained saves. Gate only resident token-row outputs.
"""


def kernel_assert(condition: bool, error_text: str):
    """Format compile-time validation consistently with the NKI helpers."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def build_kernel():
    global nl, nisa, calibrate_token_rows
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def calibrate_token_rows(values, nonnegative: bool = False):
        high = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=high, data=values, op=nl.maximum, axis=(1,))
        if not nonnegative:
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
    def forward_engine(x, up, down, up_scales, down_scales, lanes: int = 2,
                token_tile: int = 128, diagnostic: bool = False,
                fixed_x_scale: float = 0.0, packed_up=None, packed_down=None,
                save_fp8: bool = False, save_hidden_metadata: bool = False,
                compact_full_vjp: bool = False, stream_packed_weights: bool = False,
                output_batch_size: int = 0):
        """Return Y[T,C], ReLU[T,H], QDQ-X[T,C], QDQ-hidden[T,H].

        Diagnostic mode additionally returns FP32 up[T,H], down[T,C] and
        per-token X/hidden metadata. Quantization and pair rules match the
        original FP8 island; this variant changes layout and removes the
        unnecessary hidden minimum reduction. BF16 backward is a separate gate.
        Optional raw activation bytes and hidden metadata preserve values already
        produced on chip for separately gated FP8 backward consumers. Existing
        BF16 saves remain by default. The explicit compact_full_vjp mode returns
        only Y, ReLU, X bytes, hidden bytes and hidden metadata, as required by
        the complete FP8 backward. It omits both unused BF16 QDQ allocations,
        reconstruction operations and stores. Diagnostic/BF16 backward callers
        retain their original interface. This flag alone establishes no speedup.
        Optional streaming loads one
        packed output group at a time instead of keeping both full weights in
        SBUF. Its sequential output-group loops preserve each contraction and
        rescaling order, at the cost of repeated HBM reads for every token tile.
        Actual caller allocation and complete-step performance need measurement.
        A positive output_batch_size allocates Y as [B,T/B,C] while preserving
        all linear-address stores and arithmetic. This independently tests the
        opaque output-rank boundary; all retained backward saves stay two-dimensional.
        """
        kernel_assert(not stream_packed_weights and output_batch_size == 0,
                      "Scale fusion currently gates resident token-row outputs only")
        kernel_assert(not compact_full_vjp or (save_fp8 and not diagnostic),
                      "Compact saves require the ordinary full-FP8 output contract")
        kernel_assert(not stream_packed_weights or (compact_full_vjp and packed_up != None and packed_down != None),
                      "Streaming requires compact full-FP8 saves and both packed weights")
        tokens, channels = x.shape
        hidden, up_channels = up.shape
        assert up_channels == channels and down.shape == (channels, hidden)
        assert x.dtype == up.dtype == down.dtype == nl.bfloat16
        assert channels % 256 == 0 and hidden % 256 == 0
        assert lanes in (1, 2) and token_tile == 128
        assert tokens % (128 * lanes) == 0
        assert up_scales.shape == down_scales.shape == (128, 3)
        assert up_scales.dtype == down_scales.dtype == nl.float32
        # Restricted alternative: |x| <= 130/fixed_x_scale, established by
        # the caller's RMS-normalization contract. No runtime reduction here.
        # Hidden calibration remains per token. Zero preserves the old policy.
        if fixed_x_scale != 0.0:
            assert channels in (256, 1024)
            assert fixed_x_scale * fixed_x_scale * channels == 16384.0
            fixed_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=fixed_meta[:, 0:1], value=fixed_x_scale)
            nisa.memset(dst=fixed_meta[:, 1:2], value=1.0 / fixed_x_scale)
            # Third column is an input bound, not an observed maximum.
            nisa.memset(dst=fixed_meta[:, 2:3], value=130.0 / fixed_x_scale)
        up_width, down_width = min(512, hidden), min(512, channels)
        assert hidden % up_width == 0 and channels % down_width == 0
        nc, nh = channels // 128, hidden // 128
        # Source-level bound only: compiler allocation must still be profiled.
        weight_bytes = (128 * (nc * up_width + nh * down_width)
                        if stream_packed_weights else 2 * channels * hidden)
        assert weight_bytes + 8 * (channels + hidden) * 128 + 4 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        if output_batch_size != 0:
            kernel_assert(output_batch_size > 0 and tokens % output_batch_size == 0,
                          "Output batch must divide the token count")
            result = nl.ndarray((output_batch_size, tokens // output_batch_size, channels),
                                dtype=nl.bfloat16, buffer=nl.shared_hbm)
            result_rows = result.reshape((tokens, channels))
        else:
            result = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
            result_rows = result
        saved_relu = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if not compact_full_vjp:
            saved_x = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
            saved_hidden = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if save_fp8:
            saved_x_bytes = nl.ndarray(x.shape, dtype=nl.uint8, buffer=nl.shared_hbm)
            saved_hidden_bytes = nl.ndarray((tokens, hidden), dtype=nl.uint8, buffer=nl.shared_hbm)
        if diagnostic or save_fp8 or save_hidden_metadata:
            h_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        if diagnostic:
            up_stage = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
            down_stage = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
            x_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        up_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        down_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=up_meta, src=up_scales)
        nisa.dma_copy(dst=down_meta, src=down_scales)
        if not stream_packed_weights:
            up_cache = nl.ndarray((128, hidden // up_width, nc, up_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            down_cache = nl.ndarray((128, channels // down_width, nh, down_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        if packed_up != None:
            assert packed_down != None
            assert packed_up.dtype == packed_down.dtype == nl.uint8
            assert packed_up.shape == (hidden // up_width, 128, nc, up_width)
            assert packed_down.shape == (channels // down_width, 128, nh, down_width)
            # Raw legacy-E4M3 bytes were prepared once per update. The bitcast
            # has no floating-point conversion or scaling; the independent
            # cache gate checks this representation and contraction layout.
            packed_up_values = packed_up.view(nl.float8_e4m3)
            packed_down_values = packed_down.view(nl.float8_e4m3)
            if not stream_packed_weights:
                for group in nl.affine_range(hidden // up_width):
                    nisa.dma_copy(dst=up_cache[:, group, :, :], src=packed_up_values[group, :, :, :])
                for group in nl.affine_range(channels // down_width):
                    nisa.dma_copy(dst=down_cache[:, group, :, :], src=packed_down_values[group, :, :, :])
        else:
            assert packed_down == None
            for group in nl.affine_range(hidden // up_width):
                for sub in nl.affine_range(up_width // 128):
                    raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                    first = group * up_width + sub * 128
                    nisa.dma_copy(dst=raw, src=up[first:first + 128, :])
                    for ci in nl.affine_range(nc):
                        transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=transposed, data=raw[:, ci * 128:(ci + 1) * 128])
                        nisa.tensor_scalar(dst=up_cache[:, group, ci, sub * 128:(sub + 1) * 128],
                                           data=transposed, op0=nl.multiply, operand0=up_meta[:, 0:1])
            for group in nl.affine_range(channels // down_width):
                for sub in nl.affine_range(down_width // 128):
                    raw = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                    first = group * down_width + sub * 128
                    nisa.dma_copy(dst=raw, src=down[first:first + 128, :])
                    for hi in nl.affine_range(nh):
                        transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=transposed, data=raw[:, hi * 128:(hi + 1) * 128])
                        nisa.tensor_scalar(dst=down_cache[:, group, hi, sub * 128:(sub + 1) * 128],
                                           data=transposed, op0=nl.multiply, operand0=down_meta[:, 0:1])

        for tile in nl.affine_range(tokens // (128 * lanes)):
            first = (tile * lanes + nl.program_id(0)) * 128
            rows = nl.ds(first, 128)
            raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=x[rows, :])
            if fixed_x_scale != 0.0:
                x_meta = fixed_meta
            else:
                x_meta = calibrate_token_rows(raw)
            x_fp8 = nl.ndarray((128, channels), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_fp8, data=raw, op0=nl.multiply, operand0=x_meta[:, 0:1])
            if not compact_full_vjp:
                x_restored = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=x_restored, data=x_fp8, op0=nl.multiply, operand0=x_meta[:, 1:2])
                nisa.dma_copy(dst=saved_x[rows, :], src=x_restored)
            if save_fp8:
                nisa.dma_copy(dst=saved_x_bytes.view(nl.float8_e4m3)[rows, :], src=x_fp8)
            inputs = nl.ndarray((128, nc, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for ci in nl.affine_range(nc):
                transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                  data=x_fp8[:, ci * 128:(ci + 1) * 128])
                nisa.tensor_copy(dst=inputs[:, ci, :], src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            relu_rows = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            hidden_rows = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            if stream_packed_weights:
                up_group = nl.ndarray((128, nc, up_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                for group in nl.sequential_range(hidden // up_width):
                    nisa.dma_copy(dst=up_group, src=packed_up_values[group, :, :, :])
                    columns = nl.ds(group * up_width, up_width)
                    accumulator = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.psum)
                    for pair in nl.affine_range(nc // 2):
                        nisa.nc_matmul(dst=accumulator, stationary=inputs[:, pair * 2:pair * 2 + 2, :],
                                       moving=up_group[:, pair * 2:pair * 2 + 2, :],
                                       accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                    scaled = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.sbuf)
                    rounded = nl.ndarray((128, up_width), dtype=nl.bfloat16, buffer=nl.sbuf)
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
                    nisa.tensor_scalar(dst=relu_rows[:, columns], data=rounded, op0=nl.maximum, operand0=0.0)
                    nisa.tensor_tensor(dst=hidden_rows[:, columns], data1=relu_rows[:, columns],
                                       data2=relu_rows[:, columns], op=nl.multiply)
                    if diagnostic:
                        nisa.dma_copy(dst=up_stage[rows, columns], src=scaled)
            else:
                for group in nl.affine_range(hidden // up_width):
                    columns = nl.ds(group * up_width, up_width)
                    accumulator = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.psum)
                    for pair in nl.affine_range(nc // 2):
                        nisa.nc_matmul(dst=accumulator, stationary=inputs[:, pair * 2:pair * 2 + 2, :],
                                       moving=up_cache[:, group, pair * 2:pair * 2 + 2, :],
                                       accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                    scaled = nl.ndarray((128, up_width), dtype=nl.float32, buffer=nl.sbuf)
                    rounded = nl.ndarray((128, up_width), dtype=nl.bfloat16, buffer=nl.sbuf)
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
                    nisa.tensor_scalar(dst=relu_rows[:, columns], data=rounded, op0=nl.maximum, operand0=0.0)
                    nisa.tensor_tensor(dst=hidden_rows[:, columns], data1=relu_rows[:, columns],
                                       data2=relu_rows[:, columns], op=nl.multiply)
                    if diagnostic:
                        nisa.dma_copy(dst=up_stage[rows, columns], src=scaled)
            nisa.dma_copy(dst=saved_relu[rows, :], src=relu_rows)
            h_meta = calibrate_token_rows(hidden_rows, True)
            h_fp8 = nl.ndarray((128, hidden), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h_fp8, data=hidden_rows, op0=nl.multiply, operand0=h_meta[:, 0:1])
            if not compact_full_vjp:
                h_restored = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=h_restored, data=h_fp8, op0=nl.multiply, operand0=h_meta[:, 1:2])
                nisa.dma_copy(dst=saved_hidden[rows, :], src=h_restored)
            if save_fp8:
                nisa.dma_copy(dst=saved_hidden_bytes.view(nl.float8_e4m3)[rows, :], src=h_fp8)
            hidden_cache = nl.ndarray((128, nh, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for hi in nl.affine_range(nh):
                transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                  data=h_fp8[:, hi * 128:(hi + 1) * 128])
                nisa.tensor_copy(dst=hidden_cache[:, hi, :], src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            output_rows = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            if stream_packed_weights:
                down_group = nl.ndarray((128, nh, down_width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                for group in nl.sequential_range(channels // down_width):
                    nisa.dma_copy(dst=down_group, src=packed_down_values[group, :, :, :])
                    columns = nl.ds(group * down_width, down_width)
                    accumulator = nl.ndarray((128, down_width), dtype=nl.float32, buffer=nl.psum)
                    for pair in nl.affine_range(nh // 2):
                        nisa.nc_matmul(dst=accumulator, stationary=hidden_cache[:, pair * 2:pair * 2 + 2, :],
                                       moving=down_group[:, pair * 2:pair * 2 + 2, :],
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
            else:
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
            nisa.dma_copy(dst=result_rows[rows, :], src=output_rows)
            if diagnostic:
                nisa.dma_copy(dst=x_metadata[rows, :], src=x_meta)
            if diagnostic or save_fp8 or save_hidden_metadata:
                nisa.dma_copy(dst=h_metadata[rows, :], src=h_meta)
        if compact_full_vjp:
            return result, saved_relu, saved_x_bytes, saved_hidden_bytes, h_metadata
        if diagnostic:
            if save_fp8:
                return (result, saved_relu, saved_x, saved_hidden, up_stage, down_stage,
                        x_metadata, h_metadata, saved_x_bytes, saved_hidden_bytes)
            return result, saved_relu, saved_x, saved_hidden, up_stage, down_stage, x_metadata, h_metadata
        if save_fp8:
            return result, saved_relu, saved_x, saved_hidden, saved_x_bytes, saved_hidden_bytes, h_metadata
        if save_hidden_metadata:
            return result, saved_relu, saved_x, saved_hidden, h_metadata
        return result, saved_relu, saved_x, saved_hidden

    return forward_engine
