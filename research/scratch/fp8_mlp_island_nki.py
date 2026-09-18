"""A causal FP8 forward island with per-token activation scaling on NC-v3.

Both projections use legacy E4M3 double-row matmul. Weight metadata is calibrated
once per update by the separately gated weight-cache helper. Activation scales
depend only on channels of the same token, never on future tokens or other rows.
The hidden activation stays on chip between projections. BF16 saved values for
an explicit straight-through backward are still written and charged.

Unlike the storage-only probe, no weight tile is converted to BF16 before GEMM.
Contraction pairs are corresponding positions in adjacent128-channel blocks.
That order must be validated against its own empirical11-bit pair reference.
"""


def build_kernel():
    # The installed NKI parser resolves callees through module globals, not
    # Python closure cells. Keep these undecorated helpers in that namespace.
    global nl, nisa, calibrate_rows, transpose_broadcast_inverse
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def calibrate_rows(values):
        """Per-token [scale,inverse,amax], for finite BF16 amax<=2**100.

        Values are token-major. Zero clamps only the exponent calculation;
        every token retains its own metadata and therefore row independence.
        """
        high = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        low = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=high, data=values, op=nl.maximum, axis=(1,))
        nisa.tensor_reduce(dst=low, data=values, op=nl.minimum, axis=(1,))
        nisa.tensor_scalar(dst=low, data=low, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(dst=high, data1=high, data2=low, op=nl.maximum)
        nisa.tensor_scalar(dst=clamped, data=high, op0=nl.maximum,
                           operand0=7.888609052210118e-31)
        nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                           op0=nl.bitwise_and, operand0=2139095040)
        nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.multiply,
                           operand0=-1.0, op1=nl.add, operand1=2181038080.0)
        nisa.tensor_scalar(dst=inverse, data=exponent,
                           op0=nl.subtract, operand0=50331648.0)
        nisa.tensor_copy(dst=meta[:, 0:1], src=scale.view(nl.float32))
        nisa.tensor_copy(dst=meta[:, 1:2], src=inverse.view(nl.float32))
        nisa.tensor_copy(dst=meta[:, 2:3], src=high)
        return meta

    def transpose_broadcast_inverse(vector, destination):
        """Transpose128 token scales and replicate across output channels."""
        transposed = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        row = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=transposed, data=vector)
        nisa.tensor_copy(dst=row, src=transposed)
        for quadrant in nl.static_range(4):
            first = quadrant * 32
            nisa.nc_stream_shuffle(dst=destination[first:first + 32, :],
                                   src=row, shuffle_mask=[0] * 32)

    @nki.jit
    def forward(x, up, down, up_scales, down_scales, lanes: int = 2,
                token_tile: int = 512, diagnostic: bool = False):
        """Return Y, saved ReLU[H,T], quantized X[T,C], quantized hidden[T,H].

        Inputs use ordinary contiguous BF16 matrix layouts. Up/down contain
        quantized/dequantized BF16 weights shared with backward. Diagnostic
        mode additionally returns FP32 projection stages and per-token scales;
        these extra stores are never included in a production timing claim.
        """
        tokens, channels = x.shape
        hidden, up_channels = up.shape
        assert up_channels == channels and down.shape == (channels, hidden)
        assert x.dtype == up.dtype == down.dtype == nl.bfloat16
        assert channels % 256 == 0 and hidden % 256 == 0
        assert lanes in (1, 2) and token_tile in (128, 256, 512)
        assert tokens % (lanes * token_tile) == 0
        assert up_scales.shape == down_scales.shape == (128, 3)
        assert up_scales.dtype == down_scales.dtype == nl.float32
        nc, nh = channels // 128, hidden // 128
        # Weights8MiB at1024x4096. Count simultaneous BF16 hidden rows and
        # FP8 hidden cache plus input/output buffers; actual spills need profiles.
        live_estimate = 2 * channels * hidden + (3 * hidden + 4 * channels) * token_tile
        assert live_estimate + 4 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        result = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_relu = nl.ndarray((hidden, tokens), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_x = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_hidden = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if diagnostic:
            up_stage = nl.ndarray((hidden, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
            down_stage = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
            x_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
            h_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        up_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        down_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=up_meta, src=up_scales)
        nisa.dma_copy(dst=down_meta, src=down_scales)
        up_cache = nl.ndarray((128, nh, nc, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        down_cache = nl.ndarray((128, nc, nh, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for hi in nl.affine_range(nh):
            raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=up[hi * 128:(hi + 1) * 128, :])
            for ci in nl.affine_range(nc):
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed, data=raw[:, ci * 128:(ci + 1) * 128])
                nisa.tensor_scalar(dst=up_cache[:, hi, ci, :], data=transposed,
                                   op0=nl.multiply, operand0=up_meta[:, 0:1])
        for ci in nl.affine_range(nc):
            raw = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=down[ci * 128:(ci + 1) * 128, :])
            for hi in nl.affine_range(nh):
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed, data=raw[:, hi * 128:(hi + 1) * 128])
                nisa.tensor_scalar(dst=down_cache[:, ci, hi, :], data=transposed,
                                   op0=nl.multiply, operand0=down_meta[:, 0:1])

        for mi in nl.affine_range(tokens // (lanes * token_tile)):
            first = (mi * lanes + nl.program_id(0)) * token_tile
            inputs = nl.ndarray((128, nc, token_tile), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            x_inverse = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
            hidden_rows = nl.ndarray((128, token_tile // 128, hidden),
                                     dtype=nl.bfloat16, buffer=nl.sbuf)
            for ti in nl.affine_range(token_tile // 128):
                rows = nl.ds(first + ti * 128, 128)
                raw = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=raw, src=x[rows, :])
                meta = calibrate_rows(raw)
                quantized = nl.ndarray((128, channels), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                restored = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=quantized, data=raw, op0=nl.multiply, operand0=meta[:, 0:1])
                nisa.tensor_scalar(dst=restored, data=quantized, op0=nl.multiply, operand0=meta[:, 1:2])
                nisa.dma_copy(dst=saved_x[rows, :], src=restored)
                transpose_broadcast_inverse(meta[:, 1:2], x_inverse[:, ti * 128:(ti + 1) * 128])
                if diagnostic:
                    nisa.dma_copy(dst=x_metadata[rows, :], src=meta)
                for ci in nl.affine_range(nc):
                    # NC-v3's FP8 transpose requires a two-element output
                    # stride in PSUM. Allocate that layout explicitly, then
                    # compact only the initialized elements into the FP8 cache.
                    transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                      data=quantized[:, ci * 128:(ci + 1) * 128])
                    nisa.tensor_copy(dst=inputs[:, ci, ti * 128:(ti + 1) * 128],
                                     src=transposed.ap(pattern=[[256, 128], [2, 128]]))

            for hi in nl.affine_range(nh):
                accumulator = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(nc // 2):
                    nisa.nc_matmul(dst=accumulator,
                                   stationary=up_cache[:, hi, pair * 2:pair * 2 + 2, :],
                                   moving=inputs[:, pair * 2:pair * 2 + 2, :],
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                weighted = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
                scaled = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
                rounded = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                relu = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                activated = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                # Rescale sequentially; multiplying two tiny inverses first can underflow.
                nisa.tensor_scalar(dst=weighted, data=accumulator,
                                   op0=nl.multiply, operand0=up_meta[:, 1:2])
                nisa.tensor_tensor(dst=scaled, data1=weighted, data2=x_inverse, op=nl.multiply)
                nisa.tensor_copy(dst=rounded, src=scaled)
                nisa.tensor_scalar(dst=relu, data=rounded, op0=nl.maximum, operand0=0.0)
                nisa.tensor_tensor(dst=activated, data1=relu, data2=relu, op=nl.multiply)
                nisa.dma_copy(dst=saved_relu[hi * 128:(hi + 1) * 128, first:first + token_tile], src=relu)
                if diagnostic:
                    nisa.dma_copy(dst=up_stage[hi * 128:(hi + 1) * 128, first:first + token_tile], src=scaled)
                for ti in nl.affine_range(token_tile // 128):
                    transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed, data=activated[:, ti * 128:(ti + 1) * 128])
                    nisa.tensor_copy(dst=hidden_rows[:, ti, hi * 128:(hi + 1) * 128], src=transposed)

            hidden_cache = nl.ndarray((128, nh, token_tile), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            h_inverse = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
            for ti in nl.affine_range(token_tile // 128):
                rows = nl.ds(first + ti * 128, 128)
                meta = calibrate_rows(hidden_rows[:, ti, :])
                quantized = nl.ndarray((128, hidden), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                restored = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=quantized, data=hidden_rows[:, ti, :],
                                   op0=nl.multiply, operand0=meta[:, 0:1])
                nisa.tensor_scalar(dst=restored, data=quantized, op0=nl.multiply, operand0=meta[:, 1:2])
                nisa.dma_copy(dst=saved_hidden[rows, :], src=restored)
                transpose_broadcast_inverse(meta[:, 1:2], h_inverse[:, ti * 128:(ti + 1) * 128])
                if diagnostic:
                    nisa.dma_copy(dst=h_metadata[rows, :], src=meta)
                for hi in nl.affine_range(nh):
                    transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                      data=quantized[:, hi * 128:(hi + 1) * 128])
                    nisa.tensor_copy(dst=hidden_cache[:, hi, ti * 128:(ti + 1) * 128],
                                     src=transposed.ap(pattern=[[256, 128], [2, 128]]))

            output_rows = nl.ndarray((128, token_tile // 128, channels),
                                     dtype=nl.bfloat16, buffer=nl.sbuf)
            if diagnostic:
                diagnostic_rows = nl.ndarray((128, token_tile // 128, channels),
                                             dtype=nl.float32, buffer=nl.sbuf)
            for ci in nl.affine_range(nc):
                accumulator = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(nh // 2):
                    nisa.nc_matmul(dst=accumulator,
                                   stationary=down_cache[:, ci, pair * 2:pair * 2 + 2, :],
                                   moving=hidden_cache[:, pair * 2:pair * 2 + 2, :],
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                weighted = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
                scaled = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.sbuf)
                rounded = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=weighted, data=accumulator,
                                   op0=nl.multiply, operand0=down_meta[:, 1:2])
                nisa.tensor_tensor(dst=scaled, data1=weighted, data2=h_inverse, op=nl.multiply)
                nisa.tensor_copy(dst=rounded, src=scaled)
                for ti in nl.affine_range(token_tile // 128):
                    transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed, data=rounded[:, ti * 128:(ti + 1) * 128])
                    nisa.tensor_copy(dst=output_rows[:, ti, ci * 128:(ci + 1) * 128], src=transposed)
                    if diagnostic:
                        full = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_transpose(dst=full, data=scaled[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=diagnostic_rows[:, ti, ci * 128:(ci + 1) * 128], src=full)
            for ti in nl.affine_range(token_tile // 128):
                rows = nl.ds(first + ti * 128, 128)
                nisa.dma_copy(dst=result[rows, :], src=output_rows[:, ti, :])
                if diagnostic:
                    nisa.dma_copy(dst=down_stage[rows, :], src=diagnostic_rows[:, ti, :])
        if diagnostic:
            return result, saved_relu, saved_x, saved_hidden, up_stage, down_stage, x_metadata, h_metadata
        return result, saved_relu, saved_x, saved_hidden

    return forward
