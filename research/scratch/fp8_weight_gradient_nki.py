"""FP8 weight-gradient stages with explicit scale transfer and long contraction.

This is a new quantized backward algorithm, not the previous BF16 surrogate.
Row-scale transfer forms BF16(G * inverse_hidden_scale); one current-gradient
scale then factors outside the token contraction. Both products return FP32.
No autograd integration or free saved-forward-byte assumption is installed.
"""


def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def multiply_rows(values, factors, lanes: int = 2):
        rows, columns = values.shape
        assert values.dtype == nl.bfloat16 and factors.dtype == nl.float32
        assert rows % (128 * lanes) == 0 and factors.shape == (rows, 1)
        assert lanes in (1, 2) and columns <= 4096
        output = nl.ndarray(values.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for tile in nl.affine_range(rows // (128 * lanes)):
            row = nl.ds((tile * lanes + nl.program_id(0)) * 128, 128)
            raw = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            scale = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            result = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=values[row, :])
            nisa.dma_copy(dst=scale, src=factors[row, :])
            nisa.tensor_scalar(dst=result, data=raw, op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=output[row, :], src=result)
        return output

    @nki.jit
    def quantize_rows(values, metadata, lanes: int = 2, per_row: bool = False):
        """Return row-major legacy FP8 bytes with no interleaving or transpose."""
        rows, columns = values.shape
        assert values.dtype == nl.bfloat16 and metadata.dtype == nl.float32
        assert metadata.shape == ((rows, 3) if per_row else (128, 3))
        assert lanes in (1, 2) and rows % (128 * lanes) == 0 and columns <= 4096
        output = nl.ndarray(values.shape, dtype=nl.uint8, buffer=nl.shared_hbm)
        target = output.view(nl.float8_e4m3)
        common = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        if not per_row:
            nisa.dma_copy(dst=common, src=metadata)
        for tile in nl.affine_range(rows // (128 * lanes)):
            row = nl.ds((tile * lanes + nl.program_id(0)) * 128, 128)
            raw = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            result = nl.ndarray((128, columns), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            if per_row:
                nisa.dma_copy(dst=common, src=metadata[row, :])
            nisa.dma_copy(dst=raw, src=values[row, :])
            nisa.tensor_scalar(dst=result, data=raw, op0=nl.multiply, operand0=common[:, 0:1])
            nisa.dma_copy(dst=target[row, :], src=result)
        return output

    @nki.jit
    def weight_product(left_bytes, right_bytes, left_metadata, right_metadata, lanes: int = 2):
        """[T,M].T @ [T,N], double-row FP8 with sequential inverse scaling.

        The contraction pairs corresponding positions in adjacent128-token
        blocks, matching the independently characterized11-bit pair rule.
        Metadata scales are global; right_metadata is unity after transferring
        a per-token activation inverse scale into the left gradient operand.
        """
        tokens, outputs = left_bytes.shape
        other_tokens, inputs = right_bytes.shape
        assert left_bytes.dtype == right_bytes.dtype == nl.uint8
        assert tokens == other_tokens and tokens % 256 == 0
        assert outputs % (128 * lanes) == 0 and inputs % 256 == 0
        assert lanes in (1, 2)
        assert left_metadata.shape == right_metadata.shape == (128, 3)
        assert left_metadata.dtype == right_metadata.dtype == nl.float32
        width = min(512, inputs)
        assert inputs % width == 0
        left, right = left_bytes.view(nl.float8_e4m3), right_bytes.view(nl.float8_e4m3)
        output = nl.ndarray((outputs, inputs), dtype=nl.float32, buffer=nl.shared_hbm)
        left_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        right_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=left_meta, src=left_metadata)
        nisa.dma_copy(dst=right_meta, src=right_metadata)
        for row_tile in nl.affine_range(outputs // (128 * lanes)):
            rows = nl.ds((row_tile * lanes + nl.program_id(0)) * 128, 128)
            for column_tile in nl.affine_range(inputs // width):
                columns = nl.ds(column_tile * width, width)
                accumulator = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(tokens // 256):
                    lhs = nl.ndarray((128, 2, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                    rhs = nl.ndarray((128, 2, width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                    for half in nl.static_range(2):
                        contraction = nl.ds((pair * 2 + half) * 128, 128)
                        nisa.dma_copy(dst=lhs[:, half, :], src=left[contraction, rows])
                        nisa.dma_copy(dst=rhs[:, half, :], src=right[contraction, columns])
                    nisa.nc_matmul(dst=accumulator, stationary=lhs, moving=rhs,
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                first_scale = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
                result = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=first_scale, data=accumulator, op0=nl.multiply,
                                   operand0=left_meta[:, 1:2])
                nisa.tensor_scalar(dst=result, data=first_scale, op0=nl.multiply,
                                   operand0=right_meta[:, 1:2])
                nisa.dma_copy(dst=output[rows, columns], src=result)
        return output

    return multiply_rows, quantize_rows, weight_product
