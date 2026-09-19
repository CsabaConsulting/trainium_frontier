"""Two-core exact current-gradient calibration and legacy-FP8 byte preparation.

First produce small per-core maxima. A second kernel combines those maxima,
repeats the established exponent-bit policy and quantizes the original input.
Optional row-scale transfer is recomputed in SBUF on both passes, retaining
its BF16 rounding boundary while eliminating the full Gprime HBM temporary.
The kernel boundary provides completion ordering; there is no cross-core race
or unsynchronized read of a maximum still being produced.
"""


def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def partial_maximum(values, factors, lanes: int = 2, transfer_rows: bool = False):
        rows, cols = values.shape
        assert values.dtype == nl.bfloat16 and factors.dtype == nl.float32
        assert lanes == 2 and rows % (128 * lanes) == 0 and cols <= 4096
        assert not transfer_rows or factors.shape == (rows, 1)
        tiles = rows // (128 * lanes)
        assert tiles <= 512
        output = nl.ndarray((128, lanes), dtype=nl.float32, buffer=nl.shared_hbm)
        maxima = nl.ndarray((128, tiles), dtype=nl.float32, buffer=nl.sbuf)
        for tile in nl.affine_range(tiles):
            row = nl.ds((tile * lanes + nl.program_id(0)) * 128, 128)
            raw = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=values[row, :])
            if transfer_rows:
                scale = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                prepared = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=scale, src=factors[row, :])
                nisa.tensor_scalar(dst=prepared, data=raw, op0=nl.multiply, operand0=scale)
            else:
                prepared = raw
            magnitude = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=magnitude, data=prepared, op0=nl.abs,
                                      operand0=0.0, reduce_op=nl.maximum,
                                      reduce_res=maxima[:, nl.ds(tile, 1)])
        maximum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=maximum, data=maxima, op=nl.maximum, axis=(1,))
        nisa.dma_copy(dst=output[:, nl.ds(nl.program_id(0), 1)], src=maximum)
        return output

    @nki.jit
    def finish_and_quantize(values, factors, partials, lanes: int = 2,
                            transfer_rows: bool = False):
        rows, cols = values.shape
        assert values.dtype == nl.bfloat16 and factors.dtype == nl.float32
        assert lanes == 2 and rows % (128 * lanes) == 0 and cols <= 4096
        assert not transfer_rows or factors.shape == (rows, 1)
        assert partials.shape == (128, lanes) and partials.dtype == nl.float32
        output = nl.ndarray(values.shape, dtype=nl.uint8, buffer=nl.shared_hbm)
        metadata_out = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        target = output.view(nl.float8_e4m3)
        local = nl.ndarray((128, lanes), dtype=nl.float32, buffer=nl.sbuf)
        maxima = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        transposed = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        flat = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
        maximum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale_bits = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse_bits = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=local, src=partials)
        nisa.tensor_reduce(dst=maxima, data=local, op=nl.maximum, axis=(1,))
        nisa.nc_transpose(dst=transposed, data=maxima)
        nisa.tensor_copy(dst=flat, src=transposed)
        nisa.tensor_reduce(dst=maximum, data=flat, op=nl.maximum, axis=(1,))
        nisa.tensor_scalar(dst=clamped, data=maximum, op0=nl.maximum,
                           operand0=7.888609052210118e-31)
        nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                           op0=nl.bitwise_and, operand0=2139095040)
        nisa.tensor_scalar(dst=scale_bits, data=exponent, op0=nl.multiply,
                           operand0=-1.0, op1=nl.add, operand1=2181038080.0)
        nisa.tensor_scalar(dst=inverse_bits, data=exponent,
                           op0=nl.subtract, operand0=50331648.0)
        metadata = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        for quadrant in nl.static_range(4):
            start = quadrant * 32
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 0:1],
                                   src=scale_bits.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 1:2],
                                   src=inverse_bits.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 2:3],
                                   src=maximum, shuffle_mask=[0] * 32)
        # Both cores use identical global metadata, but only one writes it.
        if nl.program_id(0) == 0:
            nisa.dma_copy(dst=metadata_out, src=metadata)
        for tile in nl.affine_range(rows // (128 * lanes)):
            row = nl.ds((tile * lanes + nl.program_id(0)) * 128, 128)
            raw = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            result = nl.ndarray((128, cols), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=values[row, :])
            if transfer_rows:
                factor = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                prepared = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=factor, src=factors[row, :])
                nisa.tensor_scalar(dst=prepared, data=raw, op0=nl.multiply, operand0=factor)
            else:
                prepared = raw
            nisa.tensor_scalar(dst=result, data=prepared, op0=nl.multiply,
                               operand0=metadata[:, 0:1])
            nisa.dma_copy(dst=target[row, :], src=result)
        return output, metadata_out

    return partial_maximum, finish_and_quantize
