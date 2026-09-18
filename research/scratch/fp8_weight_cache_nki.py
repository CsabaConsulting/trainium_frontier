"""Online power-of-two calibration and legacy-E4M3 weight-cache preparation.

Calibration is device work and must be included in any cache-refresh timing.
The exponent-bit policy places a positive normal maximum in [64,128), with
headroom below legacy E4M3's240. This avoids log2/floor/exp2, not the required
maximum reduction. BF16 cache outputs let the same quantized weights be used
by a declared straight-through backward and by the storage-compression control.
"""


def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def calibrate(weight):
        """Return repeated [scale,inverse,amax] metadata for one BF16 matrix.

        Args:
            weight: Finite BF16 [R,C], R divisible128, C<=4096. The current
                probe contract requires amax<=2**100; zero is allowed.
        Returns:
            FP32 [128,3] metadata, identical on every partition.
        Notes:
            Launch on one physical core within the existing LNC2 environment.
            Clamp maxima below2**-100 only for the scale policy.
        """
        rows, cols = weight.shape
        assert weight.dtype == nl.bfloat16 and rows % 128 == 0 and cols <= 4096
        output = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        maxima = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=maxima, value=0.0)
        for row in nl.affine_range(rows // 128):
            tile = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            high = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            low = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile, src=weight[row * 128:(row + 1) * 128, :])
            # Two reductions avoid materializing a full FP32 abs tile.
            nisa.tensor_reduce(dst=high, data=tile, op=nl.maximum, axis=(1,))
            nisa.tensor_reduce(dst=low, data=tile, op=nl.minimum, axis=(1,))
            nisa.tensor_scalar(dst=low, data=low, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=high, data1=high, data2=low, op=nl.maximum)
            nisa.tensor_tensor(dst=maxima, data1=maxima, data2=high, op=nl.maximum)
        transposed = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        flat = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
        maximum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=transposed, data=maxima)
        nisa.tensor_copy(dst=flat, src=transposed)
        nisa.tensor_reduce(dst=maximum, data=flat, op=nl.maximum, axis=(1,))
        nisa.tensor_scalar(dst=clamped, data=maximum, op0=nl.maximum,
                           operand0=7.888609052210118e-31)  #2**-100
        nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                           op0=nl.bitwise_and, operand0=2139095040)
        # b is the biased exponent. scale exponent=133-b and inverse=b-133.
        # Encoded integer values are multiples of2**23, so these FP32 ALU
        # operations are exact over the declared range before the uint32 cast.
        nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.multiply,
                           operand0=-1.0, op1=nl.add, operand1=2181038080.0)
        nisa.tensor_scalar(dst=inverse, data=exponent,
                           op0=nl.subtract, operand0=50331648.0)
        metadata = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        for quadrant in nl.static_range(4):
            start = quadrant * 32
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 0:1],
                                   src=scale.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 1:2],
                                   src=inverse.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=metadata[start:start + 32, 2:3],
                                   src=maximum, shuffle_mask=[0] * 32)
        nisa.dma_copy(dst=output, src=metadata)
        return output

    @nki.jit
    def quantize_dequantize(weight, metadata, lanes: int = 2):
        """Prepare BF16 values that exactly represent scaled legacy E4M3.

        Args:
            weight: BF16 [R,C], R divisible128*lanes.
            metadata: FP32 [128,3] from calibrate.
            lanes: Matching physical-core launch grid.
        Returns:
            BF16 [R,C] quantized/dequantized weights, retaining matrix layout.
        """
        rows, cols = weight.shape
        assert weight.dtype == nl.bfloat16 and rows % (128 * lanes) == 0
        assert metadata.shape == (128, 3) and metadata.dtype == nl.float32
        assert lanes in (1, 2) and cols <= 4096
        output = nl.ndarray(weight.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=meta, src=metadata)
        for tile in nl.affine_range(rows // (128 * lanes)):
            first = (tile * lanes + nl.program_id(0)) * 128
            source = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            quantized = nl.ndarray((128, cols), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            result = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=source, src=weight[first:first + 128, :])
            nisa.tensor_scalar(dst=quantized, data=source, op0=nl.multiply, operand0=meta[:, 0:1])
            nisa.tensor_scalar(dst=result, data=quantized, op0=nl.multiply, operand0=meta[:, 1:2])
            nisa.dma_copy(dst=output[first:first + 128, :], src=result)
        return output

    return calibrate, quantize_dequantize
