"""First backward FP8 stage: per-row gradient quantization and packed-weight dot.

Computes BF16(round FP8(gradient) @ FP8(weight).T) with two sequential inverse
scales. This changes the gradient approximation; it is not the existing BF16
surrogate backward. The separate probe defines and checks the numerical policy.
No autograd integration or full-FP8 training claim is made here.
"""


def build_kernel():
    global nl, nisa, kernel_assert
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def kernel_assert(condition, message):
        assert condition, f'[NCC_INKI016] Kernel validation exception: {message}'

    @nki.jit
    def gradient_gemm(gradient, packed_weight, weight_metadata,
                      lanes: int = 2, diagnostic: bool = False):
        """Return [tokens, outputs]; optionally QDQ gradient, scales and FP32 dot.

        packed_weight is the wave84-gated uint8 legacy-E4M3 representation of
        [outputs, contraction]. For G@Wdown, pack Wdown.T once per update.
        Every row uses only its current incoming-gradient values for scale.
        Declared finite input domain has max magnitude <=2^100; the probe
        also checks zero, subnormal and a wide range of per-row exponents.
        """
        tokens, contraction = gradient.shape
        groups, partition, blocks, width = packed_weight.shape
        outputs = groups * width
        kernel_assert(gradient.dtype == nl.bfloat16, 'BF16 incoming gradient required')
        kernel_assert(packed_weight.dtype == nl.uint8, 'Packed legacy FP8 bytes required')
        kernel_assert(lanes in (1, 2) and tokens % (128 * lanes) == 0, 'Invalid token tiling')
        kernel_assert(contraction % 256 == 0 and blocks * 128 == contraction,
                      'Contraction must match double-row packed layout')
        kernel_assert(partition == 128 and width in (256, 512), 'Unsupported packed group width')
        kernel_assert(weight_metadata.shape == (128, 3) and weight_metadata.dtype == nl.float32,
                      'Use the gated online per-weight scale metadata')
        kernel_assert(contraction * outputs + 8 * contraction * 128 + 4 * 1024**2
                      <= nl.tile_size.sbuf_size_bytes, 'SBUF working set exceeds declared limit')
        result = nl.ndarray((tokens, outputs), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if diagnostic:
            saved_gradient = nl.ndarray(gradient.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
            saved_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
            dot = nl.ndarray((tokens, outputs), dtype=nl.float32, buffer=nl.shared_hbm)
        weights = nl.ndarray((128, groups, blocks, width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        raw_weights = packed_weight.view(nl.float8_e4m3)
        for group in nl.affine_range(groups):
            nisa.dma_copy(dst=weights[:, group, :, :], src=raw_weights[group, :, :, :])
        wm = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wm, src=weight_metadata)
        for tile in nl.affine_range(tokens // (128 * lanes)):
            first = (tile * lanes + nl.program_id(0)) * 128
            rows = nl.ds(first, 128)
            raw = nl.ndarray((128, contraction), dtype=nl.bfloat16, buffer=nl.sbuf)
            high = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            low = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            clamped = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            exponent = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            scale = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            inverse = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=gradient[rows, :])
            nisa.tensor_reduce(dst=high, data=raw, op=nl.maximum, axis=(1,))
            nisa.tensor_reduce(dst=low, data=raw, op=nl.minimum, axis=(1,))
            nisa.tensor_scalar(dst=low, data=low, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=high, data1=high, data2=low, op=nl.maximum)
            nisa.tensor_scalar(dst=clamped, data=high, op0=nl.maximum, operand0=2.0**-100)
            nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                               op0=nl.bitwise_and, operand0=2139095040)
            nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.multiply,
                               operand0=-1.0, op1=nl.add, operand1=2181038080.0)
            nisa.tensor_scalar(dst=inverse, data=exponent, op0=nl.subtract, operand0=50331648.0)
            quantized = nl.ndarray((128, contraction), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=quantized, data=raw, op0=nl.multiply, operand0=scale.view(nl.float32))
            transposed = nl.ndarray((128, blocks, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for block in nl.affine_range(blocks):
                padded = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                nisa.nc_transpose(dst=padded.ap(pattern=[[256, 128], [2, 128]]),
                                  data=quantized[:, block * 128:(block + 1) * 128])
                nisa.tensor_copy(dst=transposed[:, block, :],
                                 src=padded.ap(pattern=[[256, 128], [2, 128]]))
            for group in nl.affine_range(groups):
                columns = nl.ds(group * width, width)
                accumulator = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.psum)
                for pair in nl.affine_range(blocks // 2):
                    nisa.nc_matmul(dst=accumulator, stationary=transposed[:, pair * 2:pair * 2 + 2, :],
                                   moving=weights[:, group, pair * 2:pair * 2 + 2, :],
                                   accumulate=(pair > 0), perf_mode=nisa.matmul_perf_mode.double_row)
                weighted = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
                scaled = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
                rounded = nl.ndarray((128, width), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=weighted, data=accumulator, op0=nl.multiply, operand0=wm[:, 1:2])
                nisa.tensor_scalar(dst=scaled, data=weighted, op0=nl.multiply, operand0=inverse.view(nl.float32))
                nisa.tensor_copy(dst=rounded, src=scaled)
                nisa.dma_copy(dst=result[rows, columns], src=rounded)
                if diagnostic:
                    nisa.dma_copy(dst=dot[rows, columns], src=scaled)
            if diagnostic:
                restored = nl.ndarray((128, contraction), dtype=nl.bfloat16, buffer=nl.sbuf)
                meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=restored, data=quantized, op0=nl.multiply, operand0=inverse.view(nl.float32))
                nisa.tensor_copy(dst=meta[:, 0:1], src=scale.view(nl.float32))
                nisa.tensor_copy(dst=meta[:, 1:2], src=inverse.view(nl.float32))
                nisa.tensor_copy(dst=meta[:, 2:3], src=high)
                nisa.dma_copy(dst=saved_gradient[rows, :], src=restored)
                nisa.dma_copy(dst=saved_metadata[rows, :], src=meta)
        if diagnostic:
            return result, saved_gradient, saved_metadata, dot
        return result

    return gradient_gemm
