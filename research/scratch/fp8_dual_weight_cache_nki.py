"""Calibrate once and produce both exact legacy-FP8 weight orientations.

The backward-packed orientation takes the original quantized rows directly;
only the forward orientation needs a partition transpose. No HBM transpose of
the BF16 weight is produced. Both outputs retain the established byte layouts.
Each physical core repeats the same finite-input maximum reduction, then owns
disjoint row blocks. Only core zero writes the common metadata.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def prepare(weight, lanes: int = 2, resident: bool = True):
        rows, columns = weight.shape
        fw_width, bw_width = min(512, rows), min(512, columns)
        fw_groups, bw_groups = rows // fw_width, columns // bw_width
        row_blocks, column_blocks = rows // 128, columns // 128
        assert weight.dtype == nl.bfloat16
        assert rows % 128 == 0 and columns % 128 == 0
        assert rows <= 4096 and columns <= 4096
        assert rows % fw_width == 0 and columns % bw_width == 0
        assert lanes in (1, 2) and fw_groups % lanes == 0
        # The current up/down matrices are at most8MiB of BF16 storage. This
        # guards the proposed resident layout, not unrelated native operators.
        assert not resident or rows * columns * 2 <= 8 * 1024**2

        restored_hbm = nl.ndarray(weight.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        metadata_hbm = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        forward_hbm = nl.ndarray((fw_groups, 128, column_blocks, fw_width),
                                 dtype=nl.uint8, buffer=nl.shared_hbm)
        backward_hbm = nl.ndarray((bw_groups, 128, row_blocks, bw_width),
                                  dtype=nl.uint8, buffer=nl.shared_hbm)
        if resident:
            cache = nl.ndarray((128, row_blocks, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
        local_maxima = nl.ndarray((128, row_blocks), dtype=nl.float32, buffer=nl.sbuf)
        for row in nl.affine_range(row_blocks):
            if resident:
                raw = cache[:, row, :]
            else:
                raw = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw, src=weight[row * 128:(row + 1) * 128, :])
            magnitudes = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(
                dst=magnitudes, data=raw, op0=nl.abs, operand0=0.0,
                reduce_op=nl.maximum, reduce_res=local_maxima[:, nl.ds(row, 1)])

        maxima = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        transposed_maxima = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        flat = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
        maximum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        clamped = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        exponent = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        scale = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        inverse = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=maxima, data=local_maxima, op=nl.maximum, axis=(1,))
        nisa.nc_transpose(dst=transposed_maxima, data=maxima)
        nisa.tensor_copy(dst=flat, src=transposed_maxima)
        nisa.tensor_reduce(dst=maximum, data=flat, op=nl.maximum, axis=(1,))
        nisa.tensor_scalar(dst=clamped, data=maximum, op0=nl.maximum,
                           operand0=7.888609052210118e-31)
        nisa.tensor_scalar(dst=exponent, data=clamped.view(nl.uint32),
                           op0=nl.bitwise_and, operand0=2139095040)
        nisa.tensor_scalar(dst=scale, data=exponent, op0=nl.multiply,
                           operand0=-1.0, op1=nl.add, operand1=2181038080.0)
        nisa.tensor_scalar(dst=inverse, data=exponent, op0=nl.subtract, operand0=50331648.0)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        for quadrant in nl.static_range(4):
            first_partition = quadrant * 32
            nisa.nc_stream_shuffle(dst=meta[first_partition:first_partition + 32, 0:1],
                                   src=scale.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=meta[first_partition:first_partition + 32, 1:2],
                                   src=inverse.view(nl.float32), shuffle_mask=[0] * 32)
            nisa.nc_stream_shuffle(dst=meta[first_partition:first_partition + 32, 2:3],
                                   src=maximum, shuffle_mask=[0] * 32)
        if nl.program_id(0) == 0:
            nisa.dma_copy(dst=metadata_hbm, src=meta)

        for group_tile in nl.affine_range(fw_groups // lanes):
            group = group_tile * lanes + nl.program_id(0)
            packed = nl.ndarray((128, column_blocks, fw_width),
                                dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for sub in nl.affine_range(fw_width // 128):
                block_row = group * (fw_width // 128) + sub
                first = block_row * 128
                if resident:
                    source = cache[:, block_row, :]
                else:
                    source = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=source, src=weight[first:first + 128, :])
                quantized = nl.ndarray((128, columns), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                restored = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=quantized, data=source, op0=nl.multiply,
                                   operand0=meta[:, 0:1])
                nisa.tensor_scalar(dst=restored, data=quantized, op0=nl.multiply,
                                   operand0=meta[:, 1:2])
                nisa.dma_copy(dst=restored_hbm[first:first + 128, :], src=restored)
                raw_bytes = quantized.view(nl.uint8)
                for backward_group in nl.affine_range(bw_groups):
                    nisa.dma_copy(
                        dst=backward_hbm[backward_group, :, block_row, :],
                        src=raw_bytes[:, backward_group * bw_width:(backward_group + 1) * bw_width])
                for column_block in nl.affine_range(column_blocks):
                    transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                        data=quantized[:, column_block * 128:(column_block + 1) * 128])
                    nisa.tensor_copy(
                        dst=packed[:, column_block, sub * 128:(sub + 1) * 128],
                        src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            nisa.dma_copy(dst=forward_hbm[group, :, :, :], src=packed.view(nl.uint8))
        return restored_hbm, metadata_hbm, forward_hbm, backward_hbm

    return prepare
