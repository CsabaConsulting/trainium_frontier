"""Fuse ROW quantization and transposed weight layout without an HBM temporary.

The direct FP8 transpose uses the two-byte PSUM stride from AWS NKI Library
92d11f63/core/mlp/mlp_cte/basic/mlp_cte_basic_transpose.py. A second implementation
uses the already validated exact BF16 integer carrier as a diagnostic control.
Both retain source179's arithmetic and return its unchanged audit metadata.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def quantize_transposed_rows(values, direct_fp8: bool = True):
    rows, columns = values.shape
    assert values.dtype == nl.bfloat16
    assert rows % 256 == 0 and columns % 128 == 0 and 128 <= columns <= 4096
    group_rows = min(512, rows // 2)
    assert rows % (group_rows * 2) == 0 and group_rows % 128 == 0
    output = nl.ndarray((columns, rows), dtype=nl.uint8, buffer=nl.shared_hbm)
    metadata = nl.ndarray((rows, 3), dtype=nl.float32, buffer=nl.shared_hbm)
    for block in nl.affine_range(rows // (group_rows * 2)):
        group_start = (block * 2 + nl.program_id(0)) * group_rows
        # Gather adjacent source-row transposes before a contiguous DMA store.
        packed = nl.ndarray((128, columns // 128, group_rows),
                            dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for sub in nl.affine_range(group_rows // 128):
            first = group_start + sub * 128
            row = nl.ds(first, 128)
            source = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            absolute = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
            maximum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            dequant = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            reciprocal = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            quantized = nl.ndarray((128, columns), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.dma_copy(dst=source, src=values[row, :])
            nisa.tensor_scalar_reduce(dst=absolute, data=source, op0=nl.abs,
                                      operand0=0.0, reduce_op=nl.maximum, reduce_res=maximum)
            nisa.tensor_scalar(dst=dequant, data=maximum, op0=nl.multiply,
                               operand0=1.0 / 240.0, op1=nl.maximum, operand1=1e-6)
            nisa.reciprocal(dst=reciprocal, data=dequant)
            nisa.activation(dst=quantized, data=source, op=nl.copy, scale=reciprocal)
            for column_block in nl.affine_range(columns // 128):
                column = nl.ds(column_block * 128, 128)
                if direct_fp8:
                    # One-byte PE transpose uses every second PSUM byte.
                    transposed = nl.ndarray((128, 128, 2), dtype=nl.float8_e4m3, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed.ap([[256, 128], [2, 128]]),
                                      data=quantized[:, column])
                    nisa.tensor_copy(dst=packed[:, column_block, sub * 128:(sub + 1) * 128],
                                     src=transposed[:, :, 0], engine=nisa.vector_engine)
                else:
                    carrier = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                    transposed_carrier = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.tensor_copy(dst=carrier, src=quantized.view(nl.uint8)[:, column])
                    nisa.nc_transpose(dst=transposed_carrier, data=carrier)
                    nisa.tensor_copy(dst=packed.view(nl.uint8)[:, column_block, sub * 128:(sub + 1) * 128],
                                     src=transposed_carrier)
            nisa.dma_copy(dst=metadata[row, 0:1], src=reciprocal)
            nisa.dma_copy(dst=metadata[row, 1:2], src=dequant)
            nisa.dma_copy(dst=metadata[row, 2:3], src=maximum)
        for column_block in nl.affine_range(columns // 128):
            nisa.dma_copy(dst=output[nl.ds(column_block * 128, 128), nl.ds(group_start, group_rows)],
                          src=packed.view(nl.uint8)[:, column_block, :])
    return output, metadata
