"""Transpose opaque bytes through an exact BF16 integer carrier on Trn2.

Every uint8 integer0..255 is exactly representable in BF16. Numeric conversion
to that carrier avoids asking the framework to transpose a byte-sized tensor.
The TensorEngine transpose and conversion back must preserve all256 byte codes,
including byte patterns that would mean NaN if incorrectly interpreted as FP8.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def transpose_bytes(source):
    rows, columns = source.shape
    assert source.dtype == nl.uint8 and rows % 128 == 0 and columns % 256 == 0
    width = min(512, rows)
    assert rows % width == 0
    output = nl.ndarray((columns, rows), dtype=nl.uint8, buffer=nl.shared_hbm)
    for column_block in nl.affine_range(columns // 256):
        column = nl.ds((column_block * 2 + nl.program_id(0)) * 128, 128)
        for row_group in nl.affine_range(rows // width):
            packed = nl.ndarray((128, width), dtype=nl.uint8, buffer=nl.sbuf)
            for sub in nl.affine_range(width // 128):
                row = nl.ds(row_group * width + sub * 128, 128)
                raw = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.sbuf)
                carrier = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.dma_copy(dst=raw, src=source[row, column])
                nisa.tensor_copy(dst=carrier, src=raw)
                nisa.nc_transpose(dst=transposed, data=carrier)
                nisa.tensor_copy(dst=packed[:, sub * 128:(sub + 1) * 128], src=transposed)
            nisa.dma_copy(dst=output[column, nl.ds(row_group * width, width)], src=packed)
    return output
