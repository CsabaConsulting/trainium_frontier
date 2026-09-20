"""Online legacy-E4M3 row quantization for the pinned library's ROW input.

The library consumes one FP32 dequantization scale appended as four raw bytes
to each FP8 token row. Weight rows use the same quantizer before the caller's
explicit transpose and per-output scale broadcast. No corpus calibration or
cached learned values enter this producer. Every invocation reads its input.

Return metadata as [reciprocal scale, dequantization scale, absmax] for the
independent gate. Its stores are part of any reported producer timing.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def quantize_rows(values, pack_input: bool = False):
    rows, columns = values.shape
    assert values.dtype == nl.bfloat16
    assert rows % 256 == 0 and columns % 128 == 0 and 128 <= columns <= 4096
    width = columns + 4 if pack_input else columns
    output = nl.ndarray((rows, width), dtype=nl.uint8, buffer=nl.shared_hbm)
    metadata = nl.ndarray((rows, 3), dtype=nl.float32, buffer=nl.shared_hbm)
    for block in nl.affine_range(rows // 256):
        first = (block * 2 + nl.program_id(0)) * 128
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
        # Match the pinned library ROW intermediate quantizer, including its
        # zero/tiny floor. Legacy E4M3 has maximum finite magnitude240.
        nisa.tensor_scalar(dst=dequant, data=maximum, op0=nl.multiply,
                           operand0=1.0 / 240.0, op1=nl.maximum, operand1=1e-6)
        nisa.reciprocal(dst=reciprocal, data=dequant)
        nisa.activation(dst=quantized, data=source, op=nl.copy, scale=reciprocal)
        nisa.dma_copy(dst=output[row, :columns], src=quantized.view(nl.uint8))
        if pack_input:
            # The consumer loads the very same FP32 access pattern. This is
            # a byte-preserving HBM view, never a float-to-uint8 conversion.
            nisa.dma_copy(dst=output.ap(dtype=nl.float32,
                pattern=[[columns // 4 + 1, 128], [1, 1]],
                offset=first * (columns // 4 + 1) + columns // 4), src=dequant)
        nisa.dma_copy(dst=metadata[row, 0:1], src=reciprocal)
        nisa.dma_copy(dst=metadata[row, 1:2], src=dequant)
        nisa.dma_copy(dst=metadata[row, 2:3], src=maximum)
    return output, metadata
