"""Prepare a BF16 backward cache and packed legacy-FP8 forward cache together.

The uint8 HBM tensor stores raw legacy-E4M3 bytes; it is never interpreted as
PyTorch's differently encoded E4M3FN. Its layout is [group, K-inner, K-block,
output-channel]. Prepare it once per optimizer update and charge its creation.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def prepare(weight, metadata, lanes: int = 2):
        """BF16 [R,C] -> BF16 QDQ [R,C], packed uint8 [G,128,C/128,W].

        W=min(512,R), G=R/W. The caller selects one physical core for G=1,
        otherwise two when G is even. Calibration is the existing separate
        online, per-matrix operation and is not excluded from update timing.
        """
        rows, columns = weight.shape
        width = min(512, rows)
        groups, blocks = rows // width, columns // 128
        assert weight.dtype == nl.bfloat16
        assert rows % width == 0 and width % 128 == 0
        assert columns % 128 == 0 and columns <= 4096
        assert lanes in (1, 2) and groups % lanes == 0
        assert metadata.shape == (128, 3) and metadata.dtype == nl.float32
        restored_hbm = nl.ndarray(weight.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        packed_hbm = nl.ndarray((groups, 128, blocks, width), dtype=nl.uint8,
                                buffer=nl.shared_hbm)
        meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=meta, src=metadata)
        for group_tile in nl.affine_range(groups // lanes):
            group = group_tile * lanes + nl.program_id(0)
            packed = nl.ndarray((128, blocks, width), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for sub in nl.affine_range(width // 128):
                first = group * width + sub * 128
                raw = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
                quantized = nl.ndarray((128, columns), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                restored = nl.ndarray((128, columns), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=raw, src=weight[first:first + 128, :])
                nisa.tensor_scalar(dst=quantized, data=raw, op0=nl.multiply, operand0=meta[:, 0:1])
                nisa.tensor_scalar(dst=restored, data=quantized, op0=nl.multiply, operand0=meta[:, 1:2])
                nisa.dma_copy(dst=restored_hbm[first:first + 128, :], src=restored)
                for block in nl.affine_range(blocks):
                    # NC-v3 FP8 transpose requires a two-byte free stride in
                    # PSUM; compact the bytes only after the transpose.
                    transposed = nl.ndarray((128, 256), dtype=nl.float8_e4m3, buffer=nl.psum)
                    nisa.nc_transpose(dst=transposed.ap(pattern=[[256, 128], [2, 128]]),
                                      data=quantized[:, block * 128:(block + 1) * 128])
                    nisa.tensor_copy(dst=packed[:, block, sub * 128:(sub + 1) * 128],
                                     src=transposed.ap(pattern=[[256, 128], [2, 128]]))
            # Bitcast, not numeric float-to-integer conversion. Each partition
            # writes a contiguous C/128*W-byte region once the group is ready.
            nisa.dma_copy(dst=packed_hbm[group, :, :, :], src=packed.view(nl.uint8))
        return restored_hbm, packed_hbm

    return prepare
