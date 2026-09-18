"""Fuse permutation/scan with wide indirect DMA, retaining fast DMA expansion.

Each LNC2 program owns a contiguous half of channels. It loads128xwidth rows,
casts on chip, and transposes128x128 channel tiles into one complete sorted
scratch tensor. A reusable FP32 scan tile is copied back into that scratch,
then wide contiguous stores write the scan output. No nc_n_gather is used.
The final dense expansion stays a separate independently gated DMA kernel.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def permute_scan(gradient, permutation, continuation, lanes: int = 2,
                     debug: bool = False):
        tokens, channels = gradient.shape
        assert gradient.dtype == nl.float32
        assert 0 < tokens <= 8192 and tokens % 128 == 0
        assert channels % (128 * lanes) == 0
        width = channels // lanes
        assert width <= 512 and lanes in (1, 2)
        assert permutation.shape == (tokens, 1) and continuation.shape == (1, tokens)
        groups = width // 128
        base = (6 + 4 * groups) * tokens
        assert base + 6 * width + 4 <= nl.tile_size.sbuf_fmax_bytes
        c0 = nl.program_id(0) * width
        result = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        if debug:
            ordered_debug = nl.ndarray((tokens, channels), dtype=nl.bfloat16,
                                       buffer=nl.shared_hbm)
        flags = nl.ndarray((128, tokens), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, 0))
        ordered = nl.ndarray((128, groups, tokens), dtype=nl.float32, buffer=nl.sbuf,
                             address=(0, 2 * tokens))
        scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf,
                             address=(0, (2 + 4 * groups) * tokens))
        rows = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf, address=(0, base))
        rounded = nl.ndarray((128, width), dtype=nl.bfloat16, buffer=nl.sbuf,
                             address=(0, base + 4 * width))
        indices = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf,
                             address=(0, base + 6 * width))
        nisa.dma_copy(dst=flags, src=continuation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        for ti in nl.sequential_range(tokens // 128):
            nisa.dma_copy(dst=indices, src=permutation[ti * 128:(ti + 1) * 128, :])
            nisa.dma_copy(dst=rows,
                          src=gradient.ap(pattern=[[channels, 128], [1, width]], offset=c0,
                                          vector_offset=indices, indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
            nisa.tensor_copy(dst=rounded, src=rows)
            if debug:
                nisa.dma_copy(dst=ordered_debug[ti * 128:(ti + 1) * 128, c0:c0 + width],
                              src=rounded)
            for ci in nl.sequential_range(groups):
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed, data=rounded[:, ci * 128:(ci + 1) * 128])
                nisa.tensor_copy(dst=ordered[:, ci, ti * 128:(ti + 1) * 128], src=transposed)
        for ci in nl.sequential_range(groups):
            nisa.tensor_tensor_scan(dst=scanned, data0=flags, data1=ordered[:, ci, :],
                                    initial=0.0, op0=nl.multiply, op1=nl.add)
            # Explicit disjoint scan destination. Reuse the sorted input space
            # only after the complete recurrence finishes; no in-place ISA
            # aliasing assumption is required.
            nisa.tensor_copy(dst=ordered[:, ci, :], src=scanned)
        for ti in nl.sequential_range(tokens // 128):
            for ci in nl.sequential_range(groups):
                transposed = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed,
                                  data=ordered[:, ci, ti * 128:(ti + 1) * 128])
                nisa.tensor_copy(dst=rows[:, ci * 128:(ci + 1) * 128], src=transposed)
            nisa.dma_copy(dst=result[ti * 128:(ti + 1) * 128, c0:c0 + width], src=rows)
        if debug:
            return result, ordered_debug
        return result

    return permute_scan
