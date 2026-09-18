"""One-kernel embedding backward: indirect loads, resetting scan, bounded gather.

No intermediate HBM permutation/scan tensors are needed in normal execution.
The output is the ordinary dense FP32 gradient; input rounding matches the
gated native BF16 policy. Explicit disjoint SBUF addresses and512-output gathers
retain the safeguards discovered in the earlier valid-input gather failure.
Debug intermediates and the uninstrumented kernel both require independent gates.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def fused_gradient(gradient, permutation, continuation, mapping,
                       vocabulary: int, lanes: int = 2, debug: bool = False,
                       contiguous_metadata: bool = False):
        tokens, channels = gradient.shape
        assert 0 < tokens <= 8192 and tokens % 128 == 0
        assert channels % (128 * lanes) == 0 and vocabulary % 128 == 0
        assert gradient.dtype == nl.float32 and permutation.shape == (tokens, 1)
        assert continuation.shape == (1, tokens)
        assert mapping.shape == ((2, vocabulary) if contiguous_metadata else (vocabulary, 2))
        vt = min(vocabulary, 512)
        assert vocabulary % vt == 0
        base = 10 * tokens + 1024
        assert base + (18 if contiguous_metadata else 14) * vt <= nl.tile_size.sbuf_fmax_bytes
        result = nl.ndarray((vocabulary, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        if debug:
            ordered_debug = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
            scanned_debug = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        flags = nl.ndarray((128, tokens), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, 0))
        ordered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf,
                             address=(0, 2 * tokens))
        scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf,
                             address=(0, 6 * tokens))
        rows = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf,
                          address=(0, 10 * tokens))
        rounded_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf,
                                  address=(0, 10 * tokens + 512))
        indices = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf,
                             address=(0, 10 * tokens + 768))
        ends = nl.ndarray((128, vt), dtype=nl.uint32, buffer=nl.sbuf, address=(0, base))
        if contiguous_metadata:
            valid_raw = nl.ndarray((128, vt), dtype=nl.int32, buffer=nl.sbuf,
                                   address=(0, base + 4 * vt))
        valid = nl.ndarray((128, vt), dtype=nl.bfloat16, buffer=nl.sbuf,
                           address=(0, base + (8 if contiguous_metadata else 4) * vt))
        sums = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf,
                          address=(0, base + (10 if contiguous_metadata else 6) * vt))
        masked = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf,
                            address=(0, base + (14 if contiguous_metadata else 10) * vt))
        nisa.dma_copy(dst=flags, src=continuation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        for ci in nl.sequential_range(channels // (128 * lanes)):
            c0 = (ci * lanes + nl.program_id(0)) * 128
            for ti in nl.sequential_range(tokens // 128):
                nisa.dma_copy(dst=indices, src=permutation[ti * 128:(ti + 1) * 128, :])
                nisa.dma_copy(dst=rows,
                              src=gradient.ap(pattern=[[channels, 128], [1, 128]], offset=c0,
                                              vector_offset=indices, indirect_dim=0),
                              dge_mode=nisa.dge_mode.swdge)
                nisa.tensor_copy(dst=rounded_rows, src=rows)
                if debug:
                    nisa.dma_copy(dst=ordered_debug[ti * 128:(ti + 1) * 128, c0:c0 + 128],
                                  src=rounded_rows)
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed, data=rounded_rows)
                nisa.tensor_copy(dst=ordered[:, ti * 128:(ti + 1) * 128], src=transposed)
            nisa.tensor_tensor_scan(dst=scanned, data0=flags, data1=ordered,
                                    initial=0.0, op0=nl.multiply, op1=nl.add)
            if debug:
                for ti in nl.sequential_range(tokens // 128):
                    scan_t = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=scan_t, data=scanned[:, ti * 128:(ti + 1) * 128])
                    nisa.tensor_copy(dst=rows, src=scan_t)
                    nisa.dma_copy(dst=scanned_debug[ti * 128:(ti + 1) * 128, c0:c0 + 128], src=rows)
            for vi in nl.sequential_range(vocabulary // vt):
                if contiguous_metadata:
                    # The small device trace found131072 scalar metadata
                    # packets in the interleaved/casting path below. Transfer
                    # each contiguous column at its original32-bit width, then
                    # cast the presence flag in SBUF.
                    nisa.dma_copy(dst=ends, src=mapping.ap(pattern=[[0, 128], [1, vt]],
                                                          offset=vi * vt))
                    nisa.dma_copy(dst=valid_raw, src=mapping.ap(pattern=[[0, 128], [1, vt]],
                                                               offset=vocabulary + vi * vt))
                    nisa.tensor_copy(dst=valid, src=valid_raw)
                else:
                    nisa.dma_copy(dst=ends, src=mapping.ap(pattern=[[0, 128], [2, vt]],
                                                          offset=vi * vt * 2))
                    nisa.dma_copy(dst=valid, src=mapping.ap(pattern=[[0, 128], [2, vt]],
                                                           offset=vi * vt * 2 + 1))
                nisa.nc_n_gather(dst=sums, data=scanned, indices=ends)
                nisa.tensor_tensor(dst=masked, data1=sums, data2=valid, op=nl.multiply)
                for ti in nl.sequential_range(vt // 128):
                    output_t = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=output_t, data=masked[:, ti * 128:(ti + 1) * 128])
                    nisa.tensor_copy(dst=rows, src=output_t)
                    nisa.dma_copy(dst=result[vi * vt + ti * 128:vi * vt + (ti + 1) * 128,
                                            c0:c0 + 128], src=rows)
        if debug:
            return result, ordered_debug, scanned_debug
        return result

    return fused_gradient
