"""Prepacked BF16 MLP with explicit, reusable SBUF storage.

At8192x1024x4096/tile512 the declared SBUF extent is7.125MiB per physical
core. Five disjoint buffers cover the input, activation, streamed weights,
row transfers and rounded temporary. Each is one logical tensor reused by
sequential outer loops; views share that tensor rather than relying on
different allocations aliasing the same address. PSUM remains compiler-managed.
No claim about the caller's allocation follows until a full-model measurement.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def fused_mlp(x, up, down, lanes: int = 2, token_tile: int = 512,
                  contiguous_weights: bool = True, tiled_io: bool = True,
                  transposed_relu: bool = True, grouped_io: bool = True,
                  weight_mode: int = 4):
        tokens, channels = x.shape
        hidden = up.shape[1] * 128
        nc, nh = channels // 128, hidden // 128
        assert weight_mode == 4 and lanes in (1, 2)
        assert contiguous_weights and tiled_io and transposed_relu and grouped_io
        assert x.dtype == nl.bfloat16 and up.dtype == nl.bfloat16 and down.dtype == nl.bfloat16
        assert channels % 128 == 0 and hidden >= channels
        assert up.shape == (128, nh, nc, 128) and down.shape == (128, nc, nh, 128)
        assert token_tile in (128, 256, 512) and tokens % (token_tile * lanes) == 0
        # Addresses are byte offsets within each of the128 SBUF partitions.
        activation_base = 2 * nc * token_tile
        weight_base = activation_base + 2 * nh * token_tile
        rows_base = weight_base + 2 * nh * 128
        temporary_base = rows_base + 2 * (token_tile // 128) * channels
        extent = temporary_base + 2 * token_tile
        assert 128 * extent + 2 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        inputs = nl.ndarray((128, nc, token_tile), dtype=nl.bfloat16,
                             buffer=nl.sbuf, address=(0, 0))
        activated = nl.ndarray((128, nh, token_tile), dtype=nl.bfloat16,
                                buffer=nl.sbuf, address=(0, activation_base))
        weights = nl.ndarray((128, nh, 128), dtype=nl.bfloat16,
                              buffer=nl.sbuf, address=(0, weight_base))
        rows = nl.ndarray((128, token_tile // 128, channels), dtype=nl.bfloat16,
                           buffer=nl.sbuf, address=(0, rows_base))
        temporary = nl.ndarray((128, token_tile), dtype=nl.bfloat16,
                                buffer=nl.sbuf, address=(0, temporary_base))
        transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
        accumulator = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
        output = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved = nl.ndarray((hidden, tokens), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for mi in nl.sequential_range(tokens // (token_tile * lanes)):
            m0 = (mi * lanes + nl.program_id(0)) * token_tile
            for ti in nl.sequential_range(token_tile // 128):
                nisa.dma_copy(dst=rows[:, 0, :], src=x[m0 + ti * 128:m0 + (ti + 1) * 128, :])
                for ci in nl.sequential_range(nc):
                    nisa.nc_transpose(dst=transposed, data=rows[:, 0, ci * 128:(ci + 1) * 128])
                    nisa.tensor_copy(dst=inputs[:, ci, ti * 128:(ti + 1) * 128], src=transposed)
            for hi in nl.sequential_range(nh):
                nisa.dma_copy(dst=weights[:, 0:nc, :], src=up[:, hi, :, :])
                for ci in nl.affine_range(nc):
                    nisa.nc_matmul(dst=accumulator, stationary=weights[:, ci, :],
                                   moving=inputs[:, ci, :], accumulate=(ci > 0))
                nisa.tensor_copy(dst=temporary, src=accumulator)
                nisa.tensor_scalar(dst=temporary, data=temporary, op0=nl.maximum, operand0=0.0)
                nisa.tensor_tensor(dst=activated[:, hi, :], data1=temporary, data2=temporary,
                                   op=nl.multiply)
                nisa.dma_copy(dst=saved[hi * 128:(hi + 1) * 128, m0:m0 + token_tile], src=temporary)
            for ci in nl.sequential_range(nc):
                nisa.dma_copy(dst=weights, src=down[:, ci, :, :])
                for hi in nl.affine_range(nh):
                    nisa.nc_matmul(dst=accumulator, stationary=weights[:, hi, :],
                                   moving=activated[:, hi, :], accumulate=(hi > 0))
                nisa.tensor_copy(dst=temporary, src=accumulator)
                for ti in nl.sequential_range(token_tile // 128):
                    nisa.nc_transpose(dst=transposed, data=temporary[:, ti * 128:(ti + 1) * 128])
                    nisa.tensor_copy(dst=rows[:, ti, ci * 128:(ci + 1) * 128], src=transposed)
            for ti in nl.sequential_range(token_tile // 128):
                nisa.dma_copy(dst=output[m0 + ti * 128:m0 + (ti + 1) * 128, :], src=rows[:, ti, :])
        return output, saved

    return fused_mlp
