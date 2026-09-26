"""Ordered BF16 embedding backward. Metadata is produced in k2.py on device."""


def build_kernel(*, enabled=False):
    global nl, nisa
    if not enabled:
        raise RuntimeError('EXPERIMENT_DISABLED: no native module imported')
    import nki
    import nki.isa as nisa
    import nki.language as nl

    # Fixed production geometry. No flag, shape fallback, or CPU-ID route.
    vocab, tokens, channels = 8192, 32768, 1024
    partition, features, groups = 128, 512, 64

    @nki.jit
    def resident_consumer(gradient, counts, starts, positions, group_maximums):
        assert gradient.shape == (tokens, channels)
        assert counts.shape == starts.shape == (vocab, 1)
        assert positions.shape == (tokens, 1)
        assert group_maximums.shape == (groups, 1)
        assert gradient.dtype == nl.bfloat16
        assert counts.dtype == starts.dtype == positions.dtype == group_maximums.dtype == nl.int32
        assert nl.num_programs() == 2
        feature_first = nl.program_id(0) * features
        output = nl.ndarray((vocab, channels), dtype=nl.float32, buffer=nl.shared_hbm)

        # Explicitly reusable, bounded scratch. nl.where creates at most one
        # live selected FP32 tile plus small address tiles; compiler liveness
        # and allocation remain unproved until native compilation.
        remaining = nl.ndarray((partition, 1), dtype=nl.int32, buffer=nl.sbuf)
        pointer = nl.ndarray((partition, 1), dtype=nl.int32, buffer=nl.sbuf)
        maximum = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        accumulator = nl.ndarray((partition, features), dtype=nl.float32, buffer=nl.sbuf)
        active = nl.ndarray((partition, 1), dtype=nl.float32, buffer=nl.sbuf)
        token = nl.ndarray((partition, 1), dtype=nl.int32, buffer=nl.sbuf)
        value = nl.ndarray((partition, features), dtype=nl.float32, buffer=nl.sbuf)
        updated = nl.ndarray((partition, features), dtype=nl.float32, buffer=nl.sbuf)
        sentinel = nl.full((partition, 1), tokens, dtype=nl.int32, buffer=nl.sbuf)
        limit = nisa.register_alloc(0)

        # 64 bounded static row-group bodies, each containing a runtime loop.
        # Neither T nor the per-row occurrence count is statically unrolled.
        # Both programs traverse identical groups and supplied loop limits.
        for group in range(groups):
            first = group * partition
            nisa.dma_copy(dst=remaining, src=counts[first:first + partition, :])
            nisa.dma_copy(dst=pointer, src=starts[first:first + partition, :])
            nisa.dma_copy(dst=maximum, src=group_maximums[group:group + 1, :])
            nisa.register_load(limit, maximum)
            nisa.memset(dst=accumulator, value=0.0)
            for _ in nl.dynamic_range(limit):
                nisa.tensor_scalar(dst=active, data=remaining, op0=nl.greater, operand0=0.0,
                                   engine=nisa.engine.vector)
                address = nl.where(active, pointer, sentinel, dtype=nl.int32)
                # Sentinel and scratch initialization are repeated each round:
                # skipped DMA lanes must not expose preceding tokens/values.
                nisa.memset(dst=token, value=tokens)
                nisa.dma_copy(dst=token,
                              src=positions.ap(pattern=[[1, partition], [1, 1]],
                                               vector_offset=address, indirect_dim=0),
                              oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
                token_address = nl.where(active, token, sentinel, dtype=nl.int32)
                nisa.memset(dst=value, value=0.0)
                # DMA widens actual BF16 tangent elements into FP32. No cast
                # down from FP32 is accepted at this callable's boundary.
                nisa.dma_copy(dst=value,
                              src=gradient.ap(pattern=[[channels, partition], [1, features]],
                                              offset=feature_first,
                                              vector_offset=token_address, indirect_dim=0),
                              oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
                nisa.tensor_tensor(dst=updated, data1=accumulator, data2=value,
                                   op=nl.add, engine=nisa.engine.vector)
                selected = nl.where(active.ap(pattern=[[1, partition], [0, features]]),
                                    updated, accumulator, dtype=nl.float32)
                nisa.tensor_copy(dst=accumulator, src=selected)
                nisa.tensor_scalar(dst=pointer, data=pointer, op0=nl.add, operand0=1.0,
                                   engine=nisa.engine.vector)
                nisa.tensor_scalar(dst=remaining, data=remaining, op0=nl.add, operand0=-1.0,
                                   engine=nisa.engine.vector)
            # A fresh dense output per call; untouched rows store positive zero.
            # Owner0 writes c[0:512], owner1 writes c[512:1024].
            nisa.dma_copy(dst=output[first:first + partition,
                                     feature_first:feature_first + features], src=accumulator)
        return output

    return resident_consumer[2]
