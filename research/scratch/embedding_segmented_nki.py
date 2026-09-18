"""Experimental FP32 ordinary-embedding gradient, installed NKI 0.5.

Stable-sort metadata depends only on input token IDs and can be shared by WTE
and the three unigram value embeddings. The kernel loads/transposes gradients,
gathers into stable token order, resets a scan at each new token, and gathers
the final sum for each vocabulary row. No duplicate HBM scatter writes occur.
Metadata creation/upload must be charged; this is not integrated into training.

Scan arithmetic is FP32, which can differ from a BF16-autocast dense lowering.
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor_scan.html
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_n_gather.html
"""
import numpy as np
import torch


def metadata(indices, vocabulary):
    """Compact CPU metadata; sorting preserves original order within each ID."""
    values = indices.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    assert len(values) and values.min() >= 0 and values.max() < vocabulary
    order = np.argsort(values, kind="stable").astype(np.int32)
    sorted_ids = values[order]
    continuation = np.zeros(len(order), dtype=np.uint8)
    continuation[1:] = sorted_ids[1:] == sorted_ids[:-1]
    is_end = np.ones(len(order), dtype=bool)
    is_end[:-1] = sorted_ids[:-1] != sorted_ids[1:]
    last = np.zeros(vocabulary, dtype=np.int32)
    last[sorted_ids[is_end]] = np.nonzero(is_end)[0]
    present = np.zeros(vocabulary, dtype=np.uint8)
    present[sorted_ids[is_end]] = 1
    return tuple(torch.from_numpy(a).reshape(1, -1) for a in
                 (order, continuation, last, present))


def cpu_scan(gradient, meta):
    """Literal FP32 recurrence, independent of embedding/index_add lowering."""
    order, continuation, last, present = (x.numpy().reshape(-1) for x in meta)
    sorted_g = gradient.detach().float().cpu().numpy()[order]
    scanned = np.empty_like(sorted_g)
    state = np.zeros(sorted_g.shape[1], dtype=np.float32)
    for i in range(len(order)):
        state = state * continuation[i] + sorted_g[i]
        scanned[i] = state
    return torch.from_numpy(scanned[last] * present[:, None])


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def segmented_gradient(gradient, permutation, continuation, last, present,
                           lanes: int = 2, manual_sbuf: bool = False,
                           debug_intermediates: bool = False, reuse_sbuf: bool = False,
                           aligned_sbuf: bool = False, gather_tile: int = 8192):
        tokens, channels = gradient.shape
        vocabulary = last.shape[1]
        assert gradient.dtype == nl.float32
        assert lanes in (1, 2) and channels % (128 * lanes) == 0
        assert tokens <= 8192 and vocabulary % 128 == 0
        assert permutation.shape == (1, tokens) and continuation.shape == (1, tokens)
        assert present.shape == (1, vocabulary)
        gt = min(tokens, gather_tile)
        assert gt > 0 and tokens % gt == 0
        vt = min(vocabulary, 512)
        assert vocabulary % vt == 0
        # Actual-shape gather diagnostics pass with disjoint explicit scratch
        # placement, unlike automatic allocation. The installed allocator uses
        # byte offsets. Test the full uninstrumented scan, not just its gather.
        assert 18 * tokens + 14 * vt <= nl.tile_size.sbuf_fmax_bytes
        perm_address = (0, 0) if manual_sbuf else None
        cont_address = (0, 4 * tokens) if manual_sbuf else None
        raw_address = (0, 6 * tokens) if manual_sbuf else None
        ordered_address = (0, 10 * tokens) if manual_sbuf else None
        scanned_address = (0, 14 * tokens) if manual_sbuf else None
        ends_address = (0, 18 * tokens) if manual_sbuf else None
        valid_address = (0, 18 * tokens + 4 * vt) if manual_sbuf else None
        sums_address = (0, 18 * tokens + 6 * vt) if manual_sbuf else None
        masked_address = (0, 18 * tokens + 10 * vt) if manual_sbuf else None
        if aligned_sbuf:
            assert manual_sbuf
            # The passing gather reproducer placed every FP32/uint32 tile
            # at a multiple of4*T bytes. The previous mixed-dtype layout put
            # raw/scanned at6*T/14*T, crossing32KiB boundaries atT8192.
            # This reordering tests address alignment without changing live
            # bytes, operations, input values or loop identities.
            raw_address = (0, 0)
            ordered_address = (0, 4 * tokens)
            scanned_address = (0, 8 * tokens)
            perm_address = (0, 12 * tokens)
            cont_address = (0, 16 * tokens)
        result = nl.ndarray((vocabulary, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        if debug_intermediates:
            raw_debug = nl.ndarray((channels, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
            ordered_debug = nl.ndarray((channels, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
            scanned_debug = nl.ndarray((channels, tokens), dtype=nl.float32, buffer=nl.shared_hbm)
        # Compact HBM metadata is broadcast to the partition lanes by zero
        # source stride. This charges only O(T+V) host-to-device bytes.
        lane = nl.program_id(0)
        if reuse_sbuf:
            # Keep the SAME tensor identities across loops as well as fixed
            # addresses. Distinct inside-loop allocations at one address can
            # have different compiler dependency behavior.
            perm = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf, address=perm_address)
            cont = nl.ndarray((128, tokens), dtype=nl.bfloat16, buffer=nl.sbuf, address=cont_address)
            raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=raw_address)
            ordered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=ordered_address)
            scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=scanned_address)
            ends = nl.ndarray((128, vt), dtype=nl.uint32, buffer=nl.sbuf, address=ends_address)
            valid = nl.ndarray((128, vt), dtype=nl.bfloat16, buffer=nl.sbuf, address=valid_address)
            sums = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf, address=sums_address)
            masked = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf, address=masked_address)
        for block in nl.sequential_range(channels // (128 * lanes)):
            c0 = (block * lanes + lane) * 128
            # Reloading alone previously failed; retain the auto-allocation
            # control while optionally assigning every live scratch address.
            if not reuse_sbuf:
                perm = nl.ndarray((128, tokens), dtype=nl.uint32, buffer=nl.sbuf, address=perm_address)
                cont = nl.ndarray((128, tokens), dtype=nl.bfloat16, buffer=nl.sbuf, address=cont_address)
            nisa.dma_copy(dst=perm, src=permutation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
            nisa.dma_copy(dst=cont, src=continuation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
            if not reuse_sbuf:
                raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=raw_address)
                ordered = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=ordered_address)
                scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf, address=scanned_address)
            # Load [T,C] directly into [128,T], including this layout cost.
            nisa.dma_copy(dst=raw,
                          src=gradient.ap(pattern=[[1, 128], [channels, tokens]], offset=c0))
            if debug_intermediates:
                nisa.dma_copy(dst=raw_debug[c0:c0 + 128, :], src=raw)
            # The independent tiled reproducer is exact with512 outputs per
            # gather and fails with larger output widths. Preserve the full
            # source index range while making that boundary explicit here.
            for gi in nl.sequential_range(tokens // gt):
                nisa.nc_n_gather(dst=ordered[:, gi * gt:(gi + 1) * gt], data=raw,
                                 indices=perm[:, gi * gt:(gi + 1) * gt])
            if debug_intermediates:
                nisa.dma_copy(dst=ordered_debug[c0:c0 + 128, :], src=ordered)
            nisa.tensor_tensor_scan(dst=scanned, data0=cont, data1=ordered,
                                    initial=0.0, op0=nl.multiply, op1=nl.add)
            if debug_intermediates:
                nisa.dma_copy(dst=scanned_debug[c0:c0 + 128, :], src=scanned)
            for vi in nl.sequential_range(vocabulary // vt):
                if not reuse_sbuf:
                    ends = nl.ndarray((128, vt), dtype=nl.uint32, buffer=nl.sbuf, address=ends_address)
                    valid = nl.ndarray((128, vt), dtype=nl.bfloat16, buffer=nl.sbuf, address=valid_address)
                    sums = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf, address=sums_address)
                    masked = nl.ndarray((128, vt), dtype=nl.float32, buffer=nl.sbuf, address=masked_address)
                nisa.dma_copy(dst=ends, src=last.ap(pattern=[[0, 128], [1, vt]], offset=vi * vt))
                nisa.dma_copy(dst=valid, src=present.ap(pattern=[[0, 128], [1, vt]], offset=vi * vt))
                nisa.nc_n_gather(dst=sums, data=scanned, indices=ends)
                nisa.tensor_tensor(dst=masked, data1=sums, data2=valid, op=nl.multiply)
                # Each output vocabulary row/channel belongs to exactly one
                # writer. Return the normal [V,C] weight-gradient layout.
                nisa.dma_copy(dst=result.ap(pattern=[[1, 128], [channels, vt]],
                                             offset=c0 + vi * vt * channels), src=masked)
        if debug_intermediates:
            return result, raw_debug, ordered_debug, scanned_debug
        return result

    return segmented_gradient
