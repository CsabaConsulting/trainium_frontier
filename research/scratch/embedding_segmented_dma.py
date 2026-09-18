"""Segmented embedding backward without nc_n_gather.

The nc_n_gather reproducer fails for valid large indices on this installed stack;
unchanged dependent index reads and explicit scratch reuse do not repair it.
This alternative uses the already-gated indirect-DMA pattern to permute input
gradients in HBM, scans regular contiguous SBUF tiles, then gathers group ends
with indirect DMA. Intermediate reads/writes and every invocation are charged.
BF16 rounding of each input gradient matches native production matmul lowering;
the scan accumulates in FP32 and returns the usual FP32 dense weight gradient.
"""
import torch

from embedding_compact_probe import build_gather
from embedding_segmented_nki import metadata


def dma_metadata(ids, vocabulary):
    permutation, continuation, last, present = metadata(ids, vocabulary)
    mapping = torch.stack((last.reshape(-1).int(), present.reshape(-1).int()), dim=1)
    return permutation.reshape(-1, 1), continuation, mapping


def build_permutation():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def permute_gradient(gradient, permutation, lanes: int = 2):
        tokens, channels = gradient.shape
        assert tokens % 128 == 0 and channels % lanes == 0
        assert permutation.shape == (tokens, 1) and gradient.dtype == nl.float32
        width = channels // lanes
        c0 = nl.program_id(0) * width
        output = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for ti in nl.affine_range(tokens // 128):
            indices = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            values = nl.ndarray((128, width), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=indices, src=permutation[ti * 128:(ti + 1) * 128, :])
            nisa.dma_copy(dst=values,
                          src=gradient.ap(pattern=[[channels, 128], [1, width]], offset=c0,
                                          vector_offset=indices, indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
            nisa.dma_copy(dst=output[ti * 128:(ti + 1) * 128, c0:c0 + width], src=values)
        return output

    return permute_gradient


def build_scan():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def segmented_scan(ordered, continuation, lanes: int = 2):
        tokens, channels = ordered.shape
        assert tokens <= 8192 and channels % (128 * lanes) == 0
        assert continuation.shape == (1, tokens) and ordered.dtype == nl.bfloat16
        output = nl.ndarray((tokens, channels), dtype=nl.float32, buffer=nl.shared_hbm)
        flags = nl.ndarray((128, tokens), dtype=nl.bfloat16, buffer=nl.sbuf)
        raw = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
        scanned = nl.ndarray((128, tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=flags, src=continuation.ap(pattern=[[0, 128], [1, tokens]], offset=0))
        for ci in nl.sequential_range(channels // (128 * lanes)):
            c0 = (ci * lanes + nl.program_id(0)) * 128
            nisa.dma_copy(dst=raw, src=ordered.ap(pattern=[[1, 128], [channels, tokens]], offset=c0))
            nisa.tensor_tensor_scan(dst=scanned, data0=flags, data1=raw,
                                    initial=0.0, op0=nl.multiply, op1=nl.add)
            nisa.dma_copy(dst=output.ap(pattern=[[1, 128], [channels, tokens]], offset=c0), src=scanned)
        return output

    return segmented_scan


def build_kernels():
    return build_permutation(), build_scan(), build_gather()
