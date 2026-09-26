"""Tiny implementation model, never a production fallback or a host launcher.

This file deliberately does not pretend a Python loop is a device metadata
producer. Native construction fails closed until that missing component exists.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Metadata:
    ids: tuple
    counts: tuple
    starts: tuple
    positions: tuple
    maximums: tuple
    group: int


def metadata_model(ids, vocab, *, tile=4, group=4):
    """Bounded synthetic stable-counting model of the proposed device stages."""
    ids = tuple(ids)  # Ownership boundary: no loader alias survives.
    if not (0 < vocab <= 256 and 0 <= len(ids) <= 512 and tile > 0 and group > 0):
        raise ValueError('TINY_MODEL_ONLY')
    if any(type(x) is not int or not 0 <= x < vocab for x in ids):
        raise ValueError('INVALID_ID')
    counts = [0] * vocab
    prefixes = []
    for begin in range(0, len(ids), tile):
        prefixes.append(tuple(counts))
        for row in ids[begin:begin + tile]:
            counts[row] += 1
    starts, total = [], 0
    for count in counts:
        starts.append(total)
        total += count
    positions = [-1] * len(ids)
    for begin in range(0, len(ids), tile):
        for local, row in enumerate(ids[begin:begin + tile]):
            ordinal = sum(x == row for x in ids[begin:begin + local])
            destination = starts[row] + prefixes[begin // tile][row] + ordinal
            if positions[destination] != -1:
                raise RuntimeError('DUPLICATE_ROUTE')
            positions[destination] = begin + local
    result = Metadata(ids, tuple(counts), tuple(starts), tuple(positions),
                      tuple(max(counts[i:i + group]) for i in range(0, vocab, group)), group)
    validate_model(result)
    return result


def validate_model(meta):
    """Every unsafe read in the tiny model follows this full semantic check."""
    vocab, tokens = len(meta.counts), len(meta.ids)
    if len(meta.starts) != vocab or len(meta.positions) != tokens or meta.group <= 0:
        raise ValueError('METADATA_SHAPE')
    if any(type(x) is not int or not 0 <= x < vocab for x in meta.ids):
        raise ValueError('INVALID_ID')
    total = 0
    for row, count in enumerate(meta.counts):
        if type(count) is not int or not 0 <= count <= tokens or meta.starts[row] != total:
            raise ValueError('COUNTS_STARTS')
        expected = tuple(t for t, value in enumerate(meta.ids) if value == row)
        if count != len(expected) or meta.positions[total:total + count] != expected:
            raise ValueError('POSITION_ORDER_OR_OWNER')
        total += count
    maximums = tuple(max(meta.counts[i:i + meta.group]) for i in range(0, vocab, meta.group))
    if total != tokens or meta.maximums != maximums:
        raise ValueError('TOTAL_OR_MAXIMUM')


def consume_model(gradient, meta):
    """Tiny BF16 tangent -> fresh dense FP32 result with per-call recurrence."""
    import torch
    validate_model(meta)
    if gradient.device.type != 'cpu' or gradient.dtype != torch.bfloat16:
        raise ValueError('TINY_CPU_BF16_ONLY')
    if gradient.ndim != 2 or gradient.shape[0] != len(meta.ids) or gradient.numel() > 4096:
        raise ValueError('TINY_GRADIENT_SHAPE')
    result = torch.zeros((len(meta.counts), gradient.shape[1]), dtype=torch.float32)
    for first in range(0, len(meta.counts), meta.group):
        for occurrence in range(meta.maximums[first // meta.group]):
            for row in range(first, min(first + meta.group, len(meta.counts))):
                if occurrence < meta.counts[row]:
                    token = meta.positions[meta.starts[row] + occurrence]
                    result[row] = result[row] + gradient[token].float()
                # No synthetic inactive-lane addition, multiply, or clamping.
    return result


def build_native(*, enabled=False):
    if not enabled:
        raise RuntimeError('EXPERIMENT_DISABLED')
    raise RuntimeError('PRODUCTION_METADATA_VALIDATION_AND_NKI_COMPOSITION_UNIMPLEMENTED')


def production_cost():
    """Integer-only estimates; does not allocate production-sized tensors."""
    return {
        'tables': 2, 'tokens': 32768, 'vocab': 8192, 'channels': 1024,
        'snapshot_int64_bytes': 32768 * 8,
        'metadata_bytes': (8192 * 2 + 32768 + 64) * 4,
        'prefix_scratch_bytes': 8192 * 256 * 4,
        'count_comparisons': 8192 * 32768,
        'ordinal_comparisons': 256 * 128 * 128,
        'count_loop_bodies': 64 * 256,
        'accumulator_per_owner_bytes': 128 * 512 * 4,
        'obvious_fp32_live_tiles_per_owner_bytes': 4 * 128 * 512 * 4,
        'two_table_useful_bf16_tangent_bytes': 2 * 32768 * 1024 * 2,
        'two_table_dense_output_bytes': 2 * 8192 * 1024 * 4,
        'padded_adds_per_table_min': 128 * 1024 * 256,
        'padded_adds_per_table_max': 128 * 1024 * 32768,
        'unskipped_bf16_payload_per_table_max': 128 * 1024 * 32768 * 2,
        'cold_compile_seconds': None, 'steady_seconds': None,
    }
