"""Small, explicit FP32 recurrence and synthetic routing; no Torch/NKI imports.

Routing here is fixture preparation ONLY, not a production device implementation.
"""
import math
import random
import struct

V, C, T, P = 256, 128, 256, 128
CASE_NAMES = ('cross_tile_large', 'cross_tile_unit', 'unique', 'all_same',
              'counts_253_1_1_1', 'boundary_rows', 'signed_zero', 'bf16_subnormal',
              'fp32_subnormal', 'empty')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def raw(rows):
    return b''.join(struct.pack('<f', x) for row in rows for x in row)


def metadata(ids):
    require(len(ids) <= T, 'token bound')
    buckets = [[] for _ in range(V)]
    for token, row in enumerate(ids):
        require(type(row) is int and 0 <= row < V, 'invalid token ID')
        buckets[row].append(token)
    counts, starts, positions = [], [], []
    for bucket in buckets:
        counts.append(len(bucket))
        starts.append(len(positions))
        positions.extend(bucket)
    maximums = [max(counts[first:first + P]) for first in range(0, V, P)]
    positions.extend([T] * (T - len(positions)))
    result = (counts, starts, positions, maximums)
    validate_metadata(ids, result)
    return result


def validate_metadata(ids, meta):
    counts, starts, positions, maximums = meta
    require(len(counts) == len(starts) == V and len(positions) == T and len(maximums) == V // P, 'metadata shapes')
    seen, offset = [], 0
    for row in range(V):
        require(type(counts[row]) is int and counts[row] >= 0, 'invalid count')
        require(starts[row] == offset, 'invalid row offset')
        segment = positions[offset:offset + counts[row]]
        require(segment == [i for i, value in enumerate(ids) if value == row], 'unstable or incorrect route')
        seen.extend(segment)
        offset += counts[row]
    require(offset == len(ids) and sorted(seen) == list(range(len(ids))), 'route permutation')
    require(positions[offset:] == [T] * (T - offset), 'padding sentinel')
    require(maximums == [max(counts[first:first + P]) for first in range(0, V, P)], 'group loop bounds')


def dense(ids, gradients):
    require(len(ids) == len(gradients) <= T, 'input length')
    result = [[0.0] * C for _ in range(V)]
    for row, gradient in zip(ids, gradients):
        require(0 <= row < V and len(gradient) == C, 'shape or ID')
        for channel, value in enumerate(gradient):
            require(math.isfinite(value) and fp32(value) == value, 'finite FP32 input required')
            result[row][channel] = fp32(result[row][channel] + value)
    return result


def consumer(ids, gradients, meta):
    """CPU model of loop bounds, double gather, inactive skip and resident state."""
    validate_metadata(ids, meta)
    counts, starts, positions, maximums = meta
    out = [[0.0] * C for _ in range(V)]
    trace = dict(group_rounds=maximums, active_gathers=0, inactive_gathers_skipped=0,
                 output_rows_stored=0, accumulator_hbm_reads=0, visits=[])
    for group, maximum in enumerate(maximums):
        first = group * P
        accum = [[0.0] * C for _ in range(P)]
        pointers = starts[first:first + P].copy()
        remaining = counts[first:first + P].copy()
        for _ in range(maximum):
            for lane in range(P):
                active = remaining[lane] > 0
                address = pointers[lane] if active else T
                token = positions[address] if address < T else T
                token = token if active else T
                if active:
                    require(token < len(gradients) and ids[token] == first + lane, 'invalid gather')
                    trace['visits'].append((first + lane, token))
                    trace['active_gathers'] += 1
                    accum[lane] = [fp32(a + g) for a, g in zip(accum[lane], gradients[token])]
                else:
                    # Exact selection of previous state, not multiplication by zero.
                    trace['inactive_gathers_skipped'] += 1
                pointers[lane] += 1
                remaining[lane] -= 1
        for lane in range(P):
            out[first + lane] = accum[lane]
            trace['output_rows_stored'] += 1
    return out, trace


def grouped_wrong(ids, gradients):
    result = [[0.0] * C for _ in range(V)]
    for start in range(0, len(ids), P):
        part = dense(ids[start:start + P], gradients[start:start + P])
        result = [[fp32(a + b) for a, b in zip(row, tile)] for row, tile in zip(result, part)]
    return result


def fixture(name, dtype='bf16'):
    require(name in CASE_NAMES and dtype in ('bf16', 'fp32'), 'unknown fixture')
    rng = random.Random(42)
    ids = [i % 11 for i in range(T)]
    if name == 'unique':
        ids = list(range(T))
    elif name == 'all_same':
        ids = [255] * T
    elif name == 'counts_253_1_1_1':
        ids = [3] * 253 + [0, 127, 255]
    elif name == 'boundary_rows':
        ids = [0, 255] * 128
    elif name == 'empty':
        ids = []
    gradients = [[fp32(rng.randint(-256, 256) / 32) for _ in range(C)] for _ in ids]
    if name.startswith('cross_tile_'):
        ids = [7] * T
        gradients = [[0.0] * C for _ in ids]
        values = (2.0**24, 1.0, -2.0**24) if name == 'cross_tile_large' else (1.0, 2.0**-24, -1.0)
        for token, value in zip((0, 128, 129), values):
            ids[token] = 3
            gradients[token] = [value] * C
    if name == 'signed_zero':
        gradients = [[-0.0 if (token + channel) % 2 else 0.0 for channel in range(C)] for token in range(T)]
    if name.endswith('subnormal'):
        value = 2.0**(-133 if name == 'bf16_subnormal' else -149)
        gradients = [[value if (token + channel) % 3 else -value for channel in range(C)] for token in range(T)]
    cast = bf16 if dtype == 'bf16' else fp32
    gradients = [[cast(g) for g in row] for row in gradients]
    return ids, gradients, metadata(ids)
