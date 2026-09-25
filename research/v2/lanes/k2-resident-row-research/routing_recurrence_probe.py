"""Tiny stdlib mathematical probe, not an NKI kernel or production CPU gate.

No Torch, candidate import, training data, network, device, or model allocation.
FP32 rounding is explicit; the reference is an ordered scalar recurrence, not
an execution of PyTorch's dense embedding backward.
"""
import hashlib
import json
import random
import struct
import sys
import tracemalloc


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def raw(rows):
    return b''.join(struct.pack('<f', x) for row in rows for x in row)


def metadata(ids, vocabulary, tile):
    """Literal count/prefix/ordinal construction; all positions are unique."""
    blocks = (len(ids) + tile - 1) // tile
    prefix = [[0] * blocks for _ in range(vocabulary)]
    counts = [0] * vocabulary
    for block in range(blocks):
        tokens = ids[block * tile:(block + 1) * tile]
        for row in range(vocabulary):
            prefix[row][block] = counts[row]
            counts[row] += sum(token == row for token in tokens)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    positions = [-1] * len(ids)
    for block in range(blocks):
        tokens = ids[block * tile:(block + 1) * tile]
        for local, row in enumerate(tokens):
            ordinal = sum(token == row for token in tokens[:local])
            dest = offsets[row] + prefix[row][block] + ordinal
            require(positions[dest] == -1, 'duplicate route destination')
            positions[dest] = block * tile + local
    require(positions == sorted(range(len(ids)), key=lambda t: (ids[t], t)), 'stable route')
    require(sorted(positions) == list(range(len(ids))), 'permutation')
    return counts, offsets, positions


def token_reference(ids, gradients, vocabulary, channels):
    result = [[0.0] * channels for _ in range(vocabulary)]
    for row, gradient in zip(ids, gradients):
        result[row] = [fp32(a + g) for a, g in zip(result[row], gradient)]
    return result


def resident_reference(ids, gradients, vocabulary, channels, tile=128, group=8):
    counts, offsets, positions = metadata(ids, vocabulary, tile)
    result = [[0.0] * channels for _ in range(vocabulary)]
    visits, rounds = [], 0
    for first in range(0, vocabulary, group):
        rows = range(first, min(first + group, vocabulary))
        accumulators = [[0.0] * channels for _ in rows]
        iterations = max((counts[row] for row in rows), default=0)
        rounds += iterations
        for k in range(iterations):
            for lane, row in enumerate(rows):
                if k < counts[row]:
                    position = positions[offsets[row] + k]
                    require(ids[position] == row, 'wrong row')
                    visits.append((row, position))
                    accumulators[lane] = [fp32(a + g) for a, g in
                                          zip(accumulators[lane], gradients[position])]
                # An inactive real row is not updated, even by synthetic +0.
        for lane, row in enumerate(rows):
            result[row] = accumulators[lane]
    for row in range(vocabulary):
        require([p for r, p in visits if r == row] ==
                [p for p, r in enumerate(ids) if r == row], 'token sequence')
    return result, rounds


def grouped_reference(ids, gradients, vocabulary, channels, tile=128):
    result = [[0.0] * channels for _ in range(vocabulary)]
    for first in range(0, len(ids), tile):
        part = token_reference(ids[first:first + tile], gradients[first:first + tile], vocabulary, channels)
        result = [[fp32(a + b) for a, b in zip(row, sub)] for row, sub in zip(result, part)]
    return result


def main():
    tracemalloc.start()
    rng = random.Random(42)
    channels, vocabulary = 8, 256
    cases = []
    for name, values in [('review_large', [2**24, 1, -2**24]),
                         ('review_unit', [1, 2**-24, -1])]:
        ids, grads = [7] * 256, [[0.0] * channels for _ in range(256)]
        for position, value in zip((0, 128, 129), values):
            ids[position], grads[position] = 3, [float(value)] * channels
        cases.append((name, ids, grads))
    for name, ids in [('unique', list(range(256))), ('all_same', [3] * 384),
                      ('interleaved', [i % 11 for i in range(384)]),
                      ('random', [rng.randrange(vocabulary) for _ in range(384)]),
                      ('partial_tile', [i % 13 for i in range(257)]),
                      ('signed_zero', [i % 3 for i in range(17)]),
                      ('bf16_subnormal', [3] * 17), ('empty', [])]:
        grads = [[bf16(rng.randint(-1024, 1024) / 128) for _ in range(channels)] for _ in ids]
        if name == 'signed_zero':
            grads = [[-0.0 if (i + c) % 2 else 0.0 for c in range(channels)] for i in range(len(ids))]
        if name == 'bf16_subnormal':
            grads = [[(-1 if (i + c) % 2 else 1) * 2**-133 for c in range(channels)] for i in range(len(ids))]
        cases.append((name, ids, grads))
    results = []
    for name, ids, grads in cases:
        expected = token_reference(ids, grads, vocabulary, channels)
        actual, rounds = resident_reference(ids, grads, vocabulary, channels)
        require(raw(actual) == raw(expected), name + ': bytes')
        grouped_mismatch = None
        cancellation_values = None
        if name.startswith('review_'):
            wrong = grouped_reference(ids, grads, vocabulary, channels)
            grouped_mismatch = raw(wrong) != raw(expected)
            require(grouped_mismatch, 'retained counterexample disappeared')
            cancellation_values = {'resident': actual[3][0], 'dense_recurrence': expected[3][0],
                                   'old_grouped': wrong[3][0]}
        results.append({'case': name, 'T': len(ids), 'byte_exact_recurrence': True,
                        'output_sha256': hashlib.sha256(raw(actual)).hexdigest(),
                        'row_group8_rounds': rounds, 'old_grouped_mismatch': grouped_mismatch,
                        'cancellation_values': cancellation_values})
    # Analytic production-sized frequency vectors only, not gradients/models/data.
    T, V, C, P = 32768, 8192, 1024, 128
    patterns = {'balanced_four_per_row': [4] * V, 'one_hot_row': [T] + [0] * (V - 1)}
    bounds = {}
    for name, counts in patterns.items():
        Q = sum(max(counts[i:i + P]) for i in range(0, V, P))
        require(T // P <= Q <= T, 'round bound')
        bounds[name] = {'Q': Q, 'active_fraction': T / (P * Q),
                        'padded_additions': P * C * Q,
                        'padded_bf16_gradient_bytes_without_skip': 2 * P * C * Q}
    key = 8191 * T
    require(fp32(key) == fp32(key + 1), 'FP32 composite-key collision')
    _, peak = tracemalloc.get_traced_memory()
    require(peak < 4 * 1024**2, 'small-MB allocation bound')
    print(json.dumps({'scope': 'stdlib scalar mathematics only; not Torch/NKI/source/native verification',
                      'python': sys.version.split()[0], 'seed': 42, 'cases': results,
                      'production_analytic_bounds': bounds, 'metadata_prefix_bytes': V * (T // 128) * 4,
                      'persistent_metadata_bytes': 4 * (V + V + 1 + T),
                      'dense_output_bytes_per_table': 4 * V * C,
                      'useful_bf16_gradient_bytes_per_table': 2 * T * C,
                      'peak_tracemalloc_bytes': peak, 'fp32_composite_key_collision': [key, key + 1],
                      'candidate_implementation': False, 'native_acceptance': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
