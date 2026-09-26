"""Six frozen full-shape recipes; pure scalar access supports tiny local tests."""
V, T, C, P = 8192, 32768, 1024, 128
FIXTURES = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5')
DIRECT_ORDER = ('F0', 'F0', 'F1', 'F2', 'F3', 'F4', 'F1', 'F5')
F2_IDS = {0: 0, 128: 0, 129: 0, 1: 127, 130: 127, 2: 128,
          131: 128, 3: 8191, 132: 8191, 4: 4095, 133: 4095}
F2_VALUES = ({0: 0x4b80, 128: 0x3f80, 129: 0xcb80},
             {0: 0x3f80, 128: 0x3380, 129: 0xbf80},
             {1: 0x0081, 130: 0x8080}, {2: 0x8081, 131: 0x0080},
             {3: 1}, {3: 0x8001}, {4: 0x8000, 133: 0}, {4: 0x8000, 133: 0x8000})
F3_VALUES = ((0x4b80, 0x3f80, 0xcb80, 1), (0x3f80, 0x3380, 0xbf80, 0),
             (1, 0x8001, 1, 1), (0x8001, 1, 0x8001, 0x8001),
             (0x3b80,) * 4, (0xbb80,) * 4, (0x8000, 0, 0x8000, 0),
             (0x3f80, 0xbf80, 0x3f00, 0xbe80))
F5_VALUES = ({0: 0x7fc1}, {127: 0x7f80}, {128: 0xff80},
             {8191: 0x7f80, 16383: 0xff80}, {4095: 0x7f7f, 12287: 0x7f7f},
             {4096: 0xff7f, 12288: 0xff7f}, {0: 0x8000}, {1: 0x3f80})


def token_id(fixture, t):
    if fixture not in FIXTURES or not 0 <= t < T:
        raise ValueError('FIXTURE_COORDINATE')
    if fixture == 'F1':
        j = t % 4096
        return 8191 if j == 4095 else j
    if fixture == 'F2':
        return F2_IDS.get(t, t % V)
    if fixture == 'F3':
        return 8191
    if fixture == 'F4':
        return (3 * (t % V) + 17) % V
    return t % V


def tangent_bits(fixture, t, c):
    if fixture not in FIXTURES or not 0 <= t < T or not 0 <= c < C:
        raise ValueError('FIXTURE_COORDINATE')
    a = c % 8
    if fixture == 'F0':
        return 0x3800 + c
    if fixture == 'F1':
        return (0x3f80, 0x3f00, 0xbe80, 0x3e00, 1, 0x8001, 0x3380, 0xb300)[(t // 4096 + a) % 8]
    if fixture == 'F2':
        return F2_VALUES[a].get(t, 0)
    if fixture == 'F3':
        return F3_VALUES[a][t % 4]
    if fixture == 'F4':
        return (0x4000, 0xbf00, 0x3e80, 1)[(t // V + a) % 4]
    return F5_VALUES[a].get(t, 0)


def weight_bits(r, c):
    return (0x3f00 + ((3 * r + c) % 128)) << 16


def construct_metadata(ids, vocab=V, partition=P):
    # Counting/prefix/fill construction. Validation below independently walks
    # the original IDs and checks sorted occurrence lists, not these offsets.
    counts = [0] * vocab
    for row in ids:
        if type(row) is not int or not 0 <= row < vocab:
            raise ValueError('INVALID_ID')
        counts[row] += 1
    starts, offset = [], 0
    for count in counts:
        starts.append(offset)
        offset += count
    cursor, positions = starts.copy(), [-1] * len(ids)
    for token, row in enumerate(ids):
        positions[cursor[row]] = token
        cursor[row] += 1
    maximums = [max(counts[g:g + partition]) for g in range(0, vocab, partition)]
    return counts, starts, positions, maximums


def validate_metadata(ids, metadata, vocab=V, partition=P):
    rows = [[] for _ in range(vocab)]
    for token, row in enumerate(ids):
        if type(row) is not int or not 0 <= row < vocab:
            raise ValueError('INVALID_ID')
        rows[row].append(token)
    counts, starts, positions, maximums = metadata
    if len(counts) != vocab or len(starts) != vocab or len(positions) != len(ids):
        raise ValueError('METADATA_SHAPE')
    if any(type(x) is not int or not 0 <= x < len(ids) for x in positions) or any(type(x) is not int for x in maximums):
        raise ValueError('METADATA_INTEGER_RANGE')
    offset = 0
    for row, expected in enumerate(rows):
        if (type(counts[row]) is not int or type(starts[row]) is not int or
                counts[row] != len(expected) or starts[row] != offset or
                positions[offset:offset + len(expected)] != expected):
            raise ValueError('METADATA_CONTENT')
        offset += len(expected)
    expected_maxima = [max(len(x) for x in rows[g:g + partition]) for g in range(0, vocab, partition)]
    if maximums != expected_maxima or sorted(positions) != list(range(len(ids))):
        raise ValueError('METADATA_MAXIMA_OR_PERMUTATION')
    return rows
