#!/usr/bin/env python3
"""Own bounded stdlib oracle; not execution of NKI or a fresh-session gate."""
import argparse
import array
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys


def require(value, why):
    if not value:
        raise RuntimeError(why)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    return struct.unpack('<f', struct.pack('<I', (word + 32767 + ((word >> 16) & 1)) & 0xffff0000))[0]


def bits(rows):
    return b''.join(struct.pack('<f', value) for row in rows for value in row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('candidate_reference', args.candidate / 'reference.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    checks, records = 0, []
    randomizer = random.Random(731)
    for name in ('cancellation_large', 'cancellation_unit', 'one_hot_group0', 'one_hot_group1',
                 'unique', 'row_boundaries', 'random', 'signed_zero', 'subnormal', 'empty'):
        for dtype, cast in (('bf16', bf16), ('fp32', f32)):
            ids = [randomizer.randrange(256) for _ in range(256)]
            rows = [[cast(randomizer.randrange(-128, 128) / 16) for _ in range(128)] for _ in ids]
            if name.startswith('cancellation'):
                ids = [8] * 256
                rows = [[0.0] * 128 for _ in ids]
                a, b = ((2.0 ** 24, 1.0) if name.endswith('large') else (1.0, 2.0 ** -24))
                for token, value in ((0, a), (128, b), (129, -a)):
                    ids[token], rows[token] = 127, [cast(value)] * 128
            elif name.startswith('one_hot'):
                ids = [0 if name.endswith('group0') else 255] * 256
            elif name == 'unique':
                ids = list(reversed(range(256)))
            elif name == 'row_boundaries':
                ids = [0, 127, 128, 255] * 64
            elif name == 'signed_zero':
                rows = [[-0.0 if (i + j) % 2 else 0.0 for j in range(128)] for i in range(256)]
            elif name == 'subnormal':
                unit = 2.0 ** (-133 if dtype == 'bf16' else -149)
                rows = [[cast(unit * (1 if (i + j) % 3 else -1)) for j in range(128)] for i in range(256)]
            elif name == 'empty':
                ids, rows = [], []
            # Independent oracle: token-major dense indexed update with a float32
            # array store after EACH occurrence, not a segment/tree reduction.
            oracle = [array.array('f', [0.0] * 128) for _ in range(256)]
            for token, row in enumerate(ids):
                for channel in range(128):
                    oracle[row][channel] = oracle[row][channel] + rows[token][channel]
            # Independent stable buckets, not candidate metadata().
            buckets = [[i for i, row in enumerate(ids) if row == r] for r in range(256)]
            counts = [len(bucket) for bucket in buckets]
            starts, positions = [], []
            for bucket in buckets:
                starts.append(len(positions))
                positions.extend(bucket)
            positions += [256] * (256 - len(positions))
            maximums = [max(counts[:128]), max(counts[128:])]
            meta = counts, starts, positions, maximums
            actual, trace = ref.consumer(ids, rows, meta)
            require(bits(actual) == bits(oracle), name + '/' + dtype + ' accumulation differs')
            require(trace['active_gathers'] == len(ids), 'occurrence count')
            require(trace['output_rows_stored'] == 256, 'all unused rows initialized')
            require(trace['accumulator_hbm_reads'] == 0, 'CPU model residency claim')
            checks += 4
            for row in range(256):
                require([t for r, t in trace['visits'] if r == row] == buckets[row], 'unstable row visits')
                checks += 1
            wrong = None
            if name.startswith('cancellation'):
                wrong = ref.grouped_wrong(ids, rows)[127][0]
                require(wrong != oracle[127][0], 'negative control failed to distinguish grouped sum')
                checks += 1
            records.append(dict(case=name, dtype=dtype, exact=True, max_abs=0.0,
                                output_sha256=hashlib.sha256(bits(actual)).hexdigest(),
                                group_maximums=maximums, row127=actual[127][0], grouped_counterexample=wrong))
    ids = [3, 3, 128]
    good = ref.metadata(ids)
    for kind in ('reordered', 'wrong_maximum', 'wrong_pointer', 'out_of_range'):
        bad = [part.copy() for part in good]
        if kind == 'reordered':
            bad[2][0], bad[2][1] = bad[2][1], bad[2][0]
        elif kind == 'wrong_maximum':
            bad[3][0] += 1
        elif kind == 'wrong_pointer':
            bad[1][3] += 1
        else:
            bad[2][0] = 256
        try:
            ref.validate_metadata(ids, bad)
        except ValueError:
            checks += 1
        else:
            raise RuntimeError('invalid metadata accepted: ' + kind)
    for optimized in (False, True):
        command = [sys.executable] + (['-O'] if optimized else []) + ['-B', str(args.candidate / 'check_native.py')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        require(result.returncode == 0 and json.loads(result.stdout) == dict(status='DISABLED', native_imported=False, native_executed=False), 'default CLI imported native work')
        checks += 1
    require('torch' not in sys.modules and 'nki' not in sys.modules, 'unexpected tensor/native import')
    checks += 1
    encoded = json.dumps(dict(status='PASS_STDLIB_ARITHMETIC_ONLY', own_checks=checks, cases=records,
                          native_kernel_executed=False, native_legality='UNPROVEN', fresh_source_gate=False,
                          source_sha256={name: hashlib.sha256((args.candidate / name).read_bytes()).hexdigest()
                                         for name in ('consumer_nki.py', 'reference.py', 'check_native.py')}), indent=2)
    if args.output:
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
