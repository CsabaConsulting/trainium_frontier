#!/usr/bin/env python3
"""Reviewer-owned tiny CPU tests. Never builds/imports NKI or calls torch.compile.

Oracle: signed Python integers in units of 2**-149, exact addition, ties-to-even
FP32 rounding after each occurrence. Emulator: intended two-group pointer loop,
not an implementation or legality test of NKI .ap/dma/gather instructions.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path('/work')
OUT = ROOT / 'cpu_artifacts'
OUT.mkdir(exist_ok=False)
V, C, T, P = 256, 128, 256, 128

def sha(data):
    return hashlib.sha256(data).hexdigest()

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]

def units(word):
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    if exponent == 255:
        raise ValueError('nonfinite oracle input')
    magnitude = fraction if exponent == 0 else ((1 << 23) | fraction) << (exponent - 1)
    return -magnitude if word >> 31 else magnitude

def rounded_word(number, negative_zero=False):
    if number == 0:
        return 0x80000000 if negative_zero else 0
    sign = 0x80000000 if number < 0 else 0
    magnitude = abs(number)
    if magnitude < (1 << 23):
        return sign | magnitude
    shift = max(0, magnitude.bit_length() - 24)
    significand, remainder = divmod(magnitude, 1 << shift)
    if shift and (remainder > (1 << (shift - 1)) or
                  (remainder == (1 << (shift - 1)) and significand & 1)):
        significand += 1
    if significand == (1 << 24):
        significand >>= 1
        shift += 1
    exponent = shift + 1
    if exponent >= 255:
        raise OverflowError('oracle output overflow')
    return sign | (exponent << 23) | (significand & 0x7fffff)

def add(a, b):
    return rounded_word(units(a) + units(b), a == b == 0x80000000)

def pack(words):
    return struct.pack('<' + 'I' * len(words), *words)

def route(ids):
    counts = [sum(row == r for row in ids) for r in range(V)]
    starts = [sum(counts[:r]) for r in range(V)]
    positions = sorted(range(len(ids)), key=lambda token: ids[token])
    positions += [T] * (T - len(ids))
    maximums = [max(counts[:P]), max(counts[P:])]
    return counts, starts, positions, maximums

def oracle(ids, upstream):
    result = [0] * (V * C)
    for token, row in enumerate(ids):
        for channel in range(C):
            target = row * C + channel
            result[target] = add(result[target], upstream[token * C + channel])
    return result

def emulate(ids, upstream, metadata, reverse=False):
    counts, starts, positions, maxima = metadata
    positions = positions.copy()
    if reverse:
        for row in range(V):
            start, count = starts[row], counts[row]
            positions[start:start + count] = reversed(positions[start:start + count])
    result = [0] * (V * C)
    trace = {'rounds': maxima, 'active': 0, 'inactive': 0, 'rows_stored': 0,
             'sentinel': T, 'out_of_bounds_reads': 0, 'per_row_visits': [[] for _ in range(V)]}
    for group, rounds in enumerate(maxima):
        first = group * P
        remaining, pointers = counts[first:first + P].copy(), starts[first:first + P].copy()
        accumulator = [0] * (P * C)
        for iteration in range(rounds):
            for lane in range(P):
                active = remaining[lane] > 0
                address = pointers[lane] if active else T
                token = positions[address] if address < T else T
                token_address = token if active else T
                if active:
                    if not (0 <= address < T and 0 <= token_address < len(ids) and ids[token_address] == first + lane):
                        raise RuntimeError('emulator invalid gather')
                    trace['active'] += 1
                    trace['per_row_visits'][first + lane].append(token_address)
                    for channel in range(C):
                        offset = lane * C + channel
                        # Separate float/struct recurrence from integer oracle.
                        accumulator[offset] = bits(value(accumulator[offset]) + value(upstream[token_address * C + channel]))
                else:
                    if address != T or token_address != T:
                        raise RuntimeError('inactive lane did not select sentinel')
                    trace['inactive'] += 1
                    # Select retains previous bits, including zeros/subnormals.
                pointers[lane] += 1
                remaining[lane] -= 1
        result[first * C:(first + P) * C] = accumulator
        trace['rows_stored'] += P
    return result, trace

def tiled_wrong(ids, upstream):
    first = oracle(ids[:128], upstream[:128 * C])
    second = oracle(ids[128:], upstream[128 * C:])
    return [add(a, b) for a, b in zip(first, second)]

def difference(actual, expected):
    count = sum(a != b for a, b in zip(actual, expected))
    first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    return {'exact': len(actual) == len(expected) and count == 0,
            'differing_elements': count,
            'first': None if first is None else {'row': first // C, 'channel': first % C,
                                                'actual_hex': f'{actual[first]:08x}',
                                                'expected_hex': f'{expected[first]:08x}'}}

def candidate():
    path = ROOT / 'reconstructed/research/v2/lanes/k2-resident-consumer/reference.py'
    spec = importlib.util.spec_from_file_location('packet_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Only CPU Torch; force both pools and all common CPU library environment knobs.
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
flush_control_available = torch.set_flush_denormal(False)
ref = candidate()
results = {'native_executed': False, 'launch_authorized': False, 'production_adoption': False,
           'whole_step_speedup': 'unmeasured', 'oracle': 'integer exact FP32 per occurrence, round-to-nearest ties-to-even',
           'executed': ['reviewer integer oracle', 'reviewer intended-control-flow emulator',
                        'supplied reference.py functions', 'torch CPU embedding backward'],
           'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu',
                           'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                           'flush_denormal_disabled_supported': flush_control_available,
                           'thread_environment': {k: os.environ.get(k) for k in
                           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                           'max_tensor_shape': [V, C], 'tensor_storage_bound_bytes': 4 * 1024 * 1024,
                           'storage_bound_basis': 'one sequential case; <=12 live 256x128 FP32-equivalent buffers, plus small ID arrays'},
           'oracle_self_checks': [], 'cases': [], 'negative_controls': [], 'metadata_rejections': []}

def check(condition, message):
    if not condition:
        raise AssertionError(message)

def own_fixture(name):
    if name == 'permuted_unique':
        ids = [(i * 73 + 19) % V for i in range(T)]
    elif name == 'interleaved_edges':
        pattern = [127, 128, 255, 0, 128, 127, 64, 192, 128, 255, 127]
        ids = [pattern[i % len(pattern)] for i in range(T)]
    elif name == 'unequal_exhaustion':
        ids = [128] * 249 + [0, 127, 255, 128, 255, 127, 1]
    elif name == 'long_first_group':
        ids = [127] * T
    elif name == 'long_second_group':
        ids = [128] * T
    elif name == 'logical_empty':
        ids = []
    else:
        ids = [255 if i % 5 else 127 for i in range(T)]
    rows = [[((token * 13 + channel * 7) % 97 - 48) / 64 for channel in range(C)] for token in range(T)]
    if name.startswith('order_'):
        ids = [192] * T
        rows = [[0.0] * C for _ in range(T)]
        triplet = (2.0**25, 2.0, -2.0**25) if name == 'order_large' else (2.0, 2.0**-23, -2.0)
        for token, val in zip((3, 130, 251), triplet):
            ids[token] = 127
            rows[token] = [val if c % 2 == 0 else -val for c in range(C)]
    elif name == 'signed_zeros':
        rows = [[-0.0 if (t + c) % 3 else 0.0 for c in range(C)] for t in range(T)]
    elif name == 'subnormal_and_boundary':
        probes = [1, 2, 0x7fffff, 0x800000, 0x800001, 0x80000001, 0x80000002,
                  0x807fffff, 0x80800000, 0x00010000, 0x80010000, 0]
        rows = [[value(probes[(t * 7 + c) % len(probes)]) for c in range(C)] for t in range(T)]
    elif name == 'rounding_and_retention':
        ids = [0] * T
        ids[0] = ids[1] = 127
        ids[2] = 128
        rows = [[0.0] * C for _ in range(T)]
        rows[0] = [1.0] * C
        rows[1] = [2.0**-8] * C
        rows[2] = [2.0**-133] * C
    return ids, rows

def run_case(label, dtype_name, ids, rows, origin, receipt_entries=None):
    dtype = torch.bfloat16 if dtype_name == 'bf16' else torch.float32
    # Fixed input allocation, including logical empty. Cast using actual CPU Torch.
    padded = rows + [[0.0] * C for _ in range(T - len(rows))]
    grad = torch.tensor(padded, dtype=dtype)
    grad32 = grad.float()
    raw_input = grad32.numpy().tobytes()
    upstream = list(struct.unpack('<' + 'I' * (T * C), raw_input))
    metadata = route(ids)
    expected = oracle(ids, upstream)
    emulated, trace = emulate(ids, upstream, metadata)
    used_rows = [[value(upstream[t * C + c]) for c in range(C)] for t in range(len(ids))]
    supplied_meta = ref.metadata(ids)
    supplied_dense = ref.raw(ref.dense(ids, used_rows))
    supplied_output, supplied_trace = ref.consumer(ids, used_rows, metadata)
    supplied_bytes = ref.raw(supplied_output)
    weight = torch.zeros((V, C), dtype=torch.float32, requires_grad=True)
    index = torch.tensor(ids, dtype=torch.int64)
    output = F.embedding(index, weight).to(dtype)
    output.backward(grad[:len(ids)])
    check(weight.grad is not None, 'CPU gradient missing')
    cpu_bytes = weight.grad.detach().numpy().tobytes()
    expected_bytes = pack(expected)
    case_dir = OUT / f'{origin}_{label}_{dtype_name}'
    case_dir.mkdir()
    for name, data in [('input.fp32le.bin', raw_input), ('expected.fp32le.bin', expected_bytes),
                       ('emulated.fp32le.bin', pack(emulated)), ('torch_gradient.fp32le.bin', cpu_bytes),
                       ('supplied_dense.fp32le.bin', supplied_dense), ('supplied_consumer.fp32le.bin', supplied_bytes)]:
        (case_dir / name).write_bytes(data)
    (case_dir / 'inputs.json').write_text(json.dumps({'ids': ids, 'logical_tokens': len(ids),
        'storage_shape': [T, C], 'upstream_dtype': dtype_name, 'input_encoding': 'BF16 expanded to FP32 or FP32, little endian',
        'metadata': dict(zip(('counts', 'starts', 'positions', 'group_maximums'), metadata))}, indent=2) + '\n')
    (case_dir / 'trace.json').write_text(json.dumps(trace, indent=2) + '\n')
    checks = {'emulator': difference(emulated, expected),
              'torch_cpu_embedding': {'exact': cpu_bytes == expected_bytes},
              'supplied_dense': {'exact': supplied_dense == expected_bytes},
              'supplied_consumer': {'exact': supplied_bytes == expected_bytes},
              'supplied_metadata': {'exact': supplied_meta == metadata},
              'stable_visits': {'exact': trace['per_row_visits'] == [[t for t, r in enumerate(ids) if r == row] for row in range(V)]},
              'all_rows_stored': {'exact': trace['rows_stored'] == V},
              'unused_rows_positive_zero': {'exact': all(expected[r * C + c] == 0 for r in range(V) if r not in ids for c in range(C))}}
    record = {'name': label, 'dtype': dtype_name, 'origin': origin, 'input_sha256': sha(raw_input),
              'expected_sha256': sha(expected_bytes), 'torch_sha256': sha(cpu_bytes), 'checks': checks,
              'group_rounds': metadata[3], 'active_gathers': trace['active'], 'inactive_skips': trace['inactive'],
              'artifact_directory': str(case_dir.relative_to(ROOT))}
    if receipt_entries:
        wrong = ref.grouped_wrong(ids, used_rows) if label.startswith('cross_tile_') else None
        receipt_checks = []
        for receipt in receipt_entries:
            receipt_checks.append(receipt['sha256'] == sha(expected_bytes)
                and receipt['group_rounds'] == metadata[3]
                and receipt['active_gathers'] == trace['active']
                and receipt['inactive_gathers_skipped'] == trace['inactive']
                and receipt['row3'] == value(expected[3 * C])
                and receipt['byte_exact'] is True
                and receipt['rejected_grouped_row3'] == (wrong[3][0] if wrong else None))
        record['recorded_receipts_match_recomputed_fields'] = all(receipt_checks)
        checks['receipt_fields'] = {'exact': all(receipt_checks)}
    record['pass'] = all(entry['exact'] for entry in checks.values())
    results['cases'].append(record)
    (case_dir / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if origin == 'own' and label.startswith('order_'):
        for method, wrong in [('reversed_occurrences', emulate(ids, upstream, metadata, reverse=True)[0]),
                              ('two_token_tiles', tiled_wrong(ids, upstream))]:
            delta = difference(wrong, expected)
            control = {'name': label, 'dtype': dtype_name, 'mutation': method,
                       'detected': not delta['exact'], 'comparison': delta, 'sha256': sha(pack(wrong))}
            results['negative_controls'].append(control)
            (case_dir / (method + '.fp32le.bin')).write_bytes(pack(wrong))
            check(control['detected'], 'negative control escaped byte comparator')
    if origin == 'own' and label == 'rounding_and_retention':
        rounded = weight.grad.detach().to(torch.bfloat16).float().numpy().tobytes()
        control = {'name': label, 'dtype': dtype_name, 'mutation': 'round_final_gradient_to_bf16',
                   'detected': rounded != expected_bytes, 'sha256': sha(rounded)}
        results['negative_controls'].append(control)
        (case_dir / 'rounded_output.fp32le.bin').write_bytes(rounded)
        check(control['detected'], 'BF16 final-output rounding escaped comparator')
    check(record['pass'], f'case failed: {origin}/{label}/{dtype_name}')

try:
    for a, b, expected in [(0, 0x80000000, 0), (0x80000000, 0x80000000, 0x80000000),
                           (1, 1, 2), (0x7fffff, 1, 0x800000),
                           (bits(2**25), bits(2), bits(2**25)),
                           (bits(2**25 + 4), bits(2), bits(2**25 + 8)),
                           (bits(2**25), bits(-2**25), 0)]:
        actual = add(a, b)
        results['oracle_self_checks'].append({'a': f'{a:08x}', 'b': f'{b:08x}',
                                              'expected': f'{expected:08x}', 'actual': f'{actual:08x}'})
        check(actual == expected, 'integer oracle self-check')
    names = ['permuted_unique', 'interleaved_edges', 'unequal_exhaustion', 'long_first_group',
             'long_second_group', 'logical_empty', 'order_large', 'order_unit',
             'signed_zeros', 'subnormal_and_boundary', 'rounding_and_retention']
    for name in names:
        for dtype_name in ('bf16', 'fp32'):
            ids, rows = own_fixture(name)
            run_case(name, dtype_name, ids, rows, 'own')
            del ids, rows
            gc.collect()
    receipts = [json.loads((Path('/packet/raw') / name).read_text()) for name in ('checks-normal.json', 'checks-optimized.json')]
    receipt_maps = [{(x['case'], x['dtype']): x for x in receipt['cases']} for receipt in receipts]
    for name in ref.CASE_NAMES:
        for dtype_name in ('bf16', 'fp32'):
            ids, rows, meta = ref.fixture(name, dtype_name)
            run_case(name, dtype_name, ids, rows, 'supplied_fixture', [mapping[name, dtype_name] for mapping in receipt_maps])
            del ids, rows, meta
            gc.collect()
    # Validator must reject bad routing rather than hiding an occurrence reorder.
    ids, _ = own_fixture('interleaved_edges')
    good = route(ids)
    for kind in ('reordered', 'bad_count', 'bad_start', 'low_maximum', 'out_of_bounds_position'):
        meta = tuple(x.copy() for x in good)
        if kind == 'reordered':
            row = next(r for r in range(V) if meta[0][r] >= 2)
            offset = meta[1][row]
            meta[2][offset], meta[2][offset + 1] = meta[2][offset + 1], meta[2][offset]
        elif kind == 'bad_count': meta[0][0] += 1
        elif kind == 'bad_start': meta[1][0] = 1
        elif kind == 'low_maximum': meta[3][0] -= 1
        else: meta[2][0] = T
        try:
            ref.validate_metadata(ids, meta)
        except ValueError as exc:
            result = {'mutation': kind, 'rejected': True, 'error': str(exc)}
        else:
            result = {'mutation': kind, 'rejected': False}
        result['ids'] = ids
        result['metadata'] = meta
        results['metadata_rejections'].append(result)
        check(result['rejected'], 'bad metadata accepted')
    results['receipt_scope_audit'] = {
        'case_fields_identical_normal_vs_optimized': receipts[0]['cases'] == receipts[1]['cases'],
        'all_20_case_fields_recomputed': all(c.get('recorded_receipts_match_recomputed_fields', True) for c in results['cases']),
        'receipt_statuses': [r['status'] for r in receipts],
        'receipts_deny_native_execution': all(r['native_executed'] is False and r['torch_imported'] is False for r in receipts),
        'unverifiable_historical_claims': ['command actually executed', 'recorded wall-clock date', 'exit code provenance',
                                         'peak traced allocation values', 'what omitted test_reference.py checked',
                                         'actual candidate commit ancestry and repository state'],
        'scope': 'Raw JSON internal consistency and reproducible stdlib arithmetic only; not authenticated execution provenance'}
    results['pass'] = True
except Exception:
    results['pass'] = False
    results['error'] = traceback.format_exc()
    traceback.print_exc()
finally:
    results['counts'] = {'cases': len(results['cases']), 'own_cases': sum(c['origin'] == 'own' for c in results['cases']),
                         'supplied_fixture_cases': sum(c['origin'] == 'supplied_fixture' for c in results['cases']),
                         'negative_controls': len(results['negative_controls'])}
    (ROOT / 'cpu_results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pass': results['pass'], 'counts': results['counts'], 'native_executed': False}), flush=True)
raise SystemExit(0 if results['pass'] else 1)
