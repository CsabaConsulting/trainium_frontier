#!/usr/bin/env python3
"""Bounded, independent stdlib source/CPU audit of the supplied packet.

This is a hand-written source-semantic model, NOT NKI execution or a compiler
emulator. Candidate source is parsed and pinned, never enabled or modified.
The model uses binary64 addition followed by struct's binary32 conversion;
the independent token-order oracle uses integer arithmetic and explicit RNE.
No SDK imports, network, device calls, installations, or native execution.
"""
import ast
import builtins
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import runpy
import struct
import sys
import time

PACKET = Path('/packet')
WORK = Path('/work')
ART = WORK / 'cpu_artifacts'
V, C, T, P = 256, 128, 256, 128
EXPECTED_CANDIDATE = '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae'
F32 = struct.Struct('<f')
U32 = struct.Struct('<I')
LIMITS = dict(cpu_seconds=120, address_space_bytes=256 * 1024 * 1024,
              max_cases=64, max_tokens_per_case=T, output_elements_per_case=V*C,
              random_seed=260926, sdk_imports_allowed=False)


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def f32_bits(value):
    try:
        return U32.unpack(F32.pack(value))[0]
    except OverflowError:
        return 0xff800000 if value < 0 else 0x7f800000


def from_bits(value):
    return F32.unpack(U32.pack(value))[0]


def finite_bits(value):
    return (value & 0x7f800000) != 0x7f800000


def raw(words):
    return struct.pack('<' + str(len(words)) + 'I', *words)


def raw_ints(words):
    return struct.pack('<' + str(len(words)) + 'i', *words)


def bf16_round(word):
    require(finite_bits(word), 'finite BF16 conversion inputs only')
    return ((word + 0x7fff + ((word >> 16) & 1)) & 0xffff0000)


def exact_units(word):
    """Represent finite FP32 exactly as a signed integer times 2**-149."""
    require(finite_bits(word), 'nonfinite integer oracle input')
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    magnitude = fraction if exponent == 0 else ((1 << 23) | fraction) << (exponent - 1)
    return -magnitude if word >> 31 else magnitude


def oracle_add(a, b):
    total = exact_units(a) + exact_units(b)
    if total == 0:
        return 0x80000000 if a == b == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    total = abs(total)
    if total < (1 << 23):
        return sign | total
    shift = max(0, total.bit_length() - 24)
    significand, remainder = divmod(total, 1 << shift)
    if shift:
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and significand & 1):
            significand += 1
    if significand == 1 << 24:
        significand >>= 1
        shift += 1
    exponent = shift + 1
    if exponent >= 255:
        return sign | 0x7f800000
    return sign | (exponent << 23) | (significand & 0x7fffff)


def model_add(a, b):
    return f32_bits(from_bits(a) + from_bits(b))


def oracle(ids, gradient):
    result = [0] * (V * C)
    for token, row in enumerate(ids):
        for col in range(C):
            dst = row * C + col
            result[dst] = oracle_add(result[dst], gradient[token*C + col])
    return result


def make_metadata(ids):
    # Deliberately independent row-scan construction, no imported helper.
    segments = [[t for t, row in enumerate(ids) if row == r] for r in range(V)]
    counts = [len(s) for s in segments]
    starts = [sum(counts[:r]) for r in range(V)]
    packed = [t for segment in segments for t in segment]
    return counts, starts, packed + [T] * (T - len(packed)), [max(counts[:P]), max(counts[P:])]


def validate(ids, meta):
    counts, starts, positions, limits = meta
    require(len(ids) <= T and all(type(x) is int and 0 <= x < V for x in ids), 'ids')
    require([len(x) for x in meta] == [V, V, T, 2], 'metadata sizes')
    require(all(type(x) is int for part in meta for x in part), 'integer metadata')
    require(all(0 <= x <= T for x in counts), 'count bounds')
    for row in range(V):
        expected = [i for i, owner in enumerate(ids) if owner == row]
        require(starts[row] == sum(counts[:row]), 'packed start')
        require(counts[row] == len(expected), 'count')
        require(positions[starts[row]:starts[row]+counts[row]] == expected, 'stable positions')
    require(sum(counts) == len(ids), 'total count')
    require(positions[len(ids):] == [T] * (T-len(ids)), 'padding sentinel')
    require(limits == [max(counts[:P]), max(counts[P:])], 'group maxima')


def audit_source():
    data = (WORK / 'candidate_consumer_nki.py').read_bytes()
    require(sha(data) == EXPECTED_CANDIDATE, 'candidate hash changed')
    tree = ast.parse(data)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    kernel = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
    require(ast.unparse(factory.body[-1]) == 'return resident_consumer[2]', 'explicit grid')
    require(ast.unparse(factory.args.kw_defaults[0]) == 'False', 'disabled factory default')
    assignments = {n.targets[0].id: n.value for n in ast.walk(kernel)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    require(ast.unparse(assignments['feature_first']) == 'nl.program_id(0) * 64', 'feature base')
    loops = [n for n in ast.walk(kernel) if isinstance(n, ast.For)]
    require([ast.unparse(n.iter) for n in loops] == ['range(2)', 'nl.dynamic_range(limit)'], 'shared loops')
    require(not any(isinstance(n, (ast.If, ast.While, ast.Break, ast.Continue, ast.IfExp))
                    for n in ast.walk(kernel)), 'unexpected kernel branch')
    calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
    require(sum(ast.unparse(n.func) == 'nl.program_id' for n in calls) == 1, 'PID calls')
    feature_uses = [n.lineno for n in ast.walk(kernel)
                    if isinstance(n, ast.Name) and n.id == 'feature_first' and isinstance(n.ctx, ast.Load)]
    require(sorted(feature_uses) == [56, 72, 72], 'PID only affects gradient offset and output slice')
    aps = {ast.unparse(n.func.value): n for n in calls
           if isinstance(n.func, ast.Attribute) and n.func.attr == 'ap'}
    kwargs = lambda node: {k.arg: k.value for k in node.keywords}
    grad_args, pos_args, active_args = (kwargs(aps[x]) for x in ('gradient', 'positions', 'active'))
    gradient_pattern = ast.literal_eval(grad_args['pattern'])
    require(gradient_pattern == [[128, 128], [1, 64]], 'gradient strides/extent')
    require(ast.literal_eval(pos_args['pattern']) == [[1, 128], [1, 1]], 'position pattern')
    require(ast.literal_eval(active_args['pattern']) == [[1, 128], [0, 64]], 'mask broadcast')
    require(ast.unparse(grad_args['offset']) == 'feature_first', 'ordinary element offset')
    require('scalar_offset' not in grad_args and 'offset' not in pos_args, 'indirection form')
    for args, name in [(grad_args, 'token_address'), (pos_args, 'address')]:
        require(ast.literal_eval(args['indirect_dim']) == 0, 'indirect dimension')
        require(ast.unparse(args['vector_offset']) == name, 'vector index')
    for name in ('accumulator', 'updated', 'value'):
        require(ast.literal_eval(assignments[name].args[0]) == (128, 64), 'feature scratch shape')
    require(ast.literal_eval(assignments['accumulator'].args[1]) == 0.0, 'positive zero initialization')
    expected_statements = [
        'assert nl.num_programs() == 2',
        'nisa.dma_copy(dst=remaining, src=counts[first:first + 128, :])',
        'nisa.dma_copy(dst=pointer, src=starts[first:first + 128, :])',
        'nisa.dma_copy(dst=maximum, src=group_maximums[group:group + 1, :])',
        'nisa.register_load(limit, maximum)',
        'address = nl.where(active, pointer, sentinel, dtype=nl.int32)',
        'token_address = nl.where(active, token, sentinel, dtype=nl.int32)',
        'nisa.tensor_tensor(dst=updated, data1=accumulator, data2=value, op=nl.add, engine=nisa.engine.vector)',
        'selected = nl.where(active.ap(pattern=[[1, 128], [0, 64]]), updated, accumulator, dtype=nl.float32)',
        'nisa.tensor_copy(dst=accumulator, src=selected)',
        'nisa.tensor_scalar(dst=pointer, data=pointer, op0=nl.add, operand0=1.0, engine=nisa.engine.vector)',
        'nisa.tensor_scalar(dst=remaining, data=remaining, op0=nl.add, operand0=-1.0, engine=nisa.engine.vector)',
        'nisa.dma_copy(dst=output[first:first + 128, feature_first:feature_first + 64], src=accumulator)',
    ]
    dumps = {ast.dump(n, include_attributes=False) for n in ast.walk(kernel)}
    for statement in expected_statements:
        require(ast.dump(ast.parse(statement).body[0], include_attributes=False) in dumps, statement)
    indirect_dmas = [n for n in calls if ast.unparse(n.func) == 'nisa.dma_copy'
                    and any(k.arg == 'oob_mode' for k in n.keywords)]
    require(len(indirect_dmas) == 2, 'two indirect reads')
    for node in indirect_dmas:
        args = kwargs(node)
        require(ast.unparse(args['oob_mode']) == 'nisa.oob_mode.skip', 'OOB skip')
        require(ast.unparse(args['dge_mode']) == 'nisa.dge_mode.swdge', 'software DGE')
    result = dict(status='PASS_STATIC_BINDING', source_sha256=sha(data), grid=2,
                  groups=2, group_rows=P, feature_width=gradient_pattern[1][1],
                  gradient_row_stride=gradient_pattern[0][0], feature_uses_lines=sorted(feature_uses),
                  no_program_dependent_control_or_metadata=True,
                  limitation='Static AST checks plus manual review; not a compiler or full NKI interpreter.')
    save_json(ART / 'source_audit.json', result)
    return result


def source_model(ids, gradient, meta, config, mutation=None):
    """Model double gather, per-group loops, two feature halves and final stores.

    Vector indirection substitutes the indexed row; the ordinary offset is
    an additional element displacement. Skipped DMA leaves initialized state.
    Inactive tensor additions are elided because select retains the old bits.
    """
    counts, starts, positions, maximums = meta
    width, stride = config['feature_width'], config['gradient_row_stride']
    result, writers = [None] * (V*C), [0] * (V*C)
    traces = []
    for pid in range(2):
        first_col = pid * width
        traces.append([])
        groups = [pid] if mutation == 'row_shard' else range(2)
        for group in groups:
            first_row = group * P
            bound = maximums[group]
            if mutation == 'global_bound':
                bound = max(maximums)
            if mutation == 'pid_bound' and pid == 1:
                bound = max(0, bound - 1)
            record = dict(group=group, bound=bound, events=[], skipped_lanes=0,
                          active_lanes=0, inactive_bit_preservations=0)
            traces[pid].append(record)
            accum = [[0] * width for _ in range(P)]
            pointers = starts[first_row:first_row+P].copy()
            remaining = counts[first_row:first_row+P].copy()
            for step in range(bound):
                for lane in range(P):
                    is_active = remaining[lane] > 0
                    address = pointers[lane] if is_active else T
                    token = T
                    if 0 <= address < T:
                        token = positions[address]
                    token_address = token if is_active else T
                    if is_active:
                        require(0 <= address < T and 0 <= token < len(ids), 'active double gather bound')
                        require(ids[token] == first_row+lane, 'gather row identity')
                        record['events'].append([step, first_row+lane, address, token])
                        record['active_lanes'] += 1
                        offset = first_col
                        if mutation == 'missing_offset':
                            offset = 0
                        if mutation == 'scaled_offset':
                            offset *= stride
                        base = token_address * stride + offset
                        for col in range(width):
                            addr = base + col
                            value = gradient[addr] if 0 <= addr < T*C else 0
                            accum[lane][col] = model_add(accum[lane][col], value)
                    else:
                        require(address == token_address == T and token == T, 'initialized inactive sentinel')
                        require(token_address*stride+first_col >= T*C, 'gradient sentinel OOB')
                        record['skipped_lanes'] += 1
                        record['inactive_bit_preservations'] += width
                    pointers[lane] += 1
                    remaining[lane] -= 1
            for lane in range(P):
                start = (first_row+lane)*C+first_col
                for col in range(width):
                    writers[start+col] += 1
                    result[start+col] = accum[lane][col]
    return result, writers, traces


def validate_flow(ids, meta, writers, traces):
    require(all(n == 1 for n in writers), 'each output element has exactly one writer')
    require(traces[0] == traces[1], 'symmetric runtime traces')
    require([r['group'] for r in traces[0]] == [0, 1], 'both groups per program')
    require([r['bound'] for r in traces[0]] == meta[3], 'per-group dynamic bounds')
    events = [event for rec in traces[0] for event in rec['events']]
    require(len(events) == len(ids), 'one occurrence per program')
    for row in range(V):
        observed = [e[3] for e in events if e[1] == row]
        require(observed == [t for t, r in enumerate(ids) if r == row], 'original token order')


def compare(actual, expected):
    words_finite = all(x is not None and finite_bits(x) for x in actual + expected)
    valid = len(actual) == len(expected) and words_finite
    exact = valid and raw(actual) == raw(expected)
    maximum = None
    if valid:
        maximum = max((abs(from_bits(a)-from_bits(b)) for a, b in zip(actual, expected)), default=0.0)
    result = dict(exact=exact, finite=words_finite, max_abs=maximum)
    if not words_finite:
        result['metric_status'] = 'NONFINITE_OR_UNWRITTEN_NOT_EXACT'
    # Strict serialization is itself part of the comparator check.
    json.dumps(result, allow_nan=False)
    return result


def independent_fixture(name, dtype):
    rng = random.Random(LIMITS['random_seed'] + sum(map(ord, name)))
    ids = [(0, 127, 128, 255)[t % 4] for t in range(T)]
    if name == 'empty':
        ids = []
    elif name == 'unique_columns':
        ids = list(range(V))
    elif name == 'all_row0':
        ids = [0] * T
    elif name == 'all_row255':
        ids = [255] * T
    elif name == 'unequal_sparse':
        ids = [0, 128, 127, 0, 255, 0, 128, 127, 0, 0, 127]
    elif name.startswith('cross_tile_'):
        ids = [201] * T
        for t in (0, 128, 129):
            ids[t] = 127
    elif name == 'rounding_ties':
        ids = [128] * 3 + [127] * 3
    elif name.startswith('random_'):
        ids = [rng.randrange(V) if t % 3 else (127 if t % 2 else 128) for t in range(T)]
    gradient = []
    for token in range(T):
        for col in range(C):
            # Channel-distinct exactly representable values, including both halves.
            word = f32_bits(((token*29 + col*17) % 509 - 254) / 32)
            if name.startswith('cross_tile_'):
                vals = (2**24, 1.0, -(2**24)) if name.endswith('large') else (1.0, 2**-24, -1.0)
                word = f32_bits(dict(zip((0, 128, 129), vals)).get(token, 0.0))
            elif name == 'signed_zero':
                word = 0x80000000 if (token+col) % 2 else 0
            elif name == 'subnormal':
                values = (1, 2, 0x7fffff, 0x80000001, 0x80000002, 0x807fffff)
                if dtype == 'bf16':
                    values = (0x10000, 0x20000, 0x7f0000, 0x80010000, 0x80020000, 0x807f0000)
                word = values[(token+col) % len(values)]
            elif name == 'rounding_ties':
                values = (0x3f800000, 0x33800000, 0xbf800000) if col % 2 == 0 else (0x3f800001, 0x33800000, 0xbf800000)
                word = values[token % 3]
            elif name == 'bf16_conversion_ties':
                word = (0x3f808000, 0x3f818000, 0x00008000, 0x00018000,
                        0xbf808000, 0xbf818000, 0x80008000, 0x80018000)[(token+col) % 8]
            elif name.startswith('random_'):
                exponent = rng.randrange(0, 181)
                word = (rng.randrange(2) << 31) | (exponent << 23) | rng.getrandbits(23)
            if token >= len(ids):
                # Finite poison in unused physical storage; must never contribute.
                word = 0x6e123456
            gradient.append(bf16_round(word) if dtype == 'bf16' else word)
    return ids, gradient, make_metadata(ids)


def numerical_self_tests():
    known = [
        (0, 0x80000000, 0), (0x80000000, 0x80000000, 0x80000000),
        (0x3f800000, 0xbf800000, 0), (1, 1, 2), (0x80000001, 0x80000001, 0x80000002),
        (0x7fffff, 1, 0x800000), (0x800000, 0x80000001, 0x7fffff),
        (0x3f800000, 0x33800000, 0x3f800000), (0x3f800001, 0x33800000, 0x3f800002),
        (0x4b800000, 0x3f800000, 0x4b800000), (0xcb800000, 0x3f800000, 0xcb7fffff),
        (0x7f7fffff, 0x7f7fffff, 0x7f800000),
    ]
    for a, b, expected in known:
        require(oracle_add(a, b) == model_add(a, b) == expected, 'known FP32 result')
    ties = [(0x3f808000, 0x3f800000), (0x3f818000, 0x3f820000),
            (0x80008000, 0x80000000), (0x00018000, 0x00020000)]
    for a, expected in ties:
        require(bf16_round(a) == expected, 'known BF16 result')
    rng = random.Random(LIMITS['random_seed'])
    for _ in range(4096):
        a, b = rng.getrandbits(32), rng.getrandbits(32)
        if not finite_bits(a):
            a ^= 1 << 23
        if not finite_bits(b):
            b ^= 1 << 23
        require(oracle_add(a, b) == model_add(a, b), 'random FP32 arithmetic check')
    return dict(known_fp32=len(known), known_bf16=len(ties), deterministic_random_fp32_pairs=4096)


def negative_controls(config):
    results = []
    for name, fixture in [('missing_offset', 'unique_columns'), ('scaled_offset', 'unique_columns'),
                          ('row_shard', 'unique_columns'), ('global_bound', 'unequal_sparse'),
                          ('pid_bound', 'unequal_sparse')]:
        ids, gradient, meta = independent_fixture(fixture, 'fp32')
        actual, writers, traces = source_model(ids, gradient, meta, config, name)
        rejected_by = None
        try:
            validate_flow(ids, meta, writers, traces)
            require(compare(actual, oracle(ids, gradient))['exact'], 'output bytes')
        except AssertionError as exc:
            rejected_by = str(exc)
        require(rejected_by is not None, 'undetected mutant: '+name)
        results.append(dict(name=name, detected=True, reason=rejected_by))
    for name in ('cross_tile_large', 'cross_tile_unit'):
        ids, gradient, meta = independent_fixture(name, 'fp32')
        expected = oracle(ids, gradient)
        partials = [oracle(ids[start:start+P], gradient[start*C:(start+P)*C]) for start in (0, P)]
        wrong = [oracle_add(a, b) for a, b in zip(*partials)]
        require(not compare(wrong, expected)['exact'], 'tile sum mutant')
        results.append(dict(name=name+'_tile_reassociation', detected=True,
                            expected_row127_col0=hex(expected[127*C]), wrong_row127_col0=hex(wrong[127*C])))
    ids, gradient, meta = independent_fixture('cross_tile_large', 'fp32')
    backwards = [word for t in reversed(range(len(ids))) for word in gradient[t*C:(t+1)*C]]
    require(not compare(oracle(list(reversed(ids)), backwards), oracle(ids, gradient))['exact'], 'token reversal')
    results.append(dict(name='reversed_token_order', detected=True))
    ids, gradient, meta = independent_fixture('subnormal', 'fp32')
    flushed = [word & 0x80000000 if (word & 0x7f800000) == 0 else word for word in gradient]
    require(not compare(oracle(ids, flushed), oracle(ids, gradient))['exact'], 'subnormal flush')
    results.append(dict(name='flush_subnormal', detected=True))
    require(not compare([0], [0x80000000])['exact'], 'signed zero byte check')
    results.append(dict(name='signed_zero_difference', detected=True))
    for name, word in [('infinity', 0x7f800000), ('nan', 0x7fc00000)]:
        comparison = compare([word], [word])
        require(not comparison['exact'] and comparison['max_abs'] is None, 'nonfinite exactness')
        results.append(dict(name=name+'_even_identical_bytes', detected=True, comparison=comparison))
    # Selection must preserve arbitrary prior bits; these injected states are
    # primitive probes, not claims they arise from the finite +0 recurrence.
    for word in (0x80000000, 1, 0x80000001, 0x3f800000):
        selected = model_add(word, 0) if False else word
        require(selected == word, 'inactive select bit retention')
    require(model_add(0x80000000, 0) != 0x80000000, 'negative zero select witness')
    results.append(dict(name='inactive_add_without_select_witness', detected=True,
                        prior='0x80000000', added_zero='0x00000000'))
    ids, _, meta = independent_fixture('unequal_sparse', 'fp32')
    for kind in ('position_order', 'maximum', 'padding', 'start'):
        wrong = [list(part) for part in meta]
        if kind == 'position_order':
            wrong[2][0], wrong[2][1] = wrong[2][1], wrong[2][0]
        elif kind == 'maximum':
            wrong[3][1] += 1
        elif kind == 'padding':
            wrong[2][-1] = 0
        else:
            wrong[1][127] += 1
        try:
            validate(ids, wrong)
        except AssertionError:
            results.append(dict(name='invalid_metadata_'+kind, detected=True))
        else:
            raise AssertionError('metadata mutant not detected')
    save_json(ART / 'negative_controls.json', results)
    return results


def guard_checks():
    imported = []
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split('.')[0] in ('nki', 'torch', 'torch_neuronx'):
            imported.append(name)
            raise AssertionError('forbidden SDK import '+name)
        return original(name, *args, **kwargs)
    builtins.__import__ = blocked
    try:
        namespace = runpy.run_path(str(WORK / 'candidate_consumer_nki.py'))
        try:
            namespace['build_kernel']()
        except RuntimeError as exc:
            require(str(exc) == 'EXPERIMENT_DISABLED: no native module imported', 'default factory guard')
        else:
            raise AssertionError('factory unexpectedly enabled')
        namespace = runpy.run_path(str(PACKET / 'gate-support/check_native.py'))
        old_argv = sys.argv
        sys.argv = ['check_native.py']
        try:
            require(namespace['main']() == 0, 'checker default return')
        finally:
            sys.argv = old_argv
    finally:
        builtins.__import__ = original
    require(not imported, 'SDK import attempt')
    return dict(factory_default_disabled=True, checker_default_disabled=True, sdk_import_attempts=imported)


def run_case(index, origin, name, dtype, ids, gradient, meta, config, support):
    validate(ids, meta)
    require(len(gradient) == T*C, 'fixed gradient storage')
    require(all(finite_bits(w) for w in gradient), 'finite case')
    if dtype == 'bf16':
        require(all(w & 0xffff == 0 for w in gradient), 'BF16 expansion exact')
    case_name = f'{index:02d}_{origin}_{name}_{dtype}'
    case_dir = ART / case_name
    case_dir.mkdir()
    expected = oracle(ids, gradient)
    actual, writers, traces = source_model(ids, gradient, meta, config)
    validate_flow(ids, meta, writers, traces)
    comparison = compare(actual, expected)
    require(comparison['exact'], case_name+' failed exactness')
    require(all(actual[row*C+col] == 0 for row in range(V) if row not in ids for col in range(C)),
            'unused row positive zero')
    support_checks = {}
    if origin == 'supplied':
        rows = [[from_bits(w) for w in gradient[t*C:(t+1)*C]] for t in range(len(ids))]
        for label, result in [('dense', support['dense'](ids, rows)),
                              ('consumer', support['consumer'](ids, rows, meta)[0])]:
            serialized = support['raw'](result)
            require(serialized == raw(expected), 'unchanged support '+label)
            support_checks[label] = dict(exact=True, sha256=sha(serialized))
    files = {
        'gradient.fp32le': raw(gradient), 'expected.fp32le': raw(expected),
        'actual.fp32le': raw(actual), 'writers.u8': bytes(writers),
        'ids.i32le': raw_ints(ids),
    }
    for filename, data in files.items():
        (case_dir / filename).write_bytes(data)
    save_json(case_dir / 'metadata.json', dict(zip(('counts', 'starts', 'positions', 'group_maximums'), meta)))
    save_json(case_dir / 'traces.json', traces)
    record = dict(case=case_name, origin=origin, dtype=dtype, logical_tokens=len(ids),
                  group_maximums=meta[3], comparison=comparison, output_bytes=len(files['actual.fp32le']),
                  output_elements=V*C, one_writer_per_element=True, symmetric_program_traces=True,
                  stable_per_row_token_order=True, unused_rows_positive_zero=True,
                  per_program_active_occurrences=sum(r['active_lanes'] for r in traces[0]),
                  per_program_inactive_lanes=sum(r['skipped_lanes'] for r in traces[0]),
                  per_program_executed_vector_add_elements=sum(meta[3])*P*64,
                  effective_fp32_additions=len(ids)*C, support_checks=support_checks,
                  files={name:dict(bytes=len(data), sha256=sha(data)) for name,data in files.items()})
    save_json(case_dir / 'result.json', record)
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS['cpu_seconds'], LIMITS['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['address_space_bytes'], LIMITS['address_space_bytes']))
    ART.mkdir(exist_ok=False)
    start = time.monotonic()
    config = audit_source()
    arithmetic = numerical_self_tests()
    guards = guard_checks()
    negatives = negative_controls(config)
    # Only unchanged stdlib support is loaded; none of its algorithms supplies
    # the independent oracle or the candidate model.
    support = runpy.run_path(str(PACKET / 'gate-support/reference.py'))
    records = []
    names = ('empty', 'unique_columns', 'all_row0', 'all_row255', 'unequal_sparse',
             'boundary_rows', 'cross_tile_large', 'cross_tile_unit', 'signed_zero',
             'subnormal', 'rounding_ties', 'bf16_conversion_ties', 'random_0', 'random_1')
    for dtype in ('fp32', 'bf16'):
        for name in names:
            ids, gradient, meta = independent_fixture(name, dtype)
            records.append(run_case(len(records), 'independent', name, dtype, ids, gradient, meta, config, support))
    for dtype in ('fp32', 'bf16'):
        for name in support['CASE_NAMES']:
            ids, rows, meta = support['fixture'](name, dtype)
            gradient = [f32_bits(value) for row in rows for value in row]
            gradient += [0x6e120000] * (T*C-len(gradient))
            require(tuple(make_metadata(ids)) == tuple(meta), 'supplied metadata agrees independently')
            records.append(run_case(len(records), 'supplied', name, dtype, ids, gradient, meta, config, support))
    require(len(records) <= LIMITS['max_cases'], 'case bound')
    summary = dict(status='PASS_BOUNDED_SOURCE_CPU_ONLY', native_verdict='REFUSE_UNPROVEN',
                   source_sha256=EXPECTED_CANDIDATE, test_sha256=sha(Path(__file__).read_bytes()),
                   limits=LIMITS, arithmetic_self_tests=arithmetic, disabled_guard_checks=guards,
                   cases=len(records), independent_cases=sum(r['origin']=='independent' for r in records),
                   supplied_cases=sum(r['origin']=='supplied' for r in records),
                   negative_controls_detected=len(negatives), full_output_bytes_compared=sum(r['output_bytes'] for r in records),
                   effective_fp32_additions=sum(r['effective_fp32_additions'] for r in records),
                   elapsed_seconds=time.monotonic()-start,
                   cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   python=sys.version, native_executed=False,
                   limitations=['Hand-written source model, not installed NKI execution.',
                                'FP32 RNE with gradual underflow is modeled; hardware behavior remains unproved.',
                                'Bounded fixtures do not prove all finite inputs or any nonfinite input contract.',
                                'No compiler success, performance, training, or adoption conclusion.'])
    save_json(ART / 'summary.json', summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
