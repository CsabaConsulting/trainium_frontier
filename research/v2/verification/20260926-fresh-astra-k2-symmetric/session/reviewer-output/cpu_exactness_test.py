#!/usr/bin/env python3
"""Bounded, stdlib-only source/CPU checks; NOT an NKI implementation or compiler.

The scalar candidate model assumes .ap uses an element base offset and replaces
the indirect dimension's index with vector_offset, retaining that axis's stride.
The independent oracle visits original tokens directly and rounds an exact
integer sum to IEEE binary32, nearest/even. No NKI/SDK/model module is imported.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time

WORK = Path(__file__).resolve().parent
ARTIFACTS = WORK / 'cpu_artifacts'
V, C, T = 256, 128, 256
MASK = 0xFFFFFFFF


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binary32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def finite(bits):
    return (bits >> 23) & 255 != 255


def cpu_add(a, b):
    """Host scalar operation with explicit binary32 storage after each add."""
    assert finite(a) and finite(b), 'Nonfinite inputs do not establish exactness'
    try:
        result = bits32(binary32(a) + binary32(b))
    except OverflowError:
        raise AssertionError('Fixture overflow is outside this finite exactness test')
    assert finite(result), 'Nonfinite output does not establish exactness'
    return result


def lattice_integer(bits):
    """Represent a finite binary32 value as a signed multiple of 2**-149."""
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    assert exponent != 255
    magnitude = fraction if exponent == 0 else ((1 << 23) | fraction) << (exponent - 1)
    return -magnitude if bits >> 31 else magnitude


def exact_add(a, b):
    """Integer-only, round-to-nearest/even binary32 oracle for finite inputs."""
    total = lattice_integer(a) + lattice_integer(b)
    if total == 0:
        return 0x80000000 if a == b == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    shift = max(0, magnitude.bit_length() - 24)
    significand = magnitude >> shift
    remainder = magnitude - (significand << shift)
    if shift and (remainder > (1 << (shift - 1)) or
                  (remainder == (1 << (shift - 1)) and significand & 1)):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        shift += 1
    if shift == 0 and significand < 1 << 23:
        return sign | significand
    exponent = shift + 1
    assert exponent < 255, 'Oracle rejects overflow/nonfinite results'
    return sign | (exponent << 23) | (significand & 0x7FFFFF)


def packed(words):
    return struct.pack('<' + str(len(words)) + 'I', *words)


def expression(node):
    return ast.dump(node, include_attributes=False)


def matches(node, expected):
    assert expression(node) == expression(ast.parse(expected, mode='eval').body), ast.unparse(node)


def keyword(call, name):
    return next(k.value for k in call.keywords if k.arg == name)


def audit_source():
    data = (WORK / 'candidate_consumer_nki.py').read_bytes()
    manifest = json.loads(Path('/packet/INPUT_MANIFEST.json').read_text())
    assert digest(data) == manifest['source_sha256']['candidate']
    tree = ast.parse(data)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    matches(factory.args.kw_defaults[0], 'False')
    assert isinstance(factory.body[1], ast.If)
    matches(factory.body[1].test, 'not enabled')
    assert isinstance(factory.body[1].body[0], ast.Raise)
    assert all(isinstance(n, ast.Import) for n in factory.body[2:5])
    kernel = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
    matches(factory.body[-1].value, 'resident_consumer[2]')
    assertions = [n.test for n in kernel.body if isinstance(n, ast.Assert)]
    assert any(expression(n) == expression(ast.parse('nl.num_programs() == 2', mode='eval').body)
               for n in assertions)
    assignments = {n.targets[0].id: n.value for n in ast.walk(kernel)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    required_assignments = {
        'feature_first': 'nl.program_id(0) * 64',
        'first': 'group * 128',
        'output': 'nl.ndarray((256, 128), dtype=nl.float32, buffer=nl.shared_hbm)',
        'accumulator': 'nl.full((128, 64), 0.0, dtype=nl.float32, buffer=nl.sbuf)',
        'sentinel': 'nl.full((128, 1), 256, dtype=nl.int32, buffer=nl.sbuf)',
        'token': 'nl.full((128, 1), 256, dtype=nl.int32, buffer=nl.sbuf)',
        'value': 'nl.full((128, 64), 0.0, dtype=nl.float32, buffer=nl.sbuf)',
        'address': 'nl.where(active, pointer, sentinel, dtype=nl.int32)',
        'token_address': 'nl.where(active, token, sentinel, dtype=nl.int32)',
        'selected': ('nl.where(active.ap(pattern=[[1, 128], [0, 64]]), '
                     'updated, accumulator, dtype=nl.float32)'),
    }
    for name, expected in required_assignments.items():
        matches(assignments[name], expected)
    loops = [n for n in ast.walk(kernel) if isinstance(n, ast.For)]
    assert len(loops) == 2
    matches(loops[0].iter, 'range(2)')
    matches(loops[1].iter, 'nl.dynamic_range(limit)')
    assert not any(isinstance(n, (ast.If, ast.IfExp, ast.While)) for n in ast.walk(kernel))
    calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
    def find_call(name, destination=None):
        candidates = [n for n in calls if ast.unparse(n.func) == name]
        if destination is not None:
            candidates = [n for n in candidates if ast.unparse(keyword(n, 'dst')) == destination]
        assert len(candidates) == 1, (name, destination)
        return candidates[0]
    required_calls = {
        ('nisa.dma_copy', 'remaining'): 'nisa.dma_copy(dst=remaining, src=counts[first:first + 128, :])',
        ('nisa.dma_copy', 'pointer'): 'nisa.dma_copy(dst=pointer, src=starts[first:first + 128, :])',
        ('nisa.dma_copy', 'maximum'): 'nisa.dma_copy(dst=maximum, src=group_maximums[group:group + 1, :])',
        ('nisa.register_load', None): 'nisa.register_load(limit, maximum)',
        ('nisa.tensor_scalar', 'active'): ('nisa.tensor_scalar(dst=active, data=remaining, op0=nl.greater, '
                                          'operand0=0.0, engine=nisa.engine.vector)'),
        ('nisa.tensor_tensor', 'updated'): ('nisa.tensor_tensor(dst=updated, data1=accumulator, data2=value, '
                                           'op=nl.add, engine=nisa.engine.vector)'),
        ('nisa.tensor_copy', 'accumulator'): 'nisa.tensor_copy(dst=accumulator, src=selected)',
        ('nisa.tensor_scalar', 'pointer'): ('nisa.tensor_scalar(dst=pointer, data=pointer, op0=nl.add, '
                                           'operand0=1.0, engine=nisa.engine.vector)'),
        ('nisa.tensor_scalar', 'remaining'): ('nisa.tensor_scalar(dst=remaining, data=remaining, op0=nl.add, '
                                             'operand0=-1.0, engine=nisa.engine.vector)'),
    }
    for (name, destination), expected in required_calls.items():
        matches(find_call(name, destination), expected)
    gradient_ap = find_call('gradient.ap')
    positions_ap = find_call('positions.ap')
    matches(gradient_ap, 'gradient.ap(pattern=[[128, 128], [1, 64]], offset=feature_first, vector_offset=token_address, indirect_dim=0)')
    matches(positions_ap, 'positions.ap(pattern=[[1, 128], [1, 1]], vector_offset=address, indirect_dim=0)')
    for destination in ('token', 'value'):
        call = find_call('nisa.dma_copy', destination)
        matches(keyword(call, 'oob_mode'), 'nisa.oob_mode.skip')
        matches(keyword(call, 'dge_mode'), 'nisa.dge_mode.swdge')
    output_call = find_call('nisa.dma_copy', 'output[first:first + 128, feature_first:feature_first + 64]')
    matches(keyword(output_call, 'src'), 'accumulator')
    # Enumerate actual program-id/feature-first uses; no metadata/control uses.
    pid_calls = [n for n in calls if ast.unparse(n.func) == 'nl.program_id']
    assert len(pid_calls) == 1
    uses = [n for n in ast.walk(kernel) if isinstance(n, ast.Name) and
            n.id == 'feature_first' and isinstance(n.ctx, ast.Load)]
    assert sorted(n.lineno for n in uses) == [56, 72, 72]
    return dict(candidate_sha256=digest(data), source_ast_checks='PASS',
                grid=ast.literal_eval(factory.body[-1].value.slice),
                groups=ast.literal_eval(loops[0].iter.args[0]),
                group_rows=ast.literal_eval(assignments['first'].right),
                feature_width=ast.literal_eval(assignments['feature_first'].right),
                gradient_pattern=ast.literal_eval(keyword(gradient_ap, 'pattern')),
                positions_pattern=ast.literal_eval(keyword(positions_ap, 'pattern')),
                feature_first_load_lines=sorted(n.lineno for n in uses),
                limitation='AST checks link this scalar model to this source; they do not execute or validate NKI semantics.')


def metadata(rows):
    assert len(rows) == T
    buckets = [[] for _ in range(V)]
    for token, row in enumerate(rows):
        if row is not None:
            assert 0 <= row < V
            buckets[row].append(token)
    counts = [len(b) for b in buckets]
    positions, starts = [], []
    for bucket in buckets:
        starts.append(len(positions))
        assert bucket == sorted(bucket)
        positions.extend(bucket)
    assert len(positions) <= T
    positions.extend([T] * (T - len(positions)))
    maxima = [max(counts[:128]), max(counts[128:])]
    return counts, starts, positions, maxima


def input_gradient(dtype):
    words = []
    fp32 = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
            0x00800000, 0x80800000, 0x3F800000, 0xBF800000,
            0x3F000000, 0xBF000000, 0x33800000, 0xB3800000]
    bf16 = [0, 0x80000000, 0x00010000, 0x80010000, 0x007F0000, 0x807F0000,
            0x00800000, 0x80800000, 0x3F800000, 0xBF800000,
            0x3F000000, 0xBF000000, 0x33800000, 0xB3800000]
    palette = fp32 if dtype == 'fp32' else bf16
    for t in range(T):
        for c in range(C):
            # Constant column signatures reveal overlap, shifted halves, and swaps.
            if c in (2, 3, 4, 5, 63, 64, 65, 127):
                value = bits32(float(c + 1))
            elif c == 6:
                value = 0x80000000
            elif c == 7:
                value = 1 if dtype == 'fp32' else 0x00010000
            elif c == 8:
                value = 0x80000001 if dtype == 'fp32' else 0x80010000
            else:
                value = palette[(t * 7 + c * 3) % len(palette)]
            words.append(value)
    # Both dtypes exactly encode these order-sensitive and halfway sequences.
    for t, value in enumerate((2.0**24, 1.0, -2.0**24)):
        words[t * C] = bits32(value)
    for t, value in enumerate((1.0, 2.0**-24, 2.0**-24)):
        words[t * C + 1] = bits32(value)
    if dtype == 'bf16':
        assert all(v & 0xFFFF == 0 for v in words)
    assert all(finite(v) for v in words)
    return words


def reference(rows, gradient):
    """Direct original-token traversal, no grouped metadata or feature partition."""
    output = [0] * (V * C)
    for token, row in enumerate(rows):
        if row is not None:
            for feature in range(C):
                idx = row * C + feature
                output[idx] = exact_add(output[idx], gradient[token * C + feature])
    return output


def candidate_model(rows, gradient, spec, mutation=None):
    counts, starts, positions, maxima = metadata(rows)
    if mutation == 'unstable_positions':
        assert counts[0] >= 3
        positions[starts[0] + 1], positions[starts[0] + 2] = positions[starts[0] + 2], positions[starts[0] + 1]
    output = [0xDEADBEEF] * (V * C)
    writers = bytearray(V * C)
    owners = bytearray([255]) * (V * C)
    traces = []
    addition_count = 0
    skipped_rows = 0
    for pid in range(spec['grid']):
        feature_first = pid * spec['feature_width']
        if mutation == 'overlapping_halves':
            feature_first = 0
        trace = []
        for group in range(spec['groups']):
            first = group * spec['group_rows']
            remaining = counts[first:first + 128].copy()
            pointer = starts[first:first + 128].copy()
            limit = maxima[group]
            if mutation == 'pid_changes_limit' and pid == 1:
                limit = max(0, limit - 1)
            accumulator = [[0] * 64 for _ in range(128)]
            iterations = []
            for iteration in range(limit):
                active_events = []
                state = []
                for local_row in range(128):
                    active = remaining[local_row] > 0
                    address = pointer[local_row] if active else T
                    token = positions[address] if address < T else T
                    token_address = token if active else T
                    state.extend((remaining[local_row], pointer[local_row], address, token_address))
                    if active:
                        assert 0 <= address < T and 0 <= token < T
                        active_events.append([first + local_row, address, token])
                        for c in range(64):
                            row_stride = spec['gradient_pattern'][0][0]
                            feature_stride = spec['gradient_pattern'][1][0]
                            offset = feature_first
                            if mutation == 'omit_feature_offset':
                                offset = 0
                            source_index = token_address * row_stride + offset + c * feature_stride
                            assert 0 <= source_index < T * C
                            accumulator[local_row][c] = cpu_add(accumulator[local_row][c], gradient[source_index])
                            addition_count += 1
                    else:
                        assert address == token_address == T
                        skipped_rows += 1
                        # Source computes an unused update, then selects prior bits.
                        # Keeping the same words emulates that select's false arm.
                    pointer[local_row] += 1
                    remaining[local_row] -= 1
                state_bytes = struct.pack('<' + str(len(state)) + 'i', *state)
                iterations.append(dict(iteration=iteration, active=active_events,
                                       inactive_rows=128 - len(active_events),
                                       metadata_state_sha256=digest(state_bytes)))
            trace.append(dict(group=group, limit=limit, iterations=iterations))
            for r in range(128):
                for c in range(64):
                    index = (first + r) * C + feature_first + c
                    writers[index] += 1
                    owners[index] = pid
                    output[index] = accumulator[r][c]
        traces.append(trace)
    return output, writers, owners, traces, addition_count, skipped_rows


def arithmetic_checks():
    probes = [
        (0, 0x80000000, 0), (0x80000000, 0x80000000, 0x80000000),
        (1, 1, 2), (0x80000001, 0x80000001, 0x80000002),
        (0x007FFFFF, 1, 0x00800000), (0x00800000, 0x80000001, 0x007FFFFF),
        (0x3F800000, 0xBF800000, 0), (0x3F800000, 0x33800000, 0x3F800000),
        (0x3F800001, 0x33800000, 0x3F800002),
        (0x4B800000, 0x3F800000, 0x4B800000),
    ]
    for a, b, expected in probes:
        assert exact_add(a, b) == cpu_add(a, b) == expected
    state = 0x51A71E
    for _ in range(4096):
        state = (1664525 * state + 1013904223) & MASK
        a = state
        state = (1664525 * state + 1013904223) & MASK
        b = state
        # Bound exponents below 252, avoiding overflow for every generated pair.
        a = (a & 0x807FFFFF) | (((a >> 23) % 252) << 23)
        b = (b & 0x807FFFFF) | (((b >> 23) % 252) << 23)
        assert exact_add(a, b) == cpu_add(a, b), (hex(a), hex(b))
    # State-level inactive select witness, including bits not reached from +0
    # in these finite RNE accumulation fixtures. This is not a NaN exactness pass.
    prior = [0x80000000, 1, 0x80000001, 0x7FC01234]
    updated = [0, 0, 0, 0x7FC00000]
    selected = [updated[i] if False else prior[i] for i in range(len(prior))]
    assert packed(selected) == packed(prior)
    assert packed(updated) != packed(prior)
    return dict(known_bit_vectors=len(probes), bounded_generated_pairs=4096,
                inactive_select_prior_hex=[f'{x:08x}' for x in prior],
                inactive_select_result_hex=[f'{x:08x}' for x in selected],
                nonfinite_exactness_cases=0,
                nan_payload_scope='Opaque select-bit witness only, not arithmetic or native exactness.')


def main():
    started = time.monotonic()
    ARTIFACTS.mkdir(exist_ok=True)
    spec = audit_source()
    arithmetic = arithmetic_checks()
    fixtures = {
        'empty': [None] * T,
        'unequal_3_5': [0, 0, 0, 128, 128, 128, 128, 128] + [None] * 248,
        'unequal_5_3': [128, 128, 128, 0, 0, 0, 0, 0] + [None] * 248,
        'group0_253': [127] * 253 + [None] * 3,
        'group1_253': [255] * 253 + [None] * 3,
        'all256_group1': [255] * T,
        'interleaved_129_127': [r for _ in range(127) for r in (127, 128)] + [127, 127],
        'one_per_row_permuted': [(i * 73) % V for i in range(T)],
    }
    receipts = []
    for dtype in ('fp32', 'bf16'):
        gradient = input_gradient(dtype)
        native_input = (packed(gradient) if dtype == 'fp32' else
                        struct.pack('<' + str(len(gradient)) + 'H', *(v >> 16 for v in gradient)))
        # BF16 is widened exactly by shifting its stored bits, with no rounding.
        if dtype == 'bf16':
            gradient = [x[0] << 16 for x in struct.iter_unpack('<H', native_input)]
        (ARTIFACTS / (dtype + '_gradient.bin')).write_bytes(native_input)
        for name, rows in fixtures.items():
            case_name = dtype + '_' + name
            case_dir = ARTIFACTS / case_name
            case_dir.mkdir(exist_ok=True)
            counts, starts, positions, maxima = metadata(rows)
            oracle = reference(rows, gradient)
            actual, writers, owners, traces, additions, skipped = candidate_model(rows, gradient, spec)
            expected_bytes, actual_bytes = packed(oracle), packed(actual)
            assert actual_bytes == expected_bytes, case_name
            assert Counter(writers) == {1: V * C}, case_name
            assert all(owners[r * C + c] == c // 64 for r in range(V) for c in range(C))
            assert traces[0] == traces[1], case_name
            assert additions == sum(counts) * C
            assert all(actual[r * C + c] == 0 for r in range(V) if not counts[r] for c in range(C))
            for pid_trace in traces:
                observed = [[] for _ in range(V)]
                for group_trace in pid_trace:
                    for iteration in group_trace['iterations']:
                        for row, address, token in iteration['active']:
                            observed[row].append(token)
                for row in range(V):
                    assert observed[row] == [t for t, target in enumerate(rows) if target == row]
            (case_dir / 'expected.f32le.bin').write_bytes(expected_bytes)
            (case_dir / 'actual.f32le.bin').write_bytes(actual_bytes)
            (case_dir / 'writers.u8.bin').write_bytes(writers)
            (case_dir / 'owners.u8.bin').write_bytes(owners)
            json_write(case_dir / 'fixture.json', dict(original_token_rows=rows, counts=counts,
                       starts=starts, positions=positions, group_maximums=maxima,
                       gradient_file='../' + dtype + '_gradient.bin'))
            json_write(case_dir / 'schedules.json', dict(program0=traces[0], program1=traces[1]))
            receipt = dict(case=case_name, status='PASS', group_maximums=maxima,
                           occurrences=sum(counts), fp32_additions=additions,
                           skipped_row_iterations=skipped, bytes_compared=len(actual_bytes),
                           actual_sha256=digest(actual_bytes), expected_sha256=digest(expected_bytes),
                           exactly_one_writer_elements=V*C, owner_counts=dict(Counter(owners)),
                           schedules_identical=True, stable_token_order=True,
                           unused_rows_positive_zero=True)
            receipts.append(receipt)
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    rows = fixtures['unequal_3_5']
    gradient = input_gradient('fp32')
    expected = packed(reference(rows, gradient))
    controls = []
    for mutation in ('overlapping_halves', 'omit_feature_offset', 'pid_changes_limit', 'unstable_positions'):
        actual, writers, _, traces, _, _ = candidate_model(rows, gradient, spec, mutation)
        failed = []
        if packed(actual) != expected:
            failed.append('full_output_bytes')
        if Counter(writers) != {1: V*C}:
            failed.append('one_writer')
        if traces[0] != traces[1]:
            failed.append('symmetric_schedule')
        assert failed, mutation
        controls.append(dict(mutation=mutation, detected_by=failed))
    values = [bits32(2.0**24), bits32(1.0), bits32(-2.0**24)]
    ordered = 0
    for value in values:
        ordered = exact_add(ordered, value)
    reassociated = exact_add(exact_add(values[0], values[2]), values[1])
    assert ordered == 0 and reassociated == bits32(1.0)
    # The direct numerical witness is equivalent to positions [0, 2, 1].
    controls.append(dict(mutation='reassociated_additions',
                         ordered_hex=f'{ordered:08x}', changed_order_hex=f'{reassociated:08x}',
                         detected_by=['ordered_FP32_bits']))
    subnormal = exact_add(0, 1)
    assert subnormal != 0
    controls.append(dict(mutation='flush_FP32_subnormal_to_zero', expected_hex='00000001',
                         mutated_hex='00000000', detected_by=['raw_bits']))
    assert struct.pack('<I', 0) != struct.pack('<I', 0x80000000)
    controls.append(dict(mutation='negative_zero_in_unused_output',
                         detected_by=['full_output_bytes'], expected_hex='00000000', mutated_hex='80000000'))
    result = dict(status='PASS_CPU_MODEL_ONLY', python=sys.version, platform=platform.platform(),
                  source=spec, arithmetic=arithmetic, cases=receipts, negative_controls=controls,
                  total_cases=len(receipts), total_bytes_compared=sum(r['bytes_compared'] for r in receipts),
                  total_fp32_additions=sum(r['fp32_additions'] for r in receipts),
                  max_fixture_tokens=T, max_fixture_gradient_elements=T*C,
                  maximum_dynamic_bound=256, rounding='binary32 round-to-nearest, ties-to-even',
                  elapsed_seconds=time.monotonic() - started,
                  limits=['Scalar CPU model, not execution of the NKI candidate.',
                          'Assumed .ap offset, indirect addressing and skipped-DMA semantics.',
                          'No evidence about NKI lowering, device arithmetic, LNC placement, speed or training.',
                          'Finite fixtures only; no nonfinite arithmetic exactness claim.'])
    json_write(WORK / 'cpu_results.json', result)
    print(json.dumps(dict(status=result['status'], total_cases=result['total_cases'],
                          total_bytes_compared=result['total_bytes_compared'],
                          negative_controls=controls), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
