"""Independent bounded stdlib review. Synthetic NKI operations are assumptions,
not a reproduction or probe of an installed frontend/API/device.
The reconstructed packet modules are never edited. Only this model is authored.
"""
import ast
import builtins
import datetime
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import threading
import time
import traceback
import types

OUT = Path('/work')
PACKET = Path('/packet')
START = time.monotonic()
START_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
CPU = min(os.sched_getaffinity(0))
os.sched_setaffinity(0, {CPU})
resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
signal.alarm(300)
RESULT = {'started_utc': START_UTC, 'tests': [], 'cases': [],
          'expected_errors': [], 'unexpected_errors': [],
          'native_executed': False, 'model_is_installed_frontend': False}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def check(name, condition, **details):
    entry = {'name': name, 'pass': bool(condition), **details}
    RESULT['tests'].append(entry)
    print(json.dumps(entry, sort_keys=True), flush=True)
    if not condition:
        raise AssertionError(name)


def expect_error(name, exception_type, operation, contains=None):
    try:
        operation()
    except exception_type as error:
        raw = traceback.format_exc()
        RESULT['expected_errors'].append({'name': name, 'type': type(error).__name__,
                                          'message': str(error), 'traceback': raw})
        check(name, contains is None or contains in str(error), actual_error=str(error))
    else:
        check(name, False, actual_error='no exception')


def reconstruct_and_verify():
    manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_bytes())
    pins = []
    for item in manifest['included']:
        data = (PACKET / item['path']).read_bytes()
        entry = {**item, 'actual_bytes': len(data), 'actual_sha256': sha(data)}
        pins.append(entry)
        check('input_pin:' + item['path'], len(data) == item['bytes'] and sha(data) == item['sha256'])
    old, new, changes = [], [], []
    inside = False
    for line in (PACKET / 'kernel-full-context.diff').read_bytes().splitlines(keepends=True):
        if line.startswith(b'@@ '):
            check('single_complete_hunk', not inside and line == b'@@ -1,71 +1,72 @@\n')
            inside = True
            continue
        if not inside:
            continue
        if line[:1] == b' ':
            old.append(line[1:])
            new.append(line[1:])
        elif line[:1] == b'+':
            new.append(line[1:])
            changes.append(line)
        elif line[:1] == b'-':
            old.append(line[1:])
            changes.append(line)
        else:
            raise ValueError('Unsupported patch line: ' + repr(line))
    old, new = b''.join(old), b''.join(new)
    check('one_permitted_declaration', changes == [b'+    global nl, nisa\n'])
    for label, source, lines in [('baseline', old, 71), ('candidate', new, 72)]:
        check('source_pin:' + label, sha(source) == manifest['source_sha256'][label]
              and len(source.splitlines()) == lines, sha256=sha(source), bytes=len(source))
        (OUT / (label + '.py')).write_bytes(source)
    old_ast, new_ast = ast.parse(old), ast.parse(new)
    old_factory, new_factory = old_ast.body[1], new_ast.body[1]
    old_kernel = next(node for node in old_factory.body if isinstance(node, ast.FunctionDef))
    new_kernel = next(node for node in new_factory.body if isinstance(node, ast.FunctionDef))
    # Include decorator through end of function, preserving whitespace/comments.
    def kernel_bytes(source, node):
        return b''.join(source.splitlines(keepends=True)[node.decorator_list[0].lineno - 1:node.end_lineno])
    kb_old, kb_new = kernel_bytes(old, old_kernel), kernel_bytes(new, new_kernel)
    check('entire_kernel_bytes_and_ast_identical', kb_old == kb_new and
          ast.dump(old_kernel) == ast.dump(new_kernel), sha256=sha(kb_new), bytes=len(kb_new))
    check('factory_signature_identical', ast.dump(old_factory.args) == ast.dump(new_factory.args))
    declaration = new_factory.body.pop(0)
    check('entire_module_ast_minus_declaration_identical', isinstance(declaration, ast.Global)
          and declaration.names == ['nl', 'nisa'] and ast.dump(old_ast) == ast.dump(new_ast))
    check('entire_module_bytes_minus_declaration_identical',
          new.replace(b'    global nl, nisa\n', b'', 1) == old)
    receipts = []
    supplied = OUT / 'supplied'
    supplied.mkdir(exist_ok=True)
    for rank in range(4):
        raw = (PACKET / f'original-native/receipts/rank{rank}.json').read_bytes()
        r = json.loads(raw)
        (supplied / f'rank{rank}.json').write_bytes(raw)
        (supplied / f'rank{rank}.error.txt').write_text(r['error'])
        check(f'receipt_identity_and_failure:{rank}', r['rank'] == rank and r['world_size'] == 4
              and r['kernel_sha256'] == sha(old) and r['status'] == 'REFUSE_OR_UNSUPPORTED'
              and "failed to resolve name 'nl.bfloat16'" in r['error'] and r['timings'] == []
              and r['tolerance'] == {'atol': 0, 'rtol': 0, 'criterion': 'identical bytes'})
        receipts.append({k: r[k] for k in ['rank', 'kernel_sha256', 'checker_sha256',
                                          'reference_sha256', 'status', 'nki_version', 'tolerance']})
    (supplied / 'terminal.json').write_bytes((PACKET / 'original-native/terminal.json').read_bytes())
    check('checker_reference_pins_consistent_identity_only',
          len({r['checker_sha256'] for r in receipts}) == 1 and
          len({r['reference_sha256'] for r in receipts}) == 1)
    save_json('input-verification.json', {'manifest_sha256': sha((PACKET / 'INPUT_MANIFEST.json').read_bytes()),
                                        'inputs': pins, 'receipts': receipts,
                                        'checker_reference_bodies_inspected': False})
    return old, new


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]


def exact_add(a, b):
    """Integer-only finite binary32 addition, round-to-nearest, ties-to-even.
    Represent values as signed integer multiples of 2**-149, then round once.
    No use of the CPU model's float addition or f32 conversion.
    """
    def units(x):
        exponent, fraction = (x >> 23) & 255, x & 0x7fffff
        assert exponent != 255, 'finite test input required'
        magnitude = fraction if exponent == 0 else ((1 << 23) | fraction) << (exponent - 1)
        return -magnitude if x >> 31 else magnitude
    total = units(a) + units(b)
    if total == 0:
        return 0x80000000 if a == b == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    if magnitude < (1 << 23):
        return sign | magnitude
    shift = max(0, magnitude.bit_length() - 24)
    rounded = magnitude >> shift
    if shift:
        remainder = magnitude - (rounded << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and rounded & 1):
            rounded += 1
    if rounded == (1 << 24):
        rounded >>= 1
        shift += 1
    exponent = shift + 1
    if exponent >= 255:
        return sign | 0x7f800000
    return sign | (exponent << 23) | (rounded & 0x7fffff)


class Tensor:
    max_elements = 0

    def __init__(self, shape, dtype, fill=0, data=None, offset=0):
        assert len(shape) == 2 and 0 < shape[0] <= 256 and 0 < shape[1] <= 128
        self.shape, self.dtype, self.offset = shape, dtype, offset
        n = shape[0] * shape[1]
        Tensor.max_elements = max(Tensor.max_elements, n)
        self.data = [fill] * n if data is None else data

    def __getitem__(self, key):
        row, col = key
        assert isinstance(row, slice) and col == slice(None)
        start, stop, step = row.indices(self.shape[0])
        assert step == 1
        return Tensor((stop - start, self.shape[1]), self.dtype,
                      data=self.data, offset=self.offset + start * self.shape[1])

    def ap(self, *, pattern, vector_offset=None, indirect_dim=None):
        if vector_offset is None:
            assert self.shape == (128, 1) and pattern == [[1, 128], [0, 128]]
            return Broadcast(self)
        assert indirect_dim == 0
        assert (self.shape == (256, 1) and pattern == [[1, 128], [1, 1]]) or (
            self.shape == (256, 128) and pattern == [[128, 128], [1, 128]])
        return Gather(self, vector_offset)


class Broadcast:
    def __init__(self, source):
        self.source = source


class Gather:
    def __init__(self, source, addresses):
        self.source, self.addresses = source, addresses


def data(tensor):
    n = tensor.shape[0] * tensor.shape[1]
    return tensor.data[tensor.offset:tensor.offset + n]


class CPUModel:
    """Explicit assumed semantics, intentionally not the real NKI package."""
    def __init__(self):
        self.skipped_rows = 0
        self.calls = []
        self.nl = types.ModuleType('nki.language')
        self.nisa = types.ModuleType('nki.isa')
        self.nki = types.ModuleType('nki')
        self.nki.language, self.nki.isa = self.nl, self.nisa
        self.nki.jit = lambda fn: fn
        for name in ['bfloat16', 'float32', 'int32', 'shared_hbm', 'sbuf', 'greater', 'add']:
            setattr(self.nl, name, name)
        self.nl.ndarray = lambda shape, dtype, buffer: Tensor(shape, dtype, fill=None)
        self.nl.full = lambda shape, x, dtype, buffer: Tensor(shape, dtype, fill=int(x) if dtype == 'int32' else f32(x))
        self.nl.where = self.where
        self.nl.dynamic_range = lambda limit: range(limit[0])
        self.nisa.engine = types.SimpleNamespace(vector='vector')
        self.nisa.oob_mode = types.SimpleNamespace(skip='skip')
        self.nisa.dge_mode = types.SimpleNamespace(swdge='swdge')
        self.nisa.dma_copy = self.dma_copy
        self.nisa.register_alloc = lambda x: [x]
        self.nisa.register_load = lambda register, tensor: register.__setitem__(0, data(tensor)[0])
        self.nisa.tensor_scalar = self.tensor_scalar
        self.nisa.tensor_tensor = self.tensor_tensor
        self.nisa.tensor_copy = lambda dst, src: self.dma_copy(dst=dst, src=src)

    def importer(self, name, globals=None, locals=None, fromlist=(), level=0):
        self.calls.append(name)
        assert name in ('nki', 'nki.isa', 'nki.language'), name
        return self.nki

    def where(self, condition, yes, no, dtype):
        assert yes.shape == no.shape and yes.dtype == no.dtype == dtype
        if isinstance(condition, Broadcast):
            flags = [v for v in data(condition.source) for _ in range(yes.shape[1])]
        else:
            flags = data(condition)
        return Tensor(yes.shape, dtype, data=[a if c else b for c, a, b in zip(flags, data(yes), data(no), strict=True)])

    def dma_copy(self, *, dst, src, oob_mode=None, dge_mode=None):
        if isinstance(src, Gather):
            assert oob_mode == 'skip' and dge_mode == 'swdge'
            width = src.source.shape[1]
            assert dst.shape == (128, width)
            for row, address in enumerate(data(src.addresses)):
                if not 0 <= address < src.source.shape[0]:
                    self.skipped_rows += 1
                    continue
                start = src.source.offset + address * width
                target = dst.offset + row * width
                dst.data[target:target + width] = src.source.data[start:start + width]
        else:
            assert dst.shape == src.shape
            dst.data[dst.offset:dst.offset + dst.shape[0] * dst.shape[1]] = data(src)

    def tensor_scalar(self, *, dst, data, op0, operand0, engine):
        assert engine == 'vector'
        source = globals()['data'](data)
        if op0 == 'greater':
            values = [float(x > operand0) for x in source]
        else:
            assert op0 == 'add'
            values = [int(x + operand0) if dst.dtype == 'int32' else f32(x + operand0) for x in source]
        dst.data[dst.offset:dst.offset + len(values)] = values

    def tensor_tensor(self, *, dst, data1, data2, op, engine):
        assert dst.dtype == 'float32' and op == 'add' and engine == 'vector'
        dst.data[dst.offset:dst.offset + dst.shape[0] * dst.shape[1]] = [
            f32(a + b) for a, b in zip(data(data1), data(data2), strict=True)]


def load_module(source, name, importer):
    builtin_map = dict(vars(builtins))
    builtin_map['__import__'] = importer
    namespace = {'__name__': name, '__builtins__': builtin_map}
    exec(compile(source, str(OUT / (name + '.py')), 'exec'), namespace)
    return namespace


def namespace_tests(old, new):
    disabled_calls = []
    def forbidden_import(name, *args, **kwargs):
        disabled_calls.append(name)
        raise ImportError('blocked NKI import: ' + name)
    evidence = {}
    for label, source in [('baseline', old), ('candidate', new)]:
        namespace = load_module(source, label, forbidden_import)
        check(label + ':module_load_is_lazy', disabled_calls == [] and 'nl' not in namespace and 'nisa' not in namespace)
        factory = namespace['build_kernel']
        check(label + ':signature_default_false', str(inspect.signature(factory)) == '(*, enabled=False)')
        expect_error(label + ':default_disabled', RuntimeError, factory, 'EXPERIMENT_DISABLED: no native module imported')
        expect_error(label + ':explicit_disabled', RuntimeError, lambda: factory(enabled=False), 'EXPERIMENT_DISABLED')
        check(label + ':disabled_no_import_no_aliases', not disabled_calls and 'nl' not in namespace and 'nisa' not in namespace)
        expect_error(label + ':enabled_import_negative_control', ImportError, lambda: factory(enabled=True), 'blocked NKI import: nki')
        check(label + ':enabled_reaches_import', disabled_calls == ['nki'])
        disabled_calls.clear()
        model = CPUModel()
        namespace = load_module(source, label, model.importer)
        kernel = namespace['build_kernel'](enabled=True)
        closure = inspect.getclosurevars(kernel)
        evidence[label] = {'freevars': list(kernel.__code__.co_freevars),
                           'global_aliases': [k for k in ['nl', 'nisa'] if k in namespace],
                           'nonlocal_aliases': sorted(closure.nonlocals), 'imports': model.calls.copy()}
        check(label + ':enabled_imports_exact_order', model.calls == ['nki', 'nki.isa', 'nki.language'])
        if label == 'baseline':
            check('baseline_aliases_are_closure_cells', set(kernel.__code__.co_freevars) == {'nl', 'nisa'}
                  and 'nl' not in namespace and closure.nonlocals['nl'] is model.nl)
            expect_error('baseline_global_lookup_negative_control', KeyError, lambda: kernel.__globals__['nl'])
        else:
            check('candidate_aliases_are_module_globals', kernel.__code__.co_freevars == ()
                  and kernel.__globals__['nl'] is model.nl and kernel.__globals__['nisa'] is model.nisa)
        # Changed sharing consequence: an enabled rebuild can rebind aliases visible
        # to already-returned candidate functions; baseline functions retain cells.
        previous_language = model.nl
        model.nl = types.ModuleType('replacement.language')
        model.nl.bfloat16 = 'replacement_bfloat16'
        model.nki.language = model.nl
        namespace['build_kernel'](enabled=True)
        resolved = inspect.getclosurevars(kernel)
        alias = resolved.globals.get('nl', resolved.nonlocals.get('nl'))
        check(label + ':rebuild_binding_consequence', alias is (previous_language if label == 'baseline' else model.nl))
        imports_before = model.calls.copy()
        expect_error(label + ':disabled_after_enabled', RuntimeError,
                     lambda: namespace['build_kernel'](), 'EXPERIMENT_DISABLED')
        check(label + ':disabled_after_enabled_no_import', imports_before == model.calls)
        evidence[label]['alias_resolution_after_rebind'] = 'original closure' if label == 'baseline' else 'new global'
        (OUT / (label + '.dis.txt')).write_text(dis.Bytecode(kernel).dis())
    save_json('namespace-evidence.json', evidence)


def make_metadata(ids):
    assert len(ids) == 256 and all(0 <= v < 256 for v in ids)
    counts, starts, positions = [], [], []
    for row in range(256):
        tokens = [t for t, target in enumerate(ids) if target == row]
        starts.append(len(positions))
        counts.append(len(tokens))
        positions.extend(tokens)
    maximums = [max(counts[:128]), max(counts[128:])]
    return counts, starts, positions, maximums


def validate_metadata(ids, counts, starts, positions, maximums):
    assert len(counts) == len(starts) == len(positions) == 256 and len(maximums) == 2
    assert sorted(positions) == list(range(256))
    for row in range(256):
        expected = [t for t, target in enumerate(ids) if target == row]
        assert counts[row] == len(expected)
        assert positions[starts[row]:starts[row] + counts[row]] == expected
    assert maximums == [max(counts[:128]), max(counts[128:])]


def oracle(ids, gradient_bits):
    # Token-major scatter, independent of starts/positions/group loop layout.
    output = [0] * (256 * 128)
    for token, row in enumerate(ids):
        for channel in range(128):
            index = row * 128 + channel
            output[index] = exact_add(output[index], gradient_bits[token * 128 + channel])
    return output


def raw_bits(values):
    return struct.pack('<' + 'I' * len(values), *values)


def gradient_fixture(dtype, zero=False):
    # Exact binary32 encodings. BF16 values have low 16 bits zero.
    pool = [0, 0x80000000, 0x3f800000, 0xbf800000, 0x4b800000, 0xcb800000,
            0x3e800000, 0xbe800000, 0x40800000, 0xc0800000, 0x3b800000]
    if dtype == 'float32':
        pool += [0x3f800001, 0x33800000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000]
    return [(0x80000000 if (t + c) % 2 else 0) if zero else pool[(t * 7 + c * 3) % len(pool)]
            for t in range(256) for c in range(128)]


def arithmetic_tests(old, new):
    known = [
        (0x3f800000, 0x33800000, 0x3f800000),  # even halfway
        (0x3f800001, 0x33800000, 0x3f800002),  # odd halfway
        (0x4b800000, 0x3f800000, 0x4b800000),
        (0x4b800000, 0xcb800000, 0),
        (1, 1, 2), (0x007fffff, 1, 0x00800000),
        (0x80000000, 0x80000000, 0x80000000), (0, 0x80000000, 0)]
    check('integer_oracle_known_answers', all(exact_add(a, b) == expected for a, b, expected in known))
    check('float_model_known_answers', all(bits(f32(value(a) + value(b))) == expected for a, b, expected in known))
    # The reference must reject order changes and a double-precision sum rounded once.
    sequence = [0x4b800000, 0x3f800000, 0xcb800000]
    def fold(sequence):
        result = 0
        for x in sequence:
            result = exact_add(result, x)
        return result
    ordered, reordered = fold(sequence), fold([sequence[0], sequence[2], sequence[1]])
    widened = bits(f32(sum(map(value, sequence))))
    check('negative_control_reordered_and_widened_sum_detected', ordered == 0 and reordered == widened == 0x3f800000,
          ordered_hex=f'{ordered:08x}', reordered_hex=f'{reordered:08x}', widened_hex=f'{widened:08x}')
    byte_control = raw_bits([0, 0x3f800000])
    signed_zero_mutation = raw_bits([0x80000000, 0x3f800000])
    one_ulp_mutation = raw_bits([0, 0x3f800001])
    check('negative_control_signed_zero_and_one_ulp_rejected',
          byte_control != signed_zero_mutation and byte_control != one_ulp_mutation)
    controls = {'order_input_hex': [f'{b:08x}' for b in sequence], 'ordered_hex': f'{ordered:08x}',
                'reordered_hex': f'{reordered:08x}', 'widened_hex': f'{widened:08x}',
                'bytes_expected': byte_control.hex(), 'bytes_wrong_sign': signed_zero_mutation.hex(),
                'bytes_wrong_ulp': one_ulp_mutation.hex()}
    save_json('negative-controls.json', controls)
    mixed = [0 if t < 8 else 127 if t < 11 else 128 if t < 16 else 255 if t < 23 else t for t in range(256)]
    fixtures = [
        ('unique_permuted', 'bfloat16', [(t * 73) % 256 for t in range(256)], False),
        ('unique_permuted', 'float32', [(t * 73) % 256 for t in range(256)], False),
        ('mixed_boundary_rows', 'bfloat16', mixed, False),
        ('mixed_boundary_rows', 'float32', mixed, False),
        ('signed_zeros_unused_group', 'float32', [0] * 256, True),
        ('hot253_cross_token_tile', 'bfloat16', [127] * 253 + [0, 128, 255], False),
        ('hot256_second_group', 'float32', [255] * 256, False),
    ]
    artifacts = OUT / 'exactness'
    artifacts.mkdir(exist_ok=True)
    for name, dtype, ids, zero in fixtures:
        case_name = name + '_' + dtype
        print('START_CASE ' + case_name, flush=True)
        gradient_bits = gradient_fixture(dtype, zero)
        check(case_name + ':finite_and_dtype', all(((b >> 23) & 255) != 255 for b in gradient_bits)
              and (dtype != 'bfloat16' or all(b & 0xffff == 0 for b in gradient_bits)))
        counts, starts, positions, maximums = make_metadata(ids)
        validate_metadata(ids, counts, starts, positions, maximums)
        if name == 'mixed_boundary_rows' and dtype == 'float32':
            wrong_positions = positions.copy()
            wrong_positions[0], wrong_positions[1] = wrong_positions[1], wrong_positions[0]
            expect_error('unstable_metadata_negative_control', AssertionError,
                         lambda: validate_metadata(ids, counts, starts, wrong_positions, maximums))
            wrong_maximums = [max(0, maximums[0] - 1), maximums[1]]
            expect_error('understated_maximum_negative_control', AssertionError,
                         lambda: validate_metadata(ids, counts, starts, positions, wrong_maximums))
        reference = raw_bits(oracle(ids, gradient_bits))
        (artifacts / (case_name + '.reference.f32le')).write_bytes(reference)
        (artifacts / (case_name + '.gradient.f32le')).write_bytes(raw_bits(gradient_bits))
        save_json('exactness/' + case_name + '.metadata.json',
                  {'ids': ids, 'counts': counts, 'starts': starts, 'positions': positions,
                   'group_maximums': maximums, 'dtype': dtype,
                   'gradient_storage': 'expanded exact IEEE binary32 little endian; BF16 low 16 bits zero'})
        record = {'name': case_name, 'dtype': dtype, 'shape': [256, 128], 'group_maximums': maximums,
                  'reference_sha256': sha(reference), 'outputs': {}}
        for label, source in [('baseline', old), ('candidate', new)]:
            model = CPUModel()
            namespace = load_module(source, label, model.importer)
            kernel = namespace['build_kernel'](enabled=True)
            tensors = [Tensor((256, 128), dtype, data=list(map(value, gradient_bits)))]
            for shape, numbers in [((256, 1), counts), ((256, 1), starts),
                                   ((256, 1), positions), ((2, 1), maximums)]:
                tensors.append(Tensor(shape, 'int32', data=numbers.copy()))
            output = kernel(*tensors)
            actual_bits = [bits(x) for x in data(output)]
            actual = raw_bits(actual_bits)
            (artifacts / (case_name + '.' + label + '.f32le')).write_bytes(actual)
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b), None)
            check(case_name + ':' + label + ':exact_bytes', actual == reference,
                  actual_sha256=sha(actual), expected_sha256=sha(reference), first_mismatch_byte=mismatch)
            check(case_name + ':' + label + ':unused_rows_positive_zero',
                  all(actual_bits[row * 128 + c] == 0 for row in range(256) if counts[row] == 0 for c in range(128)))
            record['outputs'][label] = {'sha256': sha(actual), 'bytes': len(actual),
                                         'exact': actual == reference, 'skipped_rows': model.skipped_rows}
        RESULT['cases'].append(record)
    # Actual source assertion controls with fake imports, before arithmetic allocation.
    model = CPUModel()
    kernel = load_module(new, 'candidate', model.importer)['build_kernel'](enabled=True)
    args = [Tensor((256, 128), 'float32')] + [Tensor((256, 1), 'int32') for _ in range(3)] + [Tensor((2, 1), 'int32')]
    bad_shape = args.copy()
    bad_shape[0] = Tensor((128, 128), 'float32')
    expect_error('source_rejects_wrong_gradient_shape', AssertionError, lambda: kernel(*bad_shape))
    bad_dtype = args.copy()
    bad_dtype[0] = Tensor((256, 128), 'float16')
    expect_error('source_rejects_wrong_gradient_dtype', AssertionError, lambda: kernel(*bad_dtype))
    bad_metadata = args.copy()
    bad_metadata[1] = Tensor((256, 1), 'float32')
    expect_error('source_rejects_wrong_metadata_dtype', AssertionError, lambda: kernel(*bad_metadata))


def main():
    check('one_thread_one_cpu', threading.active_count() == 1 and os.sched_getaffinity(0) == {CPU})
    old, new = reconstruct_and_verify()
    namespace_tests(old, new)
    arithmetic_tests(old, new)
    check('largest_logical_tensor_under_64MiB', Tensor.max_elements * 4 <= 64 * 1024 * 1024,
          max_elements=Tensor.max_elements, max_logical_tensor_bytes=Tensor.max_elements * 4)


try:
    main()
    RESULT['status'] = 'PASS'
except BaseException as error:
    RESULT['status'] = 'REFUSE'
    RESULT['unexpected_errors'].append({'type': type(error).__name__, 'message': str(error),
                                         'traceback': traceback.format_exc()})
    traceback.print_exc()
finally:
    RESULT['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    RESULT['wall_seconds'] = time.monotonic() - START
    usage = resource.getrusage(resource.RUSAGE_SELF)
    RESULT['resources'] = {'python': sys.version, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                           'threads': threading.active_count(), 'cpu_seconds': usage.ru_utime + usage.ru_stime,
                           'max_rss_kib_linux': usage.ru_maxrss, 'address_space_limit_bytes': 256 * 1024 * 1024,
                           'cpu_soft_limit_seconds': 240, 'wall_alarm_seconds': 300,
                           'max_tensor_elements': Tensor.max_elements,
                           'max_logical_tensor_bytes': Tensor.max_elements * 4,
                           'storage': 'Python lists/scalars; logical tensors 4 bytes/element, list overhead included in RSS'}
    save_json('results.json', RESULT)
    print('FINAL_TEST_STATUS ' + RESULT['status'], flush=True)
sys.exit(0 if RESULT['status'] == 'PASS' else 1)
