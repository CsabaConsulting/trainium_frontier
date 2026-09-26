"""Independent tiny CPU review. Never import any native module or repository."""
import ast
import hashlib
import importlib.util
import json
import platform
import struct
import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_SHA = '05c1440c0f908b87ceeb1b7d029098fced9013bd0faaa9460a65b108084a1847'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), sort_keys=True, allow_nan=False), flush=True)


def authenticate():
    contents = (ROOT / 'INPUT_MANIFEST.json').read_bytes()
    require(sha(contents) == MANIFEST_SHA, 'manifest authentication')
    manifest = json.loads(contents)
    for item in manifest['included']:
        path = ROOT / item['path']
        require(path.is_relative_to(ROOT) and '..' not in path.parts, 'packet-only path')
        contents = path.read_bytes()
        require(len(contents) == item['bytes'] and sha(contents) == item['sha256'], str(path))
    diff = (ROOT / 'adapter.diff').read_text()
    reconstructed = ''.join(line[1:] + '\n' for line in diff.splitlines()
                            if line.startswith('+') and not line.startswith('+++'))
    require(reconstructed.count('\n') == 44, '44 lines required')
    require(reconstructed.encode() == (ROOT / 'output/adapter.py').read_bytes(), 'exact reconstruction')
    require(sha(reconstructed.encode()) == manifest['adapter_sha256'], 'adapter hash')
    emit('authentication', files=len(manifest['included']), manifest=MANIFEST_SHA,
         adapter=manifest['adapter_sha256'], adapter_lines=44)


def load_adapter():
    spec = importlib.util.spec_from_file_location('reviewed_adapter', ROOT / 'output/adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def disabled_test():
    require('torch' not in sys.modules, 'runtime started without Torch')
    adapter = load_adapter()
    try:
        adapter.build_registered_forward(None, None)
    except RuntimeError as error:
        require(str(error) == 'EXPERIMENT_DISABLED: no Torch/NKI module imported', 'disabled error')
    else:
        raise RuntimeError('default enabled unexpectedly')
    require(not any(n == 'torch' or n.startswith(('torch.', 'nki', 'torch_neuronx'))
                    for n in sys.modules), 'disabled path imported a runtime')
    emit('disabled', status='PASS', torch_imported=False, native_imported=False)


def main():
    authenticate()
    disabled_test()
    import torch
    import torch.nn.functional as F
    from torch._library.custom_ops import custom_op
    from torch._library.infer_schema import infer_schema
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'ordered'
    require(mode in ('ordered', 'zero', 'stale'), 'mode')
    emit('runtime', executable=sys.executable, python=platform.python_version(), torch=torch.__version__,
         threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
         mode=mode, optimized=not __debug__, target='Torch2.11/NKI0.5 NOT EXECUTED')

    # Execute the exact archived decorator AST; do not import its native module.
    sdk_text = (ROOT / 'sdk/nki_hop.py').read_text()
    tree = ast.parse(sdk_text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nki_op')
    original_ast = ast.dump(node, include_attributes=True)
    ns = dict(Callable=Callable, Iterable=Iterable, custom_op=custom_op, infer_schema=infer_schema)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'sdk/nki_hop.py'), 'exec'), ns)
    require(ast.dump(node, include_attributes=True) == original_ast, 'SDK AST mutated')
    emit('archived_decorator', source_sha=sha(sdk_text.encode()),
         ast_sha=sha(original_ast.encode()), first_line=node.lineno, last_line=node.end_lineno,
         native_module_imported=False)

    def raw(tensor):
        return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def fingerprint(tensor):
        return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), sha256=sha(raw(tensor)))

    def exact(actual, expected):
        return (isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype and bool(torch.isfinite(actual).all())
                and bool(torch.isfinite(expected).all()) and raw(actual) == raw(expected))

    def f32(value):
        return struct.unpack('<f', struct.pack('<f', value))[0]

    def oracle(ids, gradient):
        # Independent scalar token-order recurrence, no metadata or scatter ops.
        out = [[0.0 for _ in range(4)] for _ in range(8)]
        for token, row in enumerate(ids):
            for channel in range(4):
                out[row][channel] = f32(out[row][channel] + float(gradient[token, channel]))
        return torch.tensor(out, dtype=torch.float32)

    def metadata(ids):
        bins = [[i for i, value in enumerate(ids) if value == row] for row in range(8)]
        counts = [len(bucket) for bucket in bins]
        starts, positions = [], []
        for bucket in bins:
            starts.append(len(positions))
            positions.extend(bucket)
        require(len(positions) <= 8, 'tiny token cap')
        positions.extend([8] * (8 - len(positions)))
        maximums = [max(counts[:4]), max(counts[4:])]
        return [torch.tensor(values, dtype=torch.int32).reshape(-1, 1)
                for values in (counts, starts, positions, maximums)]

    def ordered_consumer(gradient, counts, starts, positions, maximums):
        # Vector lanes with one ordered FP32 addition per occurrence.
        # Padding also permits a local logical-empty autograd fixture.
        padded = torch.cat((gradient.to(torch.float32), gradient.new_zeros((8, 4)).float()), dim=0)
        accum = gradient.new_zeros((8, 4), dtype=torch.float32)
        groups = torch.arange(8, device=gradient.device) // 4
        for occurrence in range(8):
            active = (counts[:, 0] > occurrence) & (maximums[groups, 0] > occurrence)
            pointers = (starts[:, 0] + occurrence).clamp(0, 7).long()
            tokens = positions[pointers, 0].clamp(0, 8).long()
            selected = padded[tokens]
            accum = torch.where(active[:, None], accum + selected, accum)
        return accum

    def zero_consumer(gradient, counts, starts, positions, maximums):
        return gradient.new_zeros((8, 4), dtype=torch.float32)

    def stale_consumer(gradient, counts, starts, positions, maximums):
        # Reconstruct a stale initial unique-fixture dW; no captured real Tensor.
        return torch.arange(1, 33, dtype=torch.float32, device=gradient.device).reshape(8, 4)

    selected_consumer = dict(ordered=ordered_consumer, zero=zero_consumer, stale=stale_consumer)[mode]
    adapter = load_adapter()

    def make_original(output_dtype):
        class Original(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight, ids, counts, starts, positions, maximums):
                ctx.save_for_backward(counts, starts, positions, maximums)
                return F.embedding(ids, weight).to(output_dtype)

            @staticmethod
            def backward(ctx, gradient):
                return (ordered_consumer(gradient.contiguous(), *ctx.saved_tensors),
                        None, None, None, None, None)

        def call(weight, ids, counts, starts, positions, maximums):
            return Original.apply(weight, ids, counts, starts, positions, maximums)
        return call

    zeros = torch.zeros((8, 4), dtype=torch.float32)
    negatives = dict(missing=None, wrong_dtype=zeros.to(torch.bfloat16), wrong_shape=zeros[:7],
                     signed_zero=-zeros, nan=torch.full((8, 4), float('nan')),
                     positive_infinity=torch.full((8, 4), float('inf')),
                     negative_infinity=torch.full((8, 4), -float('inf')))
    for label, wrong in negatives.items():
        require(not exact(wrong, zeros), 'validator accepted ' + label)
    emit('validator_negatives', rejected=list(negatives), tolerance='strict bytes; finite; shape; dtype')

    initial_values = [[float(1 + 4 * token + channel) for channel in range(4)] for token in range(8)]
    changed_values = [[float(33 - 2 * token + channel) / 4 for channel in range(4)] for token in range(8)]
    signed_values = [[-0.0 if (token + channel) % 2 else 0.0 for channel in range(4)] for token in range(8)]
    cancel_values = [[v] * 4 for v in (2.0**24, 1.0, -2.0**24, 2.0**-24, 1.0, -1.0, 3.0, 5.0)]
    cases = [
        ('unique_first', list(range(8)), initial_values),
        ('changed_tangent', list(range(8)), changed_values),
        ('changed_metadata', [7, 0, 7, 0, 3, 7, 3, 0], changed_values),
        ('cancellation', [2, 2, 2, 4, 4, 4, 6, 6], cancel_values),
        ('signed_zero', [1, 1, 5, 5, 1, 1, 5, 5], signed_values),
        ('unique_revisit', list(range(8)), initial_values),
        ('logical_empty', [], []),
    ]
    counters = dict(routes=0, registered_exact=0, registered_mismatch=0, controls_exact=0)
    for dtype in (torch.float32, torch.bfloat16):
        registered = adapter.build_registered_forward(selected_consumer, dtype, enabled=True,
                                                       nki_op_factory=ns['nki_op'])
        schema = str(registered._opoverload._schema)
        handlers = registered._torch_dispatch_fns
        require(FunctionalTensorMode in handlers, 'functional decomposition absent')
        require(callable(registered._abstract_fn), 'fake absent')
        require(callable(registered._setup_context_fn) and callable(registered._backward_fn), 'autograd absent')
        require(len(registered._opoverload._schema.arguments) == 6, 'six input schema')
        require(all(a.alias_info is None for a in registered._opoverload._schema.arguments), 'mutation alias')
        emit('registrations', dtype=str(dtype), schema=schema, fake=registered._abstract_fn.__qualname__,
             functional=handlers[FunctionalTensorMode].__qualname__,
             setup=registered._setup_context_fn.__qualname__, backward=registered._backward_fn.__qualname__)
        original = make_original(dtype)
        def ordinary(weight, ids, counts, starts, positions, maximums):
            return F.embedding(ids, weight).to(dtype)
        def enclosing(weight, ids, counts, starts, positions, maximums):
            return registered(weight, ids, counts, starts, positions, maximums)
        routes = [('registered_eager', registered),
                  ('registered_AOT', torch.compile(enclosing, backend='aot_eager', fullgraph=True, dynamic=False)),
                  ('original_eager', original),
                  ('original_AOT', torch.compile(original, backend='aot_eager', fullgraph=True, dynamic=False)),
                  ('ordinary_eager', ordinary)]
        poison_detected = {name: 0 for name, _ in routes if name.startswith('registered')}
        saved_probe = []
        for name, ids_list, values in cases:
            ids = torch.tensor(ids_list, dtype=torch.int32)
            tangent = torch.tensor(values, dtype=dtype).reshape(-1, 4).t().contiguous().t()
            meta = metadata(ids_list)
            expected = oracle(ids_list, tangent)
            require(exact(ordered_consumer(tangent, *meta), expected), 'own consumer fails own scalar oracle')
            for route_name, function in routes:
                weight = (torch.arange(32, dtype=torch.float32).reshape(8, 4) / 7 - 2).requires_grad_()
                inputs = [weight, ids, *meta, tangent]
                before = [fingerprint(t) for t in inputs]
                require(sum(t.numel() * t.element_size() for t in inputs) <= 4096, 'tiny input storage cap')
                output = function(weight, ids, *meta)
                expected_output = torch.tensor([weight.detach()[row].tolist() for row in ids_list],
                                               dtype=torch.float32).reshape(-1, 4).to(dtype)
                require(exact(output, expected_output), route_name + ' forward')
                if name == 'unique_first' and route_name == 'registered_eager':
                    saved = output.grad_fn.saved_tensors
                    require(len(saved) == 4 and all(x is y for x, y in zip(saved, meta)), 'saved metadata identity')
                    saved_probe.append('four original metadata tensors only')
                gradient, = torch.autograd.grad(output, weight, grad_outputs=tangent)
                after = [fingerprint(t) for t in inputs]
                require(before == after, route_name + ' input mutation')
                is_exact = exact(gradient, expected)
                counters['routes'] += 1
                if route_name.startswith('registered'):
                    counters['registered_exact' if is_exact else 'registered_mismatch'] += 1
                    if mode == 'ordered':
                        require(is_exact, name + ' ' + route_name + ' byte difference')
                    elif not is_exact:
                        poison_detected[route_name] += 1
                else:
                    require(is_exact, name + ' ' + route_name + ' reference difference')
                    counters['controls_exact'] += 1
                emit('route', dtype=str(dtype), case=name, route=route_name, exact_gradient=is_exact,
                     output=fingerprint(output), actual=fingerprint(gradient), expected=fingerprint(expected),
                     seven_inputs_before=before, seven_inputs_after=after, tangent_contiguous=tangent.is_contiguous())
        if mode != 'ordered':
            require(all(count >= 2 for count in poison_detected.values()), 'poison failed to expose fallback')
        # Fixed T storage, empty metadata, nonzero tangent: direct zero-trip diagnostic.
        empty_meta = metadata([])
        direct = ordered_consumer(torch.ones((8, 4), dtype=dtype), *empty_meta)
        require(exact(direct, zeros), 'empty metadata consumed storage')
        emit('dtype_summary', dtype=str(dtype), saved_metadata=saved_probe,
             poisoned_consumer_mismatches=poison_detected, fixed_storage_empty_exact=True)

    require(not any(n == 'nki' or n.startswith(('nki.', 'torch_neuronx')) for n in sys.modules), 'native imported')
    emit('summary', status='PASS_CPU_SUITABILITY' if mode == 'ordered' else 'POISON_REFUSED',
         counters=counters, native_imported=False, native_tested=False,
         maximum_tensor_shape=[16, 4], tiny_tensor_allocation_bound='under 64 MiB, no model',
         all_native_and_performance_gates='PENDING')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
