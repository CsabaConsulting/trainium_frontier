#!/usr/bin/env python3
"""Opt-in tiny CPU evidence, not native correctness or an independent review.

Uses the actual archived nki_op AST and local real torch.library. The ONLY
consumer substitution is an explicitly selected tiny CPU callable. No NKI,
TorchNeuron, real data, full model, accelerator or host is imported/accessed.
"""
import argparse
import ast
from collections.abc import Callable, Iterable
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SDK = ROOT / 'research/v2/operator/20260926-nki-api-readonly-proposal/results/sources'
KERNEL = ROOT / 'research/v2/lanes/k2-symmetric-consumer/consumer_nki.py'
REFERENCE = KERNEL.with_name('reference.py')
OLD = ROOT / 'research/v2/proposals/20260926-k2-fp32-first-discriminator/check_native.py'
PINS = {
    'kernel': (KERNEL, '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae'),
    'reference': (REFERENCE, '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f'),
    'old_checker': (OLD, 'bd2bca7c29615df599094d3c618f5cdf0e4f4381e6d19213dc022691214ce788'),
    'nki_hop': (SDK / 'nki_hop.py', 'a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914'),
    'nki_kernel': (SDK / 'nki_kernel.py', '37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2'),
    'framework_kernel': (SDK / 'nki-framework/kernel.py', 'efef333640db8aae6805119ebed619f7f7114f8b7d6974114d0dc12a6c61f83a'),
    'framework_compiled': (SDK / 'nki-framework/compiled.py', 'c416d73c01fd8810509c4496758a122df941e109221fd0ee9a47d30f710b378e'),
}
V, T, C, P = 8, 8, 4, 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def extracted(path, name, namespace):
    node = next(x for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == name)
    # Original AST, decorators and function body are not rewritten.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name], sha(ast.dump(node, include_attributes=False).encode())


def load_adapter():
    spec = importlib.util.spec_from_file_location('registered_forward_adapter', HERE / 'adapter.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def statics():
    result = {}
    for name, (path, expected) in PINS.items():
        actual = sha(path.read_bytes())
        require(actual == expected, 'Protected pin mismatch: ' + name)
        result[name] = actual
    adapter = ast.parse((HERE / 'adapter.py').read_text())
    calls = [n for n in ast.walk(adapter) if isinstance(n, ast.Call)]
    require(sum(isinstance(n.func, ast.Name) and n.func.id == 'nki_op_factory'
                for n in calls) == 1, 'Exactly one forward registration factory')
    require(sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'register_autograd'
                for n in calls) == 1, 'Exactly one derivative registration')
    require(not any(isinstance(n, ast.Name) and n.id in ('wrap_nki', 'custom_op')
                    for n in ast.walk(adapter)), 'No explicit wrap or second custom op')
    kernel_ast = ast.parse(KERNEL.read_text())
    factory = next(n for n in kernel_ast.body if isinstance(n, ast.FunctionDef))
    require(ast.unparse(factory.body[-1]) == 'return resident_consumer[2]', 'LNC2 unchanged')
    return result


def metadata(ids):
    buckets = [[i for i, row in enumerate(ids) if row == v] for v in range(V)]
    counts = [len(b) for b in buckets]
    starts, positions = [], []
    for bucket in buckets:
        starts.append(len(positions))
        positions.extend(bucket)
    positions += [T] * (T - len(positions))
    maximums = [max(counts[i:i + P]) for i in range(0, V, P)]
    return [torch.tensor(x, dtype=torch.int32).reshape(-1, 1)
            for x in (counts, starts, positions, maximums)]


def cpu_consumer(gradient, counts, starts, positions, maximums):
    """Tiny traceable substitute: metadata-driven ordered FP32 additions.

    Fixed eight-iteration unrolling is CPU-test-only, NOT the NKI dynamic loop.
    Every metadata Tensor remains a runtime input; no .item()/host extraction.
    """
    accumulator = torch.zeros((V, C), dtype=torch.float32, device=gradient.device)
    for occurrence in range(T):
        address = (starts.reshape(V).long() + occurrence).clamp(0, T - 1)
        token = positions.reshape(T)[address].long().clamp(0, T - 1)
        value = gradient[token].float()
        limits = maximums.reshape(V // P).repeat_interleave(P)
        active = (counts.reshape(V) > occurrence) & (limits > occurrence)
        accumulator = torch.where(active[:, None], accumulator + value, accumulator)
    return accumulator


def dense(ids, gradient):
    result = [[0.0] * C for _ in range(V)]
    for row, values in zip(ids, gradient.tolist()):
        for col, value in enumerate(values):
            result[row][col] = struct.unpack('<f', struct.pack('<f', result[row][col] + value))[0]
    return torch.tensor(result, dtype=torch.float32)


def record(value):
    raw = value.detach().contiguous()
    return dict(shape=list(raw.shape), dtype=str(raw.dtype), finite=bool(torch.isfinite(raw).all()),
                nonzero=int(torch.count_nonzero(raw)),
                sha256=sha(raw.view(torch.uint8).numpy().tobytes()))


def compare(actual, expected):
    if actual is None:
        return dict(exact=False, missing=True, expected=record(expected))
    a, e = record(actual), record(expected)
    same = a['shape'] == e['shape'] and a['dtype'] == e['dtype']
    out = dict(exact=same and a['finite'] and e['finite'] and a['sha256'] == e['sha256'],
               actual=a, expected=e)
    if a['shape'] == e['shape'] and actual.numel():
        error = float((actual.detach().double() - expected.detach().double()).abs().max())
        out['max_abs'] = error if math.isfinite(error) else None
        if not math.isfinite(error):
            out['max_abs_nonfinite'] = repr(error)
    return out


def fixtures(dtype):
    ordinary = torch.tensor([[float(i * C + j + 1) / 8 for j in range(C)]
                             for i in range(T)], dtype=dtype)
    changed = -ordinary * 2
    duplicates = [0, 0, 4, 4, 0, 1, 4, 1]
    cancellation = torch.tensor([[2**25, 1, -0.0, 2**-10], [1, 2**25, 0.0, 2**-10],
                                 [-2**25, -2**25, -0.0, -2**-10], [3, 2, 0, 0],
                                 [0, 0, -0.0, 0], [0, 0, 0, 0],
                                 [0, 0, 0, 0], [0, 0, 0, 0]], dtype=dtype)
    return [
        ('unique_first', list(range(V)), list(range(V)), ordinary),
        ('unique_changed_tangent', list(range(V)), list(range(V)), changed),
        ('changed_metadata', duplicates, duplicates, ordinary),
        ('changed_metadata_tangent', duplicates, duplicates, changed),
        ('ordered_cancellation', [0] * T, [0] * T, cancellation),
        ('empty_metadata', [0] * T, [], torch.zeros((T, C), dtype=dtype)),
        ('signed_zero', [0, 4] * 4, [0, 4] * 4, torch.full((T, C), -0.0, dtype=dtype)),
    ]


def run(substitute):
    global torch, F
    pins = statics()
    mod = load_adapter()
    try:
        mod.build_registered_forward(None, None)
    except RuntimeError as exc:
        require('EXPERIMENT_DISABLED' in str(exc), 'Default disabled reason')
    else:
        raise ValueError('Default not disabled')
    require('torch' not in sys.modules and 'nki' not in sys.modules, 'Disabled imported runtime')
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    from torch._library.custom_ops import custom_op
    from torch._library.infer_schema import infer_schema
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    namespace = dict(torch=torch, Callable=Callable, Iterable=Iterable,
                     custom_op=custom_op, infer_schema=infer_schema)
    nki_op, nki_ast_sha = extracted(PINS['nki_hop'][0], 'nki_op', namespace)
    # The original make_wrapper AST is unchanged. Only its test namespace's
    # compile dependency selects aot_eager instead of unavailable neuron.
    class CPUOnlyTorch:
        autograd = torch.autograd

        @staticmethod
        def compile(fn, *, backend, fullgraph, dynamic):
            require(backend == 'neuron', 'Unexpected original backend')
            return torch.compile(fn, backend='aot_eager', fullgraph=fullgraph, dynamic=dynamic)

    old, old_ast_sha = extracted(OLD, 'make_wrapper', dict(torch=CPUOnlyTorch, F=F))
    routes, registrations = [], []
    for dtype in (torch.float32, torch.bfloat16):
        if substitute == 'ordered':
            kernel = cpu_consumer
        elif substitute == 'zero':
            def kernel(gradient, counts, starts, positions, maximums):
                return torch.zeros((V, C), dtype=torch.float32)
        else:
            def kernel(gradient, counts, starts, positions, maximums):
                # Deliberately replay the first fixture's dW, constructed inside
                # the traced function. No captured real Tensor (unsupported by
                # AOT fake tracing); this remains a wrong-consumer test only.
                return (torch.arange(V * C, dtype=torch.float32).reshape(V, C) + 1) / 8
        registered = mod.build_registered_forward(kernel, dtype, enabled=True, nki_op_factory=nki_op)
        compiled = torch.compile(registered, backend='aot_eager', fullgraph=True, dynamic=False)
        original, original_compiled = old(cpu_consumer, dtype)
        require(registered._abstract_fn is not None, 'Actual SDK fake registration missing')
        require(FunctionalTensorMode in registered._torch_dispatch_fns, 'Functional decomposition missing')
        require(registered._backward_fn is not None and registered._setup_context_fn is not None,
                'Registered autograd missing')
        class SavedProbe:
            def save_for_backward(self, *values):
                self.saved = values
        saved_probe = SavedProbe()
        markers = tuple(object() for _ in range(6))
        registered._setup_context_fn(saved_probe, markers, None)
        require(saved_probe.saved == markers[2:], 'Saved metadata must be exactly the last four inputs')
        with FakeTensorMode():
            fake = registered(torch.empty(V, C), torch.empty(T, dtype=torch.int32),
                              *[torch.empty(n, 1, dtype=torch.int32) for n in (V, V, T, V // P)])
            require(tuple(fake.shape) == (T, C) and fake.dtype == dtype, 'Fake shape/dtype')
        registrations.append(dict(name=registered._name, schema=str(registered._schema),
                                  fake_registered=True, functional_decomp_registered=True,
                                  autograd_registered=True, fake_shape=list(fake.shape),
                                  saves_exactly_four_metadata_inputs=True,
                                  same_compiled_callable_replayed_for_all_fixtures=True))
        for name, ids, active_ids, upstream in fixtures(dtype):
            index = torch.tensor(ids, dtype=torch.int32)
            meta = metadata(active_ids)
            weight_data = torch.arange(V * C, dtype=torch.float32).reshape(V, C) / 16
            expected_output = F.embedding(index, weight_data).to(dtype)
            expected_gradient = dense(active_ids, upstream)
            require(compare(cpu_consumer(upstream, *meta), expected_gradient)['exact'],
                    'CPU substitute versus independent ordered scalar oracle')
            for route_name, function in [('registered_eager', registered), ('registered_aot_eager', compiled),
                                         ('original_function_eager', original),
                                         ('original_function_aot_eager', original_compiled),
                                         ('ordinary_embedding', lambda w, ix, *unused: F.embedding(ix, w).to(dtype))]:
                weight = weight_data.detach().clone().requires_grad_()
                inputs = [weight, index, *meta, upstream]
                before = [record(value)['sha256'] for value in inputs]
                output = function(weight, index, *meta)
                output.backward(upstream)
                row = dict(dtype=str(dtype), case=name, route=route_name,
                           output=compare(output, expected_output), gradient=compare(weight.grad, expected_gradient),
                           inputs_before=before, inputs_after=[record(value)['sha256'] for value in inputs])
                row['inputs_unchanged'] = row['inputs_before'] == row['inputs_after']
                routes.append(row)
    negatives = {}
    expected = torch.ones((V, C), dtype=torch.float32)
    for name, value in [('zero', torch.zeros_like(expected)), ('stale', expected * 2),
                        ('nan', torch.full_like(expected, float('nan'))),
                        ('positive_inf', torch.full_like(expected, float('inf'))),
                        ('negative_inf', torch.full_like(expected, -float('inf'))), ('missing', None),
                        ('wrong_dtype', expected.to(torch.bfloat16))]:
        observed = compare(value, expected)
        require(not observed['exact'], 'Negative validator accepted: ' + name)
        negatives[name] = observed
    signed = compare(torch.full((1,), -0.0), torch.zeros(1))
    require(not signed['exact'] and signed['max_abs'] == 0, 'Signed zero byte distinction')
    negatives['signed_zero'] = signed
    controls = [r for r in routes if not r['route'].startswith('registered')]
    candidate = [r for r in routes if r['route'].startswith('registered')]
    require(all(r['output']['exact'] and r['gradient']['exact'] and r['inputs_unchanged'] for r in controls),
            'Original/ordinary CPU control failed')
    require(all(r['output']['exact'] and r['inputs_unchanged'] for r in candidate), 'Forward/mutation failure')
    if substitute == 'ordered':
        require(all(r['gradient']['exact'] for r in candidate), 'Candidate CPU dW failure')
    else:
        for dtype in ('torch.float32', 'torch.bfloat16'):
            for route in ('registered_eager', 'registered_aot_eager'):
                wrong = [r for r in candidate if r['dtype'] == dtype and r['route'] == route
                         and r['case'] == 'unique_changed_tangent']
                require(len(wrong) == 1 and not wrong[0]['gradient']['exact'],
                        'Wrong consumer did not affect registered dW: ' + dtype + '/' + route)
    require('nki' not in sys.modules and 'torch_neuronx' not in sys.modules, 'Native import forbidden')
    return dict(status='OWN_CPU_CHECKS_COMPLETED' if substitute == 'ordered' else 'WRONG_CONSUMER_REFUSED',
                native_gate='PENDING', independent_review='PENDING', substitute=substitute,
                torch_version=torch.__version__, python=sys.version, optimized=not __debug__,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                shape=dict(V=V, T=T, C=C), tensor_limit_bytes=64 * 1024 * 1024,
                largest_single_fixture_tensor_bytes=V * C * 4,
                pins=pins, nki_op_ast_sha256=nki_ast_sha, old_wrapper_ast_sha256=old_ast_sha,
                dependency_namespace=['torch', 'collections.abc.Callable', 'collections.abc.Iterable',
                                      'torch._library.custom_ops.custom_op', 'torch._library.infer_schema.infer_schema'],
                registrations=registrations, routes=routes, validation_negatives=negatives,
                original_native_failure='PRESERVED; local control exactness does not erase native failure',
                limits='Torch2.13 CPU substitute, not Torch2.11/NKI0.5/automatic HOP/native lowering')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-cpu-test', action='store_true')
    parser.add_argument('--substitute', choices=('ordered', 'zero', 'stale'), default='ordered')
    args = parser.parse_args()
    if not args.enable_cpu_test:
        print(json.dumps(dict(status='EXPERIMENT_DISABLED', torch_imported='torch' in sys.modules,
                              nki_imported='nki' in sys.modules, launch_authorized=False)))
        return 0
    try:
        result = run(args.substitute)
        code = 0
    except Exception:
        result = dict(status='REFUSED_OR_INCOMPLETE', error=traceback.format_exc(),
                      substitute=args.substitute, native_gate='PENDING')
        code = 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
