#!/usr/bin/env python3
"""DISABLED registered K2 correctness diagnostic; no production or timing path."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
KERNEL_SHA = '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae'
REFERENCE_SHA = '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f'
ADAPTER_SHA = 'd761a7274e50c21d5930bfa8b984da710ed257be73381fc17ea5f062c03f190a'
ROUTES = ('direct_consumer', 'direct_repeated_call', 'registered_eager',
          'registered_compiled', 'compiled_ordinary_embedding')
OLD_ROUTES = ('original_function_eager', 'original_function_compiled')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_wrapper(kernel, output_dtype):
    class DiagnosticEmbedding(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, counts, starts, positions, maximums):
            ctx.save_for_backward(counts, starts, positions, maximums)
            return F.embedding(ids, weight).to(output_dtype)

        @staticmethod
        def backward(ctx, gradient):
            counts, starts, positions, maximums = ctx.saved_tensors
            result = kernel(gradient.contiguous(), counts, starts, positions, maximums)
            return result, None, None, None, None, None

    def enclosing(weight, ids, counts, starts, positions, maximums):
        return DiagnosticEmbedding.apply(weight, ids, counts, starts, positions, maximums)

    return enclosing, torch.compile(enclosing, backend='neuron', fullgraph=True, dynamic=False)


def tensor_record(value):
    cpu = value.detach().cpu().contiguous()
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                finite=bool(torch.isfinite(cpu).all()), nonzero=int(torch.count_nonzero(cpu)),
                sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())


def compare(actual, expected):
    if actual is None:
        return dict(exact=False, missing=True, expected=tensor_record(expected))
    a, e = tensor_record(actual), tensor_record(expected)
    same = a['shape'] == e['shape'] and a['dtype'] == e['dtype']
    result = dict(exact=same and a['sha256'] == e['sha256'] and a['finite'] and e['finite'],
                  actual=a, expected=e)
    if a['shape'] == e['shape']:
        error = float((actual.detach().cpu().double() - expected.detach().cpu().double()).abs().max())
        result['max_abs'] = error if math.isfinite(error) else None
        if not math.isfinite(error):
            result['max_abs_nonfinite'] = repr(error)
    return result


def observe_route(name, inputs, body, expected, sync):
    """Retain mismatch/errors and post-observations without stopping later routes."""
    result = dict(name=name, status='ERROR', checks={}, inputs_before={}, inputs_after={})
    try:
        sync()
        result['inputs_before'] = {key: tensor_record(value) for key, value in inputs.items()}
        outputs = body()
        sync()
        result['checks'] = {key: compare(outputs.get(key), value) for key, value in expected.items()}
        result['status'] = 'EXACT' if all(x['exact'] for x in result['checks'].values()) else 'MISMATCH'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        try:
            sync()
            result['inputs_after'] = {key: tensor_record(value) for key, value in inputs.items()}
            result['inputs_unchanged'] = (result['inputs_before'] == result['inputs_after']
                                           and len(result['inputs_before']) == 7)
            if not result['inputs_unchanged']:
                result['status'] = 'MISMATCH_OR_ERROR'
        except Exception:
            result['inputs_unchanged'] = False
            result['post_observation_error'] = traceback.format_exc()
            result['status'] = 'ERROR'
    return result


def case_exact(case):
    return (case.get('status') == 'EXACT' and
            all(c.get('exact') is True for c in case['checks'].values()) and
            all(r.get('status') == 'EXACT' and r.get('inputs_unchanged') is True
                for r in case['routes']))


def run_case(name, dtype_name, kernel, device, sync, functions, *, changed_tangent=False):
    dtype = torch.float32 if dtype_name == 'fp32' else torch.bfloat16
    ids, rows, metadata = fixture(name, dtype_name)
    if changed_tangent:
        rows = [[-value for value in row] for row in rows]
    expected_gradient = torch.tensor(dense(ids, rows), dtype=torch.float32)
    upstream_cpu = torch.tensor(rows, dtype=dtype) if ids else torch.zeros((T, C), dtype=dtype)
    index_cpu = torch.tensor(ids if ids else [0] * T, dtype=torch.int32)
    cpu_weight = torch.randn(V, C, dtype=torch.float32).requires_grad_()
    cpu_output = F.embedding(index_cpu, cpu_weight).to(dtype)
    cpu_output.backward(upstream_cpu)
    upstream = upstream_cpu.to(device)
    index = index_cpu.to(device)
    meta = [torch.tensor(values, dtype=torch.int32).reshape(-1, 1).to(device) for values in metadata]
    case = dict(name=name, dtype=str(dtype), changed_tangent=changed_tangent,
                metadata_group_bounds=metadata[3], checks={}, routes=[], original_routes=[])
    sync()
    case['checks']['upstream_roundtrip'] = compare(upstream, upstream_cpu)
    if ids:
        case['checks']['dense_CPU_oracle'] = compare(cpu_weight.grad, expected_gradient)
    else:
        case['autograd_scope'] = 'NOT_TESTED for logical-empty input; direct fixed-storage zero-trip loop only'
    required = ROUTES if ids else ROUTES[:2]
    for route_name in (*required, *(OLD_ROUTES if ids else ())):
        weight = cpu_weight.detach().to(device).requires_grad_()
        inputs = dict(upstream=upstream, ids=index, weight=weight, counts=meta[0],
                      starts=meta[1], positions=meta[2], maximums=meta[3])
        def body():
            if route_name in ROUTES[:2]:
                return dict(gradient=kernel(upstream, *meta))
            if route_name == 'compiled_ordinary_embedding':
                output = functions[route_name](weight, index)
            else:
                output = functions[route_name](weight, index, *meta)
            output.backward(upstream)
            return dict(output=output, gradient=weight.grad)
        expected = dict(gradient=expected_gradient)
        if route_name not in ROUTES[:2]:
            expected['output'] = cpu_output.detach()
        observation = observe_route(route_name, inputs, body, expected, sync)
        case['original_routes' if route_name in OLD_ROUTES else 'routes'].append(observation)
    exact = (all(c['exact'] for c in case['checks'].values()) and
             all(r['status'] == 'EXACT' and r['inputs_unchanged'] for r in case['routes']))
    case['status'] = 'EXACT' if exact else 'MISMATCH_OR_ERROR'
    return case


def main():
    global torch, F, fixture, dense, V, T, C
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-experimental', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--expected-kernel-sha256')
    parser.add_argument('--expected-reference-sha256')
    parser.add_argument('--expected-adapter-sha256')
    args = parser.parse_args()
    if not args.enable_experimental:
        print(json.dumps(dict(status='DISABLED', native_imported=False, native_executed=False)))
        return 0
    if (not args.label or not args.out_dir or args.expected_kernel_sha256 != KERNEL_SHA or
            args.expected_reference_sha256 != REFERENCE_SHA or args.expected_adapter_sha256 != ADAPTER_SHA):
        parser.error('enabled diagnostic requires unique label/output and exact protected pins')
    for name, pin in [('consumer_nki.py', KERNEL_SHA), ('reference.py', REFERENCE_SHA), ('adapter.py', ADAPTER_SHA)]:
        if digest(HERE / name) != pin:
            raise RuntimeError('SUPPORT_HASH_MISMATCH:' + name)
    if os.environ.get('WORLD_SIZE') != '4' or os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('REQUIRES_FOUR_RANK_LNC2_OPERATOR_BINDING')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_neuronx
    from consumer_nki import build_kernel
    from reference import C, T, V, CASE_NAMES, fixture, dense
    from adapter import build_registered_forward
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    rank = int(os.environ['RANK'])
    receipt = dict(scope='REGISTERED_K2_COMPONENT_DIAGNOSTIC_ONLY', label=args.label, rank=rank,
        world_size=4, shape=dict(V=V, T=T, C=C), seed=42, argv=sys.argv[1:],
        checker_sha256=digest(Path(__file__)), kernel_sha256=KERNEL_SHA,
        reference_sha256=REFERENCE_SHA, adapter_sha256=ADAPTER_SHA,
        torch=torch.__version__, nki_version=importlib.metadata.version('nki'),
        tolerance=dict(atol=0, rtol=0, criterion='identical bytes'),
        original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',
        original_fp32_first_gate='FAILED_EXACTNESS_PRESERVED', native_qualification=False,
        source_adoption=False, accepted_speedup=False, whole_step_speedup='UNMEASURED',
        public_20m=None, no_training=True, no_eval=True, timings=[],
        first_registered_compiled_fixture='unique/fp32/nonzero', cases=[], tangent_replays=[],
        observation_scope='Host input hashes, not internal compiled backward operands; original Function observations are separate')
    initialized = False
    rc = 1
    try:
        dist.init_process_group('neuron')
        initialized = True
        device = torch.device('neuron')
        sync = torch.neuron.synchronize
        if rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        kernel = build_kernel(enabled=True)
        cache = {}
        def functions(dtype_name):
            # One operator and one compiled callable per dtype, reused throughout.
            if dtype_name not in cache:
                dtype = torch.float32 if dtype_name == 'fp32' else torch.bfloat16
                eager = build_registered_forward(kernel, dtype, enabled=True)
                compiled = torch.compile(eager, backend='neuron', fullgraph=True, dynamic=False)
                original, old_compiled = make_wrapper(kernel, dtype)
                def ordinary(weight, index):
                    return F.embedding(index, weight).to(dtype)
                cache[dtype_name] = dict(registered_eager=eager, registered_compiled=compiled,
                    original_function_eager=original, original_function_compiled=old_compiled,
                    compiled_ordinary_embedding=torch.compile(ordinary, backend='neuron', fullgraph=True, dynamic=False))
            return cache[dtype_name]
        receipt['sentinel'] = run_case('unique', 'fp32', kernel, device, sync, functions('fp32'))
        print(json.dumps(dict(phase='sentinel', result=receipt['sentinel']), allow_nan=False), flush=True)
        if not case_exact(receipt['sentinel']):
            receipt['status'] = 'REGISTERED_SENTINEL_FAILED_SUITE_NOT_STARTED'
        else:
            for dtype_name in ('bf16', 'fp32'):
                for name in CASE_NAMES:
                    case = run_case(name, dtype_name, kernel, device, sync, functions(dtype_name))
                    receipt['cases'].append(case)
                    print(json.dumps(dict(phase='full_suite', result=case), allow_nan=False), flush=True)
                replay = run_case('unique', dtype_name, kernel, device, sync, functions(dtype_name), changed_tangent=True)
                receipt['tangent_replays'].append(replay)
                print(json.dumps(dict(phase='changed_tangent', result=replay), allow_nan=False), flush=True)
            complete = (len(receipt['cases']) == 20 and len(receipt['tangent_replays']) == 2 and
                        all(case_exact(c) for c in receipt['cases'] + receipt['tangent_replays']))
            receipt['status'] = 'REGISTERED_FULL_CASES_OBSERVED_EXACT' if complete else 'REGISTERED_FULL_CASES_MISMATCH_OR_ERROR'
            rc = 0 if complete else 1
    except Exception:
        receipt['status'] = 'REGISTERED_DIAGNOSTIC_INCOMPLETE'
        receipt['error'] = traceback.format_exc()
    finally:
        receipt['returncode'] = rc
        receipt['diagnostic_elapsed_seconds'] = time.monotonic() - started
        try:
            encoded = json.dumps(receipt, indent=2, allow_nan=False)
            if args.out_dir.is_dir():
                with (args.out_dir / ('rank' + str(rank) + '.json')).open('x') as stream:
                    stream.write(encoded + '\n')
            print(json.dumps(receipt, allow_nan=False), flush=True)
        finally:
            if initialized:
                dist.destroy_process_group()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
