#!/usr/bin/env python3
"""Disabled, source-neutral FP32-first diagnostic; never a native qualification gate."""
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
ROUTES = ('direct_consumer', 'eager_custom', 'compiled_custom', 'compiled_ordinary_embedding')


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


def run_routes(kernel, device, sync, ids, rows, metadata, dense):
    """The one FP32 unique fixture; no BF16 or cancellation case is executed."""
    expected_gradient = torch.tensor(dense(ids, rows), dtype=torch.float32)
    upstream_cpu = torch.tensor(rows, dtype=torch.float32)
    ids_cpu = torch.tensor(ids, dtype=torch.int32)
    cpu_weight = torch.randn(256, 128, dtype=torch.float32).requires_grad_()
    cpu_output = F.embedding(ids_cpu, cpu_weight).to(torch.float32)
    cpu_output.backward(upstream_cpu)
    oracle = compare(cpu_weight.grad, expected_gradient)
    upstream = upstream_cpu.to(device)
    index = ids_cpu.to(device)
    meta = [torch.tensor(values, dtype=torch.int32).reshape(-1, 1).to(device) for values in metadata]
    pair = {}
    routes = []
    for name in ROUTES:
        # Preserve the existing checker's fresh-leaf expression, without a clone/cast fix.
        weight = cpu_weight.detach().to(device).requires_grad_()
        inputs = dict(upstream=upstream, ids=index, weight=weight, counts=meta[0],
                      starts=meta[1], positions=meta[2], maximums=meta[3])

        def body():
            if name == 'direct_consumer':
                return dict(gradient=kernel(upstream, *meta))
            if name == 'compiled_ordinary_embedding':
                def ordinary(w, ix):
                    return F.embedding(ix, w).to(torch.float32)
                function = torch.compile(ordinary, backend='neuron', fullgraph=True, dynamic=False)
                output = function(weight, index)
            else:
                if 'wrappers' not in pair:
                    pair['wrappers'] = make_wrapper(kernel, torch.float32)
                function = pair['wrappers'][0 if name == 'eager_custom' else 1]
                output = function(weight, index, *meta)
            output.backward(upstream)
            return dict(output=output, gradient=weight.grad)

        expected = dict(gradient=expected_gradient)
        if name != 'direct_consumer':
            expected['output'] = cpu_output.detach()
        routes.append(observe_route(name, inputs, body, expected, sync))
    return dict(case='unique', dtype='torch.float32', dense_cpu_oracle=oracle,
                metadata_group_bounds=metadata[3], routes=routes)


def classify(case):
    complete = ([r['name'] for r in case['routes']] == list(ROUTES))
    exact = complete and case['dense_cpu_oracle']['exact'] and all(
        r['status'] == 'EXACT' and r['inputs_unchanged'] for r in case['routes'])
    return ('FP32_FIRST_DIAGNOSTIC_OBSERVED_EXACT' if exact else
            'FP32_FIRST_DIAGNOSTIC_MISMATCH_OR_ERROR'), (0 if exact else 1)


def main():
    global torch, F
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-experimental', action='store_true', default=False)
    parser.add_argument('--label')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--expected-kernel-sha256')
    parser.add_argument('--expected-reference-sha256')
    args = parser.parse_args()
    if not args.enable_experimental:
        print(json.dumps(dict(status='DISABLED', native_imported=False, native_executed=False)))
        return 0
    if not args.label or not args.out_dir or args.expected_kernel_sha256 != KERNEL_SHA or args.expected_reference_sha256 != REFERENCE_SHA:
        parser.error('enabled diagnostic requires unique output/label and exact unchanged source pins')
    for name, pin in [('consumer_nki.py', KERNEL_SHA), ('reference.py', REFERENCE_SHA)]:
        if digest(HERE / name) != pin:
            raise RuntimeError('SUPPORT_HASH_MISMATCH:' + name)
    if os.environ.get('WORLD_SIZE') != '4' or os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('REQUIRES_FOUR_RANK_LNC2_OPERATOR_BINDING')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_neuronx
    from consumer_nki import build_kernel
    from reference import dense, fixture
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    rank = int(os.environ['RANK'])
    receipt = dict(scope='FP32_FIRST_SOURCE_NEUTRAL_DIAGNOSTIC_ONLY', label=args.label, rank=rank,
        world_size=4, shape=dict(V=256, T=256, C=128), seed=42, argv=sys.argv[1:],
        checker_sha256=digest(Path(__file__)), kernel_sha256=KERNEL_SHA, reference_sha256=REFERENCE_SHA,
        torch=torch.__version__, nki_version=importlib.metadata.version('nki'),
        tolerance=dict(atol=0, rtol=0, criterion='identical bytes'),
        original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED', native_qualification=False,
        source_adoption=False, accepted_speedup=False, whole_step_speedup='UNMEASURED',
        public_20m=None, no_training=True, no_eval=True,
        observation_scope='Host hashes before/after each route; NOT values inside compiled VJP; possible observer effect.',
        route_order=list(ROUTES), first_custom_compiled_fixture='unique/fp32', timings=[])
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
        ids, rows, metadata = fixture('unique', 'fp32')
        receipt['case'] = run_routes(build_kernel(enabled=True), device, sync, ids, rows, metadata, dense)
        receipt['status'], rc = classify(receipt['case'])
    except Exception:
        receipt['status'] = 'FP32_FIRST_DIAGNOSTIC_INCOMPLETE'
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
