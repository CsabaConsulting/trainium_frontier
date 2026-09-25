#!/usr/bin/env python3
"""DISABLED native consumer falsifier; launch only via the sole guarded Operator.

This is not a launcher, training run, source gate or production routing path.
No Torch/NKI import occurs without --enable-experimental.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable-experimental', action='store_true', default=False)
    p.add_argument('--label')
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--expected-kernel-sha256')
    p.add_argument('--expected-reference-sha256')
    p.add_argument('--timing-repeats', type=int, choices=range(0, 6), default=0)
    args = p.parse_args()
    if not args.enable_experimental:
        print(json.dumps(dict(status='DISABLED', native_imported=False, native_executed=False)))
        return 0
    if not args.label or not args.out_dir or not args.expected_kernel_sha256 or not args.expected_reference_sha256:
        p.error('enabled diagnostic requires unique label/output and exact support hashes')
    for file, expected in (('consumer_nki.py', args.expected_kernel_sha256), ('reference.py', args.expected_reference_sha256)):
        if digest(HERE / file) != expected:
            raise RuntimeError('SUPPORT_HASH_MISMATCH: ' + file)
    if os.environ.get('WORLD_SIZE') != '4' or os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2':
        raise RuntimeError('REQUIRES_FOUR_RANK_LNC2_OPERATOR_BINDING')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_neuronx  # Native backend registration; never fall back to CPU.
    from consumer_nki import build_kernel
    from reference import C, T, V, CASE_NAMES, dense, fixture, raw

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    rank = int(os.environ['RANK'])
    receipt = dict(scope='TINY_CONSUMER_FALSIFIER_ONLY', label=args.label, rank=rank,
                   world_size=4, shape=dict(V=V, C=C, T=T), seed=42,
                   source_adoption=False, launch_authorized=False,
                   rejected_e4d93='REFUSE_UNCHANGED', production_routing='NOT_IMPLEMENTED',
                   actual_model_autograd='NOT_TESTED', whole_step_speedup='UNMEASURED',
                   accepted_speedup=False, metadata_origin='synthetic CPU fixture, uploaded before calls',
                   metadata_cost_in_timings=False, tolerance=dict(atol=0, rtol=0, criterion='identical bytes'),
                   checker_sha256=digest(Path(__file__)), kernel_sha256=args.expected_kernel_sha256,
                   reference_sha256=args.expected_reference_sha256, argv=sys.argv[1:],
                   torch=torch.__version__, cases=[], timings=[])
    try:
        receipt['nki_version'] = importlib.metadata.version('nki')
    except importlib.metadata.PackageNotFoundError:
        receipt['nki_version'] = 'UNKNOWN'
    dist.init_process_group('neuron')
    device = torch.device('neuron')
    if not hasattr(torch, 'neuron') or not hasattr(torch.neuron, 'synchronize'):
        raise RuntimeError('NATIVE_SYNCHRONIZE_API_MISSING')
    sync = torch.neuron.synchronize

    def tensor_bytes(t):
        return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    def compare(actual, expected, name):
        cpu = actual.detach().cpu()
        ok = cpu.shape == expected.shape and cpu.dtype == expected.dtype and tensor_bytes(cpu) == tensor_bytes(expected)
        record = dict(name=name, exact=ok, actual_dtype=str(cpu.dtype), shape=list(cpu.shape),
                      finite=bool(torch.isfinite(cpu).all()), nonzero=int(torch.count_nonzero(cpu)),
                      actual_sha256=hashlib.sha256(tensor_bytes(cpu)).hexdigest(),
                      expected_sha256=hashlib.sha256(tensor_bytes(expected)).hexdigest())
        if cpu.shape == expected.shape:
            record['max_abs'] = float((cpu.double() - expected.double()).abs().max()) if cpu.numel() else 0.0
        return record

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

    error = None
    try:
        # Exclusive creation prevents overwriting another attempt's artifacts.
        if rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        kernel = build_kernel(enabled=True)
        for dtype_name, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
            eager, compiled = make_wrapper(kernel, dtype)
            for name in CASE_NAMES:
                case = dict(name=name, dtype=str(dtype), checks=[], status='STARTED')
                receipt['cases'].append(case)
                ids, rows, metadata = fixture(name, dtype_name)
                expected = torch.tensor(dense(ids, rows), dtype=torch.float32)
                # Full shape remains fixed; empty metadata consumes none of this storage.
                upstream = torch.tensor(rows, dtype=dtype) if ids else torch.zeros((T, C), dtype=dtype)
                device_upstream = upstream.to(device)
                device_meta = [torch.tensor(values, dtype=torch.int32).reshape(-1, 1).to(device) for values in metadata]
                sync()
                case['metadata_group_bounds'] = metadata[3]
                case['checks'].append(compare(device_upstream, upstream, 'upstream_roundtrip'))
                initial = time.monotonic()
                actual = kernel(device_upstream, *device_meta)
                sync()
                case['direct_first_call_seconds'] = time.monotonic() - initial
                case['checks'].append(compare(actual, expected, 'direct_consumer'))
                repeated = kernel(device_upstream, *device_meta)
                sync()
                case['checks'].append(compare(repeated, expected, 'direct_repeated_call'))
                if ids:
                    index = torch.tensor(ids, dtype=torch.int32)
                    cpu_weight = torch.randn(V, C, dtype=torch.float32).requires_grad_()
                    cpu_output = F.embedding(index, cpu_weight).to(dtype)
                    cpu_output.backward(upstream)
                    case['checks'].append(compare(cpu_weight.grad, expected, 'dense_CPU_oracle'))
                    for mode, function in (('eager_autograd', eager), ('compiled_enclosing_autograd', compiled)):
                        weight = cpu_weight.detach().to(device).requires_grad_()
                        output = function(weight, index.to(device), *device_meta)
                        output.backward(device_upstream)
                        sync()
                        case['checks'].append(compare(output, cpu_output.detach(), mode + '_forward'))
                        if weight.grad is None:
                            case['checks'].append(dict(name=mode + '_backward', exact=False, missing_gradient=True))
                        else:
                            case['checks'].append(compare(weight.grad, expected, mode + '_backward'))
                else:
                    case['autograd_scope'] = 'NOT_TESTED for logical-empty input; direct fixed-storage zero-trip loop only'
                case['status'] = 'EXACT' if all(c['exact'] for c in case['checks']) else 'FAIL_EXACTNESS'
                print(json.dumps(case, allow_nan=False), flush=True)
                if case['status'] != 'EXACT':
                    raise RuntimeError('EXACTNESS_FALSIFIED: ' + name + '/' + dtype_name)
            # Optional bounded component latency, never a production speed claim.
            for name in ('unique', 'all_same') if args.timing_repeats else ():
                ids, rows, metadata = fixture(name, dtype_name)
                g = torch.tensor(rows, dtype=dtype).to(device)
                meta = [torch.tensor(values, dtype=torch.int32).reshape(-1, 1).to(device) for values in metadata]
                for _ in range(2):
                    kernel(g, *meta)
                    sync()
                times = []
                for _ in range(args.timing_repeats):
                    begin = time.monotonic()
                    kernel(g, *meta)
                    sync()
                    times.append(time.monotonic() - begin)
                receipt['timings'].append(dict(case=name, dtype=str(dtype), seconds=times,
                                               scope='consumer_only_excludes_routing_and_transfer'))
        receipt['status'] = 'TINY_CONSUMER_EXACT_ONLY'
    except Exception:
        error = traceback.format_exc()
        receipt['status'] = 'REFUSE_OR_UNSUPPORTED'
        receipt['error'] = error
    finally:
        receipt['diagnostic_elapsed_seconds'] = time.monotonic() - started
        receipt['elapsed_scope'] = 'checker entry through receipt finalization; includes imports and CPU fixture/metadata, excludes outer launcher'
        # No end barrier: preserve each rank's error even if another rank fails.
        if args.out_dir.is_dir():
            with (args.out_dir / ('rank' + str(rank) + '.json')).open('x') as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps(receipt, allow_nan=False), flush=True)
        dist.destroy_process_group()
    if error:
        print(error, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
