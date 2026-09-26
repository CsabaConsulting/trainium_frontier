"""Disabled ordinary-only diagnostic. Integer oracle; no production imports."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import struct
import sys
import time

V, T, C = 256, 256, 128
CASES = ('sentinel', 'subnormal', 'normal', 'negated')
ROUTES = ('ordinary_eager', 'ordinary_compiled')
PINS = dict(ids='bdc67a401900ee3bbd1e23ce38227311ef06dfd415ef25311138f1379648567e',
            tangent='8e79a64b9db2dd89e7bac15c068285a6aafb12db7dcd6f208bb683e3d08cf78d',
            gradient='402743e587f15111ef1d3b927df591942db4bd96038726e4edc134a622e44085')
LABEL = 'fs-20260926-ordinary-fp32-subnormal-trn21-d1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def encode(n, power):
    """Exact n*2**power -> little-endian binary32, integer arithmetic only."""
    if n == 0:
        return b'\0' * 4  # Untouched rows are positive zero even in negative replay.
    sign, n = (0x80000000 if n < 0 else 0), abs(n)
    exponent = n.bit_length() - 1 + power
    if exponent < -126:
        require(power >= -149, 'NONEXACT_SUBNORMAL')
        bits = n << (power + 149)
        require(bits < 0x800000, 'BAD_SUBNORMAL')
    else:
        require(-126 <= exponent <= 127 and n.bit_length() <= 24, 'NONEXACT_NORMAL')
        significand = n << (24 - n.bit_length())
        bits = ((exponent + 127) << 23) | (significand & 0x7fffff)
    return struct.pack('<I', sign | bits)


def fixture(case):
    require(case in CASES, 'UNKNOWN_CASE')
    ids = list(range(T)) if case == 'sentinel' else [t % 11 for t in range(T)]
    rng = random.Random(42)
    coefficients = ([rng.randint(-256, 256) for _ in range(T*C)] if case == 'sentinel'
                    else [(-1 if (t+c) % 3 == 0 else 1) * (-1 if case == 'negated' else 1)
                          for t in range(T) for c in range(C)])
    power = -5 if case == 'sentinel' else -126 if case == 'normal' else -149
    sums = [0] * (V*C)
    for t, row in enumerate(ids):
        for c in range(C):
            sums[row*C+c] += coefficients[t*C+c]
    result = dict(ids=struct.pack('<'+'i'*T, *ids),
                  tangent=b''.join(encode(n, power) for n in coefficients),
                  gradient=b''.join(encode(n, power) for n in sums))
    if case == 'subnormal':
        require({k: sha(v) for k, v in result.items()} == PINS, 'ORIGINAL_FIXTURE_PIN')
    return result


def blob(raw, dtype, shape, blobs):
    require(type(raw) is bytes, 'RAW_BYTES_TYPE')
    require(type(dtype) is str and dtype in ('torch.float32', 'torch.int32'), 'UNSUPPORTED_DTYPE')
    require(isinstance(shape, (list, tuple)) and len(shape) > 0, 'SHAPE_CONTAINER')
    elements = 1
    for dimension in shape:
        # Every tensor in this fixed diagnostic is nonempty; bool is not an extent.
        require(type(dimension) is int and dimension > 0, 'POSITIVE_INTEGER_DIMENSION')
        elements *= dimension
    require(len(raw) == 4 * elements, 'RAW_BYTE_COUNT')
    digest = sha(raw)
    blobs.setdefault(digest, base64.b64encode(raw).decode('ascii'))
    result = dict(sha256=digest, nbytes=len(raw), dtype=dtype, shape=list(shape))
    if dtype == 'torch.float32':
        bits = struct.unpack('<'+'I'*(len(raw)//4), raw)
        result.update(nonzero=sum((b & 0x7fffffff) != 0 for b in bits),
                      finite=all((b & 0x7f800000) != 0x7f800000 for b in bits),
                      negative_zero=sum(b == 0x80000000 for b in bits),
                      nonfinite_count=sum(b & 0x7f800000 == 0x7f800000 for b in bits),
                      nonfinite=[dict(index=i, tag='NaN' if bits[i] & 0x7fffff else '-Infinity' if bits[i] >> 31 else 'Infinity')
                                 for i in [i for i,b in enumerate(bits) if b & 0x7f800000 == 0x7f800000][:16]])
    return result


def tensor_record(tensor, sync, blobs):
    if tensor is None:
        return None
    sync()
    cpu = tensor.detach().cpu().contiguous()
    return blob(cpu.view(__import__('torch').uint8).numpy().tobytes(), str(cpu.dtype), cpu.shape, blobs)


def comparison(actual, expected, blobs):
    exact = actual is not None and same(actual, expected) and actual.get('finite', True) is True
    error = None
    if actual and actual['dtype'] == expected['dtype'] == 'torch.float32' and actual['shape'] == expected['shape']:
        a = struct.unpack('<'+'f'*(actual['nbytes']//4), base64.b64decode(blobs[actual['sha256']]))
        e = struct.unpack('<'+'f'*(expected['nbytes']//4), base64.b64decode(blobs[expected['sha256']]))
        error = max((abs(x-y) for x, y in zip(a, e)), default=0.0) if actual['finite'] else 'NONFINITE'
    return dict(exact=bool(exact), max_abs=error, actual=actual, expected=expected)


def run_route(fn, route, case, weight, fixture_bytes, device, sync, blobs):
    import torch
    record = dict(route=route, case=case, status='ERROR', checks={}, error=None)
    def from_bytes(raw, dtype, shape):
        return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
    ids = from_bytes(fixture_bytes['ids'], torch.int32, (T,))
    tangent = from_bytes(fixture_bytes['tangent'], torch.float32, (T,C))
    expected = dict(ids=tensor_record(ids, lambda: None, blobs),
                    weight=tensor_record(weight, lambda: None, blobs),
                    tangent=tensor_record(tangent, lambda: None, blobs))
    record['expected_inputs'] = expected
    expected_output = tensor_record(weight[ids.to(torch.int64)], lambda: None, blobs)
    expected_grad = blob(fixture_bytes['gradient'], 'torch.float32', (V,C), blobs)
    record['expected_output'], record['expected_gradient'] = expected_output, expected_grad
    try:
        w = weight.clone().to(device).detach().requires_grad_(True)
        ix, upstream = ids.to(device), tangent.to(device)
        require(w.grad is None, 'FRESH_GRAD_REQUIRED')
        def snapshot():
            return {k: tensor_record(v, sync, blobs) for k,v in dict(ids=ix, weight=w, tangent=upstream).items()}
        record['before_forward'] = snapshot()
        output = fn(w, ix)
        record['output'] = comparison(tensor_record(output, sync, blobs), expected_output, blobs)
        record['before_backward'] = snapshot()
        output.backward(upstream)
        record['gradient'] = comparison(tensor_record(w.grad, sync, blobs), expected_grad, blobs)
        record['after_backward'] = snapshot()
        record['checks'] = {name: record[name] == expected for name in ('before_forward', 'before_backward', 'after_backward')}
        record['status'] = 'EXACT' if (all(record['checks'].values()) and record['output']['exact'] and record['gradient']['exact']) else 'MISMATCH'
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    return record


def validate_receipt(receipt, expected_rank):
    """Reject incomplete, reordered, inconsistent or byte-inexact observations."""
    require(type(expected_rank) is int and expected_rank in range(4), 'RANK')
    require(type(receipt.get('rank')) is int and receipt['rank'] == expected_rank, 'RANK')
    require(receipt.get('world_size') == 4 and receipt.get('label') == LABEL, 'IDENTITY')
    require(receipt.get('original_K2_gate') == 'FAILED_PRESERVED', 'OLD_GATE')
    require(receipt.get('error') is None and receipt.get('cleanup') == 'DESTROYED' and type(receipt.get('returncode')) is int and receipt['returncode'] == 0, 'EXIT')
    require(type(receipt.get('compiled_callable_count')) is int and receipt['compiled_callable_count'] == 1, 'CALLABLE_COUNT')
    routes, blobs = receipt['routes'], receipt['blobs']
    require(len(routes) == 8, 'ROUTE_COUNT')
    for digest, encoded in blobs.items():
        require(sha(base64.b64decode(encoded, validate=True)) == digest, 'BLOB_HASH')
    weight_record = receipt['weight']
    weight_raw = base64.b64decode(blobs[weight_record['sha256']])
    require(same(blob(weight_raw, 'torch.float32', (V,C), {}), weight_record) and weight_record['finite'] is True, 'WEIGHT')
    for entry, (case, route) in zip(routes, [(c,r) for c in CASES for r in ROUTES]):
        require((entry['case'], entry['route']) == (case,route), 'ORDER')
        require(entry['status'] == 'EXACT' and entry['error'] is None, 'ROUTE_FAILED')
        data = fixture(case)
        ids = struct.unpack('<'+'i'*T, data['ids'])
        expected = dict(ids=blob(data['ids'], 'torch.int32', (T,), {}), weight=weight_record,
                        tangent=blob(data['tangent'], 'torch.float32', (T,C), {}))
        forward = blob(b''.join(weight_raw[i*C*4:(i+1)*C*4] for i in ids), 'torch.float32', (T,C), {})
        gradient = blob(data['gradient'], 'torch.float32', (V,C), {})
        require(same(entry['expected_inputs'], expected) and same(entry['expected_output'], forward) and same(entry['expected_gradient'], gradient), 'ORACLE')
        for item in (*expected.values(), forward, gradient):
            require(item['sha256'] in blobs, 'MISSING_RAW_BYTES')
        for phase in ('before_forward', 'before_backward', 'after_backward'):
            require(same(entry[phase], expected) and entry['checks'].get(phase) is True, 'INPUT_MUTATION')
        for key, target in (('output',forward), ('gradient',gradient)):
            actual = entry[key]['actual']
            require(actual is not None, 'MISSING_TENSOR')
            raw = base64.b64decode(blobs[actual['sha256']], validate=True)
            require(same(blob(raw, actual['dtype'], actual['shape'], {}), actual), 'TENSOR_METADATA')
            require(same(entry[key], comparison(actual, target, blobs)) and entry[key]['exact'] is True, 'NOT_EXACT')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-control-only', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if not args.enable_control_only:
        print(json.dumps(dict(status='DISABLED', native_imported=False)))
        return 0
    require(args.label == LABEL and args.out_dir is not None, 'EXACT_LABEL_OUTPUT_REQUIRED')
    rank = int(os.environ.get('RANK', '-1'))
    require(rank in range(4) and os.environ.get('LOCAL_RANK') == str(rank) and os.environ.get('WORLD_SIZE') == '4', 'RANK_ENV')
    require(os.environ.get('NEURON_LOGICAL_NC_CONFIG') == '2', 'LNC2')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / ('rank' + str(rank) + '.json')
    require(not destination.exists(), 'RECEIPT_EXISTS')
    receipt = dict(label=args.label, rank=rank, world_size=4, pid=os.getpid(), argv=sys.argv[1:],
                   checker_sha256=sha(Path(__file__).read_bytes()), seed=42, routes=[], blobs={},
                   original_K2_gate='FAILED_PRESERVED', error=None, cleanup='NOT_INITIALIZED',
                   compiled_callable_count=0, returncode=1, native_qualification=False,
                   environment={k:os.environ.get(k) for k in ('NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    initialized, dist = False, None
    started = time.monotonic()
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        import torch_neuronx  # Backend registration only; no custom operator modules.
        receipt['runtime'] = dict(torch=torch.__version__, nki=importlib.metadata.version('nki'))
        require(receipt['runtime'] == dict(torch='2.11.0+cpu', nki='0.5.0+28631259367.ga768afa6'), 'RUNTIME_PIN')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        dist.init_process_group('neuron')
        initialized = True
        require(dist.get_rank() == rank and dist.get_world_size() == 4, 'ACTUAL_RANK')
        weight = torch.randn(V,C, dtype=torch.float32)
        receipt['weight'] = tensor_record(weight, lambda:None, receipt['blobs'])
        def ordinary(weight, index):
            return F.embedding(index, weight).to(torch.float32)
        compiled = torch.compile(ordinary, backend='neuron', fullgraph=True, dynamic=False)
        receipt['compiled_callable_count'] = 1
        for case in CASES:
            data = fixture(case)
            for route, fn in zip(ROUTES, (ordinary,compiled)):
                result = run_route(fn, route, case, weight, data, torch.device('neuron'), torch.neuron.synchronize, receipt['blobs'])
                receipt['routes'].append(result)
                print(json.dumps(dict(rank=rank, case=case, route=route, status=result['status'], error=result['error'])), flush=True)
        # Keep mismatch ranks alive until every rank has finished safe witnesses.
        dist.barrier()
    except Exception as exc:
        receipt['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if initialized:
            try:
                dist.destroy_process_group()
                receipt['cleanup'] = 'DESTROYED'
            except Exception as exc:
                receipt['cleanup'] = type(exc).__name__ + ': ' + str(exc)
        receipt['diagnostic_elapsed_seconds'] = time.monotonic() - started
        receipt['returncode'] = 0
        try:
            validate_receipt(receipt, rank)
        except Exception as exc:
            receipt['returncode'] = 1
            receipt['validation_error'] = type(exc).__name__ + ': ' + str(exc)
        raw = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
        require(len(raw.encode()) <= 8*1024**2, 'RECEIPT_SIZE_LIMIT')
        with destination.open('x') as stream:
            stream.write(raw)
    return receipt['returncode']


if __name__ == '__main__':
    sys.exit(main())
