"""Implementer self-checks only, not independent review or native evidence."""
import copy
import base64
import json
import struct
import subprocess
import sys
from pathlib import Path

import ordinary_subnormal_probe as p


def main():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    weight = torch.randn(p.V,p.C)
    receipt = dict(label=p.LABEL, rank=0, world_size=4, error=None, cleanup='DESTROYED',
                   original_K2_gate='FAILED_PRESERVED', compiled_callable_count=1,
                   returncode=0, routes=[], blobs={})
    receipt['weight'] = p.tensor_record(weight, lambda:None, receipt['blobs'])
    # CPU eager duplicates intentionally exercise receipt structure, never native compilation.
    for case in p.CASES:
        for route in p.ROUTES:
            receipt['routes'].append(p.run_route(lambda w,i:F.embedding(i,w).to(torch.float32),
                route, case, weight, p.fixture(case), torch.device('cpu'), lambda:None, receipt['blobs']))
    p.validate_receipt(receipt,0)
    tests = ['cpu_ordinary_eight_routes', 'integer_original_pins']
    def empty_weight_forward(r):
        # Reconstruct the independently reported all-consistent empty-payload forgery.
        empty = dict(sha256=p.sha(b''), nbytes=0, dtype='torch.float32', shape=[p.V,p.C],
                     nonzero=0, finite=True, negative_zero=0, nonfinite_count=0, nonfinite=[])
        r['blobs'][empty['sha256']] = base64.b64encode(b'').decode()
        r['weight'] = copy.deepcopy(empty)
        for route in r['routes']:
            for phase in ('expected_inputs','before_forward','before_backward','after_backward'):
                route[phase]['weight'] = copy.deepcopy(empty)
            route['expected_output'] = copy.deepcopy(empty)
            route['output'] = dict(exact=True,max_abs=0.0,actual=copy.deepcopy(empty),expected=copy.deepcopy(empty))
    mutations = {
        'consistent_empty_weight_forward': empty_weight_forward,
        'zero_gradient': lambda r:r['routes'][2]['gradient'].update(actual=p.blob(bytes(p.V*p.C*4),'torch.float32',(p.V,p.C),r['blobs'])),
        'stale_positive_on_negative': lambda r:r['routes'][6].update(gradient=r['routes'][2]['gradient']),
        'wrong_route': lambda r:r['routes'][0].update(route='invalid'),
        'wrong_rank': lambda r:r.update(rank=1),
        'wrong_case': lambda r:r['routes'][0].update(case='normal'),
        'input_mutation': lambda r:r['routes'][0]['before_backward'].update(tangent=r['routes'][2]['before_backward']['tangent']),
        'missing_gradient': lambda r:r['routes'][0]['gradient'].update(actual=None),
        'missing_route': lambda r:r['routes'].pop(),
        'cleanup_error': lambda r:r.update(cleanup='ERROR'),
        'false_exact': lambda r:r['routes'][0]['gradient'].update(exact=False),
        'wrong_blob': lambda r:r['blobs'].update({r['weight']['sha256']:'AAAA'}),
        'boolean_exit': lambda r:r.update(returncode=False),
        'boolean_error_metric': lambda r:r['routes'][0]['gradient'].update(max_abs=False),
        'missing_input_bytes': lambda r:r['blobs'].pop(r['routes'][0]['expected_inputs']['ids']['sha256']),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(receipt)
        mutate(changed)
        try:
            p.validate_receipt(changed,0)
        except (ValueError,KeyError):
            tests.append(name)
        else:
            raise RuntimeError('NEGATIVE_ACCEPTED:'+name)
    invalid_blobs = [
        ('zero_bytes',b'', 'torch.float32',(p.V,p.C)),
        ('truncated_bytes',bytes(p.V*p.C*4-4), 'torch.float32',(p.V,p.C)),
        ('overlong_bytes',bytes(p.V*p.C*4+4), 'torch.float32',(p.V,p.C)),
        ('truncated_int32',bytes(4*p.T-1), 'torch.int32',(p.T,)),
        ('wrong_dtype',bytes(4), 'torch.float64',(1,)),
        ('wrong_dtype_type',bytes(4), None,(1,)),
        ('bool_dimension',bytes(4), 'torch.float32',(True,)),
        ('float_dimension',bytes(4), 'torch.float32',(1.0,)),
        ('negative_dimension',b'', 'torch.float32',(-1,)),
        ('zero_dimension',b'', 'torch.float32',(0,)),
        ('string_dimension',bytes(4), 'torch.float32',('1',)),
        ('empty_shape',bytes(4), 'torch.float32',()),
    ]
    for name,raw,dtype,shape in invalid_blobs:
        untouched = {}
        try:
            p.blob(raw,dtype,shape,untouched)
        except ValueError:
            p.require(not untouched,'REJECT_BEFORE_RECORDING')
            tests.append(name)
        else:
            raise RuntimeError('INVALID_BLOB_ACCEPTED:'+name)
    z = p.blob(b'\0'*4,'torch.float32',(1,),{})
    blobs = {}
    negative_zero = p.blob(struct.pack('<I',0x80000000),'torch.float32',(1,),blobs)
    p.blob(b'\0'*4,'torch.float32',(1,),blobs)
    if p.comparison(negative_zero,z,blobs)['exact']:
        raise RuntimeError('SIGNED_ZERO_ACCEPTED')
    nan = p.blob(struct.pack('<I',0x7fc00000),'torch.float32',(1,),blobs)
    p.require(not nan['finite'] and nan['nonfinite'][0]['tag'] == 'NaN', 'NAN_TAG')
    json.dumps(p.comparison(nan,z,blobs), allow_nan=False)
    all_nan = p.blob(struct.pack('<I',0x7fc00000)*(p.V*p.C),'torch.float32',(p.V,p.C),blobs)
    p.require(all_nan['nonfinite_count'] == p.V*p.C and len(all_nan['nonfinite']) == 16, 'BOUNDED_NAN_TAGS')
    tests += ['signed_zero_rejected','nan_json_safe','all_nan_tags_bounded_full_bytes_retained']
    disabled = subprocess.run([sys.executable,'-I','-B',str(Path(p.__file__).resolve())], capture_output=True,text=True)
    p.require(disabled.returncode == 0 and json.loads(disabled.stdout)['native_imported'] is False, 'DISABLED')
    tests.append('default_disabled')
    print(json.dumps(dict(status='SELF_CHECK_PASS',torch=torch.__version__,native=False,
                         cpu_route_alias='both receipt names use CPU eager; no compiled claim', tests=tests,
                         fixture_pins=p.PINS, receipt_bytes=len(json.dumps(receipt)),
                         routes=[dict(case=r['case'],route=r['route'],status=r['status']) for r in receipt['routes']]),indent=2))


if __name__ == '__main__':
    main()
