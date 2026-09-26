"""Completion-only validator for the FP32-first diagnostic; never native acceptance."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROUTES = ('direct_consumer', 'eager_custom', 'compiled_custom', 'compiled_ordinary_embedding')
EXPECTED_G = 'dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671'


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def tensor(record, shape, dtype):
    require(record['shape'] == shape and record['dtype'] == dtype, 'TENSOR_SHAPE_DTYPE')
    require(record['finite'] is True and type(record['nonzero']) is int
            and 0 <= record['nonzero'] <= math.prod(shape), 'TENSOR_FINITE_COUNT')
    require(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None, 'TENSOR_SHA')


def exact(check):
    require(check['exact'] is True and check['max_abs'] == 0 and check['actual'] == check['expected'], 'NONEXACT_COMPARISON')
    tensor(check['actual'], [256, 128], 'torch.float32')


def validate(root, label, environment, hashes, checker_args):
    root = Path(root)
    cases = []
    expected_input_hashes = {
        'upstream': EXPECTED_G,
        'ids': hashlib.sha256(struct.pack('<256i', *range(256))).hexdigest(),
        'counts': hashlib.sha256(struct.pack('<256i', *([1]*256))).hexdigest(),
        'starts': hashlib.sha256(struct.pack('<256i', *range(256))).hexdigest(),
        'positions': hashlib.sha256(struct.pack('<256i', *range(256))).hexdigest(),
        'maximums': hashlib.sha256(struct.pack('<2i', 1, 1)).hexdigest(),
    }
    for rank in range(4):
        receipt = json.loads((root/'receipts'/('rank%d.json' % rank)).read_text())
        env = json.loads((root/('rank-environment-r%d.json' % rank)).read_text())
        exited = json.loads((root/('rank-wrapper-exit-r%d.json' % rank)).read_text())
        require(receipt['status'] == 'FP32_FIRST_DIAGNOSTIC_OBSERVED_EXACT' and receipt['returncode'] == 0, 'DIAGNOSTIC_NOT_EXACT')
        require(receipt['scope'] == 'FP32_FIRST_SOURCE_NEUTRAL_DIAGNOSTIC_ONLY', 'SCOPE')
        require(receipt['rank'] == rank and receipt['world_size'] == 4 and receipt['label'] == label, 'IDENTITY')
        require(receipt['shape'] == dict(V=256, T=256, C=128) and receipt['seed'] == 42, 'SHAPE_SEED')
        require(receipt['argv'] == checker_args, 'ARGV')
        for field, name in [('checker_sha256', 'check_native.py'), ('kernel_sha256', 'consumer_nki.py'), ('reference_sha256', 'reference.py')]:
            require(receipt[field] == hashes[name], 'SOURCE_PIN')
        require(receipt['tolerance'] == dict(atol=0, rtol=0, criterion='identical bytes'), 'TOLERANCE')
        require(receipt['route_order'] == list(ROUTES) and receipt['first_custom_compiled_fixture'] == 'unique/fp32', 'ORDER')
        require(receipt['native_qualification'] is False and receipt['source_adoption'] is False
                and receipt['accepted_speedup'] is False and receipt['public_20m'] is None
                and receipt['whole_step_speedup'] == 'UNMEASURED' and receipt['timings'] == [], 'NO_ACCEPTANCE')
        require(receipt['original_twenty_case_gate'] == 'FAILED_EXACTNESS_PARKED_UNCHANGED'
                and receipt['no_training'] is True and receipt['no_eval'] is True, 'OLD_GATE_AND_SCOPE')
        require(type(receipt['diagnostic_elapsed_seconds']) in (int, float) and
                math.isfinite(receipt['diagnostic_elapsed_seconds']) and receipt['diagnostic_elapsed_seconds'] >= 0, 'ELAPSED')
        require(env['rank'] == env['local_rank'] == rank and env['world_size'] == 4 and env['environment'] == environment, 'RANK_ENV')
        require(env['checker_sha256'] == hashes['check_native.py'] and env['checker_argv'] == checker_args, 'ENTRY_PIN_ARGV')
        require(exited['rank'] == rank and type(exited['returncode']) is int and exited['returncode'] == 0 and exited['error'] is None, 'EXIT')
        case = receipt['case']
        require(case['case'] == 'unique' and case['dtype'] == 'torch.float32' and case['metadata_group_bounds'] == [1, 1], 'FIXTURE')
        exact(case['dense_cpu_oracle'])
        require(case['dense_cpu_oracle']['actual']['sha256'] == EXPECTED_G, 'DENSE_ORACLE')
        require([r['name'] for r in case['routes']] == list(ROUTES), 'ALL_ROUTES')
        first_inputs = case['routes'][0]['inputs_before']
        for route in case['routes']:
            require(route['status'] == 'EXACT' and route['inputs_unchanged'] is True, 'ROUTE_NOT_EXACT')
            before, after = route['inputs_before'], route['inputs_after']
            require(before == after == first_inputs and set(before) == set(expected_input_hashes) | {'weight'}, 'INPUT_MUTATION')
            for name, value in before.items():
                shape = [256, 128] if name in ('weight', 'upstream') else [256] if name == 'ids' else [2, 1] if name == 'maximums' else [256, 1]
                tensor(value, shape, 'torch.float32' if name in ('weight', 'upstream') else 'torch.int32')
                if name != 'weight':
                    require(value['sha256'] == expected_input_hashes[name], 'FIXTURE_INPUT_HASH')
            expected_checks = {'gradient'} if route['name'] == 'direct_consumer' else {'gradient', 'output'}
            require(set(route['checks']) == expected_checks, 'CHECK_SET')
            for value in route['checks'].values():
                exact(value)
            require(route['checks']['gradient']['actual']['sha256'] == EXPECTED_G
                    and route['checks']['gradient']['actual']['nonzero'] == 32709, 'NONZERO_GRADIENT')
        cases.append(case)
    require(all(case == cases[0] for case in cases[1:]), 'CROSS_RANK_OBSERVATIONS_DIFFER')
    return dict(status='FP32_FIRST_OBSERVATIONS_EXACT_NOT_NATIVE_QUALIFICATION', ranks=4,
                routes_per_rank=4, accepted_speedup=False, native_qualification=False,
                source_adoption=False, full_run_authorized=False,
                original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED')
