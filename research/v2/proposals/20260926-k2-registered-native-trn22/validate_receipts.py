"""Strict archive completion checks; never a launch, adoption or speed verdict."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('registered_bound_reference', HERE / 'reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
ROUTES = ('direct_consumer', 'direct_repeated_call', 'registered_eager',
          'registered_compiled', 'compiled_ordinary_embedding')
OLD_ROUTES = ('original_function_eager', 'original_function_compiled')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_contract(name, dtype, changed=False):
    ids, rows, metadata = reference.fixture(name, dtype)
    if changed:
        rows = [[-v for v in row] for row in rows]
    dense = reference.dense(ids, rows)
    upstream = rows if ids else [[0.0] * 128 for _ in range(256)]
    raw = reference.raw(upstream)
    if dtype == 'bf16':
        raw = b''.join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
    expected = dict(shape=[256, 128], dtype='torch.float32', finite=True,
                    nonzero=sum(v != 0 for row in dense for v in row), sha256=sha(reference.raw(dense)))
    input_hashes = dict(upstream=sha(raw), ids=sha(struct.pack('<256i', *(ids if ids else [0] * 256))))
    for key, values in zip(('counts', 'starts', 'positions', 'maximums'), metadata):
        input_hashes[key] = sha(struct.pack('<' + str(len(values)) + 'i', *values))
    return dict(gradient=expected, inputs=input_hashes, maximums=metadata[3], logical_nonempty=bool(ids))


def tensor(record, shape, dtype):
    require(record['shape'] == shape and record['dtype'] == dtype, 'TENSOR_SHAPE_DTYPE')
    require(record['finite'] is True and type(record['nonzero']) is int
            and 0 <= record['nonzero'] <= math.prod(shape), 'TENSOR_FINITE_COUNT')
    require(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256']), 'TENSOR_SHA')


def exact(check, shape, dtype, expected=None):
    require(check['exact'] is True and type(check['max_abs']) in (int, float)
            and check['max_abs'] == 0 and check['actual'] == check['expected']
            and 'missing' not in check and 'max_abs_nonfinite' not in check, 'NONEXACT_COMPARISON')
    tensor(check['actual'], shape, dtype)
    if expected is not None:
        require(check['expected'] == expected, 'REFERENCE_VALUE_MISMATCH')


def inputs(route, contract, dtype):
    before, after = route['inputs_before'], route['inputs_after']
    require(route['inputs_unchanged'] is True and before == after
            and set(before) == set(contract['inputs']) | {'weight'}, 'INPUT_MUTATION_OR_MISSING')
    for key, value in before.items():
        shape = [256, 128] if key in ('weight', 'upstream') else [256] if key == 'ids' else [2, 1] if key == 'maximums' else [256, 1]
        kind = 'torch.float32' if key == 'weight' else 'torch.' + ('bfloat16' if dtype == 'bf16' else 'float32') if key == 'upstream' else 'torch.int32'
        tensor(value, shape, kind)
        if key != 'weight':
            require(value['sha256'] == contract['inputs'][key], 'FIXTURE_INPUT_HASH')
    return before


def validate_case(case, name, dtype, changed=False):
    require(case['name'] == name and case['dtype'] == 'torch.' + ('bfloat16' if dtype == 'bf16' else 'float32')
            and case['changed_tangent'] is changed and case['status'] == 'EXACT', 'CASE_IDENTITY_STATUS')
    contract = fixture_contract(name, dtype, changed)
    require(case['metadata_group_bounds'] == contract['maximums'], 'METADATA_GROUPS')
    checks = {'upstream_roundtrip', 'dense_CPU_oracle'} if contract['logical_nonempty'] else {'upstream_roundtrip'}
    require(set(case['checks']) == checks, 'ORIGINAL_CHECK_SET')
    upstream = case['checks']['upstream_roundtrip']
    exact(upstream, [256, 128], case['dtype'])
    require(upstream['actual']['sha256'] == contract['inputs']['upstream'], 'UPSTREAM_HASH')
    if contract['logical_nonempty']:
        exact(case['checks']['dense_CPU_oracle'], [256, 128], 'torch.float32', contract['gradient'])
    required = ROUTES if contract['logical_nonempty'] else ROUTES[:2]
    require([r['name'] for r in case['routes']] == list(required), 'ALL_REQUIRED_ROUTES')
    first = None
    for route in case['routes']:
        require(route['status'] == 'EXACT' and 'error' not in route and 'post_observation_error' not in route,
                'ROUTE_ERROR')
        before = inputs(route, contract, dtype)
        if first is None:
            first = before
        require(before == first, 'ROUTE_INPUTS_DIFFER')
        expected_checks = {'gradient'} if route['name'] in ROUTES[:2] else {'gradient', 'output'}
        require(set(route['checks']) == expected_checks, 'ROUTE_CHECK_SET')
        exact(route['checks']['gradient'], [256, 128], 'torch.float32', contract['gradient'])
        if 'output' in expected_checks:
            exact(route['checks']['output'], [256, 128], case['dtype'])
    # Old failures remain visible. They neither count as candidate exactness nor
    # suppress candidate observations. Missing/malformed observations are refused.
    old_required = OLD_ROUTES if contract['logical_nonempty'] else ()
    require([r['name'] for r in case['original_routes']] == list(old_required), 'ORIGINAL_ROUTES_MISSING')
    for route in case['original_routes']:
        require(route['status'] in ('EXACT', 'MISMATCH', 'ERROR', 'MISMATCH_OR_ERROR'), 'ORIGINAL_STATUS')
        if route['status'] in ('ERROR', 'MISMATCH_OR_ERROR'):
            require(bool(route.get('error') or route.get('post_observation_error')) or route.get('inputs_unchanged') is False,
                    'ORIGINAL_ERROR_WITHOUT_EVIDENCE')
        else:
            inputs(route, contract, dtype)
            require(set(route['checks']) == {'output', 'gradient'}, 'ORIGINAL_CHECKS_MISSING')
            observed = all(x.get('exact') is True for x in route['checks'].values())
            require(observed == (route['status'] == 'EXACT'), 'ORIGINAL_STATUS_CONTRADICTION')
            for key, value in route['checks'].items():
                if value.get('exact') is True:
                    exact(value, [256, 128], 'torch.float32' if key == 'gradient' else case['dtype'],
                          contract['gradient'] if key == 'gradient' else None)
                else:
                    require(value.get('missing') is True or 'actual' in value, 'ORIGINAL_MISMATCH_WITHOUT_EVIDENCE')


def validate(root, label, environment, hashes, checker_args):
    root = Path(root)
    observations = []
    for rank in range(4):
        receipt = json.loads((root / 'receipts' / ('rank%d.json' % rank)).read_text())
        env = json.loads((root / ('rank-environment-r%d.json' % rank)).read_text())
        exited = json.loads((root / ('rank-wrapper-exit-r%d.json' % rank)).read_text())
        require(receipt['status'] == 'REGISTERED_FULL_CASES_OBSERVED_EXACT' and
                type(receipt['returncode']) is int and receipt['returncode'] == 0 and 'error' not in receipt, 'INCOMPLETE')
        require(receipt['scope'] == 'REGISTERED_K2_COMPONENT_DIAGNOSTIC_ONLY' and receipt['label'] == label
                and type(receipt['rank']) is int and receipt['rank'] == rank and receipt['world_size'] == 4, 'IDENTITY')
        require(receipt['shape'] == dict(V=256, T=256, C=128) and receipt['seed'] == 42, 'SHAPE_SEED')
        require(receipt['argv'] == checker_args and receipt['first_registered_compiled_fixture'] == 'unique/fp32/nonzero', 'ARGV_ORDER')
        for field, filename in [('checker_sha256', 'check_native.py'), ('kernel_sha256', 'consumer_nki.py'),
                                ('reference_sha256', 'reference.py'), ('adapter_sha256', 'adapter.py')]:
            require(receipt[field] == hashes[filename], 'SOURCE_PIN')
        require(receipt['tolerance'] == dict(atol=0, rtol=0, criterion='identical bytes'), 'TOLERANCE')
        require(receipt['native_qualification'] is False and receipt['source_adoption'] is False
                and receipt['accepted_speedup'] is False and receipt['public_20m'] is None
                and receipt['whole_step_speedup'] == 'UNMEASURED' and receipt['timings'] == []
                and receipt['no_training'] is True and receipt['no_eval'] is True, 'NO_ADOPTION')
        require(receipt['original_twenty_case_gate'] == 'FAILED_EXACTNESS_PARKED_UNCHANGED'
                and receipt['original_fp32_first_gate'] == 'FAILED_EXACTNESS_PRESERVED', 'OLD_FAILURE_PRESERVED')
        require(type(receipt['diagnostic_elapsed_seconds']) in (int, float) and
                math.isfinite(receipt['diagnostic_elapsed_seconds']) and receipt['diagnostic_elapsed_seconds'] >= 0, 'ELAPSED')
        require(env['rank'] == env['local_rank'] == rank and env['world_size'] == 4 and env['environment'] == environment, 'RANK_ENV')
        require(env['checker_sha256'] == hashes['check_native.py'] and env['checker_argv'] == checker_args, 'ENTRY_PIN_ARGV')
        require(exited['rank'] == rank and type(exited['returncode']) is int
                and exited['returncode'] == 0 and exited['error'] is None, 'EXIT')
        validate_case(receipt['sentinel'], 'unique', 'fp32')
        require(len(receipt['cases']) == 20 and len(receipt['tangent_replays']) == 2, 'FULL_CASE_COVERAGE')
        for case, (name, dtype) in zip(receipt['cases'], [(n, d) for d in ('bf16', 'fp32') for n in reference.CASE_NAMES]):
            validate_case(case, name, dtype)
        for replay, dtype in zip(receipt['tangent_replays'], ('bf16', 'fp32')):
            validate_case(replay, 'unique', dtype, True)
        observations.append([receipt['sentinel'], receipt['cases'], receipt['tangent_replays']])
    # Require required-case observations to agree across ranks; old exception text
    # may include rank-local filenames and is separately preserved, not compared.
    def required_only(value):
        if isinstance(value, dict):
            return {k: required_only(v) for k, v in value.items() if k != 'original_routes'}
        if isinstance(value, list):
            return [required_only(v) for v in value]
        return value
    require(all(required_only(x) == required_only(observations[0]) for x in observations[1:]), 'CROSS_RANK_REQUIRED_DIFFER')
    return dict(status='REGISTERED_ALL4_FULL20_OBSERVATIONS_EXACT_PENDING_RESULT_REVIEW', ranks=4,
                original_checks_per_rank=150, cases_per_rank=20, sentinel_per_rank=1, tangent_replays_per_rank=2,
                native_qualification=False, source_adoption=False, accepted_speedup=False,
                full_run_authorized=False, original_gate='FAILED_EXACTNESS_PRESERVED')
