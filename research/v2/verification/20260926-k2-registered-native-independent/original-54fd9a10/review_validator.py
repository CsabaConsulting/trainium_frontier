"""Independent stdlib-only frozen native-packet validator review; never launch."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import platform
import struct
import sys
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MANIFEST = '4c2248de3a8dd8fb91954d34ab7655808e7b0452d82b0bf48b9f09c1cfb1c3b4'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), allow_nan=False, sort_keys=True), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate():
    raw = (PACKET / 'MANIFEST.sha256.json').read_bytes()
    require(sha(raw) == MANIFEST, 'manifest SHA')
    included, omitted = [], []
    for entry in json.loads(raw)['files']:
        basename = Path(entry['path']).name
        if basename in ('README.md', 'test_packet.py'):
            require(not (PACKET / basename).exists(), 'intentional omission changed')
            omitted.append(basename)
        else:
            require(sha((PACKET / basename).read_bytes()) == entry['sha256'], basename)
            included.append(basename)
    old = {'old-host_driver.py': '52ca628542afd89a1974f822ad9d4059a385c6815e5242d2a9db4990d8d5da01',
           'old-rank_entry.py': '5393415d1d6d6af8a8d5eb465746fa99ff8f70429cfdc69975ac5bb03c4685ba',
           'old-NATIVE_BINDING.disabled.json': 'a1533374c564687da0e8969fd07728d42948afffb60a146f20ffe64db43da102'}
    for name, expected in old.items():
        require(sha((PACKET / name).read_bytes()) == expected, 'old source ' + name)
    emit('authenticated', manifest=MANIFEST, subset_files=len(included), omitted=omitted,
         separately_authenticated_old=list(old), python=platform.python_version(), optimized=not __debug__)


def main():
    authenticate()
    validator = load('frozen_validator_independent', PACKET / 'validate_receipts.py')
    reference = load('frozen_reference_independent', PACKET / 'reference.py')
    binding = json.loads((PACKET / 'NATIVE_BINDING.disabled.json').read_text())

    def tensor_record(values, shape, dtype):
        flat = list(values)
        if dtype == 'torch.int32':
            raw = struct.pack('<' + str(len(flat)) + 'i', *flat)
        else:
            raw = struct.pack('<' + str(len(flat)) + 'f', *flat)
            if dtype == 'torch.bfloat16':
                raw = b''.join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
        return dict(shape=shape, dtype=dtype, finite=all(math.isfinite(v) for v in flat),
                    nonzero=sum(v != 0 for v in flat), sha256=sha(raw))

    def exact(record):
        return dict(exact=True, max_abs=0.0, actual=copy.deepcopy(record), expected=copy.deepcopy(record))

    def flatten(rows):
        return [value for row in rows for value in row]

    weight = [[(row + channel + 1) / 32 for channel in range(128)] for row in range(256)]
    weight_record = tensor_record(flatten(weight), [256, 128], 'torch.float32')
    zero = tensor_record([0.0] * 32768, [256, 128], 'torch.float32')

    def make_case(name, dtype, changed=False):
        ids, rows, metadata = reference.fixture(name, dtype)
        if changed:
            rows = [[-v for v in row] for row in rows]
        # Independent scalar recurrence: reference supplies fixture definition only.
        accum = [[0.0] * 128 for _ in range(256)]
        for token, row in enumerate(ids):
            for c, value in enumerate(rows[token]):
                accum[row][c] = struct.unpack('<f', struct.pack('<f', accum[row][c] + value))[0]
        grad = tensor_record(flatten(accum), [256, 128], 'torch.float32')
        kind = 'torch.bfloat16' if dtype == 'bf16' else 'torch.float32'
        up = tensor_record(flatten(rows) if ids else [0.0] * 32768, [256, 128], kind)
        actual_ids = ids if ids else [0] * 256
        output = [[reference.bf16(v) if dtype == 'bf16' else v for v in weight[row]] for row in actual_ids]
        forward = tensor_record(flatten(output), [256, 128], kind)
        before = dict(upstream=up, ids=tensor_record(actual_ids, [256], 'torch.int32'), weight=weight_record)
        for key, values in zip(('counts', 'starts', 'positions', 'maximums'), metadata):
            before[key] = tensor_record(values, [len(values), 1], 'torch.int32')
        routes = []
        for route_name in (validator.ROUTES if ids else validator.ROUTES[:2]):
            checks = dict(gradient=exact(grad))
            if route_name not in validator.ROUTES[:2]:
                checks['output'] = exact(forward)
            routes.append(dict(name=route_name, status='EXACT', checks=checks,
                               inputs_before=copy.deepcopy(before), inputs_after=copy.deepcopy(before),
                               inputs_unchanged=True))
        old = []
        if ids:
            for route_name in validator.OLD_ROUTES:
                route = copy.deepcopy(routes[2])
                route['name'] = route_name
                if route_name.endswith('compiled') and grad['nonzero']:
                    route['status'] = 'MISMATCH'
                    route['checks']['gradient'] = dict(exact=False, max_abs=1.0,
                                                       actual=copy.deepcopy(zero), expected=copy.deepcopy(grad))
                old.append(route)
        checks = dict(upstream_roundtrip=exact(up))
        if ids:
            checks['dense_CPU_oracle'] = exact(grad)
        return dict(name=name, dtype=kind, changed_tangent=changed, status='EXACT',
                    metadata_group_bounds=metadata[3], checks=checks, routes=routes, original_routes=old)

    sentinel = make_case('unique', 'fp32')
    cases = [make_case(name, dtype) for dtype in ('bf16', 'fp32') for name in reference.CASE_NAMES]
    replays = [make_case('unique', dtype, True) for dtype in ('bf16', 'fp32')]
    base = dict(scope='REGISTERED_K2_COMPONENT_DIAGNOSTIC_ONLY', label=binding['label'],
                world_size=4, shape=dict(V=256, T=256, C=128), seed=42,
                argv=binding['checker_args'], first_registered_compiled_fixture='unique/fp32/nonzero',
                torch='2.11.0+cpu', nki_version='0.5.0+28631259367.ga768afa6',
                tolerance=dict(atol=0, rtol=0, criterion='identical bytes'),
                status='REGISTERED_FULL_CASES_OBSERVED_EXACT', returncode=0,
                native_qualification=False, source_adoption=False, accepted_speedup=False,
                public_20m=None, whole_step_speedup='UNMEASURED', timings=[], no_training=True, no_eval=True,
                original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',
                original_fp32_first_gate='FAILED_EXACTNESS_PRESERVED', diagnostic_elapsed_seconds=1.0,
                sentinel=sentinel, cases=cases, tangent_replays=replays)
    for field, name in [('checker_sha256', 'check_native.py'), ('kernel_sha256', 'consumer_nki.py'),
                        ('reference_sha256', 'reference.py'), ('adapter_sha256', 'adapter.py')]:
        base[field] = binding['support_sha256'][name]
    documents = {}
    for rank in range(4):
        documents[f'receipts/rank{rank}.json'] = dict(copy.deepcopy(base), rank=rank)
        documents[f'rank-environment-r{rank}.json'] = dict(rank=rank, local_rank=rank, world_size=4,
            environment=binding['environment'], checker_sha256=binding['support_sha256']['check_native.py'],
            checker_argv=binding['checker_args'])
        documents[f'rank-wrapper-exit-r{rank}.json'] = dict(rank=rank, returncode=0, error=None)

    def validate(documents):
        class MemoryPath(PurePosixPath):
            def read_text(self):
                key = str(self.relative_to('/synthetic-review-only'))
                if key not in documents:
                    raise FileNotFoundError(key)
                return json.dumps(documents[key], allow_nan=True)
        with patch.object(validator, 'Path', MemoryPath):
            return validator.validate('/synthetic-review-only', binding['label'], binding['environment'],
                                      binding['support_sha256'], binding['checker_args'])

    result = validate(documents)
    emit('positive_synthetic_archive', result=result, is_native=False, generated_independently=True)
    # A standalone reproduction input is clearly labeled as synthetic test material.
    with (OUT / ('synthetic-baseline-optimized.json' if not __debug__ else 'synthetic-baseline.json')).open('x') as f:
        json.dump(dict(SYNTHETIC_ONLY=True, purpose='VALIDATOR_TEST_NOT_NATIVE_EVIDENCE', documents=documents), f)

    trials = []
    def trial(name, mutate, expect_reject=True):
        data = copy.deepcopy(documents)
        mutate(data)
        try:
            value = validate(data)
            refused, error = False, None
        except Exception as exc:
            value, refused, error = None, True, type(exc).__name__ + ': ' + str(exc)
        okay = refused if expect_reject else not refused
        record = dict(name=name, expected='REFUSE' if expect_reject else 'ACCEPT_SEPARATE_OLD_FAILURE',
                      actual='REFUSED' if refused else 'ACCEPTED', okay=okay, error=error)
        trials.append(record)
        emit('validator_mutation', **record)
        if not okay:
            suffix = '-optimized' if not __debug__ else ''
            with (OUT / ('counterexample-' + name + suffix + '.json')).open('x') as f:
                json.dump(dict(SYNTHETIC_ONLY=True, mutation=name, result=value, documents=data), f, allow_nan=True)

    def one(data):
        return data['receipts/rank0.json']

    def all_receipts(data, mutate):
        for rank in range(4):
            mutate(data[f'receipts/rank{rank}.json'])

    trial('missing_rank', lambda d: d.pop('receipts/rank3.json'))
    trial('missing_case', lambda d: one(d)['cases'].pop())
    trial('missing_replay', lambda d: one(d)['tangent_replays'].pop())
    trial('sentinel_only', lambda d: one(d).update(cases=[], tangent_replays=[]))
    trial('missing_sentinel', lambda d: one(d).pop('sentinel'))
    trial('wrong_case_order', lambda d: one(d)['cases'].reverse())
    trial('wrong_adapter_hash', lambda d: one(d).update(adapter_sha256='0' * 64))
    trial('wrong_argv', lambda d: one(d).update(argv=[]))
    trial('wrong_label', lambda d: one(d).update(label='other-label'))
    trial('rank_nonzero_exit', lambda d: d['rank-wrapper-exit-r0.json'].update(returncode=1))
    trial('rank_boolean_exit', lambda d: d['rank-wrapper-exit-r0.json'].update(returncode=False))
    trial('candidate_missing_gradient', lambda d: one(d)['sentinel']['routes'][3]['checks'].pop('gradient'))
    trial('old_route_as_candidate', lambda d: one(d)['sentinel']['routes'].__setitem__(3, one(d)['sentinel']['original_routes'][1]))
    trial('zero_both_expected_actual', lambda d: one(d)['sentinel']['routes'][3]['checks'].update(gradient=exact(zero)))
    for label, value in [('nan', float('nan')), ('positive_inf', float('inf')), ('negative_inf', -float('inf'))]:
        trial(label + '_error', lambda d, v=value: one(d)['sentinel']['routes'][3]['checks']['gradient'].update(max_abs=v))
    trial('nonfinite_tensor', lambda d: one(d)['sentinel']['routes'][3]['checks']['gradient']['actual'].update(finite=False))
    trial('changed_metadata_hash', lambda d: one(d)['sentinel']['routes'][3]['inputs_before']['counts'].update(sha256='0' * 64))
    trial('missing_old_observation', lambda d: one(d)['sentinel']['original_routes'].pop())
    trial('hidden_old_mismatch', lambda d: one(d)['sentinel']['original_routes'][1].update(status='EXACT'))
    trial('cross_rank_weight_difference', lambda d: [r[k]['weight'].update(sha256='0' * 64)
          for r in one(d)['sentinel']['routes'] for k in ('inputs_before', 'inputs_after')])
    trial('changed_tangent_stale', lambda d: one(d)['tangent_replays'].__setitem__(0, copy.deepcopy(one(d)['cases'][2])))
    trial('wrong_environment', lambda d: d['rank-environment-r0.json'].update(environment={}))
    trial('adoption_claim', lambda d: one(d).update(native_qualification=True))
    # Required forward output is claimed zero while all same-input controls claim nonzero.
    trial('contradictory_candidate_forward', lambda d: all_receipts(d, lambda r:
          r['sentinel']['routes'][3]['checks'].update(output=exact(zero))))
    # A before/after record claims a different nonzero count from the same upstream roundtrip.
    trial('contradictory_upstream_count', lambda d: all_receipts(d, lambda r:
          [route[key]['upstream'].update(nonzero=0) for route in r['sentinel']['routes']
           for key in ('inputs_before', 'inputs_after')]))
    # Neither runtime identity field can be omitted from complete native evidence.
    trial('missing_runtime_versions', lambda d: all_receipts(d, lambda r: (r.pop('torch'), r.pop('nki_version'))))
    # Distinct original error observations are legitimate and cannot mask candidate failures.
    trial('separate_old_error', lambda d: one(d)['sentinel']['original_routes'][1].update(
          status='ERROR', error='Synthetic original Function execution error', checks={}, inputs_unchanged=False), False)
    trial('malformed_old_mismatch', lambda d: all_receipts(d, lambda r:
          r['sentinel']['original_routes'][1]['checks']['gradient'].update(actual={})))
    require(not any(name == 'torch' or name == 'nki' or name.startswith(('torch.', 'nki.', 'torch_neuronx'))
                    for name in sys.modules), 'runtime import forbidden')
    failures = [t['name'] for t in trials if not t['okay']]
    emit('summary', verdict='REFUSE' if failures else 'PASS_STDLIB_VALIDATOR', trials=len(trials),
         unexpected_acceptances=failures, native_imported=False, native_executed=False)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
