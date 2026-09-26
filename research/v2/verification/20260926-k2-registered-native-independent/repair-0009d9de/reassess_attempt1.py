"""Independent input-only validator successor review. No Torch/native imports."""
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

ROOT = Path('/tmp/trainium-k2-validator-repair-review-b3ZFC3')
PACKET = ROOT / 'research/v2/proposals/20260926-k2-registered-native-trn22'
PRIOR = Path('/tmp/trainium-k2-registration-review-GS6xFb')
OLD = PRIOR / 'native-packet'
OWN = OLD / 'native-review-output'
OUT = ROOT / 'review-output'
NEW_MANIFEST = 'fcb2c40adbe5a33a93b4ded6edb544afdff9a42c1f17f86a84d7519a346fd6d2'
OLD_MANIFEST = '4c2248de3a8dd8fb91954d34ab7655808e7b0452d82b0bf48b9f09c1cfb1c3b4'
OWN_MANIFEST = '14fa1b666709863d52a87123107fba05a47ed77ba0927a9a89c8ab18b24e4511'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True, allow_nan=False), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate():
    manifest = (PACKET / 'MANIFEST.sha256.json').read_bytes()
    require(sha(manifest) == NEW_MANIFEST, 'new manifest identity')
    allowed_omissions = {'README.md', 'VALIDATOR_REPAIR.md', 'test_packet.py',
                         'replay_repair_repros.py', 'repair-evidence/original-reviewer-refuse.log'}
    present, omitted = [], []
    for item in json.loads(manifest)['files']:
        path = ROOT / item['path']
        relative = str(path.relative_to(PACKET))
        if not path.exists():
            require(relative in allowed_omissions, 'unexpected omission ' + relative)
            omitted.append(relative)
        else:
            require(sha(path.read_bytes()) == item['sha256'], 'input SHA ' + relative)
            present.append(relative)
    require(set(omitted) == allowed_omissions, 'omission subset changed')
    old_manifest = (OLD / 'MANIFEST.sha256.json').read_bytes()
    require(sha(old_manifest) == OLD_MANIFEST, 'original manifest identity')
    for item in json.loads(old_manifest)['files']:
        path = OLD / Path(item['path']).name
        if path.name not in ('README.md', 'test_packet.py'):
            require(sha(path.read_bytes()) == item['sha256'], 'original input SHA ' + path.name)
    own_manifest = (OWN / 'OUTPUT_SHA256.txt').read_bytes()
    require(sha(own_manifest) == OWN_MANIFEST, 'own original output manifest identity')
    for line in own_manifest.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        require(sha((OWN / name).read_bytes()) == digest, 'own prior output SHA ' + name)
    protected = ['check_native.py', 'adapter.py', 'consumer_nki.py', 'reference.py',
                 'rank_entry.py', 'idle_guard.py', 'dispatch_driver.py',
                 'EXPERIMENT_PROPOSAL.json', 'bound-command.txt']
    for name in protected:
        require((PACKET / name).read_bytes() == (OLD / name).read_bytes(), 'changed protected ' + name)
    oldhash = sha((OLD / 'validate_receipts.py').read_bytes())
    newhash = sha((PACKET / 'validate_receipts.py').read_bytes())
    olddriver = (OLD / 'host_driver.py').read_text()
    require(olddriver.count(oldhash) == 1, 'old driver pin occurrence')
    require(olddriver.replace(oldhash, newhash) == (PACKET / 'host_driver.py').read_text(),
            'driver changed beyond validator pin')
    oldbinding = json.loads((OLD / 'NATIVE_BINDING.disabled.json').read_text())
    newbinding = json.loads((PACKET / 'NATIVE_BINDING.disabled.json').read_text())
    oldbinding['driver_sha256'] = sha((PACKET / 'host_driver.py').read_bytes())
    oldbinding['support_sha256']['validate_receipts.py'] = newhash
    require(oldbinding == newbinding, 'binding beyond two SHA pins')
    require(not newbinding['launch_authorized'], 'binding must remain disabled')
    auth = json.loads((PACKET / 'repair-evidence/runtime-authentication.json').read_text())
    for entry in auth['runtimes']:
        path = PRIOR / 'native-fp32-first' / entry['path']
        require(sha(path.read_bytes()) == entry['sha256'], 'prior raw runtime receipt SHA')
        data = json.loads(path.read_text())
        require(data['rank'] == entry['rank'], 'runtime receipt rank')
        require({key: data[key] for key in ('torch', 'nki_version')} ==
                {key: entry[key] for key in ('torch', 'nki_version')}, 'runtime receipt values')
    emit('authenticated', new_manifest=NEW_MANIFEST, original_manifest=OLD_MANIFEST,
         own_original_manifest=OWN_MANIFEST, present_files=len(present), omitted=omitted,
         protected_unchanged=protected, driver_only_validator_pin=True, binding_only_two_pins=True,
         prior_raw_runtime_receipts=4, validator_sha256=newhash, repair_diff_sha256=sha((ROOT / 'repair.diff').read_bytes()),
         python=platform.python_version(), optimized=not __debug__)
    return newbinding


def record(values, shape, kind):
    values = list(values)
    code = 'i' if kind == 'torch.int32' else 'f'
    raw = struct.pack('<' + str(len(values)) + code, *values)
    if kind == 'torch.bfloat16':
        raw = b''.join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
    return dict(shape=shape, dtype=kind, finite=all(math.isfinite(v) for v in values),
                nonzero=sum(v != 0 for v in values), sha256=sha(raw))


def independent_exactness(validator, baseline):
    # The frozen reference supplies fixture definitions, never the accumulation or encoding oracle.
    reference = load('independent_fixture_definition', PACKET / 'reference.py')
    checked = 0
    for changed, dtype, name in [(False, d, n) for d in ('bf16', 'fp32') for n in reference.CASE_NAMES] + \
                               [(True, d, 'unique') for d in ('bf16', 'fp32')] + [(False, 'fp32', 'unique')]:
        ids, rows, metadata = reference.fixture(name, dtype)
        if changed:
            rows = [[-x for x in row] for row in rows]
        accum = [0.0] * (256 * 128)
        for token, index in enumerate(ids):
            for channel, value in enumerate(rows[token]):
                offset = index * 128 + channel
                accum[offset] = struct.unpack('<f', struct.pack('<f', accum[offset] + value))[0]
        kind = 'torch.bfloat16' if dtype == 'bf16' else 'torch.float32'
        flat = [x for row in rows for x in row] if ids else [0.0] * 32768
        inputs = dict(upstream=record(flat, [256, 128], kind),
                      ids=record(ids if ids else [0] * 256, [256], 'torch.int32'))
        for key, values in zip(('counts', 'starts', 'positions', 'maximums'), metadata):
            inputs[key] = record(values, [len(values), 1], 'torch.int32')
        expected = dict(gradient=record(accum, [256, 128], 'torch.float32'), inputs=inputs,
                        maximums=metadata[3], logical_nonempty=bool(ids))
        require(validator.fixture_contract(name, dtype, changed) == expected, 'own scalar exactness ' + name + dtype)
        emit('independent_scalar_exactness', case=name, dtype=dtype, changed_tangent=changed,
             gradient_sha256=expected['gradient']['sha256'], gradient_nonzero=expected['gradient']['nonzero'])
        checked += 1
    require(checked == 23, 'own exactness coverage')
    # Independently verify positive archive's synthetic nonzero embedding lookup forward bytes.
    for case in [baseline['receipts/rank0.json']['sentinel']] + baseline['receipts/rank0.json']['cases']:
        if not case['original_routes']:
            continue
        dtype = 'bf16' if case['dtype'] == 'torch.bfloat16' else 'fp32'
        ids, _, _ = reference.fixture(case['name'], dtype)
        values = [(index + channel + 1) / 32 for index in ids for channel in range(128)]
        if dtype == 'bf16':
            # Explicit nearest-even BF16 rounding, independent of reference.bf16.
            rounded = []
            for value in values:
                bits = struct.unpack('<I', struct.pack('<f', value))[0]
                bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
                rounded.append(struct.unpack('<f', struct.pack('<I', bits))[0])
            values = rounded
        expected = record(values, [256, 128], case['dtype'])
        require(case['routes'][2]['checks']['output']['actual'] == expected, 'own positive forward lookup bytes')
    emit('independent_exactness_complete', contracts=checked, synthetic_forward_lookup=True,
         torch_imported=False, native=False, maximum_numeric_elements=32768)


def main():
    binding = authenticate()
    validator = load('repair_validator_independent', PACKET / 'validate_receipts.py')
    baseline = json.loads((OWN / 'synthetic-baseline.json').read_text())['documents']
    independent_exactness(validator, baseline)

    def validate(documents):
        class MemoryPath(PurePosixPath):
            def read_text(self):
                return json.dumps(documents[str(self.relative_to('/synthetic-only'))], allow_nan=False)
        with patch.object(validator, 'Path', MemoryPath):
            return validator.validate('/synthetic-only', binding['label'], binding['environment'],
                                      binding['support_sha256'], binding['checker_args'])

    results = []
    def test(name, data, accept):
        try:
            result = validate(data)
            accepted, error = True, None
        except Exception as exc:
            result, accepted, error = None, False, type(exc).__name__ + ': ' + str(exc)
        okay = accepted == accept
        item = dict(name=name, expected_accept=accept, accepted=accepted, okay=okay, error=error)
        results.append(item)
        emit('own_test', **item)
        if not okay:
            suffix = '-optimized' if not __debug__ else ''
            path = OUT / ('unexpected-' + name + suffix + '.json')
            with path.open('x') as handle:
                json.dump(dict(SYNTHETIC_ONLY=True, result=result, documents=data), handle, allow_nan=False)

    test('own_positive_all4_all20_sentinel_changed_tangent', baseline, True)
    for name in ('contradictory_candidate_forward', 'contradictory_upstream_count',
                 'missing_runtime_versions', 'malformed_old_mismatch'):
        source = OWN / ('counterexample-' + name + '.json')
        test('frozen_original_' + name, json.loads(source.read_text())['documents'], False)

    def trial(name, mutate, accept=False):
        data = copy.deepcopy(baseline)
        mutate(data)
        test(name, data, accept)

    def receipt(data):
        return data['receipts/rank0.json']

    def old(data):
        return receipt(data)['sentinel']['original_routes'][1]

    def candidate(data):
        return receipt(data)['sentinel']['routes'][3]

    def replace_old_comparison(data, actual, max_abs, descriptor=None):
        route = old(data)
        expected = route['checks']['gradient']['expected']
        check = dict(exact=False, actual=actual, expected=expected, max_abs=max_abs)
        if descriptor:
            check['max_abs_nonfinite'] = descriptor
        route['checks']['gradient'] = check
        route['status'] = 'MISMATCH'

    for label, value, descriptor in [('nan', float('nan'), 'nan'), ('positive_inf', float('inf'), 'inf'),
                                     ('negative_inf', -float('inf'), 'inf')]:
        actual = record([value] * 32768, [256, 128], 'torch.float32')
        trial('retain_old_' + label, lambda d, a=actual, desc=descriptor:
              replace_old_comparison(d, a, None, desc), True)
        trial('refuse_candidate_' + label, lambda d, a=actual:
              candidate(d)['checks']['gradient'].update(actual=a), False)
    trial('retain_old_zero_gradient', lambda d: None, True)
    trial('retain_old_missing_gradient', lambda d: old(d)['checks'].update(gradient=dict(
          exact=False, missing=True, expected=old(d)['checks']['gradient']['expected'])), True)
    trial('retain_old_error_complete_inputs', lambda d: old(d).update(status='ERROR', checks={},
          error='Independent synthetic old execution error'), True)
    trial('retain_old_error_early_missing_inputs', lambda d: old(d).update(status='ERROR', checks={},
          error='Independent synthetic early old error', inputs_before={}, inputs_after={}, inputs_unchanged=False), True)
    trial('retain_old_post_observation_error', lambda d: old(d).update(status='ERROR',
          post_observation_error='Independent synthetic old observation error', inputs_after={}, inputs_unchanged=False), True)
    trial('retain_old_dtype_mismatch', lambda d: old(d)['checks']['gradient']['actual'].update(dtype='torch.float64'), True)
    def shape_mismatch(data):
        old(data)['checks']['gradient']['actual']['shape'] = [128, 256]
        old(data)['checks']['gradient'].pop('max_abs')
    trial('retain_old_shape_mismatch', shape_mismatch, True)
    def signed_zero(data):
        case = next(c for c in receipt(data)['cases'] if c['name'] == 'zero')
        route = case['original_routes'][1]
        route['status'] = 'MISMATCH'
        route['checks']['gradient'] = dict(exact=False, max_abs=0.0,
            actual=record([-0.0] * 32768, [256, 128], 'torch.float32'),
            expected=route['checks']['gradient']['expected'])
    trial('retain_old_signed_zero_byte_mismatch', signed_zero, True)
    trial('refuse_old_zero_claimed_exact', lambda d: old(d).update(status='EXACT'))
    trial('refuse_old_empty_actual_record', lambda d: old(d)['checks']['gradient'].update(actual={}))
    trial('refuse_old_missing_expected_record', lambda d: old(d)['checks']['gradient'].pop('expected'))
    trial('refuse_old_forged_expected_record', lambda d: old(d)['checks']['gradient']['expected'].update(nonzero=0))
    trial('refuse_old_false_exact_flag', lambda d: old(d)['checks']['gradient'].update(exact=True))
    trial('refuse_old_error_without_error_text', lambda d: old(d).update(status='ERROR', checks={}))
    trial('refuse_old_error_blank_text', lambda d: old(d).update(status='ERROR', checks={}, error='  '))
    trial('refuse_old_missing_claimed_exact', lambda d: old(d)['checks'].update(gradient=dict(
          exact=True, missing=True, expected=old(d)['checks']['gradient']['expected'])))
    trial('refuse_old_zero_error_count_contradiction', lambda d: old(d)['checks']['gradient'].update(max_abs=0))
    trial('refuse_old_shape_boolean', lambda d: old(d)['checks']['gradient']['actual'].update(shape=[True, 32768]))
    trial('refuse_old_nonzero_boolean', lambda d: old(d)['checks']['gradient']['actual'].update(nonzero=False))
    trial('refuse_old_nonzero_overflow', lambda d: old(d)['checks']['gradient']['actual'].update(nonzero=32769))
    trial('refuse_old_hash_malformed', lambda d: old(d)['checks']['gradient']['actual'].update(sha256='0' * 63))
    for field in ('torch', 'nki_version'):
        for label, value in [('blank', ' '), ('wrong_type', 211), ('wrong_version', 'synthetic-wrong-version')]:
            trial('refuse_' + field + '_' + label, lambda d, f=field, v=value: receipt(d).update({f: v}))
        trial('refuse_missing_' + field, lambda d, f=field: receipt(d).pop(f))
    for field, value in [('shape', [128, 256]), ('dtype', 'torch.float64'), ('finite', False),
                         ('nonzero', 0), ('sha256', '0' * 64)]:
        trial('refuse_candidate_forward_' + field, lambda d, f=field, v=value:
              [candidate(d)['checks']['output'][k].update({f: v}) for k in ('actual', 'expected')])
        trial('refuse_upstream_' + field, lambda d, f=field, v=value:
              [candidate(d)[k]['upstream'].update({f: v}) for k in ('inputs_before', 'inputs_after')])
    trial('refuse_candidate_missing_gradient', lambda d: candidate(d)['checks'].pop('gradient'))
    trial('refuse_candidate_zero_actual_and_expected', lambda d: candidate(d)['checks']['gradient'].update(
          actual=record([0.0] * 32768, [256, 128], 'torch.float32'),
          expected=record([0.0] * 32768, [256, 128], 'torch.float32')))
    trial('refuse_old_route_as_candidate', lambda d: receipt(d)['sentinel']['routes'].__setitem__(3, old(d)))
    trial('refuse_missing_rank', lambda d: d.pop('receipts/rank3.json'))
    trial('refuse_missing_case', lambda d: receipt(d)['cases'].pop())
    trial('refuse_missing_tangent_replay', lambda d: receipt(d)['tangent_replays'].pop())
    trial('refuse_sentinel_only', lambda d: receipt(d).update(cases=[], tangent_replays=[]))
    require(not any(n == 'torch' or n == 'nki' or n.startswith(('torch.', 'nki.', 'torch_neuronx'))
                    for n in sys.modules), 'forbidden runtime imported')
    failures = [t['name'] for t in results if not t['okay']]
    emit('summary', verdict='PASS_SCOPED_SUCCESSOR_VALIDATOR' if not failures else 'REFUSE',
         tests=len(results), positive=sum(t['expected_accept'] for t in results),
         negative=sum(not t['expected_accept'] for t in results), failures=failures,
         original_verdict='REFUSE_PRESERVED', native_executed=False, full_run_authorized=False)
    require(not failures, 'unexpected test outcomes')


if __name__ == '__main__':
    main()
