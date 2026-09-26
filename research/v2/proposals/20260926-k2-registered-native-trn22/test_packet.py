#!/usr/bin/env python3
"""Own stdlib/synthetic tests only; no host, tensors or claimed native evidence."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import posixpath
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CHECKS = []


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)
    CHECKS.append(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(name):
    return subprocess.check_output(['git', 'show', 'fa1a3652:research/v2/proposals/20260926-k2-fp32-first-discriminator/' + name], cwd=REPO).decode()


def load(name):
    spec = importlib.util.spec_from_file_location('test_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = json.loads((HERE / 'NATIVE_BINDING.disabled.json').read_text())
check(all(b[k] is False for k in ('stage_authorized', 'launch_authorized', 'source_adoption', 'operator_assigned', 'host_reserved')), 'disabled')
check(b['host'] == 'trn22' and b['expected_hostname'] == 'ip-172-31-32-253', 'host')
check(b['label'] == 'fs-20260926-k2-registered-forward-trn22-r1', 'unique_new_label')
check(b['timeout_seconds'] == 600 and b['term_grace_seconds'] == 60, 'watchdog')
check(b['original_lock']['device'] == 66305 and b['original_lock']['inode'] == 832589, 'original_lock')
check(b['postflight']['required'] is True and 'after any exit' in b['postflight']['owner'], 'all_outcome_postflight')
check(b['retries'] == 0 and not b['automatic_next_job'] and not b['automatic_full'] and not b['automatic_eval'], 'no_automatic_next')
for name, pin in b['support_sha256'].items():
    check(sha((HERE / name).read_bytes()) == pin, 'pin_' + name)
check(sha((HERE / 'host_driver.py').read_bytes()) == b['driver_sha256'], 'driver_pin')
check(shlex.split((HERE / 'bound-command.txt').read_text()) == b['command'], 'exact_command')
check(b['command'][:8] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '600s',
      '/home/ubuntu/workspace/native_venv/bin/torchrun', '--standalone', '--nproc_per_node=4'], 'bounded_world4')
check(b['command'][9:] == b['checker_args'] and '--timing-repeats' not in b['command'], 'no_component_timing')
for name in ('host_driver.py', 'rank_entry.py'):
    old, new = ast.parse(original(name)), ast.parse((HERE / name).read_text())
    old.body = [n for n in old.body if not isinstance(n, ast.Assign)]
    new.body = [n for n in new.body if not isinstance(n, ast.Assign)]
    check(ast.dump(old) == ast.dump(new), 'unchanged_body_' + name)
for name in ('idle_guard.py', 'dispatch_driver.py'):
    check((HERE / name).read_text() == original(name), 'unchanged_shared_' + name)
tree = ast.parse((HERE / 'host_driver.py').read_text())
constants = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
        constants[node.targets[0].id] = ast.literal_eval(value)
for var, key in [('ROOT', 'root'), ('LABEL', 'label'), ('PINS', 'support_sha256'), ('COMMAND', 'command'),
                 ('CHECKER_ARGS', 'checker_args'), ('ENV', 'environment'), ('CACHE_KEYS', 'cache_keys')]:
    check(constants[var] == b[key], 'constant_' + var)
check(len(set(b['environment'][k] for k in b['cache_keys'])) == 6, 'six_unique_caches')
check(b['environment']['NEURON_COMPILE_CACHE_URL'] == b['environment']['NEURON_COMPILE_CACHE_DIR'], 'url_dir')
check('10.5 * 1024**3' in (HERE / 'host_driver.py').read_text(), 'disk_floor')
check(not any(isinstance(n, ast.Assert) for n in ast.walk(tree)), 'driver_not_assert_based')
checker = ast.parse((HERE / 'check_native.py').read_text())
old_checker = ast.parse(original('check_native.py'))
for name in ('make_wrapper', 'tensor_record', 'compare', 'observe_route'):
    old = next(n for n in old_checker.body if isinstance(n, ast.FunctionDef) and n.name == name)
    new = next(n for n in checker.body if isinstance(n, ast.FunctionDef) and n.name == name)
    check(ast.dump(old) == ast.dump(new), 'unchanged_helper_' + name)
check(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_registered_forward'
          for n in ast.walk(checker)) == 1, 'one_registration_site')
check('if dtype_name not in cache:' in (HERE / 'check_native.py').read_text(), 'registration_cached_per_dtype')
disabled = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'check_native.py')],
                          capture_output=True, text=True, timeout=10, check=True)
check(json.loads(disabled.stdout) == dict(status='DISABLED', native_imported=False, native_executed=False), 'checker_default_disabled')
v = load('validate_receipts')


def comparison(value):
    return dict(exact=True, max_abs=0.0, actual=copy.deepcopy(value), expected=copy.deepcopy(value))


def case(name, dtype, changed=False):
    contract = v.fixture_contract(name, dtype, changed)
    kind = 'torch.bfloat16' if dtype == 'bf16' else 'torch.float32'
    input_records = {}
    for key, hashed in {**contract['inputs'], 'weight': '1' * 64}.items():
        shape = [256, 128] if key in ('weight', 'upstream') else [256] if key == 'ids' else [2, 1] if key == 'maximums' else [256, 1]
        input_records[key] = dict(shape=shape, dtype='torch.float32' if key == 'weight' else kind if key == 'upstream' else 'torch.int32',
                                  finite=True, nonzero=0, sha256=hashed)
    out = dict(shape=[256, 128], dtype=kind, finite=True, nonzero=0, sha256='2' * 64)
    checks = dict(upstream_roundtrip=comparison(input_records['upstream']))
    if contract['logical_nonempty']:
        checks['dense_CPU_oracle'] = comparison(contract['gradient'])
    result = dict(name=name, dtype=kind, changed_tangent=changed, status='EXACT',
                  metadata_group_bounds=contract['maximums'], checks=checks, routes=[], original_routes=[])
    names = v.ROUTES if contract['logical_nonempty'] else v.ROUTES[:2]
    for route in (*names, *(v.OLD_ROUTES if contract['logical_nonempty'] else ())):
        record = dict(name=route, status='EXACT', inputs_unchanged=True,
                      inputs_before=copy.deepcopy(input_records), inputs_after=copy.deepcopy(input_records),
                      checks=dict(gradient=comparison(contract['gradient'])))
        if route not in v.ROUTES[:2]:
            record['checks']['output'] = comparison(out)
        result['original_routes' if route in v.OLD_ROUTES else 'routes'].append(record)
    return result


sentinel = case('unique', 'fp32')
suite = [case(n, d) for d in ('bf16', 'fp32') for n in v.reference.CASE_NAMES]
replays = [case('unique', d, True) for d in ('bf16', 'fp32')]
files = {}
for rank in range(4):
    files[f'/fixture/receipts/rank{rank}.json'] = dict(
        status='REGISTERED_FULL_CASES_OBSERVED_EXACT', returncode=0,
        scope='REGISTERED_K2_COMPONENT_DIAGNOSTIC_ONLY', rank=rank, world_size=4,
        label=b['label'], shape=dict(V=256, T=256, C=128), seed=42, argv=b['checker_args'],
        checker_sha256=b['support_sha256']['check_native.py'], kernel_sha256=b['support_sha256']['consumer_nki.py'],
        reference_sha256=b['support_sha256']['reference.py'], adapter_sha256=b['support_sha256']['adapter.py'],
        tolerance=dict(atol=0, rtol=0, criterion='identical bytes'), native_qualification=False,
        source_adoption=False, accepted_speedup=False, public_20m=None, whole_step_speedup='UNMEASURED', timings=[],
        no_training=True, no_eval=True, original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',
        original_fp32_first_gate='FAILED_EXACTNESS_PRESERVED', first_registered_compiled_fixture='unique/fp32/nonzero',
        diagnostic_elapsed_seconds=1.0, sentinel=copy.deepcopy(sentinel), cases=copy.deepcopy(suite), tangent_replays=copy.deepcopy(replays))
    files[f'/fixture/rank-environment-r{rank}.json'] = dict(rank=rank, local_rank=rank, world_size=4,
        environment=b['environment'], checker_sha256=b['support_sha256']['check_native.py'], checker_argv=b['checker_args'])
    files[f'/fixture/rank-wrapper-exit-r{rank}.json'] = dict(rank=rank, returncode=0, error=None)


def validate(fixture):
    class VirtualPath:
        def __init__(self, path): self.path = str(path)
        def __truediv__(self, other): return VirtualPath(self.path + '/' + str(other))
        def read_text(self): return json.dumps(fixture[self.path])
    with patch.object(v, 'Path', VirtualPath):
        return v.validate('/fixture', b['label'], b['environment'], b['support_sha256'], b['checker_args'])


check(validate(files)['cases_per_rank'] == 20, 'synthetic_full_case_completion')
key = '/fixture/receipts/rank0.json'
mutations = [
    ('missing_rank', lambda f: f.pop('/fixture/receipts/rank3.json')),
    ('missing_case', lambda f: f[key]['cases'].pop()),
    ('missing_replay', lambda f: f[key]['tangent_replays'].pop()),
    ('sentinel_only', lambda f: f[key].update(cases=[])),
    ('missing_sentinel', lambda f: f[key].pop('sentinel')),
    ('reordered_case', lambda f: f[key]['cases'].reverse()),
    ('altered_hash', lambda f: f[key].update(adapter_sha256='0' * 64)),
    ('altered_argv', lambda f: f[key]['argv'].append('--fake')),
    ('altered_label', lambda f: f[key].update(label='old-label')),
    ('nonzero_exit', lambda f: f['/fixture/rank-wrapper-exit-r0.json'].update(returncode=1)),
    ('bool_exit', lambda f: f[key].update(returncode=False)),
    ('nonfinite', lambda f: f[key]['sentinel']['routes'][3]['checks']['gradient']['actual'].update(finite=False)),
    ('zero_gradient', lambda f: f[key]['sentinel']['routes'][3]['checks']['gradient']['actual'].update(nonzero=0, sha256='0' * 64)),
    ('forged_zero_both', lambda f: [f[key]['sentinel']['routes'][3]['checks']['gradient'][part].update(nonzero=0, sha256='0' * 64) for part in ('actual', 'expected')]),
    ('changed_metadata', lambda f: f[key]['sentinel']['routes'][3]['inputs_after']['counts'].update(sha256='0' * 64)),
    ('changed_fixture_hash_both', lambda f: [f[key]['sentinel']['routes'][3][part]['counts'].update(sha256='0' * 64) for part in ('inputs_before', 'inputs_after')]),
    ('missing_original', lambda f: f[key]['sentinel']['original_routes'].pop()),
    ('hidden_old_failure', lambda f: f[key].update(original_twenty_case_gate='PASS')),
    ('contradictory_old_status', lambda f: f[key]['sentinel']['original_routes'][0].update(status='MISMATCH')),
    ('missing_candidate_gradient', lambda f: f[key]['sentinel']['routes'][3]['checks'].pop('gradient')),
    ('false_exact', lambda f: f[key]['sentinel']['routes'][3]['checks']['gradient'].update(exact=False)),
    ('bool_max_abs', lambda f: f[key]['sentinel']['routes'][3]['checks']['gradient'].update(max_abs=False)),
    ('nan_elapsed', lambda f: f[key].update(diagnostic_elapsed_seconds=float('nan'))),
    ('native_adoption', lambda f: f[key].update(native_qualification=True)),
    ('wrong_environment', lambda f: f['/fixture/rank-environment-r0.json']['environment'].update(NEURON_LOGICAL_NC_CONFIG='1')),
]
for name, mutation in mutations:
    changed = copy.deepcopy(files)
    mutation(changed)
    try:
        validate(changed)
    except (RuntimeError, KeyError, TypeError, ValueError):
        CHECKS.append('refuse_' + name)
    else:
        raise RuntimeError('False completion: ' + name)
# A separately retained old zero-gradient mismatch must not mask candidate data.
changed = copy.deepcopy(files)
for rank in range(4):
    record = changed[f'/fixture/receipts/rank{rank}.json']['sentinel']['original_routes'][1]
    record['status'] = 'MISMATCH'
    record['checks']['gradient']['exact'] = False
    record['checks']['gradient']['actual'].update(nonzero=0, sha256='0' * 64)
    record['checks']['gradient']['max_abs'] = 8.0
check(validate(changed)['native_qualification'] is False, 'old_failure_separate_not_suppressing_candidate')
# Reuse unchanged mocked driver fixture, with all effects substituted locally.
old_tests = ast.parse(subprocess.check_output(['git', 'show', '710ea677:research/v2/proposals/20260926-k2-symmetric-native-trn21/test_binding.py'], cwd=REPO).decode())
fixture_fn = next(n for n in old_tests.body if isinstance(n, ast.FunctionDef) and n.name == 'fake_driver_case')
exec(compile(ast.Module(body=[fixture_fn], type_ignores=[]), '<pinned-mocked-driver-fixture>', 'exec'))
for scenario in ('success', 'wrong_host', 'stop', 'disk', 'hash', 'existing_cache', 'guard_refusal', 'child_nonzero', 'incomplete'):
    fake_driver_case(b, scenario)
check('torch' not in sys.modules and 'nki' not in sys.modules, 'no_tensor_native_import')
print(json.dumps(dict(status='OWN_STDLIB_CHECKS_COMPLETED_NOT_NATIVE', checks=CHECKS, count=len(CHECKS),
                     fixture_scope='SYNTHETIC_ONLY_NO_NATIVE_RECEIPTS', launch_authorized=False,
                     optimized=not __debug__), indent=2))
