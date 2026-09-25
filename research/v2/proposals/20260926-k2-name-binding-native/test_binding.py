#!/usr/bin/env python3
"""Stdlib-only static binding and synthetic receipt-refusal tests; no host calls."""
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
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CHECKS = []


def check(value, name):
    if not value:
        raise RuntimeError(name)
    CHECKS.append(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_validator():
    spec = importlib.util.spec_from_file_location('tested_validator', HERE / 'validate_receipts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(binding, validator):
    files = {}
    for rank in range(4):
        cases = []
        for name in validator.CASE_NAMES:
            for dtype in validator.DTYPES:
                names = validator.COMMON + (validator.AUTOGRAD if name != 'empty' else ())
                checks = []
                for test in names:
                    target_dtype = dtype if test in ('upstream_roundtrip', 'eager_autograd_forward', 'compiled_enclosing_autograd_forward') else 'torch.float32'
                    checks.append(dict(name=test, exact=True, finite=True, shape=[256, 128],
                                       actual_dtype=target_dtype, actual_sha256='0' * 64,
                                       expected_sha256='0' * 64, max_abs=0.0))
                cases.append(dict(name=name, dtype=dtype, status='EXACT', direct_first_call_seconds=0.1, checks=checks))
        files[f'/fixture/receipts/rank{rank}.json'] = dict(
            scope='TINY_CONSUMER_FALSIFIER_ONLY', status='TINY_CONSUMER_EXACT_ONLY',
            rank=rank, world_size=4, label=binding['label'], seed=42,
            shape={'V': 256, 'C': 128, 'T': 256}, argv=binding['checker_args'],
            kernel_sha256=binding['support_sha256']['consumer_nki.py'],
            checker_sha256=binding['support_sha256']['check_native.py'],
            reference_sha256=binding['support_sha256']['reference.py'],
            tolerance={'atol': 0, 'rtol': 0, 'criterion': 'identical bytes'},
            accepted_speedup=False, whole_step_speedup='UNMEASURED',
            diagnostic_elapsed_seconds=2.0, cases=cases,
            timings=[dict(case=name, dtype=dtype, seconds=[0.001] * 3,
                          scope='consumer_only_excludes_routing_and_transfer')
                     for name in ('unique', 'all_same') for dtype in validator.DTYPES])
        files[f'/fixture/rank-environment-r{rank}.json'] = dict(
            rank=rank, local_rank=rank, world_size=4, environment=binding['environment'],
            checker_sha256=binding['support_sha256']['check_native.py'],
            checker_argv=binding['checker_args'])
        files[f'/fixture/rank-wrapper-exit-r{rank}.json'] = dict(rank=rank, returncode=0, error=None)
    return copy.deepcopy(files)


def virtual_validate(validator, files, binding):
    class VirtualPath:
        def __init__(self, path):
            self.path = str(path)
        def __truediv__(self, other):
            return VirtualPath(self.path + '/' + str(other))
        def read_text(self):
            if self.path not in files:
                raise FileNotFoundError(self.path)
            return json.dumps(files[self.path])
    with patch.object(validator, 'Path', VirtualPath):
        return validator.validate('/fixture', binding['label'], binding['environment'],
                                  binding['support_sha256'], binding['checker_args'])


def fake_driver_case(binding, scenario):
    """Exercise driver control flow with all filesystem/process/guard calls fake."""
    spec = importlib.util.spec_from_file_location('tested_driver', HERE / 'host_driver.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)  # __main__ is not invoked during import.
    created, records, events = set(), {}, []
    if scenario == 'existing_cache':
        created.add(binding['root'] + '/caches')

    class FakePath:
        def __init__(self, value):
            self.value = str(value)
        def __str__(self):
            return self.value
        def __eq__(self, other):
            return str(self) == str(other)
        def __truediv__(self, other):
            return FakePath(self.value + '/' + str(other))
        def resolve(self):
            return self
        @property
        def parent(self):
            return FakePath(posixpath.dirname(self.value))
        @property
        def name(self):
            return posixpath.basename(self.value)
        def glob(self, pattern):
            return []
        def mkdir(self, exist_ok=False):
            if self.value in created:
                raise FileExistsError(self.value)
            created.add(self.value)
        def iterdir(self):
            return []
        def stat(self):
            return SimpleNamespace(st_dev=1, st_ino=len(created))
        def open(self, mode):
            return io.StringIO()

    driver.Path = FakePath
    driver.__file__ = binding['root'] + '/host_driver.py'
    driver.ROOT = FakePath(binding['root'])
    driver.PROJECT = FakePath('/home/ubuntu/project')
    driver.socket = SimpleNamespace(gethostname=lambda: 'wrong' if scenario == 'wrong_host' else binding['expected_hostname'])
    driver.os = SimpleNamespace(
        path=SimpleNamespace(lexists=lambda p: str(p) in created or
                             (scenario == 'stop' and str(p) == binding['root'] + '/STOP')),
        statvfs=lambda p: SimpleNamespace(f_bavail=0 if scenario == 'disk' else 20 * 1024**3, f_frsize=1))
    driver.sha = lambda p: 'wrong' if scenario == 'hash' else binding['support_sha256'][p.name]
    driver.record = lambda name, value: records.__setitem__(name, value)

    def guarded(*args):
        events.append('guard')
        if scenario == 'guard_refusal':
            raise RuntimeError('SYNTHETIC_BUSY_OR_UNKNOWN')
        return 9

    def validated(*args):
        events.append('validate')
        if scenario == 'incomplete':
            raise RuntimeError('SYNTHETIC_MISSING_RANK')
        return {'status': 'TINY_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ARTIFACT_REVIEW'}

    def spawned(*args, **kwargs):
        events.append('child')
        check(kwargs['pass_fds'] == (9,) and kwargs['env'] == binding['environment'], 'fake_child_fd_env_' + scenario)
        return 1 if scenario == 'child_nonzero' else 0

    driver.load_module = lambda name, path: SimpleNamespace(require_idle_under_inherited_lock=guarded) if name == 'bound_idle_guard' else SimpleNamespace(validate=validated)
    driver.subprocess = SimpleNamespace(call=spawned, DEVNULL=-3, STDOUT=-2)
    failed = False
    try:
        rc = driver.main()
    except RuntimeError:
        failed, rc = True, None
    if scenario == 'success':
        check(not failed and rc == 0 and records['terminal.json']['complete'] is True, 'fake_driver_success_requires_validation')
        check(events == ['guard', 'child', 'validate'] and len(records['fresh-caches.json']['directories']) == 6, 'fake_guard_child_validation_order')
    else:
        check(records['terminal.json']['complete'] is False, 'fake_driver_refuses_' + scenario)
        if scenario in ('child_nonzero', 'incomplete'):
            check(events.count('child') == 1, 'fake_only_one_child_' + scenario)
        else:
            check('child' not in events, 'fake_no_child_on_' + scenario)


def main():
    b = json.loads((HERE / 'NATIVE_BINDING.disabled.json').read_text())
    check(all(b[k] is False for k in ('stage_authorized', 'launch_authorized', 'source_adoption', 'operator_assigned', 'host_reserved')), 'disabled_not_assigned')
    check(len(b['pending_gates']) == 5 and b['timeout_seconds'] == 600 and b['term_grace_seconds'] == 60, 'pending_gates_and_timeout')
    check(sha((HERE / 'host_driver.py').read_bytes()) == b['driver_sha256'], 'driver_pin')
    for file, expected in b['support_sha256'].items():
        if file in ('rank_entry.py', 'validate_receipts.py'):
            actual = HERE / file
        elif file in ('idle_guard.py', 'dispatch_driver.py'):
            actual = REPO / 'research/v2/operator/20260926-common-host-idle-guard' / file
        else:
            frozen = subprocess.check_output(['git', 'show', b['source_module_commit'] + ':' + b['source_module_path'] + '/' + file], cwd=REPO)
            check(sha(frozen) == expected, 'frozen_module_' + file)
            CHECKS.append('support_pin_' + file)
            continue
        check(sha(actual.read_bytes()) == expected, 'support_pin_' + file)
    check(shlex.split((HERE / 'bound-command.txt').read_text()) == b['command'], 'bound_command_text')
    check(b['command'][:8] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '600s',
                               '/home/ubuntu/workspace/native_venv/bin/torchrun', '--standalone', '--nproc_per_node=4'], 'one_bounded_four_rank_child')
    check(b['command'][8] == b['root'] + '/rank_entry.py' and b['command'][9:] == b['checker_args'], 'wrapper_preserves_checker_arguments')
    original = json.loads(subprocess.check_output(['git', 'show', b['rebind']['parent_commit'] + ':' +
                          b['rebind']['parent_path'] + '/NATIVE_BINDING.disabled.json'], cwd=REPO))
    original_args = [arg.replace(original['label'], b['label']).replace(
        original['support_sha256']['consumer_nki.py'], b['support_sha256']['consumer_nki.py'])
        for arg in original['checker_args']]
    check(b['checker_args'] == original_args, 'frozen_checker_args_only_label_root_kernel_pin_rebound')
    tree = ast.parse((HERE / 'host_driver.py').read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path':
                value = value.args[0]
            constants[node.targets[0].id] = ast.literal_eval(value)
    for var, key in (('ROOT', 'root'), ('LABEL', 'label'), ('COMMAND', 'command'), ('CHECKER_ARGS', 'checker_args'),
                     ('ENV', 'environment'), ('PINS', 'support_sha256'), ('CACHE_KEYS', 'cache_keys')):
        check(constants[var] == b[key], 'driver_constant_' + var)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    guard = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'require_idle_under_inherited_lock']
    jobs = [n for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.value.id == 'subprocess' and n.func.attr == 'call']
    check(len(guard) == len(jobs) == 1 and guard[0].lineno < jobs[0].lineno, 'guard_before_only_child')
    check(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in jobs[0].keywords), 'same_inherited_fd')
    validation = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'validate']
    check(len(validation) == 1 and jobs[0].lineno < validation[0].lineno, 'validate_after_child')
    check(not any(isinstance(n, ast.Assert) for n in ast.walk(tree)), 'driver_checks_survive_optimized_python')
    env = b['environment']
    check(len(set(env[k] for k in b['cache_keys'])) == 6 and all(env[k].startswith(b['root'] + '/caches/') for k in b['cache_keys']), 'fresh_private_cache_paths')
    check(env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'], 'compiler_url_equals_dir')
    entry = ast.parse((HERE / 'rank_entry.py').read_text())
    keys_node = next(n for n in entry.body if isinstance(n, ast.Assign) and n.targets[0].id == 'KEYS')
    check(set(ast.literal_eval(keys_node.value)) == set(env), 'all_declared_environment_keys_captured')
    entry_calls = [n for n in ast.walk(entry) if isinstance(n, ast.Call)]
    runpy_calls = [n for n in entry_calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'run_path']
    check(len(runpy_calls) == 1, 'single_unchanged_checker_runpy')
    imports = [n for n in ast.walk(entry) if isinstance(n, (ast.Import, ast.ImportFrom))]
    check(not any('torch' in ast.unparse(n) or 'nki' in ast.unparse(n) for n in imports), 'wrapper_no_tensor_import')
    v = load_validator()
    files = fixture(b, v)
    complete = virtual_validate(v, files, b)
    check(len(complete['ranks']) == 4 and all(x['complete_cases'] == 20 and x['exact_comparisons'] == 150 and x['timing_samples'] == 12 for x in complete['ranks']), 'synthetic_complete_coverage')
    key = '/fixture/receipts/rank2.json'
    mutations = [
        ('missing_rank', lambda f: f.pop('/fixture/receipts/rank3.json')),
        ('refused_status', lambda f: f[key].update(status='REFUSE_OR_UNSUPPORTED')),
        ('wrong_scope', lambda f: f[key].update(scope='MODEL_PASS')),
        ('wrong_rank', lambda f: f[key].update(rank=0)),
        ('changed_label', lambda f: f[key].update(label='different')),
        ('changed_argv', lambda f: f[key].update(argv=[])),
        ('changed_hash', lambda f: f[key].update(kernel_sha256='1' * 64)),
        ('relaxed_tolerance', lambda f: f[key]['tolerance'].update(atol=0.01)),
        ('speed_claim', lambda f: f[key].update(accepted_speedup=True)),
        ('missing_case', lambda f: f[key]['cases'].pop()),
        ('duplicate_case', lambda f: f[key]['cases'].__setitem__(1, copy.deepcopy(f[key]['cases'][0]))),
        ('started_case', lambda f: f[key]['cases'][0].update(status='STARTED')),
        ('missing_gradient_check', lambda f: f[key]['cases'][0]['checks'].pop()),
        ('nonexact', lambda f: f[key]['cases'][0]['checks'][0].update(exact=False)),
        ('nonfinite', lambda f: f[key]['cases'][0]['checks'][0].update(finite=False)),
        ('sha_contradiction', lambda f: f[key]['cases'][0]['checks'][0].update(expected_sha256='f' * 64)),
        ('error_contradiction', lambda f: f[key]['cases'][0]['checks'][0].update(max_abs=1.0)),
        ('null_nonfinite_error', lambda f: f[key]['cases'][0]['checks'][0].update(max_abs=None, max_abs_nonfinite='nan')),
        ('refused_report_successor', lambda f: f[key].update(status='REFUSE_OR_UNSUPPORTED', error='EXACTNESS_FALSIFIED')),
        ('dtype_changed', lambda f: f[key]['cases'][0]['checks'][0].update(actual_dtype='torch.float64')),
        ('nan_elapsed', lambda f: f[key].update(diagnostic_elapsed_seconds=float('nan'))),
        ('negative_elapsed', lambda f: f[key].update(diagnostic_elapsed_seconds=-1)),
        ('bool_elapsed', lambda f: f[key].update(diagnostic_elapsed_seconds=False)),
        ('missing_timing', lambda f: f[key]['timings'].pop()),
        ('bad_timing_scope', lambda f: f[key]['timings'][0].update(scope='whole_step')),
        ('nan_timing', lambda f: f[key]['timings'][0].update(seconds=[0.1, float('nan'), 0.1])),
        ('short_timing', lambda f: f[key]['timings'][0].update(seconds=[0.1])),
        ('wrong_environment', lambda f: f['/fixture/rank-environment-r2.json']['environment'].update(NEURON_COMPILE_CACHE_URL='/shared')),
        ('bad_worker_exit', lambda f: f['/fixture/rank-wrapper-exit-r2.json'].update(returncode=1)),
        ('boolean_worker_exit', lambda f: f['/fixture/rank-wrapper-exit-r2.json'].update(returncode=False)),
    ]
    for name, mutate in mutations:
        changed = copy.deepcopy(files)
        mutate(changed)
        try:
            virtual_validate(v, changed, b)
        except (RuntimeError, FileNotFoundError, KeyError, TypeError, ValueError):
            CHECKS.append('refuse_' + name)
        else:
            raise RuntimeError('false completion: ' + name)
    for scenario in ('success', 'wrong_host', 'stop', 'disk', 'hash', 'existing_cache',
                     'guard_refusal', 'child_nonzero', 'incomplete'):
        fake_driver_case(b, scenario)
    print(json.dumps(dict(status='PASS_STDLIB_BINDING_ONLY', count=len(CHECKS), checks=CHECKS,
                          fixtures='synthetic dictionaries in memory, not run evidence',
                          actual_driver_process_executed=False, driver_control_flow_with_all_effects_faked=True,
                          native_or_tensor_imported=False,
                          host_actions=0, launch_authorized=False), indent=2))


if __name__ == '__main__':
    main()
