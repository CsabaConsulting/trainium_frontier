#!/usr/bin/env python3
"""Independent exact cold-five binding/refusal checks; stdlib only, no hosts."""
import ast
import hashlib
import io
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile
import types

REPO = pathlib.Path('/home/poojith-devan/Desktop/Trainium')
PACKET = pathlib.Path(sys.argv[1])
PACKET_COMMIT = '9881a3c8975b2504673a24caf54249b9245cae31'
PACKET_PATH = 'research/v2/proposals/20260926-base-cache-cold-five'
rows = []


def check(label, condition):
    if not condition:
        raise RuntimeError(label)
    rows.append(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=REPO)


binding_bytes = (PACKET / 'CT_BINDING.disabled.json').read_bytes()
names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PACKET_COMMIT, '--', PACKET_PATH], cwd=REPO).decode().splitlines()
check('frozen packet exists', bool(names))
for name in names:
    check('frozen bytes ' + pathlib.Path(name).name, (PACKET / pathlib.Path(name).relative_to(PACKET_PATH)).read_bytes() == git(PACKET_COMMIT, name))
b = json.loads(binding_bytes)
p = json.loads((PACKET / 'EXPERIMENT_PROPOSAL.json').read_text())
provenance = json.loads((PACKET / 'package/provenance.json').read_text())
driver = (PACKET / 'host_driver.py').read_bytes()
tree = ast.parse(driver)
guard_dir = REPO / 'research/v2/operator/20260926-common-host-idle-guard'
for field, data in (
    ('source_sha256', git(b['source_commit'], b['source_git_path'])),
    ('driver_sha256', driver),
    ('prepare_sha256', (REPO / 'prepare.py').read_bytes()),
    ('harness_sha256', git(b['harness_source_commit'], b['harness_source_git_path'])),
    ('idle_guard_sha256', (guard_dir / 'idle_guard.py').read_bytes()),
    ('dispatch_driver_sha256', (guard_dir / 'dispatch_driver.py').read_bytes()),
):
    check('hash ' + field, sha(data) == b[field])
for name, digest in b['package_files'].items():
    check('package hash ' + name, sha((PACKET / name).read_bytes()) == digest)
for name in ('command', 'training_command', 'outer_command', 'detached_command'):
    check('argv hash ' + name, sha(json.dumps(b[name], separators=(',', ':'), ensure_ascii=True).encode()) == b[name + '_sha256'])
check('scope stays disabled', b['stage_authorized'] is False and b['launch_authorized'] is False
      and p['native_launch_authorized'] is False and not b['source_adoption'])
check('no eval/full', b['no_eval'] and b['no_automatic_full'] and not p['full_run_automatic'])
check('pending exact fullsize/source gates', 'full117M' in ' '.join(b['pending_gates']) and 'fresh independent source' in ' '.join(b['pending_gates']))
check('trn22 binding only', b['host'] == 'trn22' and b['expected_hostname'] == 'ip-172-31-32-253')
check('proposal exact source', p['source_sha256'] == b['source_sha256'] == provenance['train_sha256'])
check('argv equality', b['training_command'] == p['argv'] == shlex.split(provenance['launch_command'])
      == shlex.split((PACKET / 'package/launch-command.txt').read_text()))
base = json.loads((REPO / 'research/v2/operator/20260926-ct-base-startup-only/CT_BINDING.disabled.json').read_text())
check('only causal argv addition', b['training_command'] == base['training_command'] + ['--initial-cache-refresh'])
argv = b['training_command']
for option, value in (('--n-embd', '1024'), ('--seq-len', '1024'), ('--pack-factor', '32'),
                      ('--matrix-lr', '0.015'), ('--seed', '42'), ('--max-train-seconds', '1800'), ('--num-steps', '5')):
    check('fixed ' + option, argv.count(option) == 1 and argv[argv.index(option) + 1] == value)
check('fullchip4 and noeval', '--nproc_per_node=4' in argv and '--no-eval-public' in argv)
check('target600 not budgetchange', b['target_startup_seconds'] == 600 and b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800')
check('timeout is safety bound', b['command'][:5] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '1800s'])
check('cold train only', b['command'][-3:] == ['--mode', 'train', '--fresh-caches'])
check('persistent outer flock', b['outer_command'] == ['/usr/bin/flock', '-n', '/home/ubuntu/.cache/trainium_freshstart/device0.lock', '/usr/bin/python3', '-I', '-B', b['driver_path']])
check('normal asserts active', '-O' not in b['outer_command'] and '-O' not in b['command'])
cache = b['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR']
check('URL equals new compiler DIR', b['training_environment']['NEURON_COMPILE_CACHE_URL'] == cache == b['sandbox'] + '/caches/compiler')
check('six private SDK caches', len([v for k, v in b['harness_injected_environment'].items() if k.endswith('CACHE_DIR') or k.endswith('DEBUG_DIR')]) == 6)
check('control raw fifth', json.loads((REPO / p['control']['receipt']).read_text())['observations'][4]['observed_seconds'] == p['control']['uncapped_fifth_observed_seconds'])
derivation = b['driver_derivation']
old = git(derivation['commit'], derivation['path'])
check('derivation pin', sha(old) == derivation['sha256'])
old_tree = ast.parse(old)
select = lambda t: ast.dump(ast.Module(body=[n for n in t.body if isinstance(n, (ast.FunctionDef, ast.Try))], type_ignores=[]))
check('entire executable control flow identical to guarded architecture driver', select(tree) == select(old_tree))
ns = {}
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)], type_ignores=[]), '<driver-definitions>', 'exec', optimize=0), ns)
for field, name in (('command', 'COMMAND'), ('training_environment', 'ENV'), ('source_sha256', 'SOURCE_SHA'),
                    ('prepare_sha256', 'PREP_SHA'), ('harness_sha256', 'HARNESS_SHA')):
    check('literal ' + field, ns[name] == b[field])
check('root and label literals', str(ns['ROOT']) == b['root'] and ns['LABEL'] == b['label'])

# Run exact control flow with fake files/processes. Assertions intentionally
# compiled at optimize=0 because the bound host argv is -I -B, never -O.
body = compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.Try)], type_ignores=[]), '<driver-control>', 'exec', optimize=0)


class FakePath:
    def __init__(self, value): self.value = str(value)
    def __str__(self): return self.value
    def __truediv__(self, value): return FakePath(self.value + '/' + value)
    @property
    def parent(self): return FakePath(self.value.rsplit('/', 1)[0])
    def glob(self, pattern): return []
    def is_dir(self): return True
    def open(self, mode): return io.StringIO()


for case, reason in (
    ('success', None), ('source', 'SOURCE_SHA'), ('prepare', 'PREPARE_SHA'), ('harness', 'HARNESS_SHA'),
    ('guard_hash', 'IDLE_GUARD_SHA'), ('stop', 'STOP:'), ('disk', 'DISK_BELOW_6_5_GIB'),
    ('existing_sandbox', 'SANDBOX_ALREADY_EXISTS'), ('busy_guard', 'FIXTURE_BUSY'), ('child124', None),
):
    events, records = [], []
    mapping = {b['root'] + '/package/train.py': ('source', b['source_sha256']), str(ns['PREP']): ('prepare', b['prepare_sha256']),
               str(ns['HARNESS']): ('harness', b['harness_sha256']), b['root'] + '/idle_guard.py': ('guard_hash', b['idle_guard_sha256'])}
    def fake_sha(path):
        key, digest = mapping[str(path)]
        return 'wrong' if case == key else digest
    def fake_guard(lock, path, record):
        events.append('guard')
        if case == 'busy_guard': raise RuntimeError('FIXTURE_BUSY')
        check('guard exact lock ' + case, lock == b['device_lock'] and str(path) == b['driver_path'])
        return 73
    def fake_call(command, **kwargs):
        events.append('child')
        check('child binding and inherited fd ' + case, command == b['command'] and kwargs['env'] == b['training_environment'] and kwargs['pass_fds'] == (73,))
        return 124 if case == 'child124' else 0
    module = types.SimpleNamespace(require_idle_under_inherited_lock=fake_guard)
    spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
    ctx = dict(ns, Path=FakePath, ROOT=FakePath(b['root']), PREP=FakePath(str(ns['PREP'])), HARNESS=FakePath(str(ns['HARNESS'])), sha=fake_sha,
               record=lambda name, value: records.append((name, value)),
               os=types.SimpleNamespace(path=types.SimpleNamespace(lexists=lambda path: (case == 'stop' and str(path).endswith('/STOP')) or (case == 'existing_sandbox' and str(path) == b['sandbox'])), statvfs=lambda _: types.SimpleNamespace(f_bavail=0 if case == 'disk' else 20 * 1024**3, f_frsize=1)),
               importlib=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=lambda *_: spec, module_from_spec=lambda _: module)),
               subprocess=types.SimpleNamespace(call=fake_call, DEVNULL=-3, STDOUT=-2))
    caught = None
    try: exec(body, ctx)
    except (Exception, SystemExit) as exc: caught = str(exc)
    if reason:
        check('refusal ' + case, reason in caught and 'child' not in events and records[-1][1]['status'] == 'CT_DRIVER_FAILED')
    else:
        check('lifetime ordering ' + case, events == ['guard', 'child'])
        check('terminal classification ' + case, records[-1][1]['status'] == ('CT_HARNESS_FAILED' if case == 'child124' else 'CT_HARNESS_EXITED'))
        check('terminal no auto followup ' + case, records[-1][1]['no_eval'] and records[-1][1]['no_automatic_full'])

guard = {'__name__': 'independent_guard'}
exec(compile((guard_dir / 'idle_guard.py').read_bytes(), '<guard>', 'exec', optimize=0), guard)
ps = '100 99 python3 /packet/host_driver.py\n99 1 flock /usr/bin/flock -n /lock /usr/bin/python3 -I -B /packet/host_driver.py\n1 0 init /sbin/init\n'
device = [{'neuron_device': 0, 'instance_type': 'trn2.3xlarge', 'nc_count': 4, 'logical_neuroncore_config': 2, 'neuroncore_ids': [0, 1, 2, 3], 'neuron_processes': []}]
for name, process_text, device_text, expected in (
    ('idle', ps, json.dumps(device), None),
    ('competing worker', ps + '200 1 python /venv/bin/torchrun train.py\n', json.dumps(device), 'PS_NATIVE_JOB_BUSY'),
    ('malformed neuron', ps, '{}', 'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'),
):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, process_text if command == guard['PS'] else device_text, '')
    error = None
    try: guard['check_idle'](run, 100, 99, {})
    except RuntimeError as exc: error = str(exc)
    check('guard parser ' + name, error == expected and len(calls) == 2)
with tempfile.TemporaryDirectory(prefix='base-cold-binding-refusal-') as tmp:
    lock = pathlib.Path(tmp) / 'device0.lock'
    lock.touch()
    error = None
    try: guard['inherited_lock'](lock, pathlib.Path(tmp) / 'host_driver.py')
    except RuntimeError as exc: error = str(exc)
    check('direct invocation refused', error == 'OUTER_FLOCK_COMMAND_MISMATCH')

harness = ast.parse(git(b['harness_source_commit'], b['harness_source_git_path']))
main = next(n for n in harness.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
node = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'train' and process.returncode == 0")
accounting = compile(ast.Module(body=node.body[3:], type_ignores=[]), '<exact-accounting>', 'exec', optimize=0)
for wall, fifth, expected in ((2699.9, 1000, True), (2700.1, 1000, False), (2400, 600, True), (2400.1, 600, False)):
    ctx = {'observed': [{'observed_seconds': fifth}] * 5, 'result': {'wall_seconds': wall},
           'checkpoint': types.SimpleNamespace(stat=lambda: types.SimpleNamespace(st_size=1)), 'sha': lambda _: 'fixture'}
    exec(accounting, ctx)
    check('strict through-exit charge ' + str(wall), ctx['result']['charged_through_process_exit'] == wall - min(fifth, 900)
          and ctx['result']['charged_through_process_exit_within_budget'] is expected)
print(json.dumps({'status': 'PASS', 'checks': len(rows), 'tests': rows,
                  'packet_commit': PACKET_COMMIT,
                  'binding_sha256': sha(binding_bytes), 'driver_sha256': sha(driver),
                  'source_sha256': b['source_sha256'], 'scope': 'Mechanical disabled cold-five binding only; no source/full117M/native acceptance or launch authority'}, indent=2))
