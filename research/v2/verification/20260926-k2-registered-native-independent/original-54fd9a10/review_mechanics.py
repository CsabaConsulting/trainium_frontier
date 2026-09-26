"""Independent static and fully mocked launch mechanics; no host operations."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
PACKET = OUT.parent
spec = importlib.util.spec_from_file_location('own_validator_review', OUT / 'review_validator.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
require, emit, load = common.require, common.emit, common.load
common.authenticate()
binding = json.loads((PACKET / 'NATIVE_BINDING.disabled.json').read_text())
driver = load('frozen_host_driver', PACKET / 'host_driver.py')
entry = load('frozen_rank_entry', PACKET / 'rank_entry.py')
guard = load('frozen_idle_guard', PACKET / 'idle_guard.py')
dispatcher = load('frozen_dispatch_driver', PACKET / 'dispatch_driver.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statements(path, omit):
    tree = ast.parse(path.read_bytes())
    return [ast.dump(node) for node in tree.body if not
            (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in omit for t in node.targets))]


require(statements(PACKET / 'host_driver.py', {'ROOT', 'LABEL', 'PINS', 'COMMAND', 'CHECKER_ARGS', 'ENV'}) ==
        statements(PACKET / 'old-host_driver.py', {'ROOT', 'LABEL', 'PINS', 'COMMAND', 'CHECKER_ARGS', 'ENV'}),
        'driver executable bodies changed')
require(statements(PACKET / 'rank_entry.py', {'ROOT', 'CHECKER_SHA'}) ==
        statements(PACKET / 'old-rank_entry.py', {'ROOT', 'CHECKER_SHA'}), 'entry executable bodies changed')
for filename, expected in binding['support_sha256'].items():
    require(sha(PACKET / filename) == expected == driver.PINS[filename], 'support pin ' + filename)
require(sha(PACKET / 'host_driver.py') == binding['driver_sha256'], 'driver pin')
require(driver.COMMAND == binding['command'] == shlex.split((PACKET / 'bound-command.txt').read_text()), 'commands')
require(driver.CHECKER_ARGS == binding['checker_args'] == driver.COMMAND[9:], 'checker argv')
require(driver.ENV == binding['environment'] and driver.CACHE_KEYS == binding['cache_keys'], 'bound environment')
require(driver.COMMAND[:5] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '600s'], 'watchdog')
require(driver.COMMAND[6:8] == ['--standalone', '--nproc_per_node=4'], 'world4')
require(driver.ENV['NEURON_LOGICAL_NC_CONFIG'] == '2', 'LNC2')
require(set(entry.KEYS) == set(driver.ENV), 'rank captures exact key set')
require(binding['original_lock'] == dict(path=driver.LOCK, device=66305, inode=832589,
        verification='Operator must authenticate original existing inode before dispatch; shared inherited-lock guard unchanged.'), 'original lock binding')
require(len(set(driver.ENV[k] for k in driver.CACHE_KEYS)) == 6 and
        driver.ENV['NEURON_COMPILE_CACHE_URL'] == driver.ENV['NEURON_COMPILE_CACHE_DIR'], 'six private caches')
require(all(binding[key] is False for key in ('stage_authorized', 'launch_authorized', 'source_adoption', 'operator_assigned', 'host_reserved')), 'disabled binding')
require(binding['automatic_next_job'] is False and binding['postflight']['required'] is True, 'postflight/no next')
emit('static_binding', status='PASS', bodies_unchanged=True, pins_match=True, watchdog='600+60',
     original_inode=832589, manual_all_outcome_postflight_required=True, native_executed=False)


def driver_trial(fault=None):
    events, records = [], {}
    created = set()
    class FakeFile(io.StringIO):
        def __exit__(self, *args):
            return False
    class MemoryPath(PurePosixPath):
        def resolve(self):
            return self
        def glob(self, pattern):
            events.append(('glob_mock_only', str(self), pattern))
            return []
        def mkdir(self, **kwargs):
            require(kwargs.get('exist_ok') is False, 'exclusive mkdir')
            require(str(self) not in created, 'duplicate mkdir')
            created.add(str(self))
            events.append(('mkdir', str(self)))
        def iterdir(self):
            return [self / 'unexpected'] if fault == 'dirty_created_cache' else []
        def stat(self):
            return SimpleNamespace(st_dev=66305, st_ino=999001)
        def open(self, mode):
            require(mode == 'x', 'exclusive output')
            events.append(('open', str(self)))
            return FakeFile()
    root = MemoryPath(binding['root'])
    def record(name, value):
        require(name not in records, 'record overwrite')
        records[name] = copy.deepcopy(value)
        events.append(('record', name))
    def lexists(path):
        text = str(path)
        if fault == 'stop' and text.endswith('/STOP'):
            return True
        if fault == 'existing_cache' and text == str(root / 'caches'):
            return True
        return False
    def fake_guard(lock, path, writer):
        events.append(('guard', lock, str(path)))
        require(lock == driver.LOCK and path == root / 'host_driver.py', 'lock/driver binding')
        if fault == 'guard_refusal':
            raise RuntimeError('MOCK_BUSY')
        return 41
    def fake_validate(*args):
        events.append(('validate',))
        require(args == (root, binding['label'], binding['environment'], binding['support_sha256'], binding['checker_args']), 'validation arguments')
        if fault == 'invalid_receipts':
            raise RuntimeError('MOCK_INVALID_RECEIPTS')
        return dict(status='SYNTHETIC_REVIEW_COMPLETION')
    def fake_load(name, path):
        if str(path).endswith('/idle_guard.py'):
            return SimpleNamespace(require_idle_under_inherited_lock=fake_guard)
        if str(path).endswith('/validate_receipts.py'):
            return SimpleNamespace(validate=fake_validate)
        raise RuntimeError('unexpected module load')
    def fake_child(command, **kwargs):
        events.append(('child',))
        require(command == binding['command'], 'exact child command')
        require(kwargs['cwd'] == root and kwargs['env'] == binding['environment'], 'exact child cwd/env')
        require(kwargs['pass_fds'] == (41,), 'inherited lock FD')
        require(kwargs['stdin'] == subprocess.DEVNULL and kwargs['stderr'] == subprocess.STDOUT, 'child I/O')
        if fault == 'child_exception':
            raise OSError('MOCK_EXEC_FAILURE')
        return 124 if fault == 'child_timeout' else 1 if fault == 'child_nonzero' else 0
    real_sha = {name: sha(PACKET / name) for name in driver.PINS}
    replacements = dict(ROOT=root, PROJECT=MemoryPath(driver.PROJECT), Path=MemoryPath,
        __file__=str(root / 'host_driver.py'), record=record, load_module=fake_load,
        sha=lambda p: '0' * 64 if fault == 'hash' else real_sha[p.name],
        socket=SimpleNamespace(gethostname=lambda: 'wrong-host' if fault == 'host' else binding['expected_hostname']),
        os=SimpleNamespace(path=SimpleNamespace(lexists=lexists), statvfs=lambda p:
                           SimpleNamespace(f_bavail=1 if fault == 'disk' else 20 * 1024**3, f_frsize=1)),
        subprocess=SimpleNamespace(call=fake_child, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT))
    error, code = None, None
    with patch.multiple(driver, **replacements):
        try:
            code = driver.main()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    children = sum(e[0] == 'child' for e in events)
    require(children <= 1, 'automatic second child')
    if fault in ('host', 'stop', 'disk', 'hash', 'existing_cache', 'dirty_created_cache', 'guard_refusal'):
        require(children == 0 and error is not None, 'preflight must refuse before child')
    elif fault is None:
        require(code == 0 and records['terminal.json']['complete'] is True, 'mock success')
        require(next(i for i, e in enumerate(events) if e[0] == 'guard') <
                next(i for i, e in enumerate(events) if e[0] == 'child') <
                next(i for i, e in enumerate(events) if e[0] == 'validate'), 'guard child validation order')
    else:
        require(records['terminal.json']['complete'] is False, 'failed outcome cannot be complete')
    require(records['terminal.json']['no_automatic_next_job'] is True, 'terminal no next')
    emit('mock_driver', fault=fault, code=code, error=error, child_count=children,
         terminal=records.get('terminal.json'), events=events)


for fault in (None, 'host', 'stop', 'disk', 'hash', 'existing_cache', 'dirty_created_cache',
              'guard_refusal', 'child_nonzero', 'child_timeout', 'child_exception', 'invalid_receipts'):
    driver_trial(fault)


def idle_trial(fault=None):
    calls, evidence = [], {}
    ps = '100 1 python3 python3 /packet/host_driver.py\n99 1 flock /usr/bin/flock -n /lock python3\n1 0 init /sbin/init\n'
    device = dict(neuron_device=0, instance_type='trn2.3xlarge', nc_count=4,
                  logical_neuroncore_config=2, neuroncore_ids=[0, 1, 2, 3], neuron_processes=[])
    if fault == 'busy_process':
        ps += '200 1 python3 /venv/bin/python train.py\n'
    if fault == 'busy_device':
        device['neuron_processes'] = [dict(pid=222)]
    if fault == 'wrong_lnc':
        device['logical_neuroncore_config'] = 1
    if fault == 'malformed_ps':
        ps = 'unparseable\n'
    if fault == 'missing_own':
        ps = '1 0 init /sbin/init\n'
    def run(command, **kwargs):
        calls.append(command)
        require(kwargs == dict(capture_output=True, text=True, timeout=15, env=guard.CHECK_ENV), 'fresh check arguments')
        if fault == 'ps_timeout' and command == guard.PS:
            raise subprocess.TimeoutExpired(command, 15, stderr='mock timeout')
        return SimpleNamespace(returncode=1 if fault == 'check_exit' else 0,
            stdout=ps if command == guard.PS else json.dumps([device]), stderr='noise' if fault == 'check_stderr' else '')
    error = None
    try:
        guard.check_idle(run, 100, 99, evidence)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    require((error is None) == (fault is None), 'mock idle classification')
    if fault == 'busy_process':
        require(calls == [guard.PS, guard.NEURON_LS], 'both checks even process busy')
    emit('mock_idle', fault=fault, error=error, called=calls, evidence=evidence)


for fault in (None, 'busy_process', 'busy_device', 'wrong_lnc', 'malformed_ps', 'missing_own',
              'ps_timeout', 'check_exit', 'check_stderr'):
    idle_trial(fault)


def lock_trial(fault=None):
    lock = '/review-existing-lock'
    path = '/packet/host_driver.py'
    expected = ['/usr/bin/flock', '-n', lock, '/usr/bin/python3', '-I', '-B', path]
    calls = []
    class MemoryPath(PurePosixPath):
        def lstat(self):
            return SimpleNamespace(st_mode=stat.S_IFLNK if fault == 'symlink' else stat.S_IFREG,
                st_uid=999 if fault == 'wrong_owner' else 123, st_dev=66305, st_ino=832589)
        def read_bytes(self):
            return ('\0'.join(expected if fault != 'parent_command' else ['wrong-parent']) + '\0').encode()
        def iterdir(self):
            return [self / '41', self / '42'] if fault == 'ambiguous_fd' else [self / '41']
        def read_text(self):
            return 'lock: 1: FLOCK ADVISORY WRITE 99 1:2:832589 0 EOF\n' if fault != 'not_locked' else ''
    def fake_fstat(fd):
        return SimpleNamespace(st_dev=66305, st_ino=1 if fault == 'wrong_fd_inode' else 832589)
    with patch.multiple(guard, Path=MemoryPath,
            os=SimpleNamespace(getuid=lambda:123, getppid=lambda:99, getpid=lambda:100, fstat=fake_fstat),
            fcntl=SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a:calls.append(a))):
        try:
            value = guard.inherited_lock(lock, path)
            error = None
        except Exception as exc:
            value, error = None, type(exc).__name__ + ': ' + str(exc)
    require((error is None) == (fault is None), 'inherited lock classification')
    if fault is None:
        require(value['inode'] == 832589 and value['device'] == 66305 and calls == [(41, 6)], 'same inherited FD')
    emit('mock_inherited_lock', fault=fault, result=value, error=error, flock_calls=calls)


for fault in (None, 'symlink', 'wrong_owner', 'parent_command', 'ambiguous_fd', 'not_locked', 'wrong_fd_inode'):
    lock_trial(fault)

require(not any(n == 'torch' or n == 'nki' or n.startswith(('torch.', 'nki.', 'torch_neuronx')) for n in sys.modules), 'native import')
emit('mechanics_summary', status='PASS_MOCKED_MECHANICS_ONLY', driver_trials=12, idle_trials=9,
     lock_trials=7, native_executed=False, lifecycle='Manual Operator all-outcome postflight remains mandatory; not exercised here')
