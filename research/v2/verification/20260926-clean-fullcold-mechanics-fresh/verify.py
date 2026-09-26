#!/usr/bin/env python3
"""Independent CPU-only mechanics probes; never executes scientific/native code."""
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

PACKET = Path('/home/poojith-devan/clean-fullcold-review-DrcoWi')
OUT = Path(__file__).resolve().parent
RESULTS = []

def check(name, condition, details=None):
    item = {'name': name, 'passed': bool(condition)}
    if details is not None:
        item['details'] = details
    RESULTS.append(item)
    if not condition:
        raise RuntimeError('INDEPENDENT_TEST_FAILED: ' + name)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

binding = json.loads((PACKET / 'NATIVE_BINDING.disabled.json').read_text())
proposal = json.loads((PACKET / 'EXPERIMENT_PROPOSAL.json').read_text())
manifest = json.loads((PACKET / 'MANIFEST.sha256.json').read_text())
driver_text = (PACKET / 'host_driver.py').read_text()
driver_tree = ast.parse(driver_text)
prefix = ast.Module(body=driver_tree.body[:-1], type_ignores=[])
body = ast.Module(body=driver_tree.body[-1:], type_ignores=[])
harness_text = (PACKET / 'submission_rehearsal.py').read_text()

check('supplied_manifest_entries_match', all(
    digest(PACKET / name) == expected
    for name, expected in manifest['files'].items() if (PACKET / name).exists()))
check('known_manifest_coverage_limit', sorted(name for name in manifest['files']
      if not (PACKET / name).exists()) == ['README.md', 'test_binding.py'])
for name, key in [('host_driver.py', 'driver_sha256'), ('idle_guard.py', 'idle_guard_sha256'),
                  ('dispatch_driver.py', 'dispatch_driver_sha256'),
                  ('submission_rehearsal.py', 'harness_sha256')]:
    check('pinned_' + name, digest(PACKET / name) == binding[key])
check('scientific_bytes_only_hashed_never_imported', digest(PACKET / 'package/train.py')
      == proposal['source_sha256'] == binding['source_sha256'])
check('baseline_hash', digest(PACKET / 'baseline_host_driver.py') == binding['driver_derivation']['sha256'])
transformed = (PACKET / 'baseline_host_driver.py').read_text()
replacement_counts = []
for before, after in binding['driver_derivation']['replacements']:
    replacement_counts.append([before, after, transformed.count(before)])
    transformed = transformed.replace(before, after)
check('exact_four_literal_transform', transformed == driver_text, replacement_counts)
launch = (PACKET / 'package/launch-command.txt').read_text()
tokens = shlex.split(launch)
check('literal_command_sha', digest(PACKET / 'package/launch-command.txt') == proposal['submission_command_sha256'])
check('command_identity', tokens[0] == 'NEURON_LOGICAL_NC_CONFIG=2'
      and tokens[1:] == binding['training_command']
      and proposal['argv'][1:] == tokens[2:]
      and proposal['argv'][0] == '/home/ubuntu/workspace/native_venv/bin/torchrun'
      and '--num-steps' not in tokens and '--num-steps5' not in tokens)
check('all_authority_disabled', all(proposal[k] is False for k in
      ('stage_authorized', 'launch_authorized', 'submission_authorized')) and all(
      binding[k] is False for k in ('stage_authorized', 'launch_authorized', 'submission_authorized', 'native_launch_authorized')))

def driver_ns(case, optimized=False):
    ns = {'__name__': 'independent_fixture'}
    exec(compile(prefix, 'actual_host_driver.py', 'exec', optimize=int(optimized)), ns)
    root = Path(tempfile.mkdtemp(prefix=case + '-', dir=OUT))
    ns['ROOT'] = root
    ns['TRAIN_ENV'] = dict(ns['TRAIN_ENV'], NEURON_COMPILE_CACHE_URL=str(root / 'train/caches/compiler'))
    ns['EVAL_ENV'] = dict(ns['EVAL_ENV'], NEURON_COMPILE_CACHE_URL=str(root / 'eval/caches/compiler'))
    ns['sys'] = SimpleNamespace(flags=SimpleNamespace(optimize=int(optimized)), exit=lambda rc: (_ for _ in ()).throw(SystemExit(rc)))
    ns['socket'] = SimpleNamespace(gethostname=lambda: binding['expected_hostname'])
    ns['os'] = SimpleNamespace(path=SimpleNamespace(lexists=lambda p: Path(p).is_relative_to(root) and Path(p).exists()),
        statvfs=lambda p: SimpleNamespace(f_bavail=20 * 1024**3, f_frsize=1))
    hashes = {'train.py': binding['source_sha256'], 'launch-command.txt': binding['package_launch_sha256'],
        'provenance.json': binding['package_provenance_sha256'], 'prepare.py': binding['prepare_sha256'],
        'submission_rehearsal.py': binding['harness_sha256'], 'dispatch_driver.py': binding['dispatch_driver_sha256'],
        'idle_guard.py': binding['idle_guard_sha256']}
    ns['sha'] = lambda p: hashes[p.name]
    return ns, root, hashes

def run_driver_case(case, expected_calls, expected_status, fault=None):
    ns, root, hashes = driver_ns(case)
    calls, guard_calls = [], []
    def guard(lock, driver, record):
        guard_calls.append((lock, str(driver)))
        if fault == 'guard_first' or (fault == 'guard_second' and len(guard_calls) == 2):
            raise RuntimeError('INDEPENDENT_BUSY_FAULT')
        return 10 if fault == 'fd_changed' and len(guard_calls) == 2 else 9
    ns['importlib'] = SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)),
        module_from_spec=lambda *a: SimpleNamespace(require_idle_under_inherited_lock=guard)))
    if fault == 'occupied_train': (root / 'train').mkdir()
    if fault == 'occupied_eval': (root / 'eval').mkdir()
    if fault == 'source_hash': hashes['train.py'] = 'wrong'
    def call(command, **kwargs):
        calls.append({'command': command, 'fd': kwargs['pass_fds'], 'environment': kwargs['env']})
        check(case + '_phase_fd_' + str(len(calls)), kwargs['pass_fds'] == (9,))
        if len(calls) == 1:
            (root / 'train/out').mkdir(parents=True)
            if fault != 'missing_checkpoint': (root / 'train/out/final.pt').write_bytes(b'CPU_SYNTHETIC_ONLY')
            (root / 'train/rehearsal.json').write_text(json.dumps({'returncode': 0,
                'charged_through_process_exit_within_budget': fault != 'bad_budget',
                'checkpoint_sha256': 'synthetic', 'checkpoint_bytes': 18}))
            if fault == 'eval_source_hash': hashes['train.py'] = 'wrong'
            return 19 if fault == 'train_exit' else 0
        return 23 if fault == 'eval_exit' else 0
    ns['subprocess'] = SimpleNamespace(call=call, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)
    outcome = None
    with patch.object(Path, 'glob', lambda *args: iter(())):
        try:
            exec(compile(body, 'actual_host_driver.py', 'exec', optimize=0), ns)
        except (Exception, SystemExit) as exc:
            outcome = str(exc)
    terminal = json.loads((root / 'terminal.json').read_text())
    check(case + '_call_count', len(calls) == expected_calls, {'calls': len(calls), 'guards': len(guard_calls), 'outcome': outcome})
    check(case + '_terminal', terminal['status'] == expected_status and terminal['no_automatic_next_job'] is True, terminal)
    if expected_calls == 2:
        check(case + '_phase_order', calls[0]['command'] == binding['command'] and calls[1]['command'] == binding['evaluation_command'])
        check(case + '_different_phase_urls', calls[0]['environment']['NEURON_COMPILE_CACHE_URL'] != calls[1]['environment']['NEURON_COMPILE_CACHE_URL'])
    check(case + '_max_two_phases', len(calls) <= 2)

for case, calls, status, fault in [
    ('normal_sequence', 2, 'COMPLETE_PENDING_INDEPENDENT_REVIEW', None),
    ('occupied_train', 0, 'FULL_DRIVER_FAILED', 'occupied_train'),
    ('occupied_eval', 0, 'FULL_DRIVER_FAILED', 'occupied_eval'),
    ('source_hash', 0, 'FULL_DRIVER_FAILED', 'source_hash'),
    ('guard_before_train', 0, 'FULL_DRIVER_FAILED', 'guard_first'),
    ('train_nonzero', 1, 'FULL_DRIVER_FAILED', 'train_exit'),
    ('train_budget_failed', 1, 'FULL_DRIVER_FAILED', 'bad_budget'),
    ('missing_checkpoint', 1, 'FULL_DRIVER_FAILED', 'missing_checkpoint'),
    ('source_changed_before_eval', 1, 'FULL_DRIVER_FAILED', 'eval_source_hash'),
    ('guard_before_eval', 1, 'FULL_DRIVER_FAILED', 'guard_second'),
    ('fd_changed_before_eval', 1, 'FULL_DRIVER_FAILED', 'fd_changed'),
    ('eval_nonzero', 2, 'EVAL_FAILED', 'eval_exit')]:
    run_driver_case(case, calls, status, fault)

for fault, expected in [('hostname','HOSTNAME'), ('disk','DISK_HEADROOM'), ('stop','STOP:'),
                        ('url','FRESH_URL_BINDING'), ('optimized','OPTIMIZED_DRIVER_FORBIDDEN')]:
    ns, root, hashes = driver_ns('preflight_' + fault, optimized=fault == 'optimized')
    if fault == 'hostname': ns['socket'].gethostname = lambda: 'wrong'
    if fault == 'disk': ns['os'].statvfs = lambda p: SimpleNamespace(f_bavail=10 * 1024**3, f_frsize=1)
    if fault == 'stop': ns['os'].path.lexists = lambda p: str(p).endswith('/STOP')
    if fault == 'url': ns['TRAIN_ENV']['NEURON_COMPILE_CACHE_URL'] = '/shared/compiler'
    error = ''
    with patch.object(Path, 'glob', lambda *args: iter(())):
        try: ns['preflight'](ns['TRAIN_ENV'])
        except Exception as exc: error = str(exc)
    check('preflight_refuses_' + fault, expected in error, error)
ns, root, hashes = driver_ns('disk_reserve_gap')
ns['os'].statvfs = lambda p: SimpleNamespace(f_bavail=11 * 1024**3, f_frsize=1)
with patch.object(Path, 'glob', lambda *args: iter(())):
    ns['preflight'](ns['TRAIN_ENV'])
check('observed_11GiB_passes_driver_below_13_5GiB_proposal', True, 'Executable gate is 10.5 GiB; proposal explicitly also requires 3 GiB reserve.')

# Run the actual supplied harness in a generated local fixture. All process calls
# are replaced by deterministic CPU stubs; neither prepare.py nor train.py runs.
def harness_case(mode, outside=False):
    fixture = Path(tempfile.mkdtemp(prefix='harness_' + mode + '-', dir=OUT))
    root = fixture / 'project'
    root.mkdir()
    (root / 'prepare.py').write_text('# CPU fixture: never executed\n')
    package = PACKET / 'package'
    sandbox = (fixture / 'outside') if outside else (root / 'out' / mode)
    checkpoint = fixture / 'synthetic.pt'
    checkpoint.write_bytes(b'CPU_SYNTHETIC_ONLY')
    ns = {'__name__': 'independent_fixture', '__file__': str(root / 'research/scratch/submission_rehearsal.py')}
    exec(compile(harness_text, str(PACKET / 'submission_rehearsal.py'), 'exec', optimize=0), ns)
    argv = ['actual-harness', '--package', str(package), '--sandbox', str(sandbox), '--mode', mode,
            '--fresh-caches']
    if mode == 'eval': argv += ['--checkpoint', str(checkpoint), '--eval-tokens', '20971520']
    captured = []
    ticks = iter([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0])
    def run(command, **kwargs):
        captured.append(('prepare_check', command, kwargs))
        return SimpleNamespace(returncode=0)
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            self.polls = 0
            captured.append(('child', command, kwargs))
            if mode == 'train':
                (sandbox / 'out').mkdir()
                (sandbox / 'out/final.pt').write_bytes(b'CPU_SYNTHETIC_ONLY')
                (sandbox / 'steps.jsonl').write_text('1\n2\n3\n4\n5\n')
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0
    ns['subprocess'] = SimpleNamespace(run=run, Popen=Process)
    ns['time'] = SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda seconds: None)
    ns['sys'] = SimpleNamespace(argv=argv, executable='/CPU_FIXTURE_PYTHON')
    ns['os'] = SimpleNamespace(environ={'NEURON_COMPILE_CACHE_URL': str(sandbox / 'caches/compiler'),
                                     'PYTHONPATH': '/unwanted', 'NEURON_LOGICAL_NC_CONFIG': '2'})
    error = None
    with patch.object(sys, 'argv', argv):
        try: ns['main']()
        except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    if outside:
        check('actual_harness_old_outside_rejected', error is not None and error.startswith('AssertionError')
              and not sandbox.exists() and not captured, error)
        return
    check('actual_harness_contained_' + mode, error is None and len(captured) == 2, error)
    environment = captured[1][2]['env']
    cache_values = [environment[key] for key in binding['fresh_cache_contract']['expected_training_dirs']]
    check('actual_harness_six_empty_private_caches_' + mode,
          len(set(cache_values)) == 6 and all(Path(p).is_relative_to(sandbox) and Path(p).is_dir()
              and not list(Path(p).iterdir()) for p in cache_values))
    check('actual_harness_URL_matches_DIR_' + mode, environment['NEURON_COMPILE_CACHE_URL'] == environment['NEURON_COMPILE_CACHE_DIR'])
    check('actual_harness_removes_PYTHONPATH_' + mode, 'PYTHONPATH' not in environment)
    if mode == 'train':
        check('actual_harness_exact_training_argv', captured[1][1] == binding['training_command'])
        result = json.loads((sandbox / 'rehearsal.json').read_text())
        check('actual_harness_budget_arithmetic', result['startup_allowance_observed'] == 10
              and result['charged_through_process_exit'] == 90 and result['charged_through_process_exit_within_budget'] is True)
    else:
        command = captured[1][1]
        check('actual_harness_fresh_eval_command', command[:3] == ['/CPU_FIXTURE_PYTHON','prepare.py','eval-public']
              and command[command.index('--batch-size')+1] == '1'
              and command[command.index('--seq-len')+1] == '1024'
              and command[command.index('--eval-tokens')+1] == '20971520'
              and command[command.index('--timeout-seconds')+1] == '1200')

harness_case('train', outside=True)
harness_case('train')
harness_case('eval')

guard_ns = {'__name__': 'independent_fixture'}
exec(compile((PACKET / 'idle_guard.py').read_text(), 'actual_idle_guard.py', 'exec', optimize=0), guard_ns)
device = {'neuron_device': 0, 'instance_type':'trn2.3xlarge','nc_count':4,
          'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3], 'neuron_processes':[]}
for fault in ('idle', 'native_job', 'device_busy', 'unknown_device', 'missing_parent', 'ps_error'):
    commands=[]
    def run(command, **kwargs):
        commands.append(command)
        if command == guard_ns['PS']:
            rows='100 99 python3 /fixture/host_driver.py\n99 1 flock /usr/bin/flock\n'
            if fault == 'native_job': rows+='101 1 python3 /fixture/train.py\n'
            if fault == 'missing_parent': rows='100 99 python3 /fixture/host_driver.py\n'
            return SimpleNamespace(returncode=1 if fault == 'ps_error' else 0, stdout=rows, stderr='')
        d=dict(device)
        if fault == 'device_busy': d['neuron_processes']=[{'pid':999}]
        if fault == 'unknown_device': d['nc_count']=32
        return SimpleNamespace(returncode=0, stdout=json.dumps([d]), stderr='')
    error=None
    try: guard_ns['check_idle'](run, 100, 99, {})
    except Exception as exc: error=str(exc)
    check('actual_idle_guard_' + fault, (error is None) == (fault == 'idle'), error)
    if fault == 'native_job': check('busy_ps_still_checks_device', len(commands) == 2)

lock_fd,lock_name=tempfile.mkstemp(prefix='cpu-lifetime-' + str(sys.flags.optimize) + '-', suffix='.lock',dir=OUT)
os.close(lock_fd)
lock = Path(lock_name)
probe = subprocess.run(['/usr/bin/flock','-n',str(lock),'/usr/bin/python3','-I','-B',
                        str(OUT / 'lock_probe.py')], capture_output=True, text=True, timeout=20)
check('actual_guard_local_lifetime_probe_exits_zero', probe.returncode == 0, {'stdout':probe.stdout,'stderr':probe.stderr})
local = json.loads(probe.stdout)
check('actual_lock_remains_through_two_children', local['contender_codes'] == [1,1,1,1]
      and len(set(local['guard_fds'])) == 1 and local['child_fds_ok'] == [True,True], local)
released = subprocess.run(['/usr/bin/flock','-n',str(lock),'/usr/bin/true'], capture_output=True, timeout=5)
check('local_lock_released_after_driver_exit', released.returncode == 0)
check('observed_original_inode_pin_external_only', (local['device'],local['inode']) !=
      (binding['original_lock']['device'], binding['original_lock']['inode']),
      'Exact guard accepts the local authenticated lifetime lock with a different identity; pinned original identity must be checked externally.')

# Additional outer transport supplied after the initial runtime review. Execute
# its actual remote pre-staging gate statements with fully local file fixtures
# and stubbed process/device/lock calls. Never evaluate its local network path.
stage_text = (PACKET / 'stage-dispatch.py').read_text()
check('pinned_outer_transport', digest(PACKET / 'stage-dispatch.py') ==
      'adcb67146b1f542289b7cc21ac552c2bcd558e25d40d624c461a2d8ddc5127ec')
stage_tree = ast.parse(stage_text)
remote_text = next(ast.literal_eval(node.value) for node in stage_tree.body
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in node.targets))
remote_tree = ast.parse(remote_text)
prefix_nodes = []
for node in remote_tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'mkdir':
        break
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        prefix_nodes.append(node)
remote_prefix = ast.Module(body=prefix_nodes, type_ignores=[])
check('transport_unlocks_before_dispatch', remote_text.index('fcntl.flock(lockfd,fcntl.LOCK_UN)') <
      remote_text.index('process=subprocess.Popen('),
      'Staging lock is released; exact outer flock acquires the subsequent train/eval lifetime lock. These are separate acquisitions.')

def transport_case(fault=None):
    fixture=Path(tempfile.mkdtemp(prefix='transport_' + str(fault) + '-', dir=OUT))
    project=fixture/'project'
    (project/'research/scratch').mkdir(parents=True)
    (project/'prepare.py').write_bytes(b'CPU_FIXTURE_PREPARE_NOT_EXECUTED')
    shutil.copyfile(PACKET/'submission_rehearsal.py', project/'research/scratch/submission_rehearsal.py')
    root=project/'out/fullcold'
    prior=project/'out/previous'
    (prior/'train/out').mkdir(parents=True)
    (prior/'terminal.json').write_text(json.dumps({'status':'CT_HARNESS_EXITED','returncode':0}))
    (prior/'outer-flock-exit.json').write_text(json.dumps({'returncode':0}))
    (prior/'train/rehearsal.json').write_text('{}')
    (prior/'train/out/final.pt').write_bytes(b'CPU_SYNTHETIC_ONLY')
    b=json.loads(json.dumps(binding))
    b.update(root=str(root), prepare_path=str(project/'prepare.py'), prepare_sha256=digest(project/'prepare.py'),
             harness_path=str(project/'research/scratch/submission_rehearsal.py'), sandbox=str(root/'train'), eval_sandbox=str(root/'eval'))
    for phase, envkey, cachekey in [('train','training_environment','expected_training_dirs'),('eval','evaluation_environment','expected_eval_dirs')]:
        b[envkey]['NEURON_COMPILE_CACHE_URL']=str(root/phase/'caches/compiler')
        b['fresh_cache_contract'][cachekey]={k:str(root/phase/'caches'/Path(v).name)
            for k,v in b['fresh_cache_contract'][cachekey].items()}
    P={'binding':b,'prior_root':str(prior),'prior_observed_pids':[123456789],
       'prior_terminal_sha256':digest(prior/'terminal.json'),'prior_outer_sha256':digest(prior/'outer-flock-exit.json'),
       'prior_rehearsal_sha256':digest(prior/'train/rehearsal.json'),'prior_checkpoint_sha256':digest(prior/'train/out/final.pt')}
    if fault=='outside': b['sandbox']=str(fixture/'outside')
    if fault=='used_root': root.mkdir(parents=True)
    if fault=='source_changed': b['prepare_sha256']='wrong'
    if fault=='private_url': b['evaluation_environment']['NEURON_COMPILE_CACHE_URL']='/shared'
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        if command[0]=='/usr/bin/ps':
            rows='PID PPID PGID SID COMMAND COMMAND\n1 0 1 1 init /sbin/init\n'
            if fault=='busy_ps': rows+='42 1 42 42 python train.py\n'
            if fault=='lineage': rows+='42 123456789 42 42 python innocent.py\n'
            return SimpleNamespace(returncode=0,stdout=rows,stderr='')
        if command[0]=='/opt/aws/neuron/bin/neuron-ls':
            return SimpleNamespace(returncode=0,stdout=json.dumps([{'neuron_processes':[{'pid':9}] if fault=='busy_device' else []}]),stderr='')
        if command[0]=='/usr/bin/fuser':
            return SimpleNamespace(returncode=0 if fault=='fuser' else 1,stdout='9' if fault=='fuser' else '',stderr='')
        raise RuntimeError('UNEXPECTED_PROCESS_CALL')
    fake_os=SimpleNamespace(getuid=lambda:1000, path=SimpleNamespace(lexists=lambda p:
        (str(p).endswith('/STOP') if fault=='stop' else False) or (Path(p).is_relative_to(fixture) and Path(p).exists())),
        open=lambda *args:7, O_RDWR=os.O_RDWR, O_NOFOLLOW=os.O_NOFOLLOW,
        fstat=lambda fd:SimpleNamespace(st_dev=66305,st_ino=999 if fault=='inode' else 832589),
        statvfs=lambda p:SimpleNamespace(f_bavail=(11 if fault=='disk' else 20)*1024**3,f_frsize=1))
    def flock(fd, flags):
        if fault=='lock_busy': raise BlockingIOError('CPU fixture: lock already held')
    original_exists=Path.exists
    original_is_file=Path.is_file
    def exists(path):
        if str(path).startswith('/proc/'): return False
        return original_exists(path)
    def is_file(path):
        if str(path) in ('/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/PAUSE_NEXT',
                         '/home/ubuntu/workspace/native_venv/bin/torchrun'): return True
        return original_is_file(path)
    ns={'P':P,'Path':Path,'hashlib':hashlib,'json':json,'os':fake_os,
        'socket':SimpleNamespace(gethostname=lambda:'wrong' if fault=='hostname' else 'ip-172-31-32-253'),
        'subprocess':SimpleNamespace(run=run),
        'fcntl':SimpleNamespace(flock=flock,LOCK_EX=2,LOCK_NB=4)}
    error=None
    with patch.object(Path,'exists',exists), patch.object(Path,'is_file',is_file), patch.object(Path,'glob',
        lambda self,pattern: iter([fixture/'neuron0']) if str(self)=='/dev' else iter(())):
        try: exec(compile(remote_prefix,'actual_transport_remote_gates.py','exec',optimize=0),ns)
        except Exception as exc: error=type(exc).__name__+': '+str(exc)
    check('outer_transport_' + str(fault), (error is None)==(fault is None), {'error':error,'process_checks':len(calls)})

for fault in (None,'hostname','used_root','inode','lock_busy','disk','stop','source_changed',
              'outside','private_url','busy_ps','lineage','busy_device','fuser'):
    transport_case(fault)

timeout_fd,timeout_name=tempfile.mkstemp(prefix='cpu-timeout-descendant-',suffix='.lock',dir=OUT)
os.close(timeout_fd)
timeout_lock=Path(timeout_name)
with (OUT/(timeout_lock.name+'.raw.txt')).open('x') as output:
    timed=subprocess.run(['/usr/bin/flock','-n',str(timeout_lock),'/usr/bin/python3','-I','-B',
        str(OUT/'timeout_driver_probe.py')],stdout=output,stderr=subprocess.STDOUT,timeout=5)
child=json.loads(Path(str(timeout_lock)+'.child.json').read_text())
alive=True
try: os.kill(child['pid'],0)
except ProcessLookupError: alive=False
competitor=subprocess.run(['/usr/bin/flock','-n',str(timeout_lock),'/usr/bin/true'],capture_output=True,timeout=5)
observed={'outer_returncode':timed.returncode,'child_alive_after_outer_exit':alive,
          'child_has_original_lock_fd':child['owns_lock_fd'],'new_competitor_returncode':competitor.returncode}
check('reproduced_timeout_descendant_loses_lock_protection', timed.returncode==124 and alive
      and child['owns_lock_fd'] is False and competitor.returncode==0, observed)
deadline=time.monotonic()+3
while not Path(str(timeout_lock)+'.child.done').exists() and time.monotonic()<deadline:
    time.sleep(0.05)
check('CPU_timeout_fixture_child_completed', Path(str(timeout_lock)+'.child.done').exists())
check('input_hashes_unchanged_after_tests', digest(PACKET/'host_driver.py') == binding['driver_sha256']
      and digest(PACKET/'package/train.py') == binding['source_sha256'] and digest(PACKET/'submission_rehearsal.py') == binding['harness_sha256'])
print('INDEPENDENT_RESULT ' + json.dumps({'verifier_optimize':sys.flags.optimize,'checks':len(RESULTS),
      'all_checks_passed':all(item['passed'] for item in RESULTS),'results':RESULTS}, sort_keys=True))
