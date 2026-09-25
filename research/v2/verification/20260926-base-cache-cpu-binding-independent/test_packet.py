"""Independent bounded packet checks and exact legacy-scan reproducer; stdlib only."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path('/home/poojith-devan/Desktop/Trainium-base-cache-cpu-packet-20260926')
REF = sys.argv[1] if len(sys.argv) > 1 else '5a3c5358473e3463b19aadeafeac6212a1b24f96'
BASE = 'research/v2/proposals/20260926-base-cache-exact-cpu/'
def raw(path): return subprocess.check_output(['git', '-C', str(REPO), 'show', REF + ':' + BASE + path])
def digest(data): return hashlib.sha256(data).hexdigest()
checks = []
def check(name, value):
    if not value: raise RuntimeError('FAILED: ' + name)
    checks.append(name)
plan = json.loads(raw('CPU_PLAN.disabled.json'))
check('not staging/native/adoption authority', plan['launch_authorized'] is False and plan['native_launch_authorized'] is False and plan['source_adoption'] is False)
check('trn22 exact host only', plan['proposed_host'] == 'trn22' and plan['expected_hostname'] == 'ip-172-31-32-253' and plan['host_reserved'] is False)
check('no laptop', plan['local_tensor_execution_prohibited'] is True)
check('full parameters small synthetic workload', plan['full_model']['parameters'] == 117453580 and plan['full_model']['parameter_tensors'] == 62 and plan['micro_batch'] == [2, 256] and plan['production_update_equivalent'] is False)
check('resource floors', plan['minimum_available_ram_gib'] == 64 and plan['minimum_free_disk_gib'] == 6.5)
check('reference/default/OFF/ON order', plan['ordering'] == ['reference-6ef7', 'candidate-default', 'candidate-off', 'candidate-on'])
for name, source in plan['sources'].items():
    blob = subprocess.check_output(['git', '-C', str(REPO), 'show', source['commit'] + ':' + source['git_path']]) if 'commit' in source else (REPO / source.get('git_path', 'prepare.py')).read_bytes()
    check('pin ' + name, digest(blob) == source['sha256'])

reproductions = []
normalized = []
for index, probe in enumerate(plan['probes']):
    name = probe['name']
    request = json.loads(raw(probe['request_path']))
    driver = raw(probe['driver_path'])
    tree = ast.parse(driver)
    check(name + ' request SHA', digest(raw(probe['request_path'])) == probe['request_sha256'])
    check(name + ' driver SHA', digest(driver) == probe['driver_sha256'])
    command = request['command']
    check(name + ' command SHA and exact binding', command == probe['command'] and digest(json.dumps(command, separators=(',', ':')).encode()) == probe['command_sha256'])
    extra = [] if index < 2 else ['--no-initial-cache-refresh' if index == 2 else '--initial-cache-refresh']
    check(name + ' only explicit mode flag', request['argv'] == plan['common_train_argv'] + extra and command[command.index('--')+1:] == request['argv'])
    check(name + ' actual source path and pin', request['source_sha256'] == plan['sources']['reference' if index == 0 else 'candidate']['sha256'] and command[command.index('--train-py') + 1] == plan['sources']['reference' if index == 0 else 'candidate']['stage_path'])
    check(name + ' standard bounded CPU harness flags', all(command[command.index(flag)+1] == value for flag, value in [('--steps','3'),('--nproc','4'),('--micro-rows','2'),('--micro-tokens','256'),('--timeout','3600')]) and '--keep' in command and '--stub-prepare' not in command and '--real-data' not in command)
    check(name + ' same common flock', probe['expected_outer_command'] == ['/usr/bin/flock','-n',plan['device_lock'],'/usr/bin/python3','-I','-B',probe['arm_root'] + '/host_driver.py'] and plan['device_lock'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    guard = next(n for n in calls if n.func.attr == 'require_idle_under_inherited_lock')
    popen = next(n for n in calls if n.func.attr == 'Popen')
    check(name + ' guard precedes child with inherited FD', guard.lineno < popen.lineno and any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in popen.keywords))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checks')
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'active')
    scan = ast.Module(body=function.body[start:start+3], type_ignores=[])
    class Entry:
        def __init__(self, pid, argv): self.parent = SimpleNamespace(name=str(pid)); self.argv = argv
        def read_bytes(self): return b'\0'.join(s.encode() for s in self.argv) + b'\0'
    entries = [Entry(100, ['/usr/bin/python3','-I','-B',probe['arm_root'] + '/host_driver.py']), Entry(99, probe['expected_outer_command'])]
    class Proc:
        def glob(self, pattern): return entries
    def path(value): return Proc() if value == '/proc' else Path(value)
    def require(value, message):
        if not value: raise RuntimeError(message)
    context = {'Path': path, 'os': SimpleNamespace(getpid=lambda:100, getppid=lambda:99), 'require': require}
    error = None
    try: exec(compile(scan, 'exact-legacy-process-scan', 'exec'), context)
    except RuntimeError as exc: error = str(exc)
    reproductions.append({'arm': name, 'driver_pid': 100, 'authenticated_outer_pid': 99, 'outer_argv': probe['expected_outer_command'], 'active': context['active'], 'error': error})
    check(name + ' reproduced required-parent false busy', error == 'OTHER_WORKLOAD:[99]')
    entries.append(Entry(200, ['/venv/bin/torchrun', 'train.py']))
    try: exec(compile(scan, 'exact-legacy-process-scan', 'exec'), context)
    except RuntimeError: pass
    check(name + ' real competitor remains detected', context['active'] == [99, 200])
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ROOT': node.value = ast.Constant('ARM')
    normalized.append(ast.dump(tree))
check('only ROOT differs in all drivers', len(set(normalized)) == 1)
cold = json.loads(raw('COLD_PROPOSAL.disabled.json'))
check('no automatic cold or full', cold['launch_authorized'] is False and cold['full_run_automatic'] is False and cold['num_steps'] == 5 and cold['ct_target_seconds'] == 600)
print(json.dumps({'reviewed_commit': REF, 'verdict': 'HOLD_REQUIRED_OUTER_FLOCK_FALSE_BUSY', 'checks_passed': len(checks), 'checks': checks, 'reproductions': reproductions, 'source_and_tensor_execution': False, 'host_actions': False, 'plan_sha256': digest(raw('CPU_PLAN.disabled.json')), 'note': 'Checks PASS means reproducer and bindings behaved as asserted; packet itself remains HOLD.'}, indent=2))
