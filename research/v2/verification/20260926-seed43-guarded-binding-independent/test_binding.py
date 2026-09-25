"""Independent stdlib-only exact binding and refusal checks; no host or tensor use."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/poojith-devan/Desktop/Trainium')
PACKET = 'research/v2/operator/20260926-trn21-recursive-seed43-guarded'
GUARD = 'research/v2/operator/20260926-common-host-idle-guard'
rows = []

def check(name, condition):
    if not condition:
        raise RuntimeError('FAILED: ' + name)
    rows.append(name)

def git_bytes(ref, path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', ref + ':' + path])

def sha(data):
    return hashlib.sha256(data).hexdigest()

binding_bytes = git_bytes('90e942ea', PACKET + '/SEED_BINDING.disabled.json')
proposal_bytes = git_bytes('90e942ea', PACKET + '/EXPERIMENT_PROPOSAL.json')
driver_bytes = git_bytes('90e942ea', PACKET + '/host_driver.py')
b = json.loads(binding_bytes)
p = json.loads(proposal_bytes)
tree = ast.parse(driver_bytes)
old_tree = ast.parse(git_bytes('83db5fe3', 'research/v2/operator/20260926-trn22-arch3e79-full/host_driver.py'))
check('canonical import byte identity', all(git_bytes('26e45fb6', PACKET + '/' + name) == git_bytes('90e942ea', PACKET + '/' + name) for name in ('host_driver.py', 'SEED_BINDING.disabled.json', 'EXPERIMENT_PROPOSAL.json')))
for field, data in [
    ('source_sha256', git_bytes(b['source_commit'], b['source_git_path'])),
    ('driver_sha256', driver_bytes),
    ('prepare_sha256', (REPO / 'prepare.py').read_bytes()),
    ('harness_sha256', git_bytes('00f2ca7f', 'research/scratch/submission_rehearsal.py')),
    ('idle_guard_sha256', (REPO / GUARD / 'idle_guard.py').read_bytes()),
    ('dispatch_driver_sha256', (REPO / GUARD / 'dispatch_driver.py').read_bytes()),
]:
    check('hash ' + field, sha(data) == b[field])
check('source gate hash', sha((REPO / b['source_gate']['path']).read_bytes()) == b['source_gate']['sha256'])
check('proposal argv binding', p['argv'] == b['training_command'])
check('proposal source pin', p['source_sha256'] == b['source_sha256'])
check('disabled assignment', b['launch_authorized'] is False and b['stage_authorized'] is False and p['launch_authorized'] is False)
check('no automatic next', b['no_auto_next_job'] is True)
check('submission seed42', b['submissions_seed'] == 42 and p['submission_seed42'] is True)
control = json.loads((REPO / 'research/v2/operator/20260925-trn21-l3-base-repeat-r2/completed-training/training-launch.json').read_text())
expected = list(control['command'])
expected[expected.index('--seed') + 1] = '43'
check('only scientific argv difference is seed', expected == b['training_command'])
check('same 4-rank LNC2', '--nproc_per_node=4' in expected and b['training_environment']['NEURON_LOGICAL_NC_CONFIG'] == '2')
check('source budget1800', expected[expected.index('--max-train-seconds') + 1] == '1800')
check('seed environment consistent', all(b[k]['NEURON_COMPETITION_R1_SEED'] == '43' for k in ('training_environment', 'evaluation_environment')))
check('training compiler flags unchanged', control['environment']['NEURON_CC_FLAGS'] == b['training_environment']['NEURON_CC_FLAGS'])
check('training caches unchanged', all(b['training_environment'][k] == value for k, value in control['environment'].items() if k.endswith('CACHE_DIR')))
check('two-phase algorithm AST unchanged', ast.dump(ast.Module(body=[x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.Try))], type_ignores=[])) == ast.dump(ast.Module(body=[x for x in old_tree.body if isinstance(x, (ast.FunctionDef, ast.Try))], type_ignores=[])))
ns = {}
prefix = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)], type_ignores=[])
exec(compile(prefix, 'exact-driver-definitions', 'exec', optimize=0), ns)
for field, name in [('command', 'TRAIN_COMMAND'), ('evaluation_command', 'EVAL_COMMAND'), ('training_environment', 'TRAIN_ENV'), ('evaluation_environment', 'EVAL_ENV')]:
    check('driver literal ' + field, ns[name] == b[field])
check('exact common outer flock', b['outer_command'] == ['/usr/bin/flock', '-n', '/home/ubuntu/.cache/trainium_freshstart/device0.lock', '/usr/bin/python3', '-I', '-B', b['driver_path']])
phase = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'phase')
call = next(n for n in ast.walk(phase) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'call')
check('child inherits authenticated fd', any(k.arg == 'pass_fds' and ast.dump(k.value) == ast.dump(ast.parse('(lock_fd,)', mode='eval').body) for k in call.keywords))

# Exact preflight function with controlled filesystem observations, not host access.
class PreflightPath:
    def __init__(self, value): self.value = str(value)
    def __truediv__(self, value): return PreflightPath(self.value + '/' + value)
    def __str__(self): return self.value
    @property
    def parent(self): return self
    def glob(self, _): return []
    def is_dir(self): return True
for name, bad, expected_error in [('valid pins', None, None), ('source drift', 'SOURCE', 'SOURCE_SHA'), ('prepare drift', 'PREP', 'PREPARE_SHA'), ('harness drift', 'HARNESS', 'HARNESS_SHA'), ('STOP exists', 'STOP', 'STOP:'), ('disk low', 'DISK', 'DISK_HEADROOM')]:
    ctx = dict(ns, ROOT=PreflightPath('ROOT'), PREP=PreflightPath('PREP'), HARNESS=PreflightPath('HARNESS'), Path=PreflightPath)
    def fixture_sha(path):
        key = 'SOURCE' if str(path) == 'ROOT/package/train.py' else str(path)
        return 'wrong' if key == bad else {'SOURCE': b['source_sha256'], 'PREP': b['prepare_sha256'], 'HARNESS': b['harness_sha256']}[key]
    ctx.update(sha=fixture_sha, os=SimpleNamespace(path=SimpleNamespace(lexists=lambda _: bad == 'STOP'), statvfs=lambda _: SimpleNamespace(f_bavail=0 if bad == 'DISK' else 20 * 1024**3, f_frsize=1)))
    preflight_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    exec(compile(ast.Module(body=[preflight_node], type_ignores=[]), 'exact-preflight', 'exec', optimize=0), ctx)
    caught = None
    try: ctx['preflight'](b['training_environment'])
    except Exception as exc: caught = str(exc)
    check('preflight ' + name, caught is None if expected_error is None else expected_error in caught)

# Run the exact top-level control flow with no process execution or real host paths.
body = compile(ast.Module(body=[next(n for n in tree.body if isinstance(n, ast.Try))], type_ignores=[]), 'exact-driver-control-flow', 'exec', optimize=0)
class FakePath:
    def __init__(self, path='ROOT', case=None): self.path, self.case = path, case
    def __truediv__(self, child): return FakePath(self.path + '/' + child, self.case)
    def is_file(self): return not self.case.get('missing_checkpoint', False)
    def read_text(self): return json.dumps(self.case.get('rehearsal', {'returncode': 0, 'charged_through_process_exit_within_budget': True, 'checkpoint_sha256': 'fixture', 'checkpoint_bytes': 1}))
    def __str__(self): return self.path

def scenario(name, case, phases, terminal, error=None):
    calls, records, guard_calls = [], [], []
    def fake_guard(*args):
        guard_calls.append(args)
        if len(guard_calls) == case.get('guard_fail_at'):
            raise RuntimeError('FIXTURE_BUSY')
        return 20 + (1 if case.get('changed_fd') and len(guard_calls) == 2 else 0)
    def fake_phase(name, command, env, fd):
        if fd != 20: raise RuntimeError('WRONG_FD')
        calls.append(name)
        return case.get('train_rc', 0) if name == 'training' else case.get('eval_rc', 0)
    module = SimpleNamespace(require_idle_under_inherited_lock=fake_guard)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
    ctx = dict(ns, ROOT=FakePath(case=case), preflight=lambda _: None,
               os=SimpleNamespace(path=SimpleNamespace(lexists=lambda _: False)),
               sha=lambda _: b['idle_guard_sha256'],
               importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *_: spec, module_from_spec=lambda _: module)),
               record=lambda n, r: records.append((n, r)), phase=fake_phase)
    caught = None
    try:
        exec(body, ctx)
    except SystemExit as exc:
        caught = 'SystemExit:' + str(exc.code)
    except Exception as exc:
        caught = str(exc)
    check(name + ' phase order', calls == phases)
    check(name + ' terminal', records[-1][1]['status'] == terminal)
    if error is not None: check(name + ' refusal reason', error in caught)
    if phases == ['training', 'fresh-public20m']:
        check(name + ' two same-lock guard calls', len(guard_calls) == 2 and guard_calls[0][0] == guard_calls[1][0] == b['device_lock'])

scenario('success', {}, ['training', 'fresh-public20m'], 'COMPLETE_PENDING_INDEPENDENT_REVIEW')
scenario('busy before train', {'guard_fail_at': 1}, [], 'FULL_DRIVER_FAILED', 'FIXTURE_BUSY')
scenario('training exit failure', {'train_rc': 124}, ['training'], 'FULL_DRIVER_FAILED', 'TRAIN_EXIT_124')
scenario('over budget', {'rehearsal': {'returncode': 0, 'charged_through_process_exit_within_budget': False}}, ['training'], 'FULL_DRIVER_FAILED', 'TRAIN_BUDGET_FAILED')
scenario('missing budget evidence', {'rehearsal': {'returncode': 0}}, ['training'], 'FULL_DRIVER_FAILED', 'charged_through_process_exit_within_budget')
scenario('missing checkpoint', {'missing_checkpoint': True}, ['training'], 'FULL_DRIVER_FAILED', 'MISSING_CHECKPOINT')
scenario('busy before eval', {'guard_fail_at': 2}, ['training'], 'FULL_DRIVER_FAILED', 'FIXTURE_BUSY')
scenario('changed lock fd', {'changed_fd': True}, ['training'], 'FULL_DRIVER_FAILED', 'LOCK_FD_CHANGED')
scenario('eval failure', {'eval_rc': 124}, ['training', 'fresh-public20m'], 'EVAL_FAILED', 'SystemExit:124')

# Exercise the actual guard parsers independently with synthetic host observations.
guard_ns = {'__name__': 'independent_guard'}
exec(compile((REPO / GUARD / 'idle_guard.py').read_text(), 'exact-idle-guard', 'exec', optimize=0), guard_ns)
idle_ps = '100 99 python3 /packet/host_driver.py\n99 1 flock /usr/bin/flock -n /lock /usr/bin/python3 -I -B /packet/host_driver.py\n1 0 init /sbin/init\n'
idle_device = [{'neuron_device': 0, 'instance_type': 'trn2.3xlarge', 'nc_count': 4, 'logical_neuroncore_config': 2, 'neuroncore_ids': [0, 1, 2, 3], 'neuron_processes': []}]
for name, ps, nl, wanted in [
    ('idle parser', idle_ps, json.dumps(idle_device), None),
    ('other torchrun', idle_ps + '200 1 python /venv/bin/torchrun --standalone train.py\n', json.dumps(idle_device), 'PS_NATIVE_JOB_BUSY'),
    ('malformed ps', idle_ps + 'bad row\n', json.dumps(idle_device), 'PS_PARSE_FAILED'),
    ('ambiguous neuron schema', idle_ps, '{}', 'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'),
    ('device holder', idle_ps, json.dumps([dict(idle_device[0], neuron_processes=[{'pid': 222}])]), 'NEURON_DEVICE_BUSY'),
]:
    observed, evidence = [], {}
    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, ps if command == guard_ns['PS'] else nl, '')
    caught = None
    try: guard_ns['check_idle'](fake_run, 100, 99, evidence)
    except Exception as exc: caught = str(exc)
    check(name, caught == wanted)
    if name == 'other torchrun': check('both idle checks even when ps busy', len(observed) == 2)
with tempfile.TemporaryDirectory(prefix='seed43-binding-review-') as tmp:
    lock = Path(tmp) / 'device0.lock'
    lock.touch()
    try:
        guard_ns['inherited_lock'](lock, Path(tmp) / 'host_driver.py')
        direct = None
    except RuntimeError as exc: direct = str(exc)
    check('direct no-flock invocation refuses', direct == 'OUTER_FLOCK_COMMAND_MISMATCH')

harness = ast.parse(git_bytes('00f2ca7f', 'research/scratch/submission_rehearsal.py'))
main = next(n for n in harness.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
eval_if = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'eval'")
eval_command_node = next(n.value for n in eval_if.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
eval_constants = [n.value for n in eval_command_node.elts if isinstance(n, ast.Constant)]
check('actual harness fresh eval context/batch/device', eval_constants == ['prepare.py', 'eval-public', '--train-py', 'train.py', '--checkpoint', '--device', 'neuron', '--batch-size', '1', '--seq-len', '1024', '--eval-tokens', '--timeout-seconds', '1200'])
budget_if = next(n for n in main.body if isinstance(n, ast.If) and 'args.mode == \'train\' and process.returncode == 0' == ast.unparse(n.test))
accounting = ast.Module(body=budget_if.body[3:], type_ignores=[])
class Checkpoint:
    def stat(self): return SimpleNamespace(st_size=1)
for wall, fifth, expected_pass, expected_charge in [(2699.9, 1200, True, 1799.9), (2700.1, 1200, False, 1800.1), (1900, 100, True, 1800), (1900.1, 100, False, 1800.1)]:
    context = {'observed': [{'observed_seconds': fifth}] * 5, 'result': {'wall_seconds': wall}, 'checkpoint': Checkpoint(), 'sha': lambda _: 'fixture'}
    exec(compile(accounting, 'exact-harness-accounting', 'exec', optimize=0), context)
    result = context['result']
    check('budget boundary ' + str(wall) + '/' + str(fifth), result['charged_through_process_exit_within_budget'] is expected_pass and abs(result['charged_through_process_exit'] - expected_charge) < 1e-9)
check('eval full public20M binding', b['evaluation_command'][-2:] == ['--eval-tokens', '20971520'] and b['eval']['seq_len'] == 1024)
result = {'status': 'PASS', 'checks': len(rows), 'tests': rows, 'source_packet_commit': '90e942ead710adb7abdb085213476cd6dce8feb9', 'input_hashes': {'binding': sha(binding_bytes), 'proposal': sha(proposal_bytes), 'driver': sha(driver_bytes)}, 'scope': 'stdlib binding / mocked refusal and exact harness arithmetic only; no host, native, source import, or tensor execution'}
print(json.dumps(result, indent=2))
