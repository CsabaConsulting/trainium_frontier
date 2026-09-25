"""Independent frozen-packet checks. Stdlib only; never runs a host driver."""
import ast
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/poojith-devan/Desktop/Trainium')
COMMIT = 'ce73fa8a54860d309c4ced52e50a5fb7016bb8ae'
BASE = 'research/v2/proposals/20260926-architecture-clean-cold-'
GUARD = 'research/v2/operator/20260926-common-host-idle-guard/'
checks, pins = [], {}

def check(name, value):
    if not value:
        raise RuntimeError('FAIL: ' + name)
    checks.append(name)

def blob(ref, path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', ref + ':' + path])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def parse(ref, path):
    return json.loads(blob(ref, path))

guard_bytes = (REPO / GUARD / 'idle_guard.py').read_bytes()
dispatch_bytes = (REPO / GUARD / 'dispatch_driver.py').read_bytes()
harness_bytes = blob('00f2ca7f79696e65ce52458f074dd13b871da90c', 'research/scratch/submission_rehearsal.py')
human = 'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public'
bindings = {}

for mode, binding_file in [('five', 'CT_BINDING.disabled.json'), ('full', 'NATIVE_BINDING.disabled.json')]:
    path = BASE + mode
    manifest = parse(COMMIT, path + '/MANIFEST.sha256.json')
    for name, wanted in manifest['files'].items():
        check(mode + ' manifest ' + name, sha(blob(COMMIT, path + '/' + name)) == wanted)
    b = parse(COMMIT, path + '/' + binding_file)
    p = parse(COMMIT, path + '/EXPERIMENT_PROPOSAL.json')
    bindings[mode] = b
    raw_driver = blob(COMMIT, path + '/host_driver.py')
    tree = ast.parse(raw_driver)
    pins[mode] = {'binding': sha(blob(COMMIT, path + '/' + binding_file)), 'driver': sha(raw_driver),
                  'proposal': sha(blob(COMMIT, path + '/EXPERIMENT_PROPOSAL.json')),
                  'manifest': sha(blob(COMMIT, path + '/MANIFEST.sha256.json'))}
    for name, data in [('source_sha256', blob(b['source_commit'], b['source_git_path'])),
                       ('driver_sha256', raw_driver), ('prepare_sha256', (REPO / 'prepare.py').read_bytes()),
                       ('harness_sha256', harness_bytes), ('idle_guard_sha256', guard_bytes),
                       ('dispatch_driver_sha256', dispatch_bytes)]:
        check(mode + ' immutable ' + name, sha(data) == b[name] == p[name])
    check(mode + ' canonical import unchanged', all(blob(COMMIT, path + '/' + n) == blob('d9f372dc', path + '/' + n) for n in manifest['files']))
    check(mode + ' source and gates disabled', b['source_sha256'] == '3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f' and b['launch_authorized'] is False and b['stage_authorized'] is False and not any(p['gates'].values()) and p['operator_assignment'] == 'NOT_ASSIGNED')
    command = human + (' --num-steps 5' if mode == 'five' else '')
    launch_bytes = blob(COMMIT, path + '/package/launch-command.txt')
    provenance = parse(COMMIT, path + '/package/provenance.json')
    check(mode + ' exact human command', launch_bytes.decode().strip() == b['launch_command'] == p['launch_command'] == provenance['launch_command'] == command)
    check(mode + ' exact argv', b['training_command'] == p['argv'] == shlex.split(command)[1:])
    check(mode + ' package source pin', provenance['train_sha256'] == b['source_sha256'])
    check(mode + ' independent entry uses same lock', b['outer_command'] == ['/usr/bin/flock', '-n', '/home/ubuntu/.cache/trainium_freshstart/device0.lock', '/usr/bin/python3', '-I', '-B', b['driver_path']])
    check(mode + ' unchanged fixed seed budget LNC', b['training_environment']['NEURON_COMPETITION_R1_SEED'] == '42' and b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800' and b['training_environment']['NEURON_LOGICAL_NC_CONFIG'] == '2')
    check(mode + ' fixed memory model scope', p['parameters'] == 113256724 and p['parameter_tensors'] == 65 and p['sequence_length'] == 1024 and p['pack_factor'] == 32)
    check(mode + ' fresh caches requested', b['command'][-1] == '--fresh-caches' and b['training_environment']['NEURON_COMPILE_CACHE_URL'] == b['sandbox'] + '/caches/compiler')
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)], type_ignores=[]), 'exact-driver-definitions', 'exec', optimize=0), ns)
    for field, name in [('command', 'COMMAND' if mode == 'five' else 'TRAIN_COMMAND'), ('training_environment', 'ENV' if mode == 'five' else 'TRAIN_ENV')]:
        check(mode + ' bound literal ' + name, ns[name] == b[field])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'call']
    check(mode + ' immediate job child inherits fd', len(calls) == 1 and any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in calls[0].keywords))
    if mode == 'full':
        check('full eval literal binding', ns['EVAL_COMMAND'] == b['evaluation_command'] and ns['EVAL_ENV'] == b['evaluation_environment'])
        check('full separate cold eval exact20M', b['evaluation_command'][-3:] == ['--eval-tokens', '20971520', '--fresh-caches'] and b['evaluation_environment']['NEURON_COMPILE_CACHE_URL'] == b['eval_sandbox'] + '/caches/compiler')
        check('full separate prior cold gate pending', b['cold_gate']['required_before_full_launch'] is True and b['cold_gate']['result'] is None and b['cold_gate']['threshold_seconds'] == 900)
        check('full no automatic next', b['no_auto_next_job'] is True)
    else:
        check('cold five only no automatic full or eval', b['no_eval'] is True and b['no_automatic_full'] is True and b['hard_cold_boundary_seconds'] == 900)

    # Exact control flow, with all filesystem, subprocess, and host observations replaced.
    # No driver is imported as a module or executed against actual machine paths.
    top = compile(ast.Module(body=[next(n for n in tree.body if isinstance(n, ast.Try))], type_ignores=[]), 'exact-driver-control', 'exec', optimize=0)
    for label, case, wanted_calls, wanted_error in [
        ('success', {}, 1 if mode == 'five' else 2, None),
        ('optimized refuses', {'opt': 1}, 0, 'OPTIMIZED_DRIVER_FORBIDDEN'),
        ('wrong host', {'host': 'elsewhere'}, 0, 'HOSTNAME'),
        ('STOP', {'stop': True}, 0, 'STOP:'),
        ('low disk', {'disk': 0}, 0, 'DISK_'),
        ('source changed', {'bad_hash': '/package/train.py'}, 0, 'SOURCE_SHA'),
        ('launch changed', {'bad_hash': '/package/launch-command.txt'}, 0, 'LAUNCH_SHA'),
        ('provenance changed', {'bad_hash': '/package/provenance.json'}, 0, 'PROVENANCE_SHA'),
        ('prepare changed', {'bad_hash': b['prepare_path']}, 0, 'PREPARE_SHA'),
        ('harness changed', {'bad_hash': b['harness_path']}, 0, 'HARNESS_SHA'),
        ('guard changed', {'bad_hash': '/idle_guard.py'}, 0, 'IDLE_GUARD_SHA'),
        ('dispatch changed', {'bad_hash': '/dispatch_driver.py'}, 0, 'DISPATCH_SHA'),
        ('sandbox exists', {'sandbox': True}, 0, 'SANDBOX_ALREADY_EXISTS'),
        ('busy before train', {'busy': 1}, 0, 'FIXTURE_BUSY'),
    ] + ([] if mode == 'five' else [
        ('train failure', {'train_rc': 124}, 1, 'TRAIN_EXIT_124'),
        ('budget false', {'budget': False}, 1, 'TRAIN_BUDGET_FAILED'),
        ('missing budget', {'missing_budget': True}, 1, 'charged_through_process_exit_within_budget'),
        ('missing checkpoint', {'missing_checkpoint': True}, 1, 'MISSING_CHECKPOINT'),
        ('busy before eval', {'busy': 2}, 1, 'FIXTURE_BUSY'),
        ('different fd', {'changed_fd': True}, 1, 'LOCK_FD_CHANGED'),
        ('eval nonzero', {'eval_rc': 124}, 2, 'SystemExit:124'),
    ]):
        observed_calls, records, guard_calls = [], [], []
        class FakePath:
            def __init__(self, value): self.value = str(value)
            def __str__(self): return self.value
            def __truediv__(self, value): return FakePath(self.value + '/' + str(value))
            @property
            def parent(self): return FakePath(self.value.rsplit('/', 1)[0])
            def glob(self, _): return []
            def is_dir(self): return True
            def is_file(self): return not case.get('missing_checkpoint', False)
            def open(self, *_): return io.StringIO()
            def read_text(self):
                r = {'returncode': 0, 'checkpoint_sha256': 'fixture', 'checkpoint_bytes': 1}
                if not case.get('missing_budget'): r['charged_through_process_exit_within_budget'] = case.get('budget', True)
                return json.dumps(r)
        def lexists(path):
            return (case.get('stop', False) and str(path).endswith('/STOP')) or (case.get('sandbox', False) and str(path).endswith('/train'))
        def fake_sha(path):
            value = str(path)
            if case.get('bad_hash') and value.endswith(case['bad_hash']): return 'WRONG'
            lookup = {b['prepare_path']: b['prepare_sha256'], b['harness_path']: b['harness_sha256'],
                      b['root'] + '/package/train.py': b['source_sha256'],
                      b['root'] + '/package/launch-command.txt': sha(launch_bytes),
                      b['root'] + '/package/provenance.json': sha(blob(COMMIT, path_packet + '/package/provenance.json')),
                      b['root'] + '/dispatch_driver.py': b['dispatch_driver_sha256'],
                      b['root'] + '/idle_guard.py': b['idle_guard_sha256']}
            return lookup[value]
        path_packet = BASE + mode
        def fake_guard(*args):
            guard_calls.append(args)
            if len(guard_calls) == case.get('busy'): raise RuntimeError('FIXTURE_BUSY')
            return 45 + int(case.get('changed_fd', False) and len(guard_calls) == 2)
        def fake_call(command, **kw):
            observed_calls.append(command)
            if kw['pass_fds'] != (45,): raise RuntimeError('WRONG_INHERITED_FD')
            return case.get('train_rc', 0) if len(observed_calls) == 1 else case.get('eval_rc', 0)
        module = SimpleNamespace(require_idle_under_inherited_lock=fake_guard)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
        def stop_exit(code): raise SystemExit(code)
        context = dict(ns, ROOT=FakePath(b['root']), PREP=FakePath(b['prepare_path']), HARNESS=FakePath(b['harness_path']), Path=FakePath,
                       sha=fake_sha, record=lambda n, r: records.append((n, r)),
                       os=SimpleNamespace(path=SimpleNamespace(lexists=lexists), statvfs=lambda _: SimpleNamespace(f_bavail=case.get('disk', 30 * 1024**3), f_frsize=1)),
                       socket=SimpleNamespace(gethostname=lambda: case.get('host', 'ip-172-31-32-253')),
                       sys=SimpleNamespace(flags=SimpleNamespace(optimize=case.get('opt', 0)), exit=stop_exit),
                       importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *_: spec, module_from_spec=lambda _: module)),
                       subprocess=SimpleNamespace(call=fake_call, DEVNULL=-3, STDOUT=-2))
        # Rebind exact function globals to the synthetic context.
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('preflight', 'phase')]
        exec(compile(ast.Module(body=funcs, type_ignores=[]), 'exact-functions', 'exec', optimize=0), context)
        error = None
        try: exec(top, context)
        except SystemExit as exc: error = 'SystemExit:' + str(exc.code)
        except Exception as exc: error = str(exc)
        check(mode + ' scenario ' + label, len(observed_calls) == wanted_calls and ((error == 'SystemExit:0') if wanted_error is None else wanted_error in str(error)))
        if label == 'success':
            check(mode + ' all children under fresh guard', len(guard_calls) == wanted_calls and all(str(c[0]) == b['device_lock'] for c in guard_calls))
            check(mode + ' terminal is observation not acceptance', records[-1][1]['status'] == ('CT_HARNESS_EXITED' if mode == 'five' else 'COMPLETE_PENDING_INDEPENDENT_REVIEW'))

# Authenticated shared guard parser: our own independent malformed/busy fixtures.
guard_ns = {}
exec(compile(guard_bytes, 'exact-guard', 'exec', optimize=0), guard_ns)
ps_ok = '101 100 python3 /root/host_driver.py\n100 1 flock /usr/bin/flock -n /lock /usr/bin/python3 -I -B /root/host_driver.py\n1 0 init /sbin/init\n'
dev_ok = [{'neuron_device': 0, 'instance_type': 'trn2.3xlarge', 'nc_count': 4, 'logical_neuroncore_config': 2, 'neuroncore_ids': [0, 1, 2, 3], 'neuron_processes': []}]
for label, ps, dev, err in [
    ('idle', ps_ok, json.dumps(dev_ok), None),
    ('unrelated driver rejected', ps_ok + '201 1 python3 /elsewhere/host_driver.py\n', json.dumps(dev_ok), 'PS_NATIVE_JOB_BUSY'),
    ('torchrun rejected', ps_ok + '202 1 torchrun /bin/torchrun train.py\n', json.dumps(dev_ok), 'PS_NATIVE_JOB_BUSY'),
    ('malformed neuron JSON', ps_ok, '{}', 'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'),
    ('occupied device', ps_ok, json.dumps([dict(dev_ok[0], neuron_processes=[{'pid': 33}])]), 'NEURON_DEVICE_BUSY'),
]:
    seen, evidence = [], {}
    def runner(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, ps if argv == guard_ns['PS'] else dev, '')
    error = None
    try: guard_ns['check_idle'](runner, 101, 100, evidence)
    except Exception as exc: error = str(exc)
    check('guard ' + label, error == err and len(seen) == 2)
with tempfile.TemporaryDirectory(prefix='clean-binding-local-lock-') as directory:
    lock = Path(directory) / 'device0.lock'
    lock.touch()
    error = None
    try: guard_ns['inherited_lock'](lock, Path(directory) / 'host_driver.py')
    except RuntimeError as exc: error = str(exc)
    check('direct invocation fails exact parent authentication', error == 'OUTER_FLOCK_COMMAND_MISMATCH')

# Execute the pinned rehearsal's exact arithmetic, not a rewritten budget formula.
main = next(n for n in ast.parse(harness_bytes).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
budget = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'train' and process.returncode == 0")
accounting = compile(ast.Module(body=budget.body[3:], type_ignores=[]), 'exact-budget', 'exec', optimize=0)
for wall, fifth, charge, passed in [(2699.9, 1200, 1799.9, True), (2700.1, 1200, 1800.1, False), (1900, 100, 1800, True), (1900.1, 100, 1800.1, False)]:
    ctx = {'observed': [{'observed_seconds': fifth}] * 5, 'result': {'wall_seconds': wall}, 'checkpoint': SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1)), 'sha': lambda _: 'fixture'}
    exec(accounting, ctx)
    check('through-exit budget ' + str((wall, fifth)), ctx['result']['charged_through_process_exit_within_budget'] is passed and abs(ctx['result']['charged_through_process_exit'] - charge) < 1e-9)
eval_if = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'eval'")
cmd = next(n.value for n in eval_if.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
constants = [n.value for n in cmd.elts if isinstance(n, ast.Constant)]
check('actual unchanged evaluator context1024 batch1 timeout1200', constants == ['prepare.py', 'eval-public', '--train-py', 'train.py', '--checkpoint', '--device', 'neuron', '--batch-size', '1', '--seq-len', '1024', '--eval-tokens', '--timeout-seconds', '1200'])
fresh_if = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.fresh_caches')
cache_for = next(n for n in fresh_if.body if isinstance(n, ast.For))
check('exact harness six SDK dirs only', ast.literal_eval(cache_for.iter) == (('TORCH_NEURONX_NEFF_CACHE_DIR','neff'), ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','local'), ('TORCH_NEURONX_HLO_CACHE_DIR','hlo'), ('TORCHINDUCTOR_CACHE_DIR','inductor'), ('NEURON_COMPILE_CACHE_DIR','compiler'), ('TORCH_NEURONX_DEBUG_DIR','debug')))
check('fresh harness refuses existing dirs', 'exist_ok=False' in ast.unparse(fresh_if))
print(json.dumps({'verdict': 'PASS_DISABLED_BINDING_MECHANICS_ONLY', 'checks': len(checks), 'tests': checks, 'commit': COMMIT, 'pins': pins,
                  'source_gate': 'REFUSE unchanged', 'launch_authorized': False, 'scope': 'stdlib immutable pins, exact mocked control/refusals, real guard parsers/direct-parent refusal, exact harness arithmetic; no tensors or hosts'}, indent=2))
