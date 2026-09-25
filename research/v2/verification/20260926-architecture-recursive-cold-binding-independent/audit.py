"""Independent Recursive cold-five descriptor and real-driver refusal tests. No hosts."""
import ast
import copy
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import types

REPO=pathlib.Path('/home/poojith-devan/Desktop/Trainium')
PACKET_PATH='research/v2/operator/20260926-trn21-architecture-recursive-cold-five'
PACKET_COMMIT='276e4cca'
PACKET=REPO/PACKET_PATH
rows=[]
def check(label,ok):
    if not ok:raise RuntimeError(label)
    rows.append(label)
def sha(data):return hashlib.sha256(data).hexdigest()
def git(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
binding_bytes=(PACKET/'CT_BINDING.disabled.json').read_bytes()
b=json.loads(binding_bytes);p=json.loads((PACKET/'EXPERIMENT_PROPOSAL.json').read_text())
driver=(PACKET/'host_driver.py').read_bytes()
for file in ('CT_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','host_driver.py','README.md','check_packet.py','operator-mechanical-checks.json'):
    check('frozen packet '+file,(PACKET/file).read_bytes()==git(PACKET_COMMIT,PACKET_PATH+'/'+file))
check('frozen binding pin',sha(binding_bytes)=='9b1b0c0185167d670157c4abdb0b5cb7b4af33f21ecdc7af854d668dfae06185')
guard_dir=REPO/'research/v2/operator/20260926-common-host-idle-guard'
for field,data in (
 ('source_sha256',git(b['source_commit'],b['source_git_path'])),
 ('driver_sha256',driver),('prepare_sha256',(REPO/'prepare.py').read_bytes()),
 ('harness_sha256',git('00f2ca7f','research/scratch/submission_rehearsal.py')),
 ('idle_guard_sha256',(guard_dir/'idle_guard.py').read_bytes()),
 ('dispatch_driver_sha256',(guard_dir/'dispatch_driver.py').read_bytes())):
    check('exact pin '+field,sha(data)==b[field])
old_path=REPO/b['driver_derivation']['directory']
old=json.loads((old_path/'CT_BINDING.disabled.json').read_text())
old_driver=(old_path/'host_driver.py').read_bytes()
check('exact reviewed template',sha(old_driver)==b['driver_derivation']['driver_sha256']=='03ba9c6b200a0c5ff34b5ac3a32fc90940d3f24fc2776671f304a8e057904d86')
tree=ast.parse(driver);old_tree=ast.parse(old_driver)
executable=lambda t:ast.dump(ast.Module(body=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.Try))],type_ignores=[]))
check('entire executable body unchanged',executable(tree)==executable(old_tree))
ns={}
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.Try)],type_ignores=[]),'exact-driver-definitions','exec',optimize=0),ns)
for key,field in [('ROOT','root'),('LABEL','label'),('PREP','prepare_path'),('HARNESS','harness_path'),('SOURCE_SHA','source_sha256'),('COMMAND','command'),('ENV','training_environment')]:
    value=str(ns[key]) if isinstance(ns[key],pathlib.Path) else ns[key]
    check('exact driver literal '+key,value==b[field])
cpu=json.loads((REPO/'research/v2/proposals/20260926-architecture-recursive-exact-cpu-trn21/on/request.json').read_text())
full=cpu['argv'].copy();full[full.index('--num-steps')+1]='5'
def validate(x):
    require=lambda ok,name: None if ok else (_ for _ in ()).throw(RuntimeError(name))
    require(x['host']=='trn21' and x['expected_hostname']=='ip-172-31-34-85','host')
    require(x['stage_authorized'] is x['launch_authorized'] is x['native_launch_authorized'] is False,'disabled')
    require(x['training_command']==old['training_command']+['--recursive-schedule'] and x['training_command'][4:]==full,'science')
    require(x['source_sha256']==cpu['train_py_sha256'] if 'train_py_sha256' in cpu else x['source_sha256']=='75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f','source')
    require(x['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','1800s'] and x['command'][-3:]==['--mode','train','--fresh-caches'],'cold invocation')
    require(x['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',x['driver_path']],'lock')
    require(x['training_environment']['NEURON_COMPILE_CACHE_URL']==x['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR']==x['sandbox']+'/caches/compiler','cache')
    cache=[v for k,v in x['harness_injected_environment'].items() if k.endswith('CACHE_DIR') or k.endswith('DEBUG_DIR')]
    require(len(set(cache))==6 and all(v.startswith(x['sandbox']+'/caches/') for v in cache),'six caches')
    require(x['cold_decision']['maximum_seconds']==900 and x['cold_decision']['target_seconds']==870,'cold limit')
    require(x['no_eval'] is x['no_automatic_full'] is True and x['source_gate']['fullsize_cpu_result']=='PENDING','pending gate')
validate(b);check('positive frozen binding',True)
for label,change in [
 ('host',lambda x:x.update(host='trn22')),
 ('disabled',lambda x:x.update(launch_authorized=True)),
 ('science',lambda x:x['training_command'].__setitem__(x['training_command'].index('--num-steps')+1,'6')),
 ('source',lambda x:x.update(source_sha256='0'*64)),
 ('cold invocation',lambda x:x['command'].pop()),
 ('lock',lambda x:x['outer_command'].__setitem__(2,'/tmp/other.lock')),
 ('cache',lambda x:x['training_environment'].update(NEURON_COMPILE_CACHE_URL='/tmp/other')),
 ('six caches',lambda x:x['harness_injected_environment'].update(TORCH_NEURONX_DEBUG_DIR='/tmp/other')),
 ('cold limit',lambda x:x['cold_decision'].update(maximum_seconds=901)),
 ('pending gate',lambda x:x.update(no_automatic_full=False))]:
    altered=copy.deepcopy(b);change(altered);error=None
    try:validate(altered)
    except RuntimeError as exc:error=str(exc)
    check('descriptor refuses '+label,error==label)
check('proposal argv matches',p['argv']==b['training_command'] and p['source_sha256']==b['source_sha256'])
check('fixed assertion-enabled command','-O' not in b['outer_command'] and 'PYTHONOPTIMIZE' not in b['training_environment'])
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

harness = ast.parse(git('00f2ca7f', 'research/scratch/submission_rehearsal.py'))
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
                  'source_sha256': b['source_sha256'], 'scope': 'Mechanical Recursive cold-five binding only; full113M results, actual first-five graph execution and separate release remain pending'}, indent=2))
