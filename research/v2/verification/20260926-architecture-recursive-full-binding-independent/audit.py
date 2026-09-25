"""Own stdlib-only frozen full binding/negative tests; no host or model execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
REF='7172b012c0e10e8f429c8b48755c1b03495a63c6'
PACKET='research/v2/proposals/20260926-architecture-recursive-full-trn21'
GUARD='research/v2/operator/20260926-common-host-idle-guard'
PARENT='af53137fb8a8dc547572a1552a6bbf91d090e2b6'
OLD='research/v2/operator/20260926-trn21-arch3e79-replicate'
SNAPSHOT='3a976019fec9fcf401a5139ba18f67a7e2c0ccac'
rows=[]
def check(name,condition):
    if not condition:raise RuntimeError('FAILED: '+name)
    rows.append(name)
def git_bytes(ref,path):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(data):return hashlib.sha256(data).hexdigest()
def packet(name):return git_bytes(REF,PACKET+'/'+name)
binding_bytes=packet('NATIVE_BINDING.disabled.json');proposal_bytes=packet('EXPERIMENT_PROPOSAL.json');driver_bytes=packet('host_driver.py')
b=json.loads(binding_bytes);p=json.loads(proposal_bytes);tree=ast.parse(driver_bytes)
old=json.loads(git_bytes(PARENT,OLD+'/NATIVE_BINDING.disabled.json'))
old_driver=git_bytes(PARENT,OLD+'/host_driver.py')
check('frozen binding',sha(binding_bytes)=='c8503bf7aeb06cc4f59d289eba73def9ebd4a59e46dd804a8a16b74ebc425035')
check('frozen driver',sha(driver_bytes)==b['driver_sha256']=='0b2888e84670f394847783b93256d3cfa98e9328113072db36aa7215562d6213')
check('frozen manifest',sha(packet('MANIFEST.sha256.json'))=='a8a5a02db8bf9a58e1aac24f5c180ca0c4a8256a5ada47b3e3fda0500d826020')
for name,pin in json.loads(packet('MANIFEST.sha256.json'))['files'].items():check('manifest '+name,sha(packet(name))==pin)
for field,data in [
('source_sha256',git_bytes(b['source_commit'],b['source_git_path'])),
('prepare_sha256',git_bytes(SNAPSHOT,'prepare.py')),
('harness_sha256',git_bytes('00f2ca7f','research/scratch/submission_rehearsal.py')),
('idle_guard_sha256',git_bytes(SNAPSHOT,GUARD+'/idle_guard.py')),
('dispatch_driver_sha256',git_bytes(SNAPSHOT,GUARD+'/dispatch_driver.py'))]:
    check('exact '+field,sha(data)==b[field])
check('exact source pin',b['source_sha256']=='75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f' and b['source_commit']=='fe0969fd311d654e143dc6113614416c8970a424')
mapped=old_driver.decode().replace(old['label'],b['label']).replace(old['source_sha256'],b['source_sha256']).replace('Exact cross-host architecture replicate then fresh public20M under one lifetime lock.','Exact architecture Recursive stack then fresh public20M under one lifetime lock.')
check('whole driver literal-only successor',mapped.encode()==driver_bytes and sha(old_driver)==b['derivation']['parent_driver_sha256'])
for field in ('command','evaluation_command','training_environment','evaluation_environment','outer_command','prepare_path','harness_path','device_lock','budgets','evaluation'):
    check('unchanged mechanics '+field,b[field]==json.loads(json.dumps(old[field]).replace(old['label'],b['label'])))
check('only scientific flag',b['training_command']==old['training_command']+['--recursive-schedule']==p['argv'])
for name,obj in [('binding',b),('proposal',p)]:
    check('disabled '+name,all(obj[key] is False for key in ('stage_authorized','launch_authorized','source_adoption','submission_authorized')))
check('host and no automatic job',b['host']=='trn21' and b['host_reserved'] is False and b['no_auto_next_job'] is True)
prov=json.loads(packet('package/provenance.json'))
check('launch bytes/payload source',shlex.split(packet('package/launch-command.txt').decode())==b['training_command']==shlex.split(prov['launch_command']) and prov['train_sha256']==b['source_sha256'] and prov['source_commit']==b['source_commit'])
for name,pin in b['package_support_sha256'].items():check('package '+name,sha(packet('package/'+name))==pin)
check('no optimized production entry',b['outer_command'][:6]==['/usr/bin/flock','-n',b['device_lock'],'/usr/bin/python3','-I','-B'] and '-O' not in b['dispatch_command'] and '-O' not in b['command'] and '-O' not in b['evaluation_command'])
check('three budgets distinct',b['budgets']['source_seconds']==1800 and b['budgets']['startup_cap_seconds']==900 and b['budgets']['train_outer_seconds']==2730 and b['budgets']['eval_outer_seconds']==1800 and b['budgets']['kill_grace_seconds']==60)
check('fresh 20M B1 context1024',b['evaluation']==dict(fresh_process=True,tokens=20971520,seq_len=1024,batch_size=1,device='neuron',prepare_unchanged=True,same_lifetime_lock=True,new_guard_before_eval=True))
check('no fresh-cache claim or reset', '--fresh-caches' not in b['command']+b['evaluation_command'] and 'not a fresh-cache' in b['cache_policy'])
for name in ('training_environment','evaluation_environment'):
    env=b[name]
    check('seed/LNC '+name,env['NEURON_COMPETITION_R1_SEED']=='42' and env['NEURON_LOGICAL_NC_CONFIG']=='2' and env['NEURON_COMPILE_CACHE_URL']==env['NEURON_COMPILE_CACHE_DIR'])
    check('clean explicit environment '+name,not any('TOKEN' in key or 'SECRET' in key or 'KEY' in key or key in ('PYTHONPATH','PYTHONOPTIMIZE') for key in env))
ledger=[json.loads(x) for x in git_bytes(SNAPSHOT,'research/v2/ledger.jsonl').decode().splitlines() if x.strip()]
for key,kind,score in [('raw_parent_contrast','architecture_arm',.9931291951444726),('decision_comparator','schedule_arm',.9900560897366009)]:
    named=b[key];row=next(r for r in ledger if r['label']==named['label'])
    check('literal historical '+key,row['kind']==kind and row['host']==named['host']=='trn21' and row['train_py_sha256']==named['source_sha256'] and row['public_20m']==named['public20m_bpb']==score and row['base_sha256'].startswith('c5c410'))
check('parent discarded not adopted',next(r for r in ledger if r['label']==b['raw_parent_contrast']['label'])['verdict']=='DISCARD' and 'NOT an adopted' in b['raw_parent_contrast']['role'])
adopt=json.loads(git_bytes(SNAPSHOT,b['decision_comparator']['adoption_record']))
check('confirmed Recursive actual comparator',adopt['source_sha256']==b['decision_comparator']['source_sha256'] and adopt['public_20m'][0]==b['decision_comparator']['public20m_bpb'] and adopt['result_labels'][0]==b['decision_comparator']['label'])
check('proposal contrasts consistent',p['primary_comparator']==b['decision_comparator'] and p['secondary_comparator']==b['raw_parent_contrast'])
check('no invented control row',all('eligibility' in b[key] for key in ('raw_parent_contrast','decision_comparator')) and 'same_host_control' not in b)
cold=b['cold_gate'];cb=json.loads(git_bytes(cold['raw_commit'],cold['archive']));entry=cb['files'][cold['receipt_key']];cr=json.loads(entry['text'])
check('pinned cold evidence separate',sha(entry['text'].encode())==entry['sha256']==cold['receipt_sha256'] and cr['returncode']==0 and cr['step_boundaries']==5 and cr['observations'][4]['observed_seconds']==cold['observed_uncapped_fifth_seconds']<=900)
check('cold source same',json.loads(cb['files']['preflight.json']['text'])['source_sha256']==b['source_sha256'])
check('cold wall separate from future full',abs(cr['wall_seconds']-min(cr['observations'][4]['observed_seconds'],900)-cold['charged_through_exit_seconds'])<1e-9 and len(cr['fresh_empty_sdk_caches'])==6 and b['source_gate']['source_sha256']==b['source_sha256'])

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
with tempfile.TemporaryDirectory(prefix='arch-replicate-binding-review-') as tmp:
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
check('eval full public20M binding', b['evaluation_command'][-2:] == ['--eval-tokens', '20971520'] and b['evaluation']['seq_len'] == 1024)


result={'status':'PASS_MECHANICAL_DISABLED_FULL_BINDING_ONLY','checks':len(rows),'tests':rows,'source_packet_commit':REF,'input_hashes':{'binding':sha(binding_bytes),'proposal':sha(proposal_bytes),'driver':sha(driver_bytes)},'control_snapshot':SNAPSHOT,'scope':'stdlib exact binding / own mocked refusal / original harness arithmetic only; no source-science review, provider, host, native or tensors','source_adoption':False,'launch_authorized':False,'submission_authorized':False}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

