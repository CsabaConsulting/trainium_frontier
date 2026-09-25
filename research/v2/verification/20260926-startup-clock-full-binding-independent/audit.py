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
REF='59d21dde1eb99443bf6b218af103202d4b2f78dd'
PACKET='research/v2/proposals/20260926-startup-adjusted-full-trn22'
GUARD='research/v2/operator/20260926-common-host-idle-guard'
PARENT='83db5fe36e9401fd77a19f59298e2e2aade889c0'
OLD='research/v2/operator/20260926-trn22-arch3e79-full'
SNAPSHOT='f4be8dd71007615bdf05770ca9ea641dc831f986'
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
check('frozen binding',sha(binding_bytes)=='91ed9e3f6234542b772c528c2b843ebe1e46fb8e3c4dc461ab8d480ee5e9d500')
check('frozen driver',sha(driver_bytes)==b['driver_sha256']=='27b4ac4ac93b7487a57b1a3b2d20c08152342d258cc84a276d65244fcc6c3067')
check('frozen manifest',sha(packet('MANIFEST.sha256.json'))=='aca8882400bd88354252f099b9c3c8f1bce1ddba54ecc466840745ee8213883a')
for name,pin in json.loads(packet('MANIFEST.sha256.json'))['files'].items():check('manifest '+name,sha(packet(name))==pin)
for field,data in [
('source_sha256',git_bytes(b['source_commit'],b['source_git_path'])),
('prepare_sha256',git_bytes(SNAPSHOT,'prepare.py')),
('harness_sha256',git_bytes('00f2ca7f','research/scratch/submission_rehearsal.py')),
('idle_guard_sha256',git_bytes(SNAPSHOT,GUARD+'/idle_guard.py')),
('dispatch_driver_sha256',git_bytes(SNAPSHOT,GUARD+'/dispatch_driver.py'))]:
    check('exact '+field,sha(data)==b[field])
check('exact source pin',b['source_sha256']=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb' and b['source_commit']=='7419927c7417b5666a984ec9c2b849feadabaaa9')
mapped=old_driver.decode().replace(old['label'],b['label']).replace(old['source_sha256'],b['source_sha256']).replace(b['derivation']['old_cold_label'],b['derivation']['new_cold_label'])
check('whole driver literal-only successor',mapped.encode()==driver_bytes and sha(old_driver)==b['derivation']['parent_driver_sha256'])
for field in ('command','evaluation_command','training_environment','evaluation_environment','outer_command','prepare_path','harness_path','device_lock','budgets','evaluation'):
    check('unchanged mechanics '+field,b[field]==json.loads(json.dumps(old[field]).replace(old['label'],b['label']).replace(b['derivation']['old_cold_label'],b['derivation']['new_cold_label'])))
check('only scientific flag',b['training_command']==old['training_command']+['--startup-adjusted-schedule']==p['argv'])
for name,obj in [('binding',b),('proposal',p)]:
    check('disabled '+name,all(obj[key] is False for key in ('stage_authorized','launch_authorized','source_adoption','submission_authorized')))
check('host and no automatic job',b['host']=='trn22' and b['host_reserved'] is False and b['no_auto_next_job'] is True)
prov=json.loads(packet('package/provenance.json'))
check('launch bytes/payload source',shlex.split(packet('package/launch-command.txt').decode())==b['training_command']==shlex.split(prov['launch_command']) and prov['train_sha256']==b['source_sha256'] and prov['source_commit']==b['source_commit'])
for name,pin in b['package_support_sha256'].items():check('package '+name,sha(packet('package/'+name))==pin)
check('no optimized production entry',b['outer_command'][:6]==['/usr/bin/flock','-n',b['device_lock'],'/usr/bin/python3','-I','-B'] and '-O' not in b['dispatch_command'] and '-O' not in b['command'] and '-O' not in b['evaluation_command'])
check('three budgets distinct',b['budgets']['source_seconds']==1800 and b['budgets']['startup_cap_seconds']==900 and b['budgets']['train_outer_seconds']==2730 and b['budgets']['eval_outer_seconds']==1800 and b['budgets']['kill_grace_seconds']==60)
check('fresh 20M B1 context1024',b['evaluation']==dict(fresh_process=True,tokens=20971520,seq_len=1024,batch_size=1,device='neuron',prepare_unchanged=True,same_lifetime_lock=True,new_guard_before_eval=True))
check('no fresh-cache claim or reset', '--fresh-caches' not in b['command']+b['evaluation_command'] and 'NOT a full true-cold replay' in b['cold_gate']['scope'])
for name in ('training_environment','evaluation_environment'):
    env=b[name]
    check('seed/LNC '+name,env['NEURON_COMPETITION_R1_SEED']=='42' and env['NEURON_LOGICAL_NC_CONFIG']=='2' and env['NEURON_COMPILE_CACHE_URL']==env['NEURON_COMPILE_CACHE_DIR'])
    check('clean explicit environment '+name,not any('TOKEN' in key or 'SECRET' in key or 'KEY' in key or key in ('PYTHONPATH','PYTHONOPTIMIZE') for key in env))
ledger_bytes=git_bytes(SNAPSHOT,'research/v2/ledger.jsonl')
check('historical ledger snapshot exact',b['decision_comparator']['ledger_commit']==SNAPSHOT and sha(ledger_bytes)==b['decision_comparator']['ledger_sha256'])
ledger=[json.loads(x) for x in ledger_bytes.decode().splitlines() if x.strip()]
for key,kind,score in [('raw_parent_contrast','architecture_arm',.9863403052402583),('decision_comparator','schedule_arm',.9902787975734985)]:
    named=b[key];row=next(r for r in ledger if r['label']==named['label'])
    check('literal historical '+key,row['kind']==kind and row['host']==named['host']=='trn22' and row['train_py_sha256']==named['source_sha256'] and row['public_20m']==named['public20m_bpb']==score and row['base_sha256'].startswith('c5c410'))
check('raw parent not promoted to adopted control', 'NOT an adopted' in b['raw_parent_contrast']['role'] and 'Not retrospectively a formal control' in b['raw_parent_contrast']['eligibility'])
adopt=json.loads(git_bytes(SNAPSHOT,b['decision_comparator']['adoption_record']))
check('confirmed Recursive actual comparator',adopt['source_sha256']==b['decision_comparator']['source_sha256'] and adopt['public_20m'][1]==b['decision_comparator']['public20m_bpb'] and adopt['result_labels'][1]==b['decision_comparator']['label'])
check('proposal contrasts consistent',p['primary_comparator']==b['decision_comparator'] and p['secondary_comparator']==b['raw_parent_contrast'])
check('no invented control row',all('eligibility' in b[key] for key in ('raw_parent_contrast','decision_comparator')) and 'same_host_control' not in b)
check('baseline scientific contrast explicitly limited','NOT a one-factor' in b['decision_comparator']['role'] and 'schedule_arm' in b['decision_comparator']['eligibility'])
cold=b['cold_gate']; cold_bytes=git_bytes(cold['packet_commit'],cold['binding']); cb=json.loads(cold_bytes)
check('prospective cold binding exact',sha(cold_bytes)==cold['binding_sha256']=='ddd5cfea65d4795c3d14e871f934fd07162412e56267253b94cdc27acbe90c31')
check('cold source/host/label same',cb['source_sha256']==b['source_sha256'] and cb['host']==b['host'] and cb['label']==cold['label'])
cold_argv=list(cb['training_command']); cold_argv[cold_argv.index('--num-steps')+1]='1000000'
check('cold/full science identical apart from step count',cold_argv==b['training_command'])
check('actual cold evidence remains absent',all(cold[k] is None for k in ('raw_commit','receipt_sha256','observed_uncapped_fifth_seconds','charged_through_exit_seconds','five_steps_exit0')) and 'PENDING' in cold['status'])
check('cold limits unchanged',cold['target_seconds']==870 and cold['threshold_seconds']==900)
for key,value in cb['harness_injected_environment'].items():
    if 'CACHE_DIR' in key or key=='TORCH_NEURONX_DEBUG_DIR':check('prospective cold cache binding '+key,b['training_environment'][key]==value)
sg=b['source_gate'];cpu=json.loads(git_bytes(sg['full_parameter_cpu_review_commit'],sg['full_parameter_cpu_review']))
check('CPU evidence source and scope',cpu['source_candidate']==b['source_sha256'] and cpu['status']=='PASS_FULL_PARAMETER_CPU_RESULT_ONLY' and cpu['parameters']==113256724 and cpu['tensors']==65 and cpu['updates_each']==3 and cpu['micro_batch']==[2,256] and cpu['native_correctness'] is False)
check('CPU raw evidence hash',sha(git_bytes(sg['full_parameter_cpu_raw_commit'],sg['full_parameter_cpu_raw']))==sg['full_parameter_cpu_raw_sha256'])
check('frozen proposal preceded packet',subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor','f12f7a6c',REF],check=False).returncode==0)
check('pending gates consistent across proposal',p['cold_gate']==cold and p['source_gate']==sg)
check('source not vendored or changed by packet','package/train.py' not in subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',REF,'--',PACKET]).decode())
check('no recursive or batch arm',not any(x in b['training_command'] for x in ('--recursive-schedule','--total-batch-size','--fp8-projections')))
for flag,value in [('--seed','42'),('--matrix-lr','0.015'),('--max-train-seconds','1800'),('--seq-len','1024'),('--n-embd','1024'),('--pack-factor','32'),('--attention-layers','0,2,4,6,7')]:
    check('scientific scalar '+flag,b['training_command'][b['training_command'].index(flag)+1]==value)

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
    def is_dir(self): return bad != 'CACHE'
for name, bad, expected_error in [('valid pins', None, None), ('source drift', 'SOURCE', 'SOURCE_SHA'), ('prepare drift', 'PREP', 'PREPARE_SHA'), ('harness drift', 'HARNESS', 'HARNESS_SHA'), ('STOP exists', 'STOP', 'STOP:'), ('disk low', 'DISK', 'DISK_HEADROOM'), ('cache missing', 'CACHE', 'CACHE_MISSING:')]:
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
exec(compile(git_bytes(SNAPSHOT,GUARD+'/idle_guard.py'), 'exact-idle-guard', 'exec', optimize=0), guard_ns)
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
with tempfile.TemporaryDirectory(prefix='startup-clock-full-binding-review-') as tmp:
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


result={'status':'PASS_MECHANICAL_DISABLED_FULL_BINDING_ONLY','checks':len(rows),'tests':rows,'source_packet_commit':REF,'input_hashes':{'binding':sha(binding_bytes),'proposal':sha(proposal_bytes),'driver':sha(driver_bytes)},'control_snapshot':SNAPSHOT,'scope':'stdlib exact binding / reused own control-flow fixtures and original harness arithmetic only; no source-science review, provider, host, native or tensors; cold evidence PENDING','source_adoption':False,'launch_authorized':False,'submission_authorized':False}
print(json.dumps(result,indent=2))
