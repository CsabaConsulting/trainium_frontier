"""Independent stdlib mechanical review; no hosts, training or tensors."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
REF='a50e680247954c659d9f8deed7d3bce470cf1034'
PACKET='research/v2/operator/20260926-arch3e79-new-base-controls'
GUARD='research/v2/operator/20260926-common-host-idle-guard'
SOURCE='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
rows=[]; reports={}; active_host='common'
def check(name,ok):
    if not ok:raise RuntimeError('FAIL: '+active_host+': '+name)
    rows.append(active_host+': '+name)
def require(ok,name):
    if not ok:raise RuntimeError(name)
def git_bytes(ref,path):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(data):return hashlib.sha256(data).hexdigest()
def data(path):return json.loads(git_bytes(REF,path))

files=subprocess.check_output(['git','-C',str(REPO),'diff-tree','--no-commit-id','--name-only','-r',REF],text=True).splitlines()
check('packet-only commit',len(files)==9 and all(p.startswith(PACKET+'/') for p in files))
for path in files:check('raw/canonical identity '+path,git_bytes(REF,path)==git_bytes('9c2711d2',path))
ledger=git_bytes(REF,'research/v2/ledger.jsonl')
check('no ledger mutation',ledger==git_bytes(REF+'^','research/v2/ledger.jsonl'))
ledger_labels={r['label'] for r in map(json.loads,ledger.decode().splitlines()) if r}
fixture_path='research/v2/verification/20260926-arch3e79-trn21-binding-independent/test_binding.py'
fixture=git_bytes('6d694d95',fixture_path).decode()
# Reuse this Verifier's own previously authored exact preflight/control-flow,
# real guard parser/direct-invocation and capped-budget fixtures. Inputs below
# are this packet's exact two drivers; no Operator test is used as approval.
fixture_tail=fixture.split('# Exact preflight function with controlled filesystem observations, not host access.\n',1)[1].split("result = {'status': 'PASS'",1)[0]
fixture_code=compile(fixture_tail,'independent-existing-refusal-fixtures','exec',optimize=0)
templates={
 'trn21':('af53137f','research/v2/operator/20260926-trn21-arch3e79-replicate','b4647403e8338cea1bc9d16b7077315d9f3705e29c4bb364a869ee30f398fe6c'),
 'trn22':('83db5fe3','research/v2/operator/20260926-trn22-arch3e79-full','c8957e5dbd67bbf017510d6110bcd0e0ab09d2183db098d18654a54f5498e7e0')}

def validate_binding(host,b,p,driver,old,old_driver):
    label='fs-20260926-arch3e79-base-control-'+host+'-r1-full'
    require(b['host']==p['host']==host,'HOST_BINDING')
    require(b['label']==p['label']==label and label not in ledger_labels and label!=old['label'],'NEW_LABEL')
    require(b['stage_authorized'] is b['launch_authorized'] is p['launch_authorized'] is False,'DISABLED')
    require(b['source_sha256']==p['source_sha256']==b['base_sha256']==p['base_sha256']==SOURCE,'SOURCE_BASE')
    require(b['kind']==p['kind']=='control','CONTROL_KIND')
    identity={'host':host,'label':label,'kind':'control','base_sha256':SOURCE,'source_sha256':SOURCE}
    require(b['result_row_identity']==p['result_row_identity']==identity,'NEW_ROW_IDENTITY')
    require(p['primary_comparator'] is p['secondary_comparator'] is None and b['only_science_delta']==p['only_science_delta']=={},'NO_RELABEL_OR_SCIENCE_CHANGE')
    require(driver.decode()==old_driver.decode().replace(old['label'],label),'DRIVER_LABEL_ONLY')
    require(sha(driver)==b['driver_sha256']==p['driver_sha256'],'DRIVER_PIN')
    require(b['training_command']==p['argv']==old['training_command'],'SCIENTIFIC_ARGV')
    for name in ('source_commit','source_git_path','source_sha256','prepare_path','prepare_sha256','harness_path','harness_sha256','idle_guard_sha256','dispatch_driver_sha256','budgets','evaluation','cold_gate'):
        require(b[name]==old[name],'UNCHANGED_'+name)
    require(b['device_lock']=='/home/ubuntu/.cache/trainium_freshstart/device0.lock','COMMON_LOCK')
    for name in ('root','package','sandbox','eval_sandbox','driver_path','command','evaluation_command','outer_command','training_environment','evaluation_environment'):
        require(b[name]==json.loads(json.dumps(old[name]).replace(old['label'],label)),'BOUND_'+name)
    require(b['outer_command']==['/usr/bin/flock','-n',b['device_lock'],'/usr/bin/python3','-I','-B',b['driver_path']],'OUTER_NORMAL_PYTHON')
    require(b['no_auto_next_job'] is True and b['old_queue']=='PAUSED_AND_EXITED; no re-arm','NO_DUAL_LAUNCHER')
    for env in (b['training_environment'],b['evaluation_environment']):
        require('PYTHONOPTIMIZE' not in env and env['NEURON_COMPILE_CACHE_URL']==env['NEURON_COMPILE_CACHE_DIR'],'ENV_ASSERTIONS_AND_CACHE')
    require(b['historical_context']['original_architecture']['kind']=='architecture_arm','HISTORY_NOT_CONTROL')
    require(not any(k in b for k in ('public_20m','score','steps','same_host_control','cross_host_original')),'NO_RESULT_FABRICATION')

for active_host,(old_ref,old_path,old_sha) in templates.items():
    path=PACKET+'/'+active_host
    binding_bytes=git_bytes(REF,path+'/NATIVE_BINDING.disabled.json')
    proposal_bytes=git_bytes(REF,path+'/EXPERIMENT_PROPOSAL.json')
    driver_bytes=git_bytes(REF,path+'/host_driver.py')
    b=json.loads(binding_bytes);p=json.loads(proposal_bytes)
    old=json.loads(git_bytes(old_ref,old_path+'/NATIVE_BINDING.disabled.json'));old_driver=git_bytes(old_ref,old_path+'/host_driver.py')
    check('authentic already-reviewed same-host template',sha(old_driver)==old_sha==b['preparation_template']['driver_sha256'])
    validate_binding(active_host,b,p,driver_bytes,old,old_driver)
    check('positive exact binding',True)
    mutations=[
      ('wrong host',lambda x,y:x.update(host='wrong'),'HOST_BINDING'),
      ('reused old label',lambda x,y:x.update(label=old['label']),'NEW_LABEL'),
      ('source drift',lambda x,y:x.update(source_sha256='0'*64),'SOURCE_BASE'),
      ('old base relabel',lambda x,y:x.update(base_sha256='c5c'),'SOURCE_BASE'),
      ('arm instead of control',lambda x,y:x.update(kind='architecture_arm'),'CONTROL_KIND'),
      ('old row reuse',lambda x,y:x['result_row_identity'].update(label=old['label']),'NEW_ROW_IDENTITY'),
      ('proposal source drift',lambda x,y:y.update(source_sha256='bad'),'SOURCE_BASE'),
      ('driver pin drift',lambda x,y:x.update(driver_sha256='0'*64),'DRIVER_PIN'),
      ('matrix LR change',lambda x,y:x['training_command'].__setitem__(x['training_command'].index('--matrix-lr')+1,'0.012'),'SCIENTIFIC_ARGV'),
      ('Recursive addition',lambda x,y:x['training_command'].append('--recursive-schedule'),'SCIENTIFIC_ARGV'),
      ('per-run lock',lambda x,y:x.update(device_lock='/tmp/other.lock'),'COMMON_LOCK'),
      ('direct driver bypass',lambda x,y:x.update(outer_command=['python3',x['driver_path']]),'BOUND_outer_command'),
      ('other host cache',lambda x,y:x['training_environment'].update(NEURON_COMPILE_CACHE_DIR='/tmp/cache'),'BOUND_training_environment'),
      ('evaluation target drift',lambda x,y:x['evaluation_command'].__setitem__(-1,'2097152'),'BOUND_evaluation_command'),
      ('second launcher allowed',lambda x,y:x.update(old_queue='REARM'),'NO_DUAL_LAUNCHER'),
      ('auto next',lambda x,y:x.update(no_auto_next_job=False),'NO_DUAL_LAUNCHER'),
      ('armed packet',lambda x,y:x.update(launch_authorized=True),'DISABLED')]
    for label,mutate,wanted in mutations:
        bad,prop=copy.deepcopy(b),copy.deepcopy(p);mutate(bad,prop);error=None
        try:validate_binding(active_host,bad,prop,driver_bytes,old,old_driver)
        except Exception as exc:error=str(exc)
        check('binding refuses '+label,error==wanted)
    for field,content in [
      ('source_sha256',git_bytes(b['source_commit'],b['source_git_path'])),
      ('prepare_sha256',(REPO/'prepare.py').read_bytes()),
      ('harness_sha256',git_bytes('00f2ca7f','research/scratch/submission_rehearsal.py')),
      ('idle_guard_sha256',(REPO/GUARD/'idle_guard.py').read_bytes()),
      ('dispatch_driver_sha256',(REPO/GUARD/'dispatch_driver.py').read_bytes())]:check('support hash '+field,sha(content)==b[field])
    tree=ast.parse(driver_bytes);ns={}
    exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.Try)],type_ignores=[]),'exact-driver-definitions','exec',optimize=0),ns)
    for name,field in [('ROOT','root'),('PREP','prepare_path'),('HARNESS','harness_path'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:
        value=str(ns[name]) if isinstance(ns[name],Path) else ns[name]
        check('driver literal '+name,value==b[field])
    phase=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='phase')
    call=next(n for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='call')
    check('immediate child inherits original FD',any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in call.keywords))
    check('no close/unlock/new queue operations',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('close','flock','execv','execve','send_message_to_thread') for n in ast.walk(tree)))
    # The unchanged production command is explicitly non-optimized. Keep its
    # asserts enabled when these fixtures themselves run under python -O.
    exec(fixture_code,globals())
    reports[active_host]={'binding_sha256':sha(binding_bytes),'proposal_sha256':sha(proposal_bytes),'driver_sha256':sha(driver_bytes),'template_commit':old_ref,'template_driver_sha256':old_sha,'label':b['label']}
print(json.dumps({'verdict':'PASS_DISABLED_NEW_CONTROL_BINDINGS_ONLY','checks':len(rows),'tests':rows,'packet_commit':REF,'raw_commit':'9c2711d2','packets':reports,'own_reused_fixture_sha256':sha(fixture.encode()),'host_identity_scope':'binding validation only; Operator must independently verify actual host before staging','host_actions':False,'tensor_allocations':False,'ledger_mutations':False,'launch_authorized':False,'controls_completed':False},indent=2))
