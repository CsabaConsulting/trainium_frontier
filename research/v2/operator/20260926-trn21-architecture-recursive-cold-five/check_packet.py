"""Operator mechanical preparation checks only; not independent review or authority."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
CANONICAL=Path('/home/poojith-devan/Desktop/Trainium')
OLD=ROOT.parent/'20260926-ct-architecture-startup-only'
count=0
def check(ok,message):
    global count
    if not ok:raise RuntimeError(message)
    count+=1
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
b=json.loads((ROOT/'CT_BINDING.disabled.json').read_text())
q=json.loads((ROOT/'EXPERIMENT_PROPOSAL.json').read_text())
old=json.loads((OLD/'CT_BINDING.disabled.json').read_text())
source='75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f'
oldproject='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1'
newproject='/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2'
label='fs-20260926-ct-arch75b9-recursive-cold-trn21-r1'
def transform(text):return text.replace(old['label'],label).replace(oldproject,newproject).replace(old['source_sha256'],source)
check(b['label']==q['label']==label,'LABEL')
check(b['host']==q['host']=='trn21','HOST')
for obj in (b,q):
    for key in ('stage_authorized','launch_authorized','native_launch_authorized'):check(obj[key] is False,'DISABLED_'+key)
check(b['source_sha256']==q['source_sha256']==source,'SOURCE')
check(b['source_commit']==q['source_commit']=='fe0969fd311d654e143dc6113614416c8970a424','SOURCE_COMMIT')
raw=subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=CANONICAL)
check(hashlib.sha256(raw).hexdigest()==source,'SOURCE_BLOB')
check(sha(OLD/'host_driver.py')=='03ba9c6b200a0c5ff34b5ac3a32fc90940d3f24fc2776671f304a8e057904d86','AUTHENTIC_TEMPLATE')
driver=(ROOT/'host_driver.py').read_text()
check(driver==transform((OLD/'host_driver.py').read_text()),'ONLY_REQUESTED_DRIVER_CONSTANTS')
check(sha(ROOT/'host_driver.py')==b['driver_sha256']==q['driver_sha256'],'DRIVER_PIN')
check(b['training_command']==q['argv']==old['training_command']+['--recursive-schedule'],'REQUESTED_COLD_ARGV')
on=json.loads((CANONICAL/'research/v2/proposals/20260926-architecture-recursive-exact-cpu-trn21/on/request.json').read_text())
expected=on['argv'].copy();expected[expected.index('--num-steps')+1]='5'
check(b['training_command'][4:]==expected,'FULL_CPU_ON_PROFILE_ONLY_NUMSTEPS5')
for flag,value in [('--max-train-seconds','1800'),('--num-steps','5'),('--seed','42'),('--matrix-lr','0.015'),('--seq-len','1024')]:
    check(q['argv'][q['argv'].index(flag)+1]==value,'VALUE_'+flag)
for key in ('root','package','sandbox','prepare_path','harness_path','command','training_environment','harness_injected_environment','driver_path','outer_command','result','boundary_log'):
    check(json.dumps(b[key],sort_keys=True)==transform(json.dumps(old[key],sort_keys=True)),'ONLY_REQUESTED_BINDING_'+key)
for key in ('prepare_sha256','harness_sha256','idle_guard_sha256','dispatch_driver_sha256','device_lock','cache_setup','step_observation','no_eval','no_automatic_full'):
    check(b[key]==old[key],'PRESERVED_'+key)
check(b['training_environment']['NEURON_COMPILE_CACHE_URL']==b['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR'],'CACHE_URL_DIR')
caches=[v for k,v in b['harness_injected_environment'].items() if k.endswith('CACHE_DIR') or k=='TORCH_NEURONX_DEBUG_DIR']
check(len(caches)==6 and len(set(caches))==6 and all(v.startswith(b['sandbox']+'/caches/') for v in caches),'SIX_PRIVATE_CACHES')
check('--fresh-caches' in b['command'],'FRESH_HARNESS')
check(b['source_gate']['fullsize_cpu_result']=='PENDING' and q['fullsize_cpu_raw_commit'] is None,'NO_FAKE_CPU_GATE')
check(len(b['pending_gates'])==4 and b['pending_gates']==q['pending_gates'],'GATES')
check(b['cold_decision']['maximum_seconds']==900 and b['cold_decision']['target_seconds']==870,'UNCHANGED_COLD_GATE')
values={}
for node in ast.parse(driver).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        value=node.value
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='Path':value=value.args[0]
        values[node.targets[0].id]=ast.literal_eval(value)
for key,name in [('ROOT','root'),('PREP','prepare_path'),('HARNESS','harness_path'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('PREP_SHA','prepare_sha256'),('HARNESS_SHA','harness_sha256'),('COMMAND','command'),('ENV','training_environment')]:check(values[key]==b[name],'DRIVER_BINDING_'+key)
guard=ROOT.parent/'20260926-common-host-idle-guard'
check(sha(guard/'idle_guard.py')==b['idle_guard_sha256'],'GUARD')
check(sha(guard/'dispatch_driver.py')==b['dispatch_driver_sha256'],'DISPATCH')
print(json.dumps({'status':'OPERATOR_MECHANICAL_PASS','checks':count,'not_independent_review':True,'no_host_actions':True,'no_tensors':True,'binding_sha256':sha(ROOT/'CT_BINDING.disabled.json'),'proposal_sha256':sha(ROOT/'EXPERIMENT_PROPOSAL.json'),'driver_sha256':sha(ROOT/'host_driver.py')},sort_keys=True))
