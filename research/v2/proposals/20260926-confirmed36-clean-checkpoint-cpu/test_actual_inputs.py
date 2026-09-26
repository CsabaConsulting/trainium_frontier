"""Local stdlib-only actual artifact binding checks, not numerical acceptance."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];checks=[]
def need(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)
rawpath=REPO/'research/v2/operator/20260926-trn21-confirmed36-clean-cpu/completed/results-bundle.json'
rawbytes=rawpath.read_bytes();raw=json.loads(rawbytes)
b=json.loads((HERE/'OPERATOR_BINDING.disabled.json').read_bytes());pending=json.loads((HERE/'OPERATOR_BINDING.pending.json').read_bytes())
it=(HERE/'INPUTS.json').read_bytes();inputs=json.loads(it);ih=hashlib.sha256(it).hexdigest()
need(hashlib.sha256(rawbytes).hexdigest()==b['fullg2_raw_sha256'],'immutable actual raw SHA')
need(ih==b['actual_inputs_sha256'],'actual INPUTS SHA')
need(inputs['status']=='FROZEN_REAL_INPUTS' and inputs['root']==raw['root']==b['old_readonly_root'],'real pair root')
need(len(inputs['files'])==17,'exact17 actual input files')
for name,pin in inputs['files'].items():
    actual=raw['retained_input_identities'][name]
    need(pin=={'sha256':actual['sha256'],'bytes':actual['bytes']} and actual['stable'] is True,'unchanged actual file '+name)
for arm in ['reference','candidate']:
    c=raw['checkpoint_records'][arm]['record'];i=inputs['arms'][arm]
    need(i=={'checkpoint_path':c['path'],'checkpoint_sha256':c['sha256'],'checkpoint_bytes':c['bytes']},'exact retained '+arm)
    need(b['checkpoint_paths'][arm]==c['path'] and b['checkpoint_sha256'][arm]==c['sha256'] and b['checkpoint_bytes'][arm]==c['bytes'],'binding retained '+arm)
before=(HERE/'host_driver.disabled.py').read_text();after=(HERE/'host_driver.py').read_text()
need(after==before.replace("'INPUTS.json':None","'INPUTS.json':'"+ih+"'"),'whole driver only real input hash filled')
ast.parse(after)
need(hashlib.sha256(after.encode()).hexdigest()==b['driver_sha256'],'actual driver SHA')
need(b['command_template'][:-1]==pending['command_template'][:-1] and b['command_template'][-1]==ih,'command only real input hash filled')
for key in ['helper_commit','helper_git_path','helper_sha256','original_lock','resources','prepare_path','prepare_sha256','idle_guard_sha256','dispatch_driver_sha256','outer_command_template','detached_command_template']:
    need(b[key]==pending[key],'no other runtime change '+key)
for key in ['stage_authorized','launch_authorized','native_launch_authorized','source_adoption','host_reserved']:
    need(b[key] is False,'still disabled '+key)
need(raw['postflight']['all_observed_pids_absent'] and not raw['postflight']['related_rows'] and raw['postflight']['neuron_device_idle'] and raw['postflight']['lock_competitor_rc']==0,'archived explicit postflight not fresh preflight')
print(json.dumps({'status':'ACTUAL_BINDING_MECHANICS_PASS_NOT_NUMERIC_ACCEPTANCE','checks':len(checks),'inputs_sha256':ih,'driver_sha256':b['driver_sha256'],'launch_authorized':False,'host_calls':False,'tensors':False},indent=2))
