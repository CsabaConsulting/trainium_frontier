"""Local stdlib/Git mechanics only; no scientific imports, host calls or tensors."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];checks=[]
def need(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def blob(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=REPO)
def sha(v):return hashlib.sha256(v).hexdigest()
b=json.loads((HERE/'OPERATOR_BINDING.pending.json').read_bytes())
d=(HERE/'host_driver.disabled.py').read_bytes();tree=ast.parse(d)
r=json.loads((HERE/'OPERATOR_DRIVER_DERIVATION.json').read_bytes())
parent=blob(r['parent_commit'],r['parent_driver_path'])
need(sha(parent)==r['parent_driver_sha256'],'proven driver parent')
s=parent.decode()
for a,z in r['literal_replacements']:
    need(a in s,'literal exists '+str(len(checks)));s=s.replace(a,z)
need(s.count(r['removed_old_four_arm_result_parser'])==1,'old four-arm parser exact deletion')
s=s.replace(r['removed_old_four_arm_result_parser'],r['replacement_status_scope_parser'])
need(s.encode()==d,'whole driver exact declared transform')
need(sha(d)==b['driver_sha256'],'disabled driver hash')
helper=blob(b['helper_commit'],b['helper_git_path'])
need(sha(helper)==b['helper_sha256']=='65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8','exact accepted helper bytes no execution')
path=b['helper_git_path'].rsplit('/',1)[0]+'/'
need(sha(blob(b['helper_commit'],path+'EXPERIMENT_PROPOSAL.json'))==b['proposal_sha256'],'proposal pin')
pending=blob(b['helper_commit'],path+'INPUTS.pending.json')
need(sha(pending)==b['pending_inputs_sha256'],'unchanged pending inputs pin')
need(json.loads(pending)['files'] is None,'actual file manifest absent not fabricated')
for field in ['actual_inputs_path','actual_inputs_sha256']:
    need(b[field] is None,'unknown '+field)
for field in ['checkpoint_paths','checkpoint_sha256','checkpoint_bytes']:
    need(b[field]=={'reference':None,'candidate':None},'unknown '+field)
for field in ['stage_authorized','launch_authorized','native_launch_authorized','source_adoption','host_reserved']:
    need(b[field] is False,'disabled '+field)
need(b['host']=='trn21' and b['expected_hostname']=='ip-172-31-34-85','exact host')
need(b['original_lock']=={'device':66305,'inode':596823,'path':'/home/ubuntu/.cache/trainium_freshstart/device0.lock'},'original lifetime lock')
need(b['old_readonly_root']=='/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1','exact retained pair root')
need(b['resources']['minimum_available_ram_bytes']==64*1024**3,'64GiB RAM')
need(b['resources']['minimum_free_disk_bytes']==int(10.5*1024**3) and b['resources']['additional_footprint_bytes']==1024**3,'disk plus extra footprint')
need(b['resources']['threads']==1 and b['resources']['timeout_seconds']==600 and b['resources']['kill_grace_seconds']==60,'unchanged bounded CPU resources')
for field,p in [('idle_guard_sha256','idle_guard.py'),('dispatch_driver_sha256','dispatch_driver.py')]:
    need(sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/p).read_bytes())==b[field],'unchanged support '+p)
need(b['command_template'][-5:]==[b['root']+'/compare_checkpoints.py','--inputs',b['root']+'/INPUTS.json','--inputs-sha256',None],'exact pending helper CLI')
func={n.name:ast.get_source_segment(d.decode(),n) for n in tree.body if isinstance(n,ast.FunctionDef)}
pt=ast.parse(parent);pf={n.name:ast.get_source_segment(parent.decode(),n) for n in pt.body if isinstance(n,ast.FunctionDef)}
need(func['input_identities']==pf['input_identities'],'retained individual hash identity machinery unchanged')
need("input_identities(inputs['files'])" in d.decode(),'new manifest files mapping only')
need("'REAL_INPUTS_PENDING'" in d.decode() and "'INPUTS.json':None" in d.decode(),'unmaterialized driver refuses actual execution')
need('FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY' not in func['validate_result'],'old four-arm acceptance not inherited')
need('CPU_CHECKPOINT_SYNTHETIC_QUALIFICATION_COMPLETE' in func['validate_result'],'existing helper status only')
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
popens=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='Popen']
need(len(popens)==1,'one bounded helper child')
need(ast.dump(next(k.value for k in popens[0].keywords if k.arg=='pass_fds'))=="Tuple(elts=[Name(id='lock_fd', ctx=Load())], ctx=Load())",'same authenticated lock fd inherited')
need('PPID/PGID/SID' in b['postflight'] and 'HOLD' in b['postflight'],'explicit external postflight obligation')
print(json.dumps({'status':'LOCAL_MECHANICS_ONLY_PENDING_ACTUAL_INPUTS','checks':len(checks),'names':checks,'host_calls':False,'source_imports':False,'tensors':False,'stage_authorized':False,'launch_authorized':False},indent=2))
