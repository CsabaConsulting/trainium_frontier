"""Independent frozen binding/refusal audit; stdlib, no tensors or host calls."""
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
REF='ec88382c58b71f37455e5e13a3c9b221846c85e4'
PREFIX='research/v2/proposals/20260926-startup-adjusted-exact-cpu-trn21/'
PARENT='9cb0d53ef81cd4668b9b3687fbacf5f6cc98ea2c'
OLD='research/v2/proposals/20260926-architecture-recursive-exact-cpu-trn21/'
checks=[]
def check(name,ok):
    if not ok: raise RuntimeError('FAIL: '+name)
    checks.append(name)
def blob(path,ref=REF): return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def data(name): return json.loads(blob(PREFIX+name))
def sha(value): return hashlib.sha256(value).hexdigest()
def fn(tree,name): return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
plan=data('CPU_PLAN.disabled.json'); proposal=data('EXPERIMENT_PROPOSAL.json')
driver=blob(PREFIX+'host_driver.py'); tree=ast.parse(driver)
check('frozen plan',sha(blob(PREFIX+'CPU_PLAN.disabled.json'))=='1b5d385b797adb735ca4df79b9673462187bf00b1758b9cfda1b35d15dfe3dc6')
check('frozen driver',sha(driver)==plan['driver_sha256']=='78a9d8f6a64b523f87bc6e1b797e2d36a0b1832e216f11e5d0ded3028b86907f')
check('disabled scope',plan['launch_authorized'] is plan['native_launch_authorized'] is plan['source_adoption'] is proposal['launch_authorized'] is False)
check('bound host resources',plan['proposed_host']=='trn21' and plan['expected_hostname']=='ip-172-31-34-85' and plan['minimum_available_ram_gib']==64 and plan['minimum_free_disk_gib']==6.5 and plan['local_tensor_execution_prohibited'])
check('full shape not full update',plan['full_model']['parameters']==113256724 and plan['full_model']['parameter_tensors']==65 and plan['diagnostic_tokens_all_ranks_per_update']==2048 and plan['source_global_tokens_per_update']==131072 and plan['production_update_equivalent'] is False)
for name,pin in data('MANIFEST.sha256.json')['files'].items(): check('manifest '+name,sha(blob(PREFIX+name))==pin)
for name,source in plan['sources'].items():
    content=blob(source['git_path'],source['commit']) if 'commit' in source else (REPO/source.get('git_path','prepare.py')).read_bytes()
    check('source/tool '+name,sha(content)==source['sha256'])
old_driver=blob(OLD+'host_driver.py',PARENT).decode()
old_plan=json.loads(blob(OLD+'CPU_PLAN.disabled.json',PARENT))
rebound=old_driver.replace(old_plan['stage_root'],plan['stage_root']).replace(old_plan['prepare_dir'],plan['prepare_dir']).replace(old_plan['expected_hostname'],plan['expected_hostname'])
requests={p['name']:data(p['request_path']) for p in plan['probes']}
for name,request in requests.items():
    old_request=blob(OLD+name+'/request.json',PARENT)
    rebound=rebound.replace(sha(old_request),sha(blob(PREFIX+name+'/request.json')))
    old_req=json.loads(old_request)
    check('only scientific flag rebound '+name,request['argv']==[x.replace('recursive-schedule','startup-adjusted-schedule') for x in old_req['argv']])
check('only driver rebind literals/pins changed',rebound.encode()==driver)
for name in ('validate_receipts.py','compare_checkpoints.py'):
    check('helper bytes unchanged '+name,blob(PREFIX+name)==blob(OLD+name,PARENT))
scored=json.loads(blob('research/v2/operator/20260926-trn22-arch3e79-full/NATIVE_BINDING.disabled.json','83db5fe3'))
baseargv=scored['training_command'][4:]
for probe in plan['probes']:
    name=probe['name']; req=requests[name]; command=probe['command']
    expected=baseargv+({'off':['--no-startup-adjusted-schedule'],'on':['--startup-adjusted-schedule']}.get(name,[]))
    check('exact scientific argv '+name,req['argv']==expected)
    check('request/command pins '+name,sha(blob(PREFIX+probe['request_path']))==probe['request_sha256'] and command==req['command'] and sha(json.dumps(command,separators=(',',':')).encode())==probe['command_sha256'])
    check('bound source path/pin '+name,command[command.index('--train-py')+1]==plan['sources'][name]['stage_path'] and req['source_sha256']==plan['sources'][name]['sha256'])
    check('G2 prepare options '+name,command[2]==plan['sources']['g2']['stage_path'] and req['g2_tool_sha256']==plan['sources']['g2']['sha256'] and req['prepare_sha256']==plan['sources']['prepare']['sha256'] and command[command.index('--prepare-dir')+1]==plan['prepare_dir'])
    check('CPU diagnostic bounded options '+name,all(command[command.index(k)+1]==v for k,v in [('--steps','3'),('--nproc','4'),('--micro-rows','2'),('--micro-tokens','256'),('--timeout','3600')]) and '--real-data' not in command and '--stub-prepare' not in command and command[command.index('--')+1:]==req['argv'])
check('original common outer flock',plan['expected_outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',plan['stage_root']+'/host_driver.py'])
for field,pin in [('expected_outer_command','outer_command_sha256'),('detached_dispatch_command','detached_command_sha256')]: check(field+' hash',sha(json.dumps(plan[field],separators=(',',':')).encode())==plan[pin])
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
children=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in ('Popen','call')]
check('both immediate child sites inherit lock',len(children)==2 and all(any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in n.keywords) for n in children))
main=next(n for n in tree.body if isinstance(n,ast.Try)); loops=[n for n in main.body if isinstance(n,ast.For)]
arms=next(n for n in loops if ast.unparse(n.target)=='name')
check('sequential four arms',ast.literal_eval(arms.iter)==('reference','default','off','on'))
for loop in (arms,next(n for n in loops if ast.unparse(n.target)=='(phase, command)')):
    guard_line=next(n.lineno for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_idle_under_inherited_lock')
    check('fresh guard before child at '+str(loop.lineno),guard_line<min(n.lineno for n in children if n.lineno>loop.lineno))
check('no unlock or automatic next',not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('close','flock','execv','execve') for n in calls) and 'CPU_FOUR_ARMS_COMPLETE_PENDING_INDEPENDENT_REVIEW' in driver.decode())
check('bounded checkpoint child',"'600s'" in driver.decode() and "'--kill-after=60s'" in driver.decode())
harness=blob(plan['sources']['g2']['git_path'],plan['sources']['g2']['commit']).decode()
check('pinned harness CPU no-compile',all(s in harness for s in ('"--device-type", "cpu"','"--no-compile"','"compute_dtype_for"','"_submission_ce_install"')))

# Exact checks() function with an independently constructed synthetic proc tree.
check_fn=fn(tree,'checks'); req=requests['on']
for label,extra,parent,host,mem,disk,fuser,wanted in [
 ('own authenticated pair only',[],99,plan['expected_hostname'],65,7,1,None),
 ('unrelated same-name driver',[(22,b'python\0/tmp/host_driver.py\0')],99,plan['expected_hostname'],65,7,1,'OTHER_WORKLOAD'),
 ('torchrun contender',[(22,b'/bin/torchrun\0train.py\0')],99,plan['expected_hostname'],65,7,1,'OTHER_WORKLOAD'),
 ('changed parent',[],98,plan['expected_hostname'],65,7,1,'AUTHENTICATED_PARENT_CHANGED'),
 ('wrong host',[],99,'wrong',65,7,1,'HOSTNAME'),
 ('low RAM',[],99,plan['expected_hostname'],63,7,1,'AVAILABLE_RAM_FLOOR'),
 ('low disk',[],99,plan['expected_hostname'],65,6,1,'DISK_FLOOR'),
 ('device holder',[],99,plan['expected_hostname'],65,7,0,'NEURON_NOT_IDLE'),
 ('fuser error',[],99,plan['expected_hostname'],65,7,2,'NEURON_NOT_IDLE')]:
    process_bytes={100:b'python\0host_driver.py\0',99:b'flock\0host_driver.py\0',**dict(extra)}
    class P:
        def __init__(self,path): self.value=str(path)
        def __str__(self): return self.value
        def __truediv__(self,part): return P(self.value+'/'+str(part))
        @property
        def name(self): return self.value.rsplit('/',1)[-1]
        @property
        def parent(self): return P(self.value.rsplit('/',1)[0])
        def is_dir(self): return True
        def glob(self,pattern):
            if self.value=='/proc': return [P('/proc/'+str(p)+'/cmdline') for p in process_bytes]
            if self.value=='/dev': return [P('/dev/neuron0')]
            return []
        def read_bytes(self): return process_bytes[int(self.parent.name)]
        def read_text(self): return 'MemAvailable: '+str(mem*1024**2)+' kB\n'
    pins={req['command'][2]:req['g2_tool_sha256'],req['command'][req['command'].index('--train-py')+1]:req['source_sha256'],plan['prepare_dir']+'/prepare.py':req['prepare_sha256']}
    ns={'Path':P,'REQUEST':req,'PREP':P(plan['prepare_dir']+'/prepare.py'),'ROOT':P(plan['stage_root']),'utc':lambda:'fixture','sha':lambda path:pins[str(path)],'require':lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg)),
        'socket':SimpleNamespace(gethostname=lambda:host),'os':SimpleNamespace(getppid=lambda:parent,getpid=lambda:100,path=SimpleNamespace(lexists=lambda _:False),statvfs=lambda _:SimpleNamespace(f_bavail=disk*1024**3,f_frsize=1)),
        'subprocess':SimpleNamespace(run=lambda *a,**k:SimpleNamespace(returncode=fuser,stdout='fixture'))}
    exec(compile(ast.Module(body=[check_fn],type_ignores=[]),'exact-checks','exec'),ns)
    error=None
    try: ns['checks'](99)
    except Exception as exc: error=str(exc)
    check('host refusal '+label,error is None if wanted is None else wanted in str(error))

# Execute exact validator against full-shape raw-derived fixtures, then mutations.
validator={'__name__':'bound_validator'}; exec(compile(blob(PREFIX+'validate_receipts.py'),'validator','exec'),validator)
baseline=json.loads(blob('research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/results/candidate-on/g2.json','abe8c373'))
check('full-shape fixture ownership',baseline['plan_sha256']==plan['expected_plan_sha256'])
class VirtualPath:
    def __init__(self,files,name='root'):self.files,self.name=files,name
    def __truediv__(self,part):return VirtualPath(self.files,self.name+'/'+part)
    def read_text(self):return json.dumps(self.files[self.name])
files={}
for name,req in requests.items():
    report=copy.deepcopy(baseline);report['argv'],report['train_py_sha256']=copy.deepcopy(req['argv']),req['source_sha256']
    files.update({'root/'+name+'/request.json':req,'root/'+name+'/g2.json':report,'root/'+name+'/terminal.json':{'exit_code':0,'verdict':'PASS'},'root/'+name+'/checkpoint-identity.json':{'bytes':1,'sha256':'1'*64}})
def receipt_case(label,arm,mutation,error=None):
    fixture=copy.deepcopy(files);mutation(fixture['root/'+arm+'/g2.json']);caught=None
    try:validator['validate'](VirtualPath(fixture))
    except Exception as exc:caught=str(exc)
    check('receipt '+label,caught is None if error is None else error in str(caught))
receipt_case('matching reference/default/OFF', 'default',lambda r:None)
receipt_case('ON intentional loss/parameter difference permitted','on',lambda r:(r['ranks'][0]['losses'].__setitem__(0,99),r['ranks'][0]['final_digests'].__setitem__(next(iter(r['ranks'][0]['final_digests'])),'changed')))
for label,mutation,error in [
 ('timeout',lambda r:r.update(timed_out=True),'EXIT_TIMEOUT'),('source',lambda r:r.update(train_py_sha256='wrong'),'SOURCE_ARGV'),
 ('stub',lambda r:r['prepare'].update(stub=True),'PREPARE'),('tiny mislabeled',lambda r:r.update(micro_batch=[1,16]),'SHAPE'),
 ('plan',lambda r:r.update(plan_sha256='wrong'),'FULL_PLAN'),('missing rank',lambda r:r['ranks'].pop(),'RANKS'),
 ('no update',lambda r:r['ranks'][0].update(n_changed=64),'ALL65_UPDATED'),('rank mismatch',lambda r:r['ranks'][1].update(cross_rank_mismatches=['x']),'RANK_CONSISTENCY'),
 ('NaN',lambda r:r['ranks'][0]['losses'].__setitem__(0,float('nan')),'FINITE_LOSSES'),('loss delta',lambda r:r['ranks'][0]['losses'].__setitem__(0,99),'LOSSES_PARAMS_PLAN_IDENTITY'),
 ('missing digest',lambda r:r['ranks'][0]['final_digests'].pop(next(iter(r['ranks'][0]['final_digests']))),'ALL65_DIGESTS'),('reload',lambda r:r['ranks'][0]['checkpoint_check'].update(ok=False),'RELOAD')]:
    receipt_case(label,'off',mutation,error)
receipt_case('ON still requires all updates','on',lambda r:r['ranks'][3].update(n_changed=64),'ALL65_UPDATED')

# Exact comparator functions with tiny stdlib doubles: control-flow evidence only.
cmp_tree=ast.parse(blob(PREFIX+'compare_checkpoints.py'))
ns={'hashlib':hashlib,'json':json,'sys':SimpleNamespace(modules={})}
exec(compile(ast.Module(body=[n for n in cmp_tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'comparator-functions','exec'),ns)
class Tensor:
    def __init__(self,value=1.,count=1,dtype='fp32',shape=(1,)):self.value,self.count,self.dtype,self.shape=value,count,dtype,shape
    def numel(self):return self.count
    def reshape(self,*a):return self
    def view(self,dtype):return struct.pack('<f',self.value)
class NoGrad:
    def __enter__(self):pass
    def __exit__(self,*args):pass
class MemoryPath:
    def __init__(self,path='root'):self.path=str(path)
    def __str__(self):return self.path
    def __truediv__(self,part):return MemoryPath(self.path+'/'+part)
    def open(self,*args):return io.BytesIO(b'checkpoint')
    def read_bytes(self):return b'source'
    def read_text(self):return json.dumps({'source_sha256':sha(b'source')} if self.path.endswith('request.json') else {'path':self.path.replace('checkpoint-identity.json','final.pt'),'sha256':sha(b'checkpoint')})
def checkpoint_case(label,mutate=lambda p,c:None,error=None):
    state={str(i):Tensor(count=113256660 if i==0 else 1) for i in range(65)}
    payloads={n:{'model_config':{'n_layer':8,'n_embd':1024},'model_state':copy.deepcopy(state)} for n in ('reference','default','off')};changed={};mutate(payloads,changed)
    class Model:
        def __init__(self,name):self.name=name
        def __call__(self,x):return Tensor(2. if changed.get('logit') and self.name=='off' else 1.,shape=(1,256,8192))
    torch=SimpleNamespace(set_num_threads=lambda _:None,randint=lambda *a,**k:Tensor(),Generator=lambda:SimpleNamespace(manual_seed=lambda _:None),load=lambda path,**k:payloads[str(path).split('/')[1]],device=lambda x:x,bfloat16='bf16',uint8='u8',no_grad=NoGrad,isfinite=lambda t:SimpleNamespace(all=lambda:math.isfinite(t.value)),__version__='stdlib-double',equal=lambda a,b:a==b if isinstance(a,bytes) else (a.value==b.value and a.shape==b.shape))
    def module(spec):return SimpleNamespace(load_for_eval=lambda path,device:Model(spec.name.removeprefix('checkpoint_')))
    ns.update(torch=torch,Path=MemoryPath,importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda name,path:SimpleNamespace(name=name,loader=SimpleNamespace(exec_module=lambda _:None)),module_from_spec=module)))
    caught=None
    try:ns['run'](MemoryPath())
    except Exception as exc:caught=str(exc)
    check('checkpoint '+label,caught is None if error is None else error in str(caught))
checkpoint_case('matching states/logits')
checkpoint_case('one byte',lambda p,c:setattr(p['off']['model_state']['1'],'value',2.),'STATE_BYTES:1')
checkpoint_case('signed zero',lambda p,c:(setattr(p['reference']['model_state']['1'],'value',0.),setattr(p['default']['model_state']['1'],'value',-0.)),'STATE_BYTES:1')
checkpoint_case('dtype',lambda p,c:setattr(p['off']['model_state']['1'],'dtype','bf16'),'STATE_BYTES:1')
checkpoint_case('nonfinite matching state',lambda p,c:[setattr(v['model_state']['1'],'value',float('inf')) for v in p.values()],'FINITE_STATE:1')
checkpoint_case('missing state',lambda p,c:p['off']['model_state'].pop('1'),'ALL65_STATE_KEYS')
checkpoint_case('wrong config',lambda p,c:p['off']['model_config'].update(n_embd=128),'FULL_CONFIG')
checkpoint_case('logit delta',lambda p,c:c.update(logit=True),'RELOADED_B1_LOGITS_EXACT')
load=next(n for n in ast.walk(cmp_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load')
check('weights_only CPU loads',{k.arg:ast.literal_eval(k.value) for k in load.keywords}=={'map_location':'cpu','weights_only':True})
check('comparison explicitly CPU BF16 shim',"mod.compute_dtype_for=lambda device:torch.bfloat16" in blob(PREFIX+'compare_checkpoints.py').decode())
check('scope real five but only three updates',plan['startup_boundary_scope']['real_prepare_startup_steps_excluded']==5 and plan['steps']==3 and plan['startup_boundary_scope']['fifth_boundary_exercised'] is False and plan['startup_boundary_scope']['no_stub_prepare'] is True)
check('prepare actual constant five',any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_STARTUP_STEPS_EXCLUDED' for t in n.targets) and ast.literal_eval(n.value)==5 for n in ast.parse((REPO/'prepare.py').read_text()).body))
check('no extra source feature defaults',requests['reference']['argv']==requests['default']['argv'] and not any('recursive' in x for r in requests.values() for x in r['argv']))
receipt_case('wrong argv','default',lambda r:r['argv'].append('--unbound'),'SOURCE_ARGV')
receipt_case('positive infinity','off',lambda r:r['ranks'][0]['losses'].__setitem__(0,float('inf')),'FINITE_LOSSES')
receipt_case('negative infinity','on',lambda r:r['ranks'][0]['losses'].__setitem__(0,float('-inf')),'FINITE_LOSSES')
receipt_case('non-identical default digest','default',lambda r:r['ranks'][0]['final_digests'].__setitem__(next(iter(r['ranks'][0]['final_digests'])),'different'),'LOSSES_PARAMS_PLAN_IDENTITY')
result={'verdict':'PASS_MECHANICAL_CPU_PACKET_ONLY','checks':len(checks),'tests':checks,'packet_commit':REF,'plan_sha256':sha(blob(PREFIX+'CPU_PLAN.disabled.json')),'driver_sha256':sha(driver),'host_actions':False,'tensors':False,'actual_checkpoint_comparator_with_torch':False,'fresh_source_review_separate':True,'full_cpu_results_pending':True,'native_launch_authorized':False}
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
