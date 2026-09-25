"""Independent stdlib binding/refusal tests. No Torch import, tensors, or hosts."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = Path('/home/poojith-devan/Desktop/Trainium')
REF = 'e96bcfe33177a7a326014875199caa09a908a660'
PREFIX = 'research/v2/proposals/20260926-architecture-clean-exact-cpu/'
checks = []

def check(name, value):
    if not value:
        raise RuntimeError('FAILED: ' + name)
    checks.append(name)

def raw(path, ref=REF):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', ref + ':' + path])

def data(name):
    return json.loads(raw(PREFIX + name))

def sha(value):
    return hashlib.sha256(value).hexdigest()

plan = data('CPU_PLAN.disabled.json')
proposal = data('EXPERIMENT_PROPOSAL.json')
driver = raw(PREFIX + 'host_driver.py')
tree = ast.parse(driver)
check('exact frozen plan', sha(raw(PREFIX + 'CPU_PLAN.disabled.json')) == 'db5181dff39733ba9974fe740323df9aa4fecce987d611a701162244082cb976')
check('disabled CPU/native/adoption', plan['launch_authorized'] is plan['native_launch_authorized'] is plan['source_adoption'] is False and proposal['launch_authorized'] is False)
check('trn22 only bounded resources', plan['proposed_host'] == 'trn22' and plan['minimum_available_ram_gib'] == 64 and plan['minimum_free_disk_gib'] == 6.5 and plan['local_tensor_execution_prohibited'])
check('full parameter/synthetic update distinction', plan['full_model']['parameters'] == 113256724 and plan['full_model']['parameter_tensors'] == 65 and plan['diagnostic_tokens_all_ranks_per_update'] == 2048 and plan['source_global_tokens_per_update'] == 131072 and plan['production_update_equivalent'] is False)
check('driver hash', sha(driver) == plan['driver_sha256'])
for name, pin in plan['support_sha256'].items():
    check('support hash ' + name, sha(raw(PREFIX + name)) == pin)
for name, source in plan['sources'].items():
    content = raw(source['git_path'], source['commit']) if 'commit' in source else (REPO / source.get('git_path', 'prepare.py')).read_bytes()
    check('source/tool hash ' + name, sha(content) == source['sha256'])
requests = {probe['name']: data(probe['request_path']) for probe in plan['probes']}
long_command = shlex.split(raw('workflow/candidates/architecture3e79-116188/launch-command.txt', 'bd9e33ee').decode())
check('candidate exact LONG flags', requests['candidate']['argv'] == long_command[long_command.index('train.py') + 1:])
scored = json.loads(raw('research/v2/operator/20260926-trn22-arch3e79-full/NATIVE_BINDING.disabled.json', '83db5fe3'))
check('reference exact scored flags', requests['reference']['argv'] == scored['training_command'][4:])
for probe in plan['probes']:
    req = requests[probe['name']]
    command = probe['command']
    check('request pin ' + probe['name'], sha(raw(PREFIX + probe['request_path'])) == probe['request_sha256'])
    check('command exact ' + probe['name'], command == req['command'] and sha(json.dumps(command, separators=(',', ':')).encode()) == probe['command_sha256'])
    check('science argv exact ' + probe['name'], command[command.index('--') + 1:] == req['argv'])
    check('standard G2 source path ' + probe['name'], command[2] == plan['sources']['g2']['stage_path'] and command[command.index('--train-py') + 1] == plan['sources'][probe['name']]['stage_path'])
    check('CPU scope options ' + probe['name'], all(command[command.index(k)+1] == v for k,v in [('--steps','3'),('--nproc','4'),('--micro-rows','2'),('--micro-tokens','256'),('--timeout','3600')]) and '--real-data' not in command and '--stub-prepare' not in command)
    check('source/request binding ' + probe['name'], req['source_sha256'] == plan['sources'][probe['name']]['sha256'] and req['prepare_sha256'] == plan['sources']['prepare']['sha256'])
old = ast.parse(raw(plan['driver_derivation']['path'], plan['driver_derivation']['commit']))
fn = lambda t,n: next(x for x in t.body if isinstance(x, ast.FunctionDef) and x.name == n)
check('repaired checks function AST unchanged', ast.dump(fn(tree, 'checks')) == ast.dump(fn(old, 'checks')))
check('single common outer command', plan['expected_outer_command'] == ['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',plan['stage_root']+'/host_driver.py'])
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
children = [n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in ('Popen','call')]
check('both immediate child sites inherit live lock', len(children) == 2 and all(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in n.keywords) for n in children))
main = next(n for n in tree.body if isinstance(n, ast.Try))
loops = [n for n in main.body if isinstance(n, ast.For)]
pair = next(n for n in loops if ast.unparse(n.target) == 'name')
check('exact sequential two-arm order', ast.literal_eval(pair.iter) == ('reference','candidate'))
check('fresh guard before each arm and comparison', all(next(n.lineno for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'require_idle_under_inherited_lock') < min(n.lineno for n in children if n.lineno > loop.lineno) for loop in (pair, next(n for n in loops if ast.unparse(n.target) == '(phase, command)'))))
check('no unlock/next/native dispatch', not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('close','flock','execv','execve') for n in calls) and 'CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW' in driver.decode())
check('bounded checkpoint child', "'600s'" in driver.decode() and "'--kill-after=60s'" in driver.decode())

# Exact stdlib receipt validator, with independent malformed-receipt cases.
validator = {'__name__': 'bound_receipt_validator'}
exec(compile(raw(PREFIX + 'validate_receipts.py'), 'validate_receipts.py', 'exec'), validator)
baseline = json.loads(raw('research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/results/candidate-on/g2.json','abe8c373'))
check('fixture is full parameter source receipt', baseline['plan_sha256'] == plan['expected_plan_sha256'])
class VirtualPath:
    def __init__(self, files, name='root'): self.files,self.name=files,name
    def __truediv__(self, part): return VirtualPath(self.files,self.name+'/'+part)
    def read_text(self): return json.dumps(self.files[self.name])
files = {}
for name, req in requests.items():
    report = copy.deepcopy(baseline)
    report['argv'], report['train_py_sha256'] = req['argv'], req['source_sha256']
    files['root/'+name+'/request.json'] = req
    files['root/'+name+'/g2.json'] = report
    files['root/'+name+'/terminal.json'] = {'exit_code':0,'verdict':'PASS'}
    files['root/'+name+'/checkpoint-identity.json'] = {'bytes':1,'sha256':'1'*64}
def receipt_case(name, change, error=None):
    fixture=copy.deepcopy(files)
    change(fixture)
    caught=None
    try: result=validator['validate'](VirtualPath(fixture))
    except Exception as exc: caught=str(exc)
    check('receipt '+name, caught is None if error is None else error in caught)
receipt_case('valid bounded evidence',lambda f:None)
key='root/candidate/g2.json'
for label, mutate, error in [
 ('timeout',lambda r:r.update(timed_out=True),'EXIT_TIMEOUT'),
 ('wrong source',lambda r:r.update(train_py_sha256='wrong'),'SOURCE_ARGV'),
 ('stub prepare',lambda r:r['prepare'].update(stub=True),'PREPARE'),
 ('tiny shape mislabeled',lambda r:r.update(micro_batch=[1,16]),'SHAPE'),
 ('ownership mismatch',lambda r:r.update(plan_sha256='wrong'),'FULL_PLAN'),
 ('missing rank',lambda r:r['ranks'].pop(),'RANKS'),
 ('unchanged parameter',lambda r:r['ranks'][0].update(n_changed=64),'ALL65_UPDATED'),
 ('cross rank mismatch',lambda r:r['ranks'][1].update(cross_rank_mismatches=['x']),'RANK_CONSISTENCY'),
 ('NaN loss',lambda r:r['ranks'][0]['losses'].__setitem__(0,float('nan')),'FINITE_LOSSES'),
 ('positive loss delta',lambda r:r['ranks'][0]['losses'].__setitem__(0,99),'LOSSES_PARAMS_PLAN_IDENTITY'),
 ('missing digest',lambda r:r['ranks'][0]['final_digests'].pop(next(iter(r['ranks'][0]['final_digests']))),'ALL65_DIGESTS'),
 ('reload false',lambda r:r['ranks'][0]['checkpoint_check'].update(ok=False),'RELOAD'),
]: receipt_case(label,lambda f,m=mutate:m(f[key]),error)

# Check exact new checkpoint comparator without importing Torch: tiny byte-value
# doubles exercise failure branches, not numerical/native correctness.
cmp_tree = ast.parse(raw(PREFIX+'compare_checkpoints.py'))
cmp_ns = {'hashlib':hashlib,'json':json,'sys':SimpleNamespace(modules={})}
exec(compile(ast.Module(body=[n for n in cmp_tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'checkpoint-functions','exec'),cmp_ns)
class Tensor:
    def __init__(self, value=1.0, count=1, dtype='fp32', shape=(1,)):
        self.value,self.count,self.dtype,self.shape=value,count,dtype,shape
    def numel(self): return self.count
    def view(self, dtype): return struct.pack('<f',self.value)
class Finite:
    def __init__(self,tensor): self.tensor=tensor
    def all(self): return math.isfinite(self.tensor.value)
class NoGrad:
    def __enter__(self): pass
    def __exit__(self,*args): pass
class MemoryPath:
    def __init__(self,path='root'): self.path=str(path)
    def __truediv__(self,part): return MemoryPath(self.path+'/'+part)
    def __str__(self): return self.path
    def open(self,*args): return io.BytesIO(b'checkpoint')
    def read_bytes(self): return b'source'
    def read_text(self):
        return json.dumps({'source_sha256':sha(b'source')} if self.path.endswith('request.json') else {'path':self.path.replace('checkpoint-identity.json','final.pt'),'sha256':sha(b'checkpoint')})
def checkpoint_case(label, mutation=None, error=None):
    state={str(i):Tensor(count=113256660 if i==0 else 1) for i in range(65)}
    payloads={n:{'model_config':{'n_layer':8,'n_embd':1024},'model_state':copy.deepcopy(state)} for n in ('reference','candidate')}
    changed={'logit':False}
    if mutation: mutation(payloads,changed)
    class Model:
        def __init__(self,name): self.name=name
        def __call__(self,x): return Tensor(2.0 if changed['logit'] and self.name=='candidate' else 1.0,shape=(1,256,8192))
    def module(spec):
        name=spec.name.removeprefix('checkpoint_')
        return SimpleNamespace(GPT=Model,load_for_eval=lambda path,device:Model(name))
    torch=SimpleNamespace(set_num_threads=lambda _:None,randint=lambda *a,**k:Tensor(),Generator=lambda:SimpleNamespace(manual_seed=lambda _:None),load=lambda path,**k:payloads['candidate' if 'candidate' in str(path) else 'reference'],device=lambda x:x,bfloat16='bf16',uint8='u8',no_grad=NoGrad,isfinite=Finite,__version__='stdlib-double')
    torch.equal=lambda a,b:a==b if isinstance(a,bytes) else (a.value==b.value and a.shape==b.shape)
    cmp_ns.update(torch=torch,Path=MemoryPath,importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda name,path:SimpleNamespace(name=name,loader=SimpleNamespace(exec_module=lambda _:None)),module_from_spec=module)))
    caught=None
    try: result=cmp_ns['run'](MemoryPath())
    except Exception as exc: caught=str(exc)
    check('checkpoint comparator '+label,caught is None if error is None else error in caught)
checkpoint_case('matching state/logits')
checkpoint_case('one state byte differs',lambda p,c:setattr(p['candidate']['model_state']['1'],'value',2.0),'STATE_BYTES:1')
checkpoint_case('signed zero bytes differ',lambda p,c:(setattr(p['reference']['model_state']['1'],'value',0.0),setattr(p['candidate']['model_state']['1'],'value',-0.0)),'STATE_BYTES:1')
checkpoint_case('dtype differs',lambda p,c:setattr(p['candidate']['model_state']['1'],'dtype','bf16'),'STATE_BYTES:1')
checkpoint_case('nonfinite matching state',lambda p,c:(setattr(p['reference']['model_state']['1'],'value',float('inf')),setattr(p['candidate']['model_state']['1'],'value',float('inf'))),'FINITE_STATE:1')
checkpoint_case('missing state',lambda p,c:p['candidate']['model_state'].pop('1'),'ALL65_STATE_KEYS')
checkpoint_case('wrong config',lambda p,c:p['candidate']['model_config'].update(n_embd=128),'FULL_CONFIG')
checkpoint_case('unequal reload logits',lambda p,c:c.update(logit=True),'RELOADED_B1_LOGITS_EXACT')
load_call=next(n for n in ast.walk(cmp_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load')
check('checkpoint loads CPU weights_only', {k.arg:ast.literal_eval(k.value) for k in load_call.keywords} == {'map_location':'cpu','weights_only':True})
check('checkpoint diagnostic no compile/native forward', "torch.device('cpu')" in raw(PREFIX+'compare_checkpoints.py').decode() and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='compile' for n in ast.walk(cmp_tree)))
print(json.dumps({'verdict':'PASS_MECHANICAL_CPU_PACKET_ONLY','checks':len(checks),'tests':checks,'packet_commit':REF,'plan_sha256':sha(raw(PREFIX+'CPU_PLAN.disabled.json')),'driver_sha256':sha(driver),'source_review_separate':True,'host_or_tensor_execution':False,'actual_checkpoint_comparator_execution_with_torch':False,'native_launch_authorized':False},indent=2))
