"""Independent tiny exactness/reference and CPU binding checks; no model/host use."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
COMMIT='d524de5cb974aa77883037cdf858fcfddba3113d'
PACKET='research/v2/proposals/20260926-clean-checkpoint-route-diagnostic/'
checks=[]
def check(name,value):
    if not value:raise RuntimeError('FAIL: '+name)
    checks.append(name)
def blob(path):return subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':'+PACKET+path])
def digest(data):return hashlib.sha256(data).hexdigest()
manifest=json.loads(blob('MANIFEST.sha256.json'))
for name,want in manifest['files'].items():check('manifest '+name,digest(blob(name))==want)
check('canonical exact byte identity',all(blob(name)==subprocess.check_output(['git','-C',str(REPO),'show','135b19e8:'+PACKET+name]) for name in manifest['files']))
probe=ast.parse(blob('probe.py')); run=next(n for n in probe.body if isinstance(n,ast.FunctionDef) and n.name=='run')
helpers=[n for n in probe.body if isinstance(n,ast.FunctionDef) and n.name in ('require','finite_metric','verdict')]+[n for n in run.body if isinstance(n,ast.FunctionDef) and n.name in ('raw','tensor_record','snapshot','compare')]

# Actual four/eight-element tensors; scalar Python/struct reference is independently written.
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
receipt={'cases':[]}
ctx={'torch':torch,'hashlib':hashlib,'math':math,'receipt':receipt,'T':1024}
exec(compile(ast.Module(body=helpers,type_ignores=[]),'exact-extracted-probe-helpers','exec',optimize=0),ctx)
rows=[]
for label,av,bv,causal,want in [
    ('exact',[1.,-2.,0.,4.],[1.,-2.,0.,4.],False,True),
    ('one exact binary perturbation',[1.,-2.,0.,4.125],[1.,-2.,0.,4.],False,False),
    ('signed zero bytes',[0.,1.,2.,3.],[-0.,1.,2.,3.],False,False),
    ('small diff strict',[0.005,1.,2.,3.],[0.,1.,2.,3.],False,False),
    ('same small diff causal',[0.005,1.,2.,3.],[0.,1.,2.,3.],True,True),
    ('large causal diff',[0.02,1.,2.,3.],[0.,1.,2.,3.],True,False),
    ('NaN retained',[float('nan'),1.,2.,3.],[0.,1.,2.,3.],False,False),
    ('Inf retained',[float('inf'),1.,2.,3.],[0.,1.,2.,3.],False,False),
]:
    a=torch.tensor(av,dtype=torch.float32);b=torch.tensor(bv,dtype=torch.float32)
    ctx['compare'](label,a,b,causal=causal)
    row=receipt['cases'][-1]
    # Scalar reference uses rounded actual FP32 values, not the probe's torch math.
    aa=a.tolist();bb=b.tolist();ab=struct.pack('<4f',*aa);bbb=struct.pack('<4f',*bb)
    check(label+' classification',row['passed'] is want)
    check(label+' raw exact reference',row['exact_bytes']==(ab==bbb))
    check(label+' byte mismatch reference',row['byte_mismatch_count']==sum(x!=y for x,y in zip(ab,bbb)))
    if all(math.isfinite(x) for x in aa+bb):
        max_abs=max(abs(x-y) for x,y in zip(aa,bb))
        check(label+' independent absolute error',row['max_abs']==max_abs)
        check(label+' mismatch count',row['element_mismatch_count']==sum(x!=y for x,y in zip(aa,bb)))
    else:
        check(label+' strict JSON-safe error',row['metric_nonfinite'] is True and row['actual']['finite'] is False)
    json.dumps(row,allow_nan=False)
    rows.append(row)
check('any failed case prevents success',ctx['verdict'](receipt['cases'])=='HOLD_NUMERIC_OR_STATE_DIFFERENCE')
check('no cases prevents success',ctx['verdict']([])=='HOLD_NUMERIC_OR_STATE_DIFFERENCE')
a=torch.tensor([[1.,2.],[3.,4.]])
ctx['compare']('shape mismatch',a,a.reshape(4));check('shape mismatch HOLD',receipt['cases'][-1]['passed'] is False)
ctx['compare']('dtype mismatch',a,a.double());check('dtype mismatch HOLD',receipt['cases'][-1]['passed'] is False)
# Distinct deterministic toy logits: caught batch dependence and causal leakage.
x=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]])
pure=x.cumsum(1)
ctx['compare']('pure rows',pure,torch.cat([x[:1].cumsum(1),x[1:].cumsum(1)]))
check('independent causal row model exact',receipt['cases'][-1]['passed'] is True)
bad=x+x.mean(0,keepdim=True)
ctx['compare']('batch dependent negative',bad,torch.cat([x[:1]*2,x[1:]*2]))
check('batch-dependent negative detected',receipt['cases'][-1]['passed'] is False)
y=x.clone();y[:,2:]=y[:,2:].flip(1)
ctx['compare']('pure causal prefix',x.cumsum(1)[:,:1],y.cumsum(1)[:,:1],causal=True)
check('causal reference prefix exact',receipt['cases'][-1]['passed'] is True)
ctx['compare']('future-token leak',x.flip(1)[:,:1],y.flip(1)[:,:1],causal=True)
check('noncausal negative detected',receipt['cases'][-1]['passed'] is False)
# Before/after state and hook refusal, without using either candidate model.
model=torch.nn.Linear(2,2,bias=False).eval().requires_grad_(False)
before=ctx['snapshot'](model)
with torch.no_grad():model.weight[0,0].add_(1)
check('state mutation fingerprint changes',before!=ctx['snapshot'](model))
hook=model.register_forward_hook(lambda *_:None)
try:
    ctx['snapshot'](model);caught=False
except RuntimeError as exc:caught='MODEL_HOOKS:' in str(exc)
finally:hook.remove()
check('hook negative refuses',caught)
model.train()
try:
    ctx['snapshot'](model);caught=False
except RuntimeError as exc:caught='EVAL_MODE:' in str(exc)
check('training-mode negative refuses',caught)

# Frozen CPU binding: exact pins and no native/training dispatch.
b=json.loads(blob('cpu/BINDING.disabled.json'));driver_bytes=blob('cpu/host_driver.py');tree=ast.parse(driver_bytes)
ns={'__name__':'inspection_only'}
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.If)],type_ignores=[]),'cpu-driver-definitions','exec',optimize=0),ns)
check('CPU binding literals',ns['COMMAND']==b['command'] and ns['ENV']==b['environment'] and ns['PINS']==b['source_pins'])
check('driver pin',digest(driver_bytes)==b['driver_sha256'])
check('CPU600+15 exactly one inference child',b['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','600s'] and b['command'][b['command'].index('--backend')+1]=='cpu' and not any('torchrun' in x or 'eval-public' in x for x in b['command']))
check('CPU thread env',all(b['environment'][k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))
check('disabled and preserved REFUSE',b['launch_authorized'] is False and b['stage_authorized'] is False and b['source_review']=='REFUSE_UNCHANGED' and b['full_training']==b['upload']=='HOLD' and b['no_auto_next_stage'] is True)
check('common outer flock exact',b['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',b['root']+'/host_driver.py'])
for source in ('candidate','reference'):
    src=b['source_blobs'][source]
    check('actual immutable source '+source,digest(subprocess.check_output(['git','-C',str(REPO),'show',src['commit']+':'+src['path']]))==b['source_pins'][source+'.py'])
for source in ('idle_guard.py','dispatch_driver.py'):
    check('unchanged '+source,digest((REPO/'research/v2/operator/20260926-common-host-idle-guard'/source).read_bytes())==b['source_pins'][source])
check('probe/validator pins',digest(blob('probe.py'))==b['source_pins']['probe.py'] and digest(blob('validate_receipts.py'))==b['source_pins']['validate_receipts.py'])
check('exact retained checkpoint',b['checkpoint_bytes']==453048594 and b['checkpoint_sha256']=='068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25')
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
check('one child inherited lifetime fd',len(calls)==1 and any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in calls[0].keywords))
check('no source monkeypatch or update in probe',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','register_forward_hook','register_forward_pre_hook','save','train') for n in ast.walk(probe)))
check('no dtype/table floor patch',not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in ('COMPUTE_DTYPE','EVAL_MIN_SEQ_LEN','forward') for t in n.targets) for n in ast.walk(probe)))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
text=ast.unparse(main)
check('resource floors',"64 * 1024 ** 3" in text and "6.5 * 1024 ** 3" in text)
check('guard precedes actual child',text.index('guard.require_idle_under_inherited_lock')<text.index('subprocess.Popen'))
check('checkpoint before-after rehash',text.count("hashlib.file_digest(stream, 'sha256')")==2 and "checkpoint_after == checkpoint_before" in text)

# Execute exact require calls on independent binding/preflight false fixtures.
reqns={};req=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require')
exec(compile(ast.Module(body=[req],type_ignores=[]),'exact-fail-closed-require','exec',optimize=0),reqns)
for name in ('WRONG_HOST','RAM_BELOW_64_GIB','DISK_HEADROOM','CHECKPOINT_SHA','SUPPORT_SHA:probe.py','COMPILER_URL_DIR','ARTIFACT_CAP_STOPPED_DIAGNOSTIC'):
    try:reqns['require'](False,name);caught=False
    except RuntimeError as exc:caught=str(exc)==name
    check('refuse '+name,caught)

# Actual scalar validator: success and independently corrupted receipt cases.
validator={}
exec(compile(blob('validate_receipts.py'),'exact-validator','exec',optimize=0),validator)
base={'rank':0,'world_size':1,'status':validator['EXACT'],'backend':'cpu','label':'fixture','candidate_sha256':validator['CANDIDATE'],'reference_sha256':validator['REFERENCE'],'checkpoint_sha256':validator['CHECKPOINT'],'call_order':validator['ORDER'],'cases':[{'name':name,'passed':True} for name in sorted(validator['CASE_NAMES'])],'outputs':{s:{k:{'sha256':'fixture','finite':True} for k in validator['ORDER']} for s in ('reference','candidate')},'inputs':{'A':'fixture'},'source_review':'REFUSE_UNCHANGED','upload':'HOLD'}
with tempfile.TemporaryDirectory(prefix='independent-checkpoint-validator-') as tmp:
    root=Path(tmp);(root/'receipts').mkdir();out=root/'receipts/rank0.json'
    for label,change,want in [('complete',lambda r:None,None),('rank',lambda r:r.update(rank=1),'RANK_WORLD'),('missing case',lambda r:r['cases'].pop(),'ALL25_CASES'),('failed case',lambda r:r['cases'][0].update(passed=False),'ALL25_CASES'),('source changed',lambda r:r.update(candidate_sha256='wrong'),'INPUT_PINS'),('order changed',lambda r:r['call_order'].reverse(),'CALL_MATRIX'),('source accepted falsely',lambda r:r.update(source_review='PASS'),'PRESERVED_HOLD')]:
        r=json.loads(json.dumps(base));change(r);out.write_text(json.dumps(r))
        error=None
        try:validator['validate'](root,'fixture','cpu')
        except RuntimeError as exc:error=str(exc)
        check('validator '+label,error==want)

print(json.dumps({'verdict':'PASS_CPU_DIAGNOSTIC_SOURCE_AND_BINDING_ONLY','commit':COMMIT,'checks':len(checks),'tests':checks,'tiny_reference_rows':rows,'torch':torch.__version__,'threads':torch.get_num_threads(),'largest_independent_tensor_elements':8,'probe_sha256':digest(blob('probe.py')),'cpu_binding_sha256':digest(blob('cpu/BINDING.disabled.json')),'cpu_driver_sha256':digest(driver_bytes),'manifest_sha256':digest(blob('MANIFEST.sha256.json')),'source_review':'REFUSE_UNCHANGED','native_review':'NOT_ACCEPTED_BY_THIS_RECEIPT','host_actions':False,'full_model_imported':False,'full_training':'HOLD','upload':'HOLD'},indent=2,allow_nan=False))
