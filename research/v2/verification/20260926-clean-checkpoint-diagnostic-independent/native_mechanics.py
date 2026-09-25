"""Independent native-binding static/mocked review; stdlib only, no native imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
COMMIT='d524de5cb974aa77883037cdf858fcfddba3113d'
P='research/v2/proposals/20260926-clean-checkpoint-route-diagnostic/'
checks=[]
def check(name,value):
 if not value:raise RuntimeError('FAIL: '+name)
 checks.append(name)
def blob(name):return subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':'+P+name])
def sha(x):return hashlib.sha256(x).hexdigest()
b=json.loads(blob('neuron/BINDING.disabled.json'));c=json.loads(blob('cpu/BINDING.disabled.json'))
driver=ast.parse(blob('neuron/host_driver.py'));cpu=ast.parse(blob('cpu/host_driver.py'))
check('native driver pinned',sha(blob('neuron/host_driver.py'))==b['driver_sha256'])
check('identical actual control-flow functions',ast.dump(ast.Module(body=[n for n in driver.body if isinstance(n,ast.FunctionDef)],type_ignores=[]))==ast.dump(ast.Module(body=[n for n in cpu.body if isinstance(n,ast.FunctionDef)],type_ignores=[])))
check('same source/checkpoint/support pins',b['source_pins']==c['source_pins'] and b['source_blobs']==c['source_blobs'] and b['checkpoint_sha256']==c['checkpoint_sha256'] and b['checkpoint_path']==c['checkpoint_path'])
for key in ('command','outer_command','detached_command'):
 check(key+' exact hash',sha(json.dumps(b[key],separators=(',',':'),ensure_ascii=True).encode())==b[key+'_sha256'])
ns={};exec(compile(ast.Module(body=[n for n in driver.body if not isinstance(n,ast.If)],type_ignores=[]),'native-definitions-only','exec',optimize=0),ns)
check('native literal argv/env/pins',ns['COMMAND']==b['command'] and ns['ENV']==b['environment'] and ns['PINS']==b['source_pins'] and ns['BACKEND']=='neuron')
check('native900+15 four-rank torchrun',b['command'][:8]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','900s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'] and b['command'][b['command'].index('--backend')+1]=='neuron')
check('four LNC2 ranks',b['world_size']==4 and b['lnc']==2 and b['environment']['NEURON_LOGICAL_NC_CONFIG']=='2')
check('separate unassigned native scope',b['launch_authorized'] is False and b['stage_authorized'] is False and b['source_review']=='REFUSE_UNCHANGED' and b['upload']==b['full_training']=='HOLD' and 'independently accepted prior CPU diagnostic receipts' in b['pending_gates'])
check('no automatic promotion',b['no_auto_next_stage'] is True and b['root']!=c['root'])
check('same original exclusive flock',b['outer_command']==['/usr/bin/flock','-n',c['device_lock'],'/usr/bin/python3','-I','-B',b['root']+'/host_driver.py'])
check('one-thread native CPU helpers',all(b['environment'][k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))
check('private compiler URL/DIR',b['environment']['NEURON_COMPILE_CACHE_URL']==b['environment']['NEURON_COMPILE_CACHE_DIR']==b['root']+'/caches/compiler')
check('six private native caches',len(ns['CACHE_KEYS'])==6 and all(b['environment'][key].startswith(b['root']+'/caches/') for key in ns['CACHE_KEYS']))
probe=ast.parse(blob('probe.py'))
run=next(n for n in probe.body if isinstance(n,ast.FunctionDef) and n.name=='run')
native_if=next(n for n in run.body if isinstance(n,ast.If) and ast.unparse(n.test)=="args.backend == 'neuron'")
native_body=ast.Module(body=[n for n in native_if.body if not isinstance(n,(ast.Import,ast.ImportFrom))],type_ignores=[])
check('native init matches prepare backend mapping',"dist.init_process_group('neuron')" in ast.unparse(native_body) and "device = torch.device(args.backend)" in ast.unparse(run))
check('no forward collective in probe call',not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='dist' for f in run.body if isinstance(f,ast.FunctionDef) and f.name=='call' for n in ast.walk(f)))
check('sync before CPU observation',"torch.neuron.synchronize()" in ast.unparse(run))
def require(ok,message):
 if not ok:raise RuntimeError(message)
for name,world,lnc,sync,want,init_expected in [('correct', '4','2',True,None,True),('wrong world','3','2',True,'NATIVE_FOUR_RANK_LNC2',False),('wrong LNC','4','1',True,'NATIVE_FOUR_RANK_LNC2',False),('sync unavailable','4','2',False,'NATIVE_SYNC_MISSING',True)]:
 seen=[]
 ctx={'require':require,'os':SimpleNamespace(environ={'WORLD_SIZE':world,'NEURON_LOGICAL_NC_CONFIG':lnc}),
      'dist':SimpleNamespace(init_process_group=lambda backend:seen.append(backend)),
      'torch':SimpleNamespace(neuron=SimpleNamespace(**({'synchronize':lambda:None} if sync else {})))}
 err=None
 try:exec(compile(native_body,'exact-native-init-mocked','exec',optimize=0),ctx)
 except RuntimeError as exc:err=str(exc)
 check('runtime init '+name,err==want and bool(seen)==init_expected and (not seen or seen==['neuron']))

# Exact finally cleanup branch: successful destruction, no init and failed cleanup.
main=next(n for n in probe.body if isinstance(n,ast.FunctionDef) and n.name=='main')
outer=next(n for n in main.body if isinstance(n,ast.Try))
cleanup=next(n for n in outer.finalbody if isinstance(n,ast.Try))
for name,initialized,fail,want in [('initialized',True,False,None),('not initialized',False,False,None),('cleanup error',True,True,'UNRESOLVED_EXECUTION_OR_EVIDENCE')]:
 seen=[];receipt={'status':'BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE'}
 def destroy():
  seen.append('destroy')
  if fail:raise RuntimeError('fixture cleanup')
 ctx={'sys':SimpleNamespace(modules={'torch.distributed':SimpleNamespace(is_initialized=lambda:initialized,destroy_process_group=destroy)}),'receipt':receipt}
 exec(compile(ast.Module(body=[cleanup],type_ignores=[]),'exact-cleanup-mocked','exec',optimize=0),ctx)
 check('cleanup '+name,bool(seen)==initialized and (receipt['status']==want if want else receipt['status']=='BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE'))

# All four actual scalar receipts, no fake PASS inheritance from just one rank.
v={};exec(compile(blob('validate_receipts.py'),'exact-native-validator','exec',optimize=0),v)
base={'world_size':4,'status':v['EXACT'],'backend':'neuron','label':'native-fixture','candidate_sha256':v['CANDIDATE'],'reference_sha256':v['REFERENCE'],'checkpoint_sha256':v['CHECKPOINT'],'call_order':v['ORDER'],'cases':[{'name':n,'passed':True} for n in sorted(v['CASE_NAMES'])],'outputs':{s:{k:{'shape':[1 if k in ('A','B','A_repeat') else 2,1024,8192],'dtype':'torch.float32','finite':True,'sha256':'same'} for k in v['ORDER']} for s in ('reference','candidate')},'inputs':{'A':{'tokens':[[1,2,3]],'sha256':'same'}},'source_review':'REFUSE_UNCHANGED','upload':'HOLD'}
scenarios=[('four complete',None,None),('missing rank','missing','RANK_RECEIPT_COUNT'),('wrong rank','rank','RANK_WORLD'),('wrong world','world','RANK_WORLD'),('rank3 missing case','missing_case','ALL25_CASES'),('rank2 nonfinite case fails','nonfinite','ALL25_CASES'),('rank1 duplicate case','duplicate','ALL25_CASES'),('rank3 output differs','output','CROSS_RANK_OUTPUT_AND_INPUT_BYTES'),('rank2 input differs','input','CROSS_RANK_OUTPUT_AND_INPUT_BYTES'),('cleanup unresolved','cleanup','RECEIPT_STATUS'),('source refusal erased','gate','PRESERVED_HOLD')]
for label,change,want in scenarios:
 with tempfile.TemporaryDirectory(prefix='native-validator-only-') as directory:
  root=Path(directory);(root/'receipts').mkdir()
  rs=[dict(copy.deepcopy(base),rank=rank) for rank in range(4)]
  if change=='rank':rs[2]['rank']=0
  elif change=='world':rs[2]['world_size']=1
  elif change=='missing_case':rs[3]['cases'].pop()
  elif change=='nonfinite':rs[2]['cases'][0].update(passed=False)
  elif change=='duplicate':rs[1]['cases'][0]=copy.deepcopy(rs[1]['cases'][1])
  elif change=='output':rs[3]['outputs']['candidate']['AB']['sha256']='different'
  elif change=='input':rs[2]['inputs']['A']['tokens']=[[3,2,1]]
  elif change=='cleanup':rs[3]['status']='UNRESOLVED_EXECUTION_OR_EVIDENCE'
  elif change=='gate':rs[3]['source_review']='PASS'
  for rank,r in enumerate(rs[:3] if change=='missing' else rs):(root/'receipts'/('rank'+str(rank)+'.json')).write_text(json.dumps(r))
  err=None
  try:v['validate'](root,'native-fixture','neuron')
  except RuntimeError as exc:err=str(exc)
  check('validator '+label,err==want)

# Resource termination is unchanged shared control flow; strict accounting has no waiver.
text=ast.unparse(next(n for n in driver.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
check('child retains lock and session',"pass_fds=(lock_fd,)" in text and 'start_new_session=True' in text)
check('global cap stops owned group',"total > 256 * 1024 ** 2" in text and 'os.killpg(child.pid, signal.SIGTERM)' in text and 'os.killpg(child.pid, signal.SIGKILL)' in text and 'child.wait(timeout=15)' in text)
check('cap poll250ms final validator', 'time.sleep(0.25)' in text and 'validator.validate(ROOT, LABEL, BACKEND)' in text)
check('probe receipt16MiB cap', '16 * 1024 ** 2' in ast.unparse(main) and "'RECEIPT_SIZE_CAP'" in ast.unparse(main))
check('fresh ps/neuron guard before native',text.index('guard.require_idle_under_inherited_lock')<text.index('subprocess.Popen'))
check('native fixed pins immutable since CPU review',sha(blob('probe.py'))=='d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1')
print(json.dumps({'verdict':'PASS_NATIVE_DIAGNOSTIC_BINDING_MECHANICS_ONLY_CONDITIONAL','commit':COMMIT,'checks':len(checks),'tests':checks,'binding_sha256':sha(blob('neuron/BINDING.disabled.json')),'driver_sha256':sha(blob('neuron/host_driver.py')),'source_review':'REFUSE_UNCHANGED','real_cpu_receipts':'PENDING_INDEPENDENT_ACCEPTANCE','native_execution':'NOT_RUN','launch_authorized':False,'full_training':'HOLD','upload':'HOLD','tensors':False,'host_actions':False},indent=2))
