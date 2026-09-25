"""Independent mechanical audit only. No source execution, SDK, hosts or tensors."""
import ast
import copy
import hashlib
import json
import subprocess
import types

REPO='/home/poojith-devan/Desktop/Trainium'
COMMIT='710ea6772d5525d97d990ecdf7f835f7f65d8380'
P='research/v2/proposals/20260926-k2-symmetric-native-trn21/'
OLD='research/v2/proposals/20260926-k2-name-binding-native/'
PARENT='84d9f94f6b0341efcb7ddcd8ccaf2944481dc1e3'
checks=[]
negatives={}
def need(v,name):
    if not v:raise RuntimeError(name)
    checks.append(name)
def raw(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(b):return hashlib.sha256(b).hexdigest()
b=json.loads(raw(COMMIT,P+'NATIVE_BINDING.disabled.json'))
q=json.loads(raw(PARENT,OLD+'NATIVE_BINDING.disabled.json'))
m=json.loads(raw(COMMIT,P+'MANIFEST.sha256.json'))
for name,pin in m['files'].items():need(sha(raw(COMMIT,P+name))==pin,'manifest:'+name)
need(b['source_module_commit']=='9ce2e6a037212aaa76bd85b9b4ab7d311a481757','exact source commit')
for name,pin in b['support_sha256'].items():
    if name in ('consumer_nki.py','check_native.py','reference.py'):
        value=raw(b['source_module_commit'],b['source_module_path']+'/'+name)
    elif name in ('idle_guard.py','dispatch_driver.py'):
        value=raw(COMMIT,'research/v2/operator/20260926-common-host-idle-guard/'+name)
    else:value=raw(COMMIT,P+name)
    need(sha(value)==pin,'runtime pin:'+name)
    if name in ('check_native.py','reference.py'):
        need(value==raw(q['source_module_commit'],q['source_module_path']+'/'+name),'unchanged source:'+name)
need(sha(raw(COMMIT,P+'host_driver.py'))==b['driver_sha256'],'driver hash')
mapping=[(q['root'].split('/out/')[0],b['root'].split('/out/')[0]),(q['label'],b['label']),
         (q['support_sha256']['consumer_nki.py'],b['support_sha256']['consumer_nki.py']),
         (q['support_sha256']['rank_entry.py'],b['support_sha256']['rank_entry.py']),
         (q['expected_hostname'],b['expected_hostname'])]
def mapped(value):
    for old,new in mapping:value=value.replace(old,new)
    return value
for name in ('host_driver.py','rank_entry.py','bound-command.txt'):
    need(mapped(raw(PARENT,OLD+name).decode())==raw(COMMIT,P+name).decode(),'exact runtime mapping:'+name)
need(raw(PARENT,OLD+'validate_receipts.py')==raw(COMMIT,P+'validate_receipts.py'),'validator byte identity')
for key in ('command','checker_args','environment','outer_command','dispatch_command'):
    need(json.loads(mapped(json.dumps(q[key])))==b[key],'exact binding mapping:'+key)
for key in ('cache_keys','timeout_seconds','term_grace_seconds','postflight','completion_rule'):
    need(q[key]==b[key],'unchanged:'+key)
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',PARENT,'--',OLD],cwd=REPO).decode().splitlines()
for path in paths:need(raw(PARENT,path)==raw(COMMIT,path),'original preserved:'+path.rsplit('/',1)[-1])
for key in ('stage_authorized','launch_authorized','source_adoption','operator_assigned','host_reserved','automatic_full','automatic_eval','automatic_next_job'):
    need(b[key] is False,'disabled:'+key)
need(b['host']=='trn21' and b['expected_hostname']=='ip-172-31-34-85','host')
need((b['original_lock']['device'],b['original_lock']['inode'])==(66305,596823),'original lock external gate')
need(b['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',b['root']+'/host_driver.py'],'exact outer lock command')
need(b['command'][:8]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'],'600+60 four ranks')
need(b['environment']['NEURON_LOGICAL_NC_CONFIG']=='2','LNC2')
need(b['checker_args'][-2:]==['--timing-repeats','3'],'inherited 3 repeats explicit')
need(len(b['cache_keys'])==6 and len({b['environment'][k] for k in b['cache_keys']})==6,'six unique caches')
need(b['environment']['NEURON_COMPILE_CACHE_URL']==b['environment']['NEURON_COMPILE_CACHE_DIR'],'compiler URL DIR')
need(all(b['environment'][k].startswith(b['root']+'/caches/') for k in b['cache_keys']),'all caches per label')
need('fresh20M' in b['prerequisites']['current_training'] and '27768550' in b['prerequisites']['assigned_profile'],'ordering gates')
need(b['postflight']['required'] and b['postflight']['artifact']=='postflight-explicit-pids.json','explicit descendant postflight')
driver=raw(COMMIT,P+'host_driver.py').decode()
tree=ast.parse(driver)
constants={}
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        try:constants[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError):pass
for name,key in [('COMMAND','command'),('CHECKER_ARGS','checker_args'),('ENV','environment'),('PINS','support_sha256'),('CACHE_KEYS','cache_keys')]:
    need(constants[name]==b[key],'driver constants:'+name)
need(driver.count('subprocess.call(')==1 and 'pass_fds=(lock_fd,)' in driver,'one child retains live lock FD')
need("require(socket.gethostname() == EXPECTED_HOSTNAME, 'WRONG_HOST')" in driver,'runtime host fail closed')
need(">= 10.5 * 1024**3" in driver,'disk floor unchanged')
need("'DIAGNOSTIC_CHILD_FAILED'" in driver and 'if child_returncode != 0:' in driver,'child failure terminal branch')
need("validator.validate(ROOT, LABEL, ENV, PINS, CHECKER_ARGS)" in driver,'exit0 validated not accepted alone')

# Build our own synthetic four-rank receipt schema, independent of supplied fixtures.
names=['cross_tile_large','cross_tile_unit','unique','all_same','counts_253_1_1_1','boundary_rows','signed_zero','bf16_subnormal','fp32_subnormal','empty']
dtypes=['torch.bfloat16','torch.float32']
common=['upstream_roundtrip','direct_consumer','direct_repeated_call']
autograd=['dense_CPU_oracle','eager_autograd_forward','eager_autograd_backward','compiled_enclosing_autograd_forward','compiled_enclosing_autograd_backward']
files={}
for rank in range(4):
    cases=[]
    for name in names:
        for dtype in dtypes:
            comparisons=[]
            for field in common+(autograd if name!='empty' else []):
                out_dtype=dtype if field in ('upstream_roundtrip','eager_autograd_forward','compiled_enclosing_autograd_forward') else 'torch.float32'
                comparisons.append(dict(name=field,exact=True,finite=True,shape=[256,128],actual_dtype=out_dtype,actual_sha256='a'*64,expected_sha256='a'*64,max_abs=0))
            cases.append(dict(name=name,dtype=dtype,status='EXACT',direct_first_call_seconds=.1,checks=comparisons))
    receipt=dict(status='TINY_CONSUMER_EXACT_ONLY',scope='TINY_CONSUMER_FALSIFIER_ONLY',rank=rank,world_size=4,label=b['label'],shape=dict(V=256,C=128,T=256),seed=42,argv=b['checker_args'],kernel_sha256=b['support_sha256']['consumer_nki.py'],checker_sha256=b['support_sha256']['check_native.py'],reference_sha256=b['support_sha256']['reference.py'],tolerance=dict(atol=0,rtol=0,criterion='identical bytes'),accepted_speedup=False,whole_step_speedup='UNMEASURED',diagnostic_elapsed_seconds=1,cases=cases,timings=[dict(case=name,dtype=dtype,seconds=[.1,.2,.3],scope='consumer_only_excludes_routing_and_transfer') for name in ('unique','all_same') for dtype in dtypes])
    files['receipts/rank'+str(rank)+'.json']=receipt
    files['rank-environment-r'+str(rank)+'.json']=dict(rank=rank,local_rank=rank,world_size=4,environment=b['environment'],checker_sha256=b['support_sha256']['check_native.py'],checker_argv=b['checker_args'])
    files['rank-wrapper-exit-r'+str(rank)+'.json']=dict(rank=rank,returncode=0,error=None)
# Model separate serialized rank files, not shared in-memory environment/argv objects.
files={name:json.loads(json.dumps(value)) for name,value in files.items()}
v=types.ModuleType('exact_frozen_validator')
exec(compile(raw(COMMIT,P+'validate_receipts.py'),'frozen-validator','exec'),v.__dict__)
class MemoryPath:
    current={}
    def __init__(self,path):self.path=str(path)
    def __truediv__(self,other):return MemoryPath(self.path+'/'+str(other))
    def read_text(self):return json.dumps(self.current[self.path.removeprefix('fixture/')])
v.Path=MemoryPath
def validate(data):
    MemoryPath.current=data
    return v.validate('fixture',b['label'],b['environment'],b['support_sha256'],b['checker_args'])
positive=validate(files)
need([x['exact_comparisons'] for x in positive['ranks']]==[150]*4,'all four independent 150-comparison fixtures')
need([x['complete_cases'] for x in positive['ranks']]==[20]*4,'all four 20 cases')
need([x['timing_samples'] for x in positive['ranks']]==[12]*4,'all four 12 timing samples')
need(positive['accepted_speedup'] is False and positive['full_run_authorized'] is False and positive['source_adoption'] is False,'completion not adoption')
def rejected(name,mutate):
    changed=copy.deepcopy(files);mutate(changed)
    need(all(changed[key]==value for key,value in files.items() if 'rank3.json' not in key and '-r3.json' not in key),'isolated rank3 mutation:'+name)
    try:validate(changed)
    except (RuntimeError,KeyError,TypeError) as exc:
        negatives[name]=type(exc).__name__+': '+str(exc);checks.append('negative:'+name)
    else:raise RuntimeError('FALSE_ACCEPT:'+name)
r='receipts/rank3.json';e='rank-environment-r3.json';x='rank-wrapper-exit-r3.json'
rejected('missing rank3',lambda f:f.pop(r))
rejected('rank3 refused',lambda f:f[r].update(status='REFUSE_OR_UNSUPPORTED'))
rejected('rank3 wrong kernel',lambda f:f[r].update(kernel_sha256='0'*64))
rejected('rank3 env wrong LNC',lambda f:f[e]['environment'].update(NEURON_LOGICAL_NC_CONFIG='1'))
rejected('rank3 argv extra',lambda f:f[r]['argv'].append('--unbound'))
rejected('rank3 label wrong',lambda f:f[r].update(label='other'))
rejected('rank3 nonzero exit',lambda f:f[x].update(returncode=1))
rejected('rank3 missing case',lambda f:f[r]['cases'].pop())
rejected('rank3 started case',lambda f:f[r]['cases'][0].update(status='STARTED'))
rejected('rank3 missing backward',lambda f:f[r]['cases'][0]['checks'].pop())
rejected('rank3 nonfinite gradient',lambda f:f[r]['cases'][0]['checks'][-1].update(finite=False,max_abs=None))
rejected('rank3 signed zero byte difference',lambda f:f[r]['cases'][0]['checks'][-1].update(actual_sha256='b'*64,max_abs=0))
rejected('rank3 exact flag false',lambda f:f[r]['cases'][0]['checks'][-1].update(exact=False))
rejected('rank3 nonzero max error',lambda f:f[r]['cases'][0]['checks'][-1].update(max_abs=1e-30))
rejected('rank3 relaxed tolerance',lambda f:f[r]['tolerance'].update(atol=.01))
rejected('rank3 wrong dtype',lambda f:f[r]['cases'][0]['checks'][-1].update(actual_dtype='torch.bfloat16'))
rejected('rank3 bad tensor digest',lambda f:f[r]['cases'][0]['checks'][-1].update(actual_sha256='x',expected_sha256='x'))
rejected('rank3 timing zero',lambda f:f[r]['timings'][0]['seconds'].__setitem__(0,0))
rejected('rank3 timing NaN',lambda f:f[r]['timings'][0]['seconds'].__setitem__(0,float('nan')))
rejected('rank3 fewer repeats',lambda f:f[r]['timings'][0]['seconds'].pop())
rejected('rank3 claimed speedup',lambda f:f[r].update(accepted_speedup=True))
rejected('rank3 elapsed infinity',lambda f:f[r].update(diagnostic_elapsed_seconds=float('inf')))
print(json.dumps(dict(status='PASS_MECHANICS_ONLY',reviewed_commit=COMMIT,checks=len(checks),check_names=checks,
                      own_positive=positive,own_negative_outcomes=negatives,
                      pins=dict(driver=b['driver_sha256'],binding=sha(raw(COMMIT,P+'NATIVE_BINDING.disabled.json')),manifest=sha(raw(COMMIT,P+'MANIFEST.sha256.json')),source=b['support_sha256']['consumer_nki.py']),
                      source_fresh_review_separate=True,no_host_native_tensors=True,no_launch_authority=True,
                      descendant_cleanup='Explicit Operator postflight remains mandatory after every exit; not proved by driver completion'),indent=2,sort_keys=True))
