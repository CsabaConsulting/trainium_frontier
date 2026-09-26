"""Independent stdlib-only mechanics review. All driver effects are mocked."""
import ast, copy, hashlib, importlib.util, io, json, math, random, struct, subprocess
from pathlib import Path
from types import SimpleNamespace

R=Path(__file__).resolve().parents[4]
P=R/'research/v2/proposals/20260926-k2-fp32-first-discriminator'
B=json.loads((P/'NATIVE_BINDING.disabled.json').read_text())
checks=[]
def ok(value,label):
    if not value: raise RuntimeError(label)
    checks.append(label)
def sha(x):return hashlib.sha256(x).hexdigest()
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=R)
prefix='research/v2/proposals/20260926-k2-fp32-first-discriminator/'
for name,pin in {**B['support_sha256'],'host_driver.py':B['driver_sha256']}.items():
    raw=(P/name).read_bytes();ok(sha(raw)==pin,'pin '+name);ok(raw==blob('fa1a3652',prefix+name),'frozen '+name)
manifest=json.loads((P/'MANIFEST.sha256.json').read_text())
for name,pin in manifest['files'].items():ok(sha((P/name).read_bytes())==pin,'manifest '+name)
for name in ['host_driver.py','rank_entry.py']:
    old=ast.parse(blob('710ea677','research/v2/proposals/20260926-k2-symmetric-native-trn21/'+name))
    new=ast.parse((P/name).read_text())
    def body(t):return ast.dump(ast.Module(body=[n for n in t.body if not isinstance(n,ast.Assign)],type_ignores=[]))
    ok(body(old)==body(new),'exact functional AST '+name)
    def literals(t):return {n.targets[0].id for n in t.body if isinstance(n,ast.Assign)}
    ok(literals(old)==literals(new),'same constant names '+name)
constants={}
for node in ast.parse((P/'host_driver.py').read_text()).body:
    if isinstance(node,ast.Assign):
        value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
        constants[node.targets[0].id]=ast.literal_eval(value)
for field,key in [('ROOT','root'),('LABEL','label'),('EXPECTED_HOSTNAME','expected_hostname'),('PINS','support_sha256'),('COMMAND','command'),('CHECKER_ARGS','checker_args'),('ENV','environment'),('CACHE_KEYS','cache_keys')]:ok(constants[field]==B[key],'bound driver constant '+field)
entry={}
for node in ast.parse((P/'rank_entry.py').read_text()).body:
    if isinstance(node,ast.Assign):
        value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
        entry[node.targets[0].id]=ast.literal_eval(value)
ok(entry['ROOT']==B['root'] and entry['CHECKER_SHA']==B['support_sha256']['check_native.py'],'entry root checker pin')
ok(set(entry['KEYS'])==set(B['environment']),'entry captures every bound environment key')
ok(B['command'][:8]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'],'one four-rank bounded child')
ok(B['command'][9:]==B['checker_args'] and '--timing-repeats' not in B['command'],'exact diagnostic argv no timing')
ok(B['outer_command']==['/usr/bin/flock','-n',B['original_lock']['path'],'/usr/bin/python3','-I','-B',B['root']+'/host_driver.py'],'original lifetime flock command')
ok(B['original_lock']['inode']==832589 and B['original_lock']['device']==66305,'original trn22 lock identity')
ok(B['host']=='trn22' and B['expected_hostname']=='ip-172-31-32-253','exact host')
ok(all(B[k] is False for k in ['stage_authorized','launch_authorized','operator_assigned','host_reserved','source_adoption','automatic_full','automatic_eval','automatic_next_job']),'disabled separate gates')
ok(B['postflight']['required'] is True and 'any exit' in B['postflight']['owner'] and 'ps AND neuron-ls' in B['postflight']['method'],'explicit every-outcome postflight requirement')
ok(B['environment']['NEURON_COMPILE_CACHE_URL']==B['environment']['NEURON_COMPILE_CACHE_DIR'] and len(B['cache_keys'])==6,'six cache/url identity')
for k in B['cache_keys']:ok(B['environment'][k].startswith(B['root']+'/caches/'),'private cache '+k)

spec=importlib.util.spec_from_file_location('reviewed_validator',P/'validate_receipts.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
rng=random.Random(42)
g=struct.pack('<32768f',*(rng.randint(-256,256)/32 for _ in range(32768)))
ok(sha(g)==v.EXPECTED_G,'own scalar fixture expected gradient hash')
def t(shape,dtype,digest,n):return dict(shape=shape,dtype=dtype,sha256=digest,nonzero=n,finite=True)
def tensor_bytes(vals,fmt):return sha(struct.pack('<'+str(len(vals))+fmt,*vals))
grad=t([256,128],'torch.float32',sha(g),32709)
weight=t([256,128],'torch.float32',tensor_bytes([1.0]*32768,'f'),32768)
inputs={'upstream':grad,'weight':weight,'ids':t([256],'torch.int32',tensor_bytes(list(range(256)),'i'),255),'counts':t([256,1],'torch.int32',tensor_bytes([1]*256,'i'),256),'starts':t([256,1],'torch.int32',tensor_bytes(list(range(256)),'i'),255),'positions':t([256,1],'torch.int32',tensor_bytes(list(range(256)),'i'),255),'maximums':t([2,1],'torch.int32',tensor_bytes([1,1],'i'),2)}
def cmp(value):return dict(exact=True,max_abs=0.0,actual=copy.deepcopy(value),expected=copy.deepcopy(value))
case=dict(case='unique',dtype='torch.float32',metadata_group_bounds=[1,1],dense_cpu_oracle=cmp(grad),routes=[])
for route in v.ROUTES:
    comparisons={'gradient':cmp(grad)}
    if route!='direct_consumer':comparisons['output']=cmp(weight)
    case['routes'].append(dict(name=route,status='EXACT',inputs_unchanged=True,inputs_before=copy.deepcopy(inputs),inputs_after=copy.deepcopy(inputs),checks=comparisons))
files={}
for rank in range(4):
    files[f'/fake/receipts/rank{rank}.json']=dict(status='FP32_FIRST_DIAGNOSTIC_OBSERVED_EXACT',returncode=0,scope='FP32_FIRST_SOURCE_NEUTRAL_DIAGNOSTIC_ONLY',rank=rank,world_size=4,label=B['label'],shape=dict(V=256,T=256,C=128),seed=42,argv=B['checker_args'],checker_sha256=B['support_sha256']['check_native.py'],kernel_sha256=B['support_sha256']['consumer_nki.py'],reference_sha256=B['support_sha256']['reference.py'],tolerance=dict(atol=0,rtol=0,criterion='identical bytes'),route_order=list(v.ROUTES),first_custom_compiled_fixture='unique/fp32',native_qualification=False,source_adoption=False,accepted_speedup=False,public_20m=None,whole_step_speedup='UNMEASURED',timings=[],original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',no_training=True,no_eval=True,diagnostic_elapsed_seconds=1.0,case=copy.deepcopy(case))
    files[f'/fake/rank-environment-r{rank}.json']=dict(rank=rank,local_rank=rank,world_size=4,environment=copy.deepcopy(B['environment']),checker_sha256=B['support_sha256']['check_native.py'],checker_argv=B['checker_args'])
    files[f'/fake/rank-wrapper-exit-r{rank}.json']=dict(rank=rank,returncode=0,error=None)
def validate(f):
    class VPath:
        def __init__(self,p):self.p=str(p)
        def __truediv__(self,x):return VPath(self.p+'/'+str(x))
        def read_text(self):return json.dumps(f[self.p])
    old=v.Path;v.Path=VPath
    try:return v.validate('/fake',B['label'],B['environment'],B['support_sha256'],B['checker_args'])
    finally:v.Path=old
ok(validate(files)['native_qualification'] is False,'own synthetic exact completion remains diagnostic')
key='/fake/receipts/rank2.json'
mutations=[]
for field,bad in [('status','PASS'),('returncode',False),('returncode',1),('rank',3),('world_size',3),('label','wrong'),('seed',41),('argv',[]),('checker_sha256','0'*64),('kernel_sha256','0'*64),('reference_sha256','0'*64),('tolerance',dict(atol=1,rtol=0,criterion='identical bytes')),('native_qualification',True),('public_20m',.9),('timings',[1]),('original_twenty_case_gate','PASS'),('no_eval',False),('diagnostic_elapsed_seconds',float('nan')),('diagnostic_elapsed_seconds',float('inf')),('diagnostic_elapsed_seconds',-1)]:
    mutations.append((field+repr(bad),lambda f,k=field,x=bad:f[key].update({k:x})))
mutations += [('missing rank',lambda f:f.pop('/fake/receipts/rank3.json')),('missing exit',lambda f:f.pop('/fake/rank-wrapper-exit-r1.json')),('failed wrapper',lambda f:f['/fake/rank-wrapper-exit-r2.json'].update(returncode=1)),('boolean wrapper rc',lambda f:f['/fake/rank-wrapper-exit-r2.json'].update(returncode=False)),('wrong env',lambda f:f['/fake/rank-environment-r2.json']['environment'].update(NEURON_LOGICAL_NC_CONFIG='1')),('wrong entry argv',lambda f:f['/fake/rank-environment-r2.json'].update(checker_argv=[])),('missing route',lambda f:f[key]['case']['routes'].pop()),('duplicate route',lambda f:f[key]['case']['routes'][2].update(name='eager_custom')),('error route',lambda f:f[key]['case']['routes'][2].update(error='x')),('input drift',lambda f:f[key]['case']['routes'][1]['inputs_after']['weight'].update(sha256='0'*64)),('wrong fixture',lambda f:f[key]['case'].update(dtype='torch.bfloat16'))]
for value in [None,float('nan'),float('inf'),False,.01]:mutations.append(('bad error '+repr(value),lambda f,x=value:f[key]['case']['routes'][2]['checks']['gradient'].update(max_abs=x)))
for field,value in [('finite',False),('nonzero',0),('nonzero',False),('sha256','0'*64),('dtype','torch.bfloat16'),('shape',[128,256])]:mutations.append(('bad gradient '+field,lambda f,k=field,x=value:f[key]['case']['routes'][2]['checks']['gradient']['actual'].update({k:x})))
mutations.append(('cross-rank consistent local weight drift',lambda f:[x.update(sha256='0'*64) for route in f[key]['case']['routes'] for x in [route['inputs_before']['weight'],route['inputs_after']['weight']]]))
for label,mutation in mutations:
    changed=copy.deepcopy(files);mutation(changed)
    try:validate(changed)
    except (RuntimeError,KeyError,ValueError,TypeError):checks.append('refuses '+label)
    else:raise RuntimeError('FALSE_COMPLETION '+label)

def driver_case(scenario):
    records={};events=[];created=set();calls=[]
    class VP:
        def __init__(self,p):self.p=str(p)
        def __str__(self):return self.p
        def __truediv__(self,x):return VP(self.p+'/'+str(x))
        def __eq__(self,x):return str(self)==str(x)
        def resolve(self):return self
        @property
        def parent(self):return VP(self.p.rsplit('/',1)[0])
        def glob(self,p):return []
        def is_dir(self):return True
        def mkdir(self,exist_ok=False):
            if self.p in created:raise FileExistsError(self.p)
            created.add(self.p)
        def iterdir(self):return iter([])
        def stat(self):return SimpleNamespace(st_dev=66305,st_ino=123)
        def open(self,*args,**kw):return io.StringIO()
    ns={'__name__':'review_only','__file__':B['root']+'/host_driver.py'}
    exec(compile((P/'host_driver.py').read_text(),'<exact-driver>','exec'),ns)
    ns['Path']=VP;ns['ROOT']=VP(B['root']);ns['PROJECT']=VP(str(ns['PROJECT']))
    ns['socket']=SimpleNamespace(gethostname=lambda:'wrong' if scenario=='host' else B['expected_hostname'])
    ns['os']=SimpleNamespace(path=SimpleNamespace(lexists=lambda p:(scenario=='stop' and str(p).endswith('/STOP')) or (scenario=='cache' and str(p)==B['root']+'/caches')),statvfs=lambda p:SimpleNamespace(f_bavail=0 if scenario=='disk' else 11*1024**3,f_frsize=1))
    ns['sha']=lambda p:'wrong' if scenario=='hash' else B['support_sha256'][str(p).rsplit('/',1)[1]]
    ns['record']=lambda n,x:records.update({n:copy.deepcopy(x)})
    def guard(*a):
        events.append('guard')
        if scenario=='guard':raise RuntimeError('BUSY')
        return 7
    def complete(*a):
        events.append('validate')
        if scenario=='validator':raise RuntimeError('INCOMPLETE')
        return dict(status='FP32_FIRST_OBSERVATIONS_EXACT_NOT_NATIVE_QUALIFICATION')
    ns['load_module']=lambda n,p:SimpleNamespace(require_idle_under_inherited_lock=guard) if n=='bound_idle_guard' else SimpleNamespace(validate=complete)
    def call(command,**kw):
        events.append('child');calls.append((command,kw));return 17 if scenario=='child' else 0
    ns['subprocess']=SimpleNamespace(call=call,DEVNULL=-3,STDOUT=-2)
    error=None
    try:rc=ns['main']()
    except RuntimeError as e:error=str(e);rc=None
    if scenario in ('host','stop','disk','hash','cache','guard'):
        ok(not calls and error is not None and records['terminal.json']['complete'] is False,'driver refuses before child '+scenario)
    elif scenario=='child':
        ok(rc==17 and events==['guard','child'] and records['terminal.json']['returncode']==17,'nonzero child preserved skips validator')
    elif scenario=='validator':
        ok(error=='INCOMPLETE' and records['terminal.json']['complete'] is False,'validator refusal propagates')
    else:
        ok(rc==0 and events==['guard','child','validate'] and records['terminal.json']['complete'] is True,'one sequential guarded child then validator')
        ok(len(created)==8 and len(records['fresh-caches.json']['directories'])==6,'absent six fresh caches and metrics before child')
    if calls:
        ok(len(calls)==1 and calls[0][0]==B['command'] and calls[0][1]['pass_fds']==(7,) and calls[0][1]['env']==B['environment'],'exact single child and inherited lock '+scenario)
    return dict(scenario=scenario,returncode=rc,error=error,events=events,terminal=records['terminal.json'])
drivers=[driver_case(x) for x in ['success','host','stop','disk','hash','cache','guard','child','validator']]
print(json.dumps(dict(status='MECHANICS_CHECKS_PASS_NOT_SOURCE_OR_NATIVE_APPROVAL',checks=len(checks),check_names=checks,driver_fixtures=drivers,packet_commit='fa1a3652',binding_sha256=sha((P/'NATIVE_BINDING.disabled.json').read_bytes()),effects='stdlib only; no tensors or host; driver effects mocked',original_failure_preserved=True,launch_authorized=False),indent=2,allow_nan=False))
