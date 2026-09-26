"""Independent scoped mechanics review; no Go/SDK/host/profile access."""
import argparse, ast, base64, hashlib, json, resource, subprocess, sys, types
from pathlib import Path
REF='99d7c261316a6597d8324c7804912365e67ce02b'
TARGET=Path('/home/poojith-devan/Desktop/Trainium-p-readiness-rlimit-repair-20260926')
REL='research/v2/proposals/20260926-p-offline-decoder-readiness-r2-trn21/'
OLD='research/v2/proposals/20260926-p-offline-decoder-trn21/'
checks=[];low=536870912;high=17179869184
def ck(name,ok):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def git(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=TARGET)
def sha(b):return hashlib.sha256(b).hexdigest()
raw=git(REF,REL+'host_driver.py');tree=ast.parse(raw)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='readiness_guard')
def wrap(res,guard,receipt):
    ns={'resource':res,'LOCK':'/unchanged/lock','ROOT':Path('/new/r2')}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-wrapper','exec'),ns)
    return ns['readiness_guard'](types.SimpleNamespace(require_idle_under_inherited_lock=guard),{'parent_address_space_bytes':low,'decoder_address_space_bytes':high},receipt)
if len(sys.argv)==3 and sys.argv[1]=='--actual-worker':
    mode=sys.argv[2];resource.setrlimit(resource.RLIMIT_AS,(low,high));receipts=[];observed=[]
    def actual_guard(lock,path,cb):
        output=subprocess.check_output([sys.executable,'-I','-B','-c','import resource,json;print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))'],timeout=5)
        observed.append(json.loads(output));ck('real child inherited ceiling',observed[-1]==[high,high])
        if mode=='failure':raise RuntimeError('READINESS_REFUSAL_SENTINEL')
        return 11
    err=None;result=None
    try:result=wrap(resource,actual_guard,lambda n,v:receipts.append((n,v)))
    except BaseException as e:err=e
    ck('real parent restored',resource.getrlimit(resource.RLIMIT_AS)==(low,high))
    ck('actual error preserved',(mode=='failure' and isinstance(err,RuntimeError) and str(err)=='READINESS_REFUSAL_SENTINEL') or (mode=='success' and err is None and result==11))
    ck('restoration receipt',len(receipts)==1 and receipts[0][1]['restored']==[low,high])
    print(json.dumps({'mode':mode,'checks':checks,'inherited':observed,'final':resource.getrlimit(resource.RLIMIT_AS),'error':None if err is None else str(err)}))
    sys.exit(0)
P=json.loads(git(REF,REL+'EXPERIMENT_PROPOSAL.json'));Q=json.loads(git('b529468f',OLD+'EXPERIMENT_PROPOSAL.json'))
B=json.loads(git(REF,REL+'OFFLINE_BINDING.disabled.json'));M=json.loads(git(REF,REL+'MANIFEST.sha256.json'))
ck('proposal first exact',git('7519c517',REL+'EXPERIMENT_PROPOSAL.json')==git(REF,REL+'EXPERIMENT_PROPOSAL.json'))
for name,digest in M['files'].items():
    b=git(REF,REL+name);ck('manifest '+name,sha(b)==digest and b==(TARGET/REL/name).read_bytes())
ck('driver exact',sha(raw)=='616fc37c756a5be2cc04f8fb4c13e551a8e0bae4e07e2eeda00475fd1db5e1df')
ck('binding exact',sha(git(REF,REL+'OFFLINE_BINDING.disabled.json'))=='3c08ce4b48d98596bd8c8594fc2f9f8fcb92e41669779cd93732de335ce856cb')
ck('manifest exact',sha(git(REF,REL+'MANIFEST.sha256.json'))=='4e6bc2f3264d9d481713fe88ec8d2ce1e1ecd221f3370ddb9286bed14b6a40d7')
for k in ['source_root','inputs','input_count','neff_count','input_bytes','limits','binary','expected_version','source_profile_sha256','source_inventory_sha256','capability_help_sha256']:
    ck('unchanged '+k,P[k]==Q[k])
ck('all86/all80/bytes',len(P['inputs'])==86 and sum(x['path'].endswith('.neff') for x in P['inputs'])==80 and sum(x['bytes'] for x in P['inputs'])==4310546731)
ck('only decoder session prefix',[v.replace(Q['root'],P['root']) for v in Q['core_argv']]==P['core_argv']==B['core_argv'])
ck('new retry root and label',P['root']!=Q['root'] and P['label']!=Q['label'] and P['root']==B['root'] and P['label']==B['label'])
ck('disabled',all(P[k] is False and B[k] is False for k in ['stage_authorized','launch_authorized','host_reserved']))
ck('unchanged support',git(REF,REL+'decoder_support.py')==git('b529468f',OLD+'decoder_support.py'))
for name,h in B['runtime_sha256'].items():
    b=git(REF,REL+name) if name not in ['idle_guard.py','dispatch_driver.py'] else git(REF,'research/v2/operator/20260926-common-host-idle-guard/'+name)
    ck('runtime '+name,sha(b)==h)
original=git('b529468f',OLD+'host_driver.py').decode().replace(Q['root'],P['root']).replace(sha(git('b529468f',OLD+'EXPERIMENT_PROPOSAL.json')),sha(git(REF,REL+'EXPERIMENT_PROPOSAL.json')))
original=original.replace("guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py', record)","readiness_guard(guard, limits, record)").replace("guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py',\n","readiness_guard(guard, limits,\n")
ck('full remaining AST identity',ast.dump(ast.Module(body=[n for n in tree.body if n is not fn],type_ignores=[]))==ast.dump(ast.parse(original)))
ck('exactly pre/post wrappers',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='readiness_guard' for n in ast.walk(tree))==2)
ck('unchanged FD propagation',b'lock_fd' in raw and b'if post_fd != lock_fd:' in raw)
# Independent failure matrix, including silent restoration failure absent from supplied tests.
class R:
    RLIMIT_AS=9
    def __init__(self,mode):self.mode=mode;self.value=(low+1,high) if mode=='initial' else (low,high);self.events=[]
    def getrlimit(self,k):ck('AS only get',k==9);return self.value
    def setrlimit(self,k,v):
        ck('AS only mutation',k==9);self.events.append(v)
        if self.mode=='elevate' and v==(high,high):raise OSError('ELEVATION')
        if self.mode=='restore' and v==(low,high):raise OSError('RESTORATION')
        if self.mode=='silent_restore' and v==(low,high):return
        self.value=v
outcomes=[]
for mode in ['ok','refusal','baseexception','receipt','elevate','restore','silent_restore','initial']:
    rr=R(mode);receipts=[];calls=[]
    def guard(lock,path,callback):
        ck('guard elevated exactly',rr.value==(high,high));calls.append((lock,str(path)))
        if mode=='refusal':raise ValueError('GUARD_ORIGINAL')
        if mode=='baseexception':raise KeyboardInterrupt('SIGNAL_ORIGINAL')
        return 27
    def receipt(n,v):
        ck('receipt after restore',rr.value==(low,high));receipts.append((n,v))
        if mode=='receipt':raise OSError('RECEIPT_ORIGINAL')
    got=None;err=None
    try:got=wrap(rr,guard,receipt)
    except BaseException as e:err=e
    ck('failure never succeeds '+mode,(err is None)==(mode=='ok'))
    if mode not in ['restore','silent_restore','initial']:ck('finally restoration '+mode,rr.value==(low,high))
    if mode in ['elevate','initial']:ck('no readiness after failed limit '+mode,not calls)
    if mode in ['restore','silent_restore','initial']:ck('no false restoration receipt '+mode,not receipts)
    if mode=='refusal':ck('same refusal exception',type(err) is ValueError and str(err)=='GUARD_ORIGINAL')
    if mode=='baseexception':ck('BaseException propagates',type(err) is KeyboardInterrupt and str(err)=='SIGNAL_ORIGINAL')
    if mode=='silent_restore':ck('readback mismatch refuses',str(err)=='READINESS_PARENT_AS_NOT_RESTORED')
    if mode=='ok':ck('same FD and context',got==27 and calls==[('/unchanged/lock','/new/r2/host_driver.py')])
    outcomes.append(dict(mode=mode,error=None if err is None else type(err).__name__+': '+str(err),final=list(rr.value),guard_calls=len(calls),receipts=len(receipts)))
actual=[]
for mode in ['success','failure']:
    cp=subprocess.run([sys.executable,'-I','-B']+(['-O'] if not __debug__ else [])+[str(Path(__file__).resolve()),'--actual-worker',mode],capture_output=True,text=True,timeout=10)
    ck('real inherited/restored '+mode,cp.returncode==0 and not cp.stderr);actual.append(json.loads(cp.stdout))
failure=json.loads(git('b29da2cf','research/v2/operator/20260926-trn21-p-offline-decoder/completed/results-bundle.json'))
for name,item in failure['files'].items():
    decoded=base64.b64decode(item['base64'],validate=True)
    ck('original archive hash '+name,sha(decoded)==item['sha256'] and len(decoded)==item['bytes'])
for name in ['terminal.json','under-lock-idle.json']:
    ck('original raw preserved '+name,git(REF,REL+'original-failure/'+name)==base64.b64decode(failure['files'][name]['base64'],validate=True))
ck('original no decoder',json.loads(git(REF,REL+'original-failure/explicit-process-postflight.json'))['unstarted_output_paths']==dict.fromkeys(['input','home','tmp','version.stdout','version.stderr','decoder.stdout','decoder.stderr'],False))
print(json.dumps(dict(status='PASS_SCOPED_MECHANICS_NOT_NATIVE_GO_OR_LAUNCH',checks=len(checks),check_names=checks,mocked_outcomes=outcomes,actual_linux_inheritance=actual,driver_sha256=sha(raw),limits=P['limits'],notes=['I authored original decoder/helper, not this r2 patch; this independently tests only the new limit wrapper plus unchanged scope. Prior independent containment review22233aea remains separate.','No Go binary, SDK, native device, original trace, model or host accessed. Real local tests allocate no large memory.','Installed neuron-ls16GiB compatibility and decoder ingestion remain unproved; failure stays refusal with no larger retry.']),indent=2))
