"""Exact CPU driver main with wholly synthetic host/files/processes; stdlib only."""
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

repo=Path('/home/poojith-devan/Desktop/Trainium')
commit='d524de5cb974aa77883037cdf858fcfddba3113d'
packet='research/v2/proposals/20260926-clean-checkpoint-route-diagnostic/'
def blob(name):return subprocess.check_output(['git','-C',str(repo),'show',commit+':'+packet+name])
b=json.loads(blob('cpu/BINDING.disabled.json'))
tree=ast.parse(blob('cpu/host_driver.py'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
prefix=ast.Module(body=[n for n in tree.body if not isinstance(n,ast.If)],type_ignores=[])
base={};exec(compile(prefix,'exact-definitions-only','exec',optimize=0),base)
checks=[]
def check(name,value):
    if not value:raise RuntimeError('FAIL: '+name)
    checks.append(name)
scenarios=[
 ('complete',{},1,None),
 ('wrong packet root',{'wrong_root':True},0,'WRONG_PACKET_ROOT'),
 ('wrong host',{'host':'bad'},0,'WRONG_HOST'),
 ('STOP',{'stop':True},0,'STOP:'),
 ('RAM below64',{'ram':63*1024**2},0,'RAM_BELOW_64_GIB'),
 ('disk below6.5',{'disk':6*1024**3},0,'DISK_HEADROOM'),
 ('prepare drift',{'prepare':True},0,'PREPARE_SHA'),
 ('checkpoint SHA drift',{'checkpoint':True},0,'CHECKPOINT_SHA'),
 ('checkpoint size drift',{'bytes':1},0,'CHECKPOINT_SHA'),
 ('probe drift',{'support':'probe.py'},0,'SUPPORT_SHA:probe.py'),
 ('guard drift',{'support':'idle_guard.py'},0,'SUPPORT_SHA:idle_guard.py'),
 ('existing receipts',{'exists':'receipts'},0,'ALREADY_EXISTS:receipts'),
 ('nonempty cache',{'cache_nonempty':True},0,'CACHE_NOT_EMPTY:'),
 ('compiler URL mismatch',{'url':True},0,'COMPILER_URL_DIR'),
 ('guard rejects busy or unauthenticated',{'guard':True},0,'FIXTURE_GUARD_REFUSE'),
 ('nonzero inference child',{'child_rc':124},1,None),
 ('checkpoint changed after read',{'changed':True},1,'CHECKPOINT_CHANGED'),
 ('missing or failed raw case',{'validator':True},1,'FIXTURE_ALL25_CASES'),
 ('artifact cap kills process group',{'cap':True},1,'ARTIFACT_CAP_STOPPED_DIAGNOSTIC'),
]
for label,case,children,wanted_error in scenarios:
    records=[];events=[];kills=[];hash_count=[0]
    class P:
        def __init__(self,value):self.value=str(value)
        def __str__(self):return self.value
        def __eq__(self,other):return str(self)==str(other)
        def __truediv__(self,value):return P(self.value+'/'+str(value))
        @property
        def parent(self):return P(self.value.rsplit('/',1)[0])
        @property
        def parts(self):return tuple(self.value.split('/'))
        @property
        def name(self):return self.value.rsplit('/',1)[-1]
        def resolve(self):return P('/wrong/host_driver.py') if case.get('wrong_root') and self.value.endswith('host_driver.py') else self
        def relative_to(self,root):return P(self.value[len(str(root))+1:])
        def glob(self,_):return []
        def rglob(self,_):return [self/'diagnostic.log'] if case.get('cap') else []
        def is_dir(self):return True
        def is_file(self):return True
        def mkdir(self,**kw):events.append('mkdir:'+self.name)
        def iterdir(self):return [self/'unexpected'] if case.get('cache_nonempty') else []
        def open(self,*args):return io.StringIO()
        def read_text(self):return 'MemAvailable: '+str(case.get('ram',65*1024**2))+' kB\n'
        def stat(self):return SimpleNamespace(st_size=case.get('bytes',453048594) if self.value.endswith('final.pt') else 257*1024**2,st_dev=1,st_ino=2)
    def fake_sha(path):
        if str(path).endswith('/prepare.py'):return 'wrong' if case.get('prepare') else b['prepare_sha256']
        return 'wrong' if case.get('support')==path.name else b['source_pins'][path.name]
    def fake_digest(*_):
        hash_count[0]+=1
        wrong=case.get('checkpoint') or (case.get('changed') and hash_count[0]>1)
        return SimpleNamespace(hexdigest=lambda:'wrong' if wrong else b['checkpoint_sha256'])
    def guard(*args):
        events.append('guard')
        if case.get('guard'):raise RuntimeError('FIXTURE_GUARD_REFUSE')
        return 71
    def validate(*args):
        events.append('validate')
        if case.get('validator'):raise RuntimeError('FIXTURE_ALL25_CASES')
        return {'status':'BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE'}
    def module(name,path):return SimpleNamespace(require_idle_under_inherited_lock=guard) if name=='bound_idle_guard' else SimpleNamespace(validate=validate)
    class Child:
        pid=4321
        def __init__(self):self.polls=0
        def poll(self):
            self.polls+=1
            return None if self.polls==1 else case.get('child_rc',0)
        def wait(self,**kwargs):return case.get('child_rc',0)
    def popen(command,**kwargs):
        events.append('child')
        check(label+' exact child argv/fd/session',command==b['command'] and kwargs['pass_fds']==(71,) and kwargs['start_new_session'] is True)
        return Child()
    env=dict(b['environment'])
    if case.get('url'):env['NEURON_COMPILE_CACHE_URL']='bad'
    fake_os=SimpleNamespace(path=SimpleNamespace(lexists=lambda p:(case.get('stop') and str(p).endswith('/STOP')) or p.name==case.get('exists')),statvfs=lambda _:SimpleNamespace(f_bavail=case.get('disk',7*1024**3),f_frsize=1),killpg=lambda pid,sig:kills.append((pid,sig)))
    ctx=dict(base,Path=P,ROOT=P(b['root']),PROJECT=P(b['prepare_path']).parent,__file__=b['root']+'/host_driver.py',
             socket=SimpleNamespace(gethostname=lambda:case.get('host',b['expected_hostname'])),os=fake_os,
             sha=fake_sha,hashlib=SimpleNamespace(file_digest=fake_digest),ENV=env,
             record=lambda name,obj:records.append((name,obj)),load_module=module,
             time=SimpleNamespace(monotonic=lambda:10.,sleep=lambda _:None),
             subprocess=SimpleNamespace(Popen=popen,DEVNULL=-3,STDOUT=-2,TimeoutExpired=subprocess.TimeoutExpired))
    exec(compile(ast.Module(body=[main],type_ignores=[]),'exact-main-synthetic-host','exec',optimize=0),ctx)
    caught=None;rc=None
    try:rc=ctx['main']()
    except Exception as exc:caught=str(exc)
    check(label+' outcome',events.count('child')==children and (caught is None if wanted_error is None else wanted_error in str(caught)))
    check(label+' terminal retained',records[-1][0]=='terminal.json')
    if children:check(label+' authenticated guard precedes child',events.index('guard')<events.index('child'))
    if label=='complete':
        check('complete checkpoint before-after and validator',hash_count[0]==2 and 'validate' in events and rc==0)
        check('success remains diagnostic not quality',records[-1][1]['status']=='BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE' and records[-1][1]['accepted_speedup'] is False and records[-1][1]['no_automatic_next_job'] is True)
    elif label=='nonzero inference child':
        check('nonzero preserved no acceptance',rc==124 and hash_count[0]==2 and 'validate' not in events and records[-1][1]['status']=='DIAGNOSTIC_CHILD_FAILED')
    elif label=='artifact cap kills process group':
        check('cap stop keeps incomplete',bool(kills) and records[-1][1]['complete'] is False)

# Exercise the authentic guard's busy/malformed refusal, with synthetic ps/device text.
guard_path=repo/'research/v2/operator/20260926-common-host-idle-guard/idle_guard.py'
g={};exec(compile(guard_path.read_text(),'exact-shared-guard','exec',optimize=0),g)
ps='10 9 python3 /root/host_driver.py\n9 1 flock /usr/bin/flock -n /lock /usr/bin/python3 -I -B /root/host_driver.py\n'
idle=[{'neuron_device':0,'instance_type':'trn2.3xlarge','nc_count':4,'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3],'neuron_processes':[]}]
for label,ps_text,device,rc,err in [('idle',ps,json.dumps(idle),0,None),('torchrun busy',ps+'22 1 torchrun /bin/torchrun train.py\n',json.dumps(idle),0,'PS_NATIVE_JOB_BUSY'),('bad device JSON',ps,'{}',0,'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'),('ps failed',ps,json.dumps(idle),1,'PS_COMMAND_FAILED')]:
    seen=[]
    def runner(command,**kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command,rc if command==g['PS'] else 0,ps_text if command==g['PS'] else device,'')
    error=None
    try:g['check_idle'](runner,10,9,{})
    except Exception as exc:error=str(exc)
    check('exact guard '+label,error==err)
    if label=='torchrun busy':check('both ps and neuron-ls run on busy',len(seen)==2)
print(json.dumps({'verdict':'PASS_CPU_DRIVER_MOCKED_REFUSALS_ONLY','commit':commit,'checks':len(checks),'tests':checks,'host_actions':False,'tensors':False,'native_review':'NOT_INCLUDED'},indent=2))
