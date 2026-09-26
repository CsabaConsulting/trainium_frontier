import sys, os, json, struct, random, hashlib, types, threading, socket, time, tempfile
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
ROOT=Path('/home/poojith-devan/k2-f0f2-trn21-binding-review-HiaXuP')
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'packet'))
import common, oracle, fixtures, host_driver, coordinator, evidence, worker, consumer_nki
import torch
import monitor

def need(x,msg):
    if not x: raise RuntimeError(msg)

def value(w):
    e=(w>>23)&255; f=w&0x7fffff
    return (-1 if w>>31 else 1)*Fraction((f|(0x800000 if e else 0)))*Fraction(2)**(e-150 if e else -149)

def ref(a,b):
    x=value(a)+value(b)
    if not x:return 0x80000000 if a==b==0x80000000 else 0
    sign=0x80000000 if x<0 else 0; x=abs(x)
    lo,hi=0,0x7f7fffff
    while lo<hi:
        mid=(lo+hi+1)//2
        if value(mid)<=x:lo=mid
        else:hi=mid-1
    if x==value(lo):return sign|lo
    if lo==0x7f7fffff:
        return sign|(0x7f800000 if x>=value(lo)+Fraction(2)**103 else lo)
    dl=x-value(lo); dh=value(lo+1)-x
    return sign|(lo if dl<dh or dl==dh and lo%2==0 else lo+1)

def scalar():
    recipes=({0:0x4b80,128:0x3f80,129:0xcb80},{0:0x3f80,128:0x3380,129:0xbf80},
             {1:0x81,130:0x8080},{2:0x8081,131:0x80},{3:1},{3:0x8001},{4:0x8000,133:0},{4:0x8000,133:0x8000})
    changes={0:0,128:0,129:0,1:127,130:127,2:128,131:128,3:8191,132:8191,4:4095,133:4095}
    rows=[[] for _ in range(8192)]
    for t in range(32768):
        r=changes.get(t,t%8192);need(fixtures.token_id('F2',t)==r,'id recipe');rows[r].append(t)
    n=0
    for c in range(1024):
        out=0
        for _ in range(4):out=ref(out,(0x3800+c)<<16)
        need(out==oracle.recurrence([0x3800+c]*4),'F0 recurrence');n+=1
    f2={}
    for r in (0,1,2,3,4,127,128,129,130,131,132,133,4095,4096,8191):
        f2[r]=[]
        for c in range(8):
            words=[recipes[c].get(t,0) for t in rows[r]]
            need(words==[fixtures.tangent_bits('F2',t,c) for t in rows[r]],'F2 scalar fixture')
            out=0
            for w in words:out=ref(out,w<<16)
            need(out==oracle.recurrence(words),'F2 recurrence');f2[r].append(hex(out));n+=1
    rng=random.Random(42)
    pairs=[(0x80000000,0x80000000),(0x80000000,0),(0x7f7fffff,0x7f7fffff)]
    pairs += [(rng.randrange(0x7f800000)|(rng.randrange(2)<<31),rng.randrange(0x7f800000)|(rng.randrange(2)<<31)) for _ in range(1024)]
    for a,b in pairs:need(oracle.add32(a,b)==ref(a,b),'independent addition')
    return dict(recurrences=n,additional_finite_pairs=len(pairs),f2=f2,method='Fraction exact sum and monotonic binary32 lattice nearest-even search')

def disabled():
    errors=[]
    for change in ({},{'launch_authorized':True},{'launch_authorized':True,'packet_root':str(common.HERE)}):
        b=dict(host_driver.BINDING,**change)
        with patch.object(host_driver,'BINDING',b):
            try:host_driver.main()
            except RuntimeError as e:errors.append(str(e))
            else:raise RuntimeError('disabled accepted')
    host_driver.require_original_lock(dict(device=66305,inode=596823))
    for d,i in ((66306,596823),(66305,596824),(None,None)):
        try:host_driver.require_original_lock(dict(device=d,inode=i))
        except RuntimeError:pass
        else:raise RuntimeError('wrong lock accepted')
    return errors

# Actual unedited worker.main, coordinator.route, Invocation and blob handling.
# CPU seams: SDK initialization/compile/embedding, tensor loads/native device,
# tiny transported tensor payloads (16 B) with declared production domain,
# and metadata revalidation. This is routing/fault-injection, NOT native proof.
class Tensor:
    def __init__(self,words,dtype='torch.float32',shape=(8192,1024),device='neuron'):
        self.raw=torch.tensor(words,dtype=torch.int32);self.dtype=getattr(torch,dtype.split('.')[-1]);self.shape=shape
        self.device=types.SimpleNamespace(type=device);self.grad=None
    def numel(self):return 4
    def element_size(self):return 4
    def detach(self):return self
    def cpu(self):return self.raw
    def requires_grad_(self,v):return self

def worker_routes(mode):
    root=Path(tempfile.mkdtemp(prefix=('opt-' if sys.flags.optimize else 'normal-')+mode+'-',dir=OUT))
    writer=evidence.Writer(root/'artifacts')
    f0=[0x3f800000]*4;f2=[0x00010000,0x80010000-2**32,0,0]
    mapping={};cases={}
    for fixture,words in [('F0',f0),('F2',f2)]:
        expected=writer.put([struct.pack('<4i',*words)])
        inp=writer.put([struct.pack('<4i',*( [10 if fixture=='F0' else 20]*4))])
        mapping[inp['sha256']]=(fixture,words)
        cases[fixture]=dict(expected=expected,tangent=inp,ids=inp,weight=inp,forward=inp)
    cases['F2']['weight']=cases['F0']['weight']
    left,right=socket.socketpair();left.settimeout(5);right.settimeout(5)
    class Sock:
        def connect(self,x):pass
        def __getattr__(self,k):return getattr(right,k)
    counts=dict(kernel=0,eager=0,compiled=0,build=0); state={}; errors=[]; receipts=[]
    def load(root,item,name,ledger):
        fixture,words=mapping[item['blob']['sha256']]
        t=Tensor([10 if fixture=='F0' else 20]*4,item['dtype'],item['shape']);t.fixture=fixture
        if name=='weight':state['weight']=t
        return t
    def kernel(t,*args):
        counts['kernel']+=1;words=f0 if t.fixture=='F0' else f2
        if mode=='skip_return_none':return None
        if mode=='cpu_oracle':return Tensor(words,device='cpu')
        if mode=='zero_native':return Tensor([0]*4)
        if mode=='stale_f0':return Tensor(f0)
        if mode=='missing_owner':return Tensor(words[:2]+[0,0])
        if mode=='one_bit':return Tensor([words[0]^1]+words[1:])
        if mode=='nonfinite':return Tensor([0x7f800000]+words[1:])
        if mode=='wrong_dtype':return Tensor(words,dtype='torch.bfloat16')
        if mode=='wrong_shape':return Tensor(words,shape=(1024,8192))
        if mode=='signed_zero' and t.fixture=='F2':return Tensor(words[:2]+[-2**31,0])
        return Tensor(words)
    def build(**kw):need(kw=={'enabled':True},'enabled kernel');counts['build']+=1;return kernel
    def embedding(ids,weight):
        counts[state.get('route_kind','eager')]+=1
        fixture=ids.fixture
        f=Tensor([10 if fixture=='F0' else 20]*4,'torch.bfloat16',(32768,1024))
        def backward(t):weight.grad=Tensor(f0 if fixture=='F0' else f2)
        f.backward=backward;f.to=lambda dtype:f;return f
    def compile_(fn,**kw):
        def call(*a):state['route_kind']='compiled';v=fn(*a);state['route_kind']='eager';return v
        return call
    def domains(case,ordinary):
        x=dict(tangent=(case['tangent'],'torch.bfloat16',[32768,1024]),ids=(case['ids'],'torch.int32',[32768]))
        if ordinary:x['weight']=(case['weight'],'torch.float32',[8192,1024])
        else:
            for k in ['counts','starts','positions','maximums']:x[k]=(case['ids'],'torch.int32',[4,1])
        return x
    env={'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'4','NEURON_LOGICAL_NC_CONFIG':'2'}
    def run():
        try:worker.main()
        except BaseException as e:errors.append(type(e).__name__+': '+str(e));right.close()
    original_send=worker.send
    original_capture=worker.capture
    def send_(sock,message):
        if mode=='forged_output_nonce' and message.get('event')=='blob' and message.get('role')=='output':
            message=dict(message,nonce='not-the-issued-invocation')
        return original_send(sock,message)
    def capture_(*args):
        if mode=='missing_output_blob' and args[2]=='output':return
        return original_capture(*args)
    from contextlib import ExitStack
    with ExitStack() as stack:
        for ctx in [patch.dict(os.environ,env),patch.object(sys,'argv',['worker.py',str(root)]),
                    patch.dict(sys.modules,{'torch_neuronx':types.ModuleType('torch_neuronx')}),
                    patch.object(worker.socket,'socket',lambda *a:Sock()),
                    patch.object(worker,'load_tensor',load),patch.object(consumer_nki,'build_kernel',build),
                    patch.object(worker,'send',send_),patch.object(worker,'capture',capture_),
                    patch.object(coordinator,'inputs',domains),patch.object(coordinator,'revalidate',lambda *a:None),
                    patch.object(torch,'neuron',types.SimpleNamespace(synchronize=lambda:None),create=True),
                    patch.object(torch,'compile',compile_),patch.object(torch.nn.functional,'embedding',embedding),
                    patch.object(torch.distributed,'init_process_group',lambda *a:None),
                    patch.object(torch.distributed,'destroy_process_group',lambda:None),
                    patch.object(torch,'set_num_interop_threads',lambda *a:None),
                    patch.object(worker.importlib.metadata,'version',lambda x:'CPU_TEST_SEAM')]:stack.enter_context(ctx)
        thread=threading.Thread(target=run);thread.start()
        hello=common.receive(left)
        try:
            for ordinal,(fixture,kind) in enumerate(common.ROUTES):
                receipts.append(coordinator.route(left,writer,0,common.identity(os.getpid()),ordinal,cases[fixture],cases['F0'],time.monotonic()+5,common.source_pins()))
            common.send(left,{'event':'finish'})
        except BaseException as e:errors.append('coordinator: '+type(e).__name__+': '+str(e))
        finally:left.close();thread.join(6)
        need(not thread.is_alive(),'test worker leaked')
    right.close()
    return dict(mode=mode,artifact_root=str(root),accepted_ordinals=[r['ordinal'] for r in receipts],calls=counts,errors=errors,
                caveat='CPU SDK/load/device seams; metadata revalidation bypassed for tiny transport. Source hashes checked unmodified. No admission/native result claimed.')

def caps_and_metadata():
    base=dict(at=2,memavailable=4*common.GIB,rss=20*common.GIB,free=[int(6.5*common.GIB)],cache=6*common.GIB,artifacts=2*common.GIB,logs=128*common.MIB)
    monitor.guards(base,0); failures=[]
    for name,change in [('at',2.00001),('memavailable',base['memavailable']-1),('rss',base['rss']+1),('free',[base['free'][0]-1]),('cache',base['cache']+1),('artifacts',base['artifacts']+1),('logs',base['logs']+1)]:
        try:monitor.guards(dict(base,**{name:change}),0)
        except RuntimeError as e:failures.append(str(e))
        else:raise RuntimeError('cap accepted '+name)
    ids=[0,1,0,1]; m=fixtures.construct_metadata(ids,2,1);fixtures.validate_metadata(ids,m,2,1)
    import copy
    for index in range(4):
        bad=copy.deepcopy(m);bad[index][0]+=1
        try:fixtures.validate_metadata(ids,bad,2,1)
        except ValueError:pass
        else:raise RuntimeError('metadata accepted')
    baseline=json.loads((ROOT/'packet/PROTOTYPE_PINS.json').read_text());now=common.source_pins()
    changed=sorted(k for k in baseline if now.get(k)!=baseline[k])
    need(changed==['binding.py','common.py','host_driver.py','test_repairs.py'],'unexpected copied delta')
    return dict(guard_negatives=failures,metadata_negatives=4,changed_prototype_files=changed)

results={'pins':common.source_pins(),'scalar':scalar(),'disabled':disabled(),'caps_and_metadata':caps_and_metadata(),'routes':[]}
for mode in ('reference_native_seam','skip_return_none','cpu_oracle','zero_native','stale_f0','missing_owner','one_bit','nonfinite','wrong_dtype','wrong_shape','signed_zero','forged_output_nonce','missing_output_blob'):
    result=worker_routes(mode);results['routes'].append(result)
    expected={'reference_native_seam':list(range(7)),'stale_f0':list(range(4)),'signed_zero':list(range(4))}.get(mode,[])
    need(result['accepted_ordinals']==expected,'unexpected route outcomes '+str(result))
need(common.source_pins()==results['pins'],'source changed')
print(json.dumps(results,indent=2,sort_keys=True))
