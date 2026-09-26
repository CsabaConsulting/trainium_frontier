import ast, base64, hashlib, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

B=Path('/home/poojith-devan/clean36-transport-race-repair-fresh-Fw03lW')
OUT=Path(__file__).parent/sys.argv[1]; OUT.mkdir()
def check(x,msg):
    if not x: raise RuntimeError(msg)
def h(b): return hashlib.sha256(b).hexdigest()
def pin(b): return {'bytes':len(b),'sha256':h(b)}
spec=importlib.util.spec_from_file_location('review_transport',B/'transport/transport.py')
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
old=ast.parse((B/'baseline/prior-transport.py').read_text()); new=ast.parse((B/'transport/transport.py').read_text())
of={n.name:ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef)}
nf={n.name:ast.dump(n) for n in new.body if isinstance(n,ast.FunctionDef)}
check([k for k in of if of[k]!=nf[k]]==['retrieve'],'unexpected changed function')
check(set(nf)-set(of)=={'authenticated_chunks'},'unexpected new function')
ob=json.loads((B/'baseline/PRIOR_TRANSPORT_BINDING.disabled.json').read_bytes()); nb=json.loads((B/'transport/TRANSPORT_BINDING.disabled.json').read_bytes())
check(nb.pop('transport_source_sha256')==h((B/'transport/transport.py').read_bytes()),'source pin')
ob.pop('transport_source_sha256'); check(ob==nb,'binding delta')
files={name:(B/'original-packet'/name).read_bytes() for name in t.PACKET_NAMES}
t.packet_modules(files)
results=[]
def case(name, mode, big=False, cap=False):
    root=OUT/name; root.mkdir(); dest=OUT/(name+'-received')
    good=(bytes(range(256))*2400 if big else b'A'*32)
    bad=b'Z'*len(good)
    values={'artifact.log':good}
    if mode=='valid': values.update({'receipts/.rank0.partial':b'partial0','receipts/rank1.json':b'rank1','hidden/.temp':b'temp','caches/cache':b'cache bytes'})
    for n,b in values.items():
        p=root/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
    def record(p): return pin(p.read_bytes())
    d=SimpleNamespace(RUNTIME_PATHS=[],retained_inputs=lambda *a:{}, file_record=record, ARTIFACT_CAP=1 if cap else 10000000,RANK_CAP=10000000)
    census={'lock_identity':list(t.LOCK_ID),'ps':{},'neuron_ls':{},'fuser':{}}
    wire=[]; mutated=False; reads=[]; orig_pread=os.pread; orig_open=os.open
    target=root/'artifact.log'
    def mutate():
        nonlocal mutated
        if mutated:return
        mutated=True
        if 'replace' in mode:
            replacement=OUT/(name+'-replacement'); replacement.write_bytes(bad); os.replace(replacement,target)
        else:
            with target.open('r+b') as f: f.write(bad); f.flush()
    def emit(item):
        wire.append(item)
        if item['kind']=='file' and item['name']=='artifact.log' and mode in ('replace_header','mutate_header'): mutate()
        if item['kind']=='chunk' and mode=='mutate_between': mutate()
    def op(path,flags,*args,**kwargs):
        if Path(path)==target and mode=='replace_before_open':mutate()
        return orig_open(path,flags,*args,**kwargs)
    def pread(fd,count,offset):
        reads.append(count); check(count<=t.CHUNK,'oversize read')
        raw=orig_pread(fd,count,offset)
        if mode=='mutate_after_read':mutate()
        return raw
    error=None; receiver_error=None
    with patch.object(t,'ROOT',root),patch.object(t,'CHECKPOINT_SHAS',{h(bad)}),patch.object(t,'decode_payload',return_value=({'INPUTS.json':b'{}'},b'{}',{},d,None,None)),patch.object(t,'census',return_value=census),patch.object(os,'pread',side_effect=pread),patch.object(os,'open',side_effect=op):
        try: receipt=t.retrieve({},'fixture',emit)
        except Exception as e:error=str(e)
        try:t.receive([json.dumps(x).encode() for x in wire],dest)
        except Exception as e:receiver_error=str(e)
    transmitted=b''.join(base64.b64decode(x['b64']) for x in wire if x['kind']=='chunk')
    retained=(dest/'files/artifact.log').read_bytes() if (dest/'files/artifact.log').exists() else b''
    check(b'Z' not in transmitted and b'Z' not in retained if not big else bad not in transmitted and bad not in retained,'prohibited bytes copied')
    # For binary fixtures compare each emitted artifact byte to its original offset.
    cursor=0; active=None
    for item in wire:
        if item['kind']=='file':active=item['name'];cursor=0
        if item['kind']=='chunk':
            raw=base64.b64decode(item['b64']);check(raw==values[active][cursor:cursor+len(raw)],'unapproved chunk');cursor+=len(raw)
    if mode=='valid':
        check(error is None and receiver_error is None,'valid failure')
        expected={n:pin(b) for n,b in values.items() if not n.startswith('caches/')}
        check(receipt['files']==expected,'reference files');check(receipt['artifact_bytes']==sum(len(values[n]) for n in expected),'reference total')
        check(receipt['cache_bytes']==11 and receipt['rank_receipt_bytes']=={'0':8,'1':5,'2':0,'3':0},'reference hidden/cache')
        check(receipt['status']==('COMPLETE_COPY_FAILED_GATE' if cap else 'COMPLETE_COPY_PENDING_INDEPENDENT_REVIEW'),'cap behavior')
        for n in expected:check((dest/'files'/n).read_bytes()==values[n],'roundtrip bytes')
    else:
        check(error is not None and receiver_error=='WIRE_INCOMPLETE','race not incomplete')
        check((dest/'retrieval-failure.json').is_file(),'missing failure receipt')
    check(sum(reads)<=3*sum(map(len,values.values())),'unbounded read amplification')
    results.append({'case':name,'first_error':error,'receiver_error':receiver_error,'read_bytes':sum(reads),'max_read':max(reads,default=0),'wire_bytes':len(transmitted),'retained_artifact_bytes':len(retained)})

case('valid_multichunk','valid',True)
case('cap_failed_hidden_partial','valid',True,True)
for mode in ('replace_before_open','replace_header','mutate_header','mutate_after_read','mutate_between'):
    case(mode,mode,mode=='mutate_between')

# Exercise deeper recursion with tiny CHUNK and a streaming digest oracle.
p=OUT/'deep.bin'; raw=bytes(range(251))*700; p.write_bytes(raw)
fd=os.open(p,os.O_RDONLY); counts=[]; orig=os.pread
def counted(fd,n,off): counts.append(n); return orig(fd,n,off)
try:
    with patch.object(t,'CHUNK',32),patch.object(os,'pread',side_effect=counted):
        digest=hashlib.sha256(); total=0
        for chunk in t.authenticated_chunks(fd,0,len(raw),h(raw)):digest.update(chunk);total+=len(chunk)
finally:os.close(fd)
check(total==len(raw) and digest.hexdigest()==h(raw),'deep reference')
check(max(counts)<=32 and sum(counts)==4*len(raw),'deep bounded reads')
results.append({'case':'deep_recursion','bytes':total,'read_bytes':sum(counts),'max_read':max(counts)})
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'verdict':'ACCEPT_NARROW_REPAIR','tests':results},indent=2))
