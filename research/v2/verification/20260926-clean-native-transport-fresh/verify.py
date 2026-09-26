import base64, contextlib, fcntl, hashlib, importlib.util, json, os, sys
from pathlib import Path
from unittest.mock import patch

B = Path('/home/poojith-devan/clean36-trn21-transport-fresh-scdhUJ')
OUT = Path(__file__).parent / (sys.argv[1] if len(sys.argv)>1 else ('optimized' if sys.flags.optimize else 'normal'))
OUT.mkdir()
def check(ok, msg):
    if not ok: raise RuntimeError(msg)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def load(path):
    spec = importlib.util.spec_from_file_location('reviewed_transport', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
t = load(B/'transport/transport.py')
files = {n:(B/'original-packet'/n).read_bytes() for n in t.PACKET_NAMES}
d,g,idle = t.packet_modules(files)
def reject(fn, text):
    try: fn()
    except Exception as e:
        check(text in str(e), ('wrong failure', text, repr(e))); return str(e)
    raise RuntimeError('failed to reject '+text)
def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: f.write(raw)
def oracle(values):
    return {k:{'bytes':len(v),'sha256':digest(v)} for k,v in values.items()}

check(digest((B/'SHA256SUMS').read_bytes())=='6e19cf83732b502f1bac32bd7b7d47baa6a624f6c6055fc0c06af1f5d2a32309','bundle manifest')
for line in (B/'SHA256SUMS').read_text().splitlines():
    pin,name=line.split('  ',1); check(digest((B/name).read_bytes())==pin,'input pin '+name)
check(len(json.loads(files['INPUTS.json'])['files'])==17,'17 inputs')
check(t.CHECKPOINT_SHAS=={a['checkpoint_sha256'] for a in json.loads(files['INPUTS.json'])['arms'].values()},'checkpoint pins')
bad=dict(files); bad['INPUTS.json']+=b' '
reject(lambda:t.packet_modules(bad),'PACKET_FILE')
reject(lambda:t.authorization({'label':t.LABEL,'host':'trn21','submission_authorized':False},'stage','f'*64),'TRANSPORT_DISABLED')
print('PASS frozen inputs, exact payload, disabled authority', flush=True)

# Independent byte/count oracle uses explicit fixture bytes, no production scan/stat logic.
values={'.hidden':b'hidden', 'receipts/.rank0.partial.json':b'partial', 'receipts/rank1.json':b'final', 'binary':bytes(range(256))*1100}
root=OUT/'source'; root.mkdir()
for n,raw in values.items(): put(root/n,raw)
put(root/'caches/cache/data',b'cache-only')
lock=OUT/'original.lock'; put(lock,b'unchanged lock')
sid=lock.stat(); lockid=(sid.st_dev,sid.st_ino)
class Driver:
    RUNTIME_PATHS=(); ARTIFACT_CAP=256*1024*1024; RANK_CAP=16*1024*1024
    file_record=staticmethod(d.file_record)
    def retained_inputs(self,*a): return {'tiny_fixture':'unchanged'}
fake=Driver()
def decode(*a): return files,b'{}',{},fake,g,idle
def census(*a): return {'lock_identity':list(lockid),'ps':{},'neuron_ls':{},'fuser':{}}
with patch.object(t,'ROOT',root), patch.object(t,'LOCK',str(lock)), patch.object(t,'LOCK_ID',lockid), patch.object(t,'decode_payload',decode), patch.object(t,'census',census):
    events=[]; receipt=t.retrieve({},'f'*64,events.append)
    check(receipt['files']==oracle(values),'independent hash oracle')
    check(receipt['artifact_bytes']==sum(map(len,values.values())),'byte oracle')
    check(receipt['rank_receipt_bytes']=={'0':7,'1':5,'2':0,'3':0},'rank oracle')
    check(receipt['cache_bytes']==10,'cache oracle')
    dest=OUT/'received'; t.receive([json.dumps(e) for e in events],dest)
    for n,raw in values.items(): check((dest/'files'/n).read_bytes()==raw,'exact bytes '+n)
    check(not (dest/'files/caches').exists(),'no cache copied')
    reject(lambda:t.receive([json.dumps(e) for e in events],dest),'File exists')
    incomplete=OUT/'incomplete'
    reject(lambda:t.receive([json.dumps(e) for e in events[:-1]],incomplete),'WIRE_INCOMPLETE')
    check((incomplete/'retrieval-failure.json').is_file(),'failure retained')
    corrupt=[dict(e) for e in events]
    next(e for e in corrupt if e['kind']=='chunk')['b64']=base64.b64encode(b'WRONG!').decode()
    reject(lambda:t.receive([json.dumps(e) for e in corrupt],OUT/'corrupt'),'WIRE_FILE_HASH')
    traversal=[events[0],{'kind':'file','name':'../escape','bytes':0,'sha256':digest(b'')}]
    reject(lambda:t.receive(map(json.dumps,traversal),OUT/'traversal'),'RELATIVE_NAME')
    trailing=events+[events[0]]
    reject(lambda:t.receive(map(json.dumps,trailing),OUT/'trailing'),'TRAILING_WIRE_RECORD')
    for failing in (root,root/'receipts'):
        original=os.scandir
        def denied(path):
            if Path(path)==failing: raise PermissionError('reviewer injected scan failure')
            return original(path)
        with patch('os.scandir',denied): reject(lambda:list(t.scan(root)),'reviewer injected scan failure')
    link=root/'symlink'; link.symlink_to(root/'binary')
    reject(lambda:list(t.scan(root)),'RETRIEVAL_NONREGULAR')
    # Preserve the symlink fixture in a separate location; no fixture deletion.
    link.rename(OUT/'retained-symlink')
    with patch.object(fake,'ARTIFACT_CAP',1):
        capped=[]; result=t.retrieve({},'f'*64,capped.append)
        check(result['status']=='COMPLETE_COPY_FAILED_GATE' and result['files']==oracle(values),'cap failure must not truncate')
    print('PASS independent transfer oracle, hidden/partial/cache accounting, new-only destination, wire faults, scan faults, cap recovery',flush=True)

# Actual local flock, used by the real staging function; host observations are isolated.
stage_root=OUT/'stage'
d.ROOT=stage_root
def held():
    fd=os.open(lock,os.O_RDWR)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        raise RuntimeError('lock not held during stage write')
    finally: os.close(fd)
class StageDriver:
    plain=staticmethod(d.plain); file_record=staticmethod(d.file_record)
    def staging_preflight(self,*a): held(); return {'local_fixture':True}
    def record(self,name,obj): held(); t.write_new(stage_root/name,t.json_bytes(obj))
    def retained_inputs(self): held(); return {}
    def resources(self): held(); return {}
    def resource_gate(self,*a): held()
    def stops(self): held()
    def validate_release(self,*a): held()
    def device_idle(self,*a): held()
    def sizes(self,*a): return {}
    def artifact_gate(self,*a): held()
    def utc(self): return 'fixture'
runtime=b'fixture'; release={'runtime_inspection_sha256':digest(runtime),'packet_manifest_sha256':t.MANIFEST_SHA,'runtime_pins':{}}
sd=StageDriver()
def stage_decode(*a): return files,b'{}',release,sd,g,idle
with patch.object(t,'ROOT',stage_root),patch.object(t,'LOCK',str(lock)),patch.object(t,'LOCK_ID',lockid),patch.object(t,'decode_payload',stage_decode):
    original_write=t.write_new
    def checked_write(path,raw,mode=0o444):
        held()
        if path.name=='RELEASE.json': check((stage_root/'staging-complete.json').exists(),'release last')
        return original_write(path,raw,mode)
    with patch.object(t,'write_new',checked_write): t.stage({'runtime_inspection':base64.b64encode(runtime)},'f'*64)
    check(all((stage_root/n).read_bytes()==raw for n,raw in files.items()),'exact staged packet')
    reject(lambda:t.stage({'runtime_inspection':base64.b64encode(runtime)},'f'*64),'STAGING_ROOT_ALREADY_EXISTS')
check(lock.read_bytes()==b'unchanged lock' and (lock.stat().st_dev,lock.stat().st_ino)==lockid,'same untouched lock')
print('PASS actual flock across stage writes, exact payload, release last, existing root refusal',flush=True)

# Adversarial replacement occurs exactly after pre-transfer hashing and before open.
# The tiny replacement stands in for a prohibited checkpoint by its exact hash.
race=OUT/'race-source'; race.mkdir(); target=race/'artifact.log'
checkpoint=b'tiny prohibited checkpoint bytes'
put(target,b'x'*len(checkpoint))
emitted=[]
def swapping_emit(event):
    emitted.append(event)
    if event['kind']=='file':
        replacement=race/'replacement'; put(replacement,checkpoint); os.replace(replacement,target)
with patch.object(t,'ROOT',race),patch.object(t,'LOCK_ID',lockid),patch.object(t,'decode_payload',decode),patch.object(t,'census',census),patch.object(t,'CHECKPOINT_SHAS',{digest(checkpoint)}):
    error=reject(lambda:t.retrieve({},'f'*64,swapping_emit),'RETRIEVAL_FILE_CHANGED')
    wire=b''.join(base64.b64decode(e['b64']) for e in emitted if e['kind']=='chunk')
    evidence={'failure':error,'prohibited_hash':digest(checkpoint),'emitted_hash':digest(wire),'emitted_bytes':len(wire),'checkpoint_emitted':wire==checkpoint,'complete_emitted':any(e['kind']=='complete' for e in emitted)}
    (OUT/'checkpoint-race.json').write_text(json.dumps(evidence,indent=2)+'\n')
    check(wire==checkpoint,'race reproduction changed; investigate')
    print('FINDING: prohibited checkpoint fixture emitted before '+error,flush=True)
    # Receiver detects the changed advertised hash, but already writes partial bytes.
    local=OUT/'race-received'
    reject(lambda:t.receive(map(json.dumps,emitted),local),'WIRE_INCOMPLETE')
    check((local/'retrieval-failure.json').exists(),'race failure retained')
    check((local/'files/artifact.log').read_bytes()==checkpoint,'checkpoint bytes retained by receiver')
print('VERDICT REFUSE: checkpoint exclusion is subject to hash/open replacement race',flush=True)
