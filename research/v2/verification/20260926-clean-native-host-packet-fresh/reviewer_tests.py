import os, sys, json, time, hashlib, importlib.util, subprocess, signal, fcntl, struct
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
BASE=Path('/home/poojith-devan/clean36-native-trn21-packet-fresh-O2iY7F')
OUT=Path(__file__).parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d=load('driver',BASE/'packet/host_driver.py')
g=load('guard',BASE/'packet/lifetime_guard.py')
def check(ok,label):
    if not ok: raise RuntimeError(label)
def rejects(fn,label):
    try: fn()
    except (RuntimeError,ValueError,OSError): return
    raise RuntimeError('accepted negative '+label)
def worker(mode,root):
    d.ROOT=root
    lock=root/'lock'; fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX)
    st=os.fstat(fd); guard=g.Guardian(fd,lock,(st.st_dev,st.st_ino),root/'events.jsonl',grace=.1,kill_wait=1)
    (root/'ready').touch()
    if mode=='unknown':
        guard.quiesce=lambda reason: (_ for _ in ()).throw(RuntimeError('unknown test'))
        guard.finish(lambda:None)
    if mode=='prespawn':
        guard.finish(lambda:None); return
    child="import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.setsid(); Path('child.pid').write_text(str(os.getpid())); time.sleep(20)"
    leader="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(.2)"
    command=[sys.executable,'-B','-c',leader,child]
    if mode in ('timeout','disk','artifact','hidden','audit','postflight'):
        command=[sys.executable,'-B','-c',leader.replace('time.sleep(.2)','time.sleep(20)'),child]
    d.COMMAND=command; d.ENV=dict(os.environ); d.CHILD_SECONDS=.5
    d.resources=lambda:{'free_disk_bytes':d.INITIAL_DISK+1,'available_ram_bytes':d.RAM}
    if mode=='disk': d.resources=lambda:{'free_disk_bytes':d.RUNTIME_DISK-1 if (root/'child.pid').exists() else d.INITIAL_DISK+1,'available_ram_bytes':d.RAM}
    if mode in ('artifact','hidden'):
        original=d.sizes
        def fake_sizes(r):
            if (root/'child.pid').exists() and not (root/'injected').exists():
                (root/'injected').touch(); p=root/('receipts/.rank0.interrupted.tmp' if mode=='hidden' else 'oversize.log'); p.parent.mkdir(exist_ok=True)
                with p.open('wb') as f: f.truncate((d.RANK_CAP if mode=='hidden' else d.ARTIFACT_CAP)+1)
            return original(r)
        d.sizes=fake_sizes
    if mode=='audit':
        class Bad:
            def write(self,*a): raise OSError('test ENOSPC')
            def flush(self): pass
        original=guard.event
        def event(name,**kw):
            if name=='phase_spawned': guard.receipt=Bad()
            original(name,**kw)
        guard.event=event
    try:
        if mode=='normal':
            with (root/'child.log').open('w') as f: rc=guard.run(command,cwd=root,env=os.environ,out=f)
        else: rc=d.run_child(guard)
        (root/'result').write_text(str(rc))
    except BaseException as e: (root/'result').write_text(type(e).__name__+': '+str(e))
    finally:
        def post():
            if mode=='postflight': raise RuntimeError('test postflight unknown')
            if mode in ('disk','artifact','hidden'):
                d.resource_gate(d.resources(),False); d.artifact_gate(d.sizes(root))
        guard.finish(post)
    (root/'done').touch()

def main():
    mode='optimized' if sys.flags.optimize else 'normal'
    root=OUT/mode; root.mkdir()
    checks=[]
    def passed(label): checks.append(label); print('PASS',label,flush=True)
    manifest=(BASE/'SHA256SUMS').read_bytes()
    check(hashlib.sha256(manifest).hexdigest()=='9d966a95dedd5bb4d1bc68514544db2b05e92e8b35d896c08499fbbef034363b','manifest pin')
    for line in manifest.decode().splitlines():
        sha,path=line.split('  '); check(hashlib.sha256((BASE/path).read_bytes()).hexdigest()==sha,path)
    check(len(manifest.decode().splitlines())==20,'20 pins'); passed('20 input packet hashes')
    b=json.loads((BASE/'packet/NATIVE_BINDING.disabled.json').read_text())
    for key,val in [('child_argv',d.CHILD_ARGV),('command',d.COMMAND),('outer_command',d.OUTER),('environment',d.ENV),('support_pins',d.PINS),('lock_identity',list(d.LOCK_ID))]: check(b[key]==val,key)
    check(d.LOCK_ID==(66305,596823) and d.CHILD_SECONDS==900 and d.TERM_SECONDS==15,'bounds')
    check(all(b[k] is False for k in ('stage_authorized','launch_authorized','native_launch_authorized','submission_authorized')),'disabled')
    check(all(x is None for x in b['runtime_pins'].values()),'runtime unresolved')
    for p,pin in json.loads((BASE/'packet/MANIFEST.json').read_text()).items(): check(d.file_record(BASE/'packet'/p)==pin,'support manifest '+p)
    check(subprocess.run([sys.executable,'-B',str(BASE/'packet/host_driver.py')],capture_output=True).returncode==2,'disabled invocation'); passed('exact binding and disabled entry')
    check(sum(bb*tt*8192*4 for bb,tt in ((1,256),(2,256),(3,256),(1,257),(2,257),(3,257)))==100859904,'independent reference bytes')
    check(3*(16777216//3)==16777215,'three receipt names bound')
    check((d.INITIAL_DISK,d.RUNTIME_DISK,d.RAM)==(14495514624,11274289152,68719476736),'resource exact')
    for initial in (False,True):
        threshold=d.INITIAL_DISK if initial else d.RUNTIME_DISK
        d.resource_gate({'free_disk_bytes':threshold,'available_ram_bytes':d.RAM},initial)
        rejects(lambda:d.resource_gate({'free_disk_bytes':threshold-1,'available_ram_bytes':d.RAM},initial),'disk byte')
    rejects(lambda:d.resource_gate({'free_disk_bytes':d.INITIAL_DISK,'available_ram_bytes':d.RAM-1},True),'RAM byte')
    s=root/'sizes'; (s/'receipts').mkdir(parents=True); (s/'caches').mkdir()
    for name,size in [('receipts/rank0.json',3),('receipts/.rank0.x.tmp',7),('receipts/rank0.partial.json',11),('.hidden',13),('caches/.tmp',17)]: (s/name).write_bytes(b'x'*size)
    check(d.sizes(s)=={'artifact_bytes':34,'cache_bytes':17,'rank_receipt_bytes':{'0':21,'1':0,'2':0,'3':0}},'independent recursive count')
    rejects(lambda:d.artifact_gate({'artifact_bytes':d.ARTIFACT_CAP+1,'rank_receipt_bytes':{}}),'total cap')
    rejects(lambda:d.artifact_gate({'artifact_bytes':0,'rank_receipt_bytes':{'0':d.RANK_CAP+1}}),'rank cap')
    (s/'bad').symlink_to(s/'.hidden'); rejects(lambda:d.sizes(s),'symlink'); passed('size oracle thresholds hidden partials symlinks')
    f=root/'input'; f.write_bytes(b'original'); check(d.file_record(f)=={'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()},'input hash oracle')
    link=root/'inputlink'; link.symlink_to(f); rejects(lambda:d.file_record(link),'input symlink')
    rejects(lambda:d.retained_inputs(b'{}'),'manifest changed')
    check(len(json.loads((BASE/'packet/INPUTS.json').read_text())['files'])==17,'17 inputs'); passed('input hash and rejection')
    lock=root/'identity'; fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600); st=os.fstat(fd)
    rejects(lambda:g.require_original_lock(fd,lock,(st.st_dev,st.st_ino)),'unheld')
    fcntl.flock(fd,fcntl.LOCK_EX); g.require_original_lock(fd,lock,(st.st_dev,st.st_ino))
    rejects(lambda:g.require_original_lock(fd,lock,(st.st_dev,st.st_ino+1)),'wrong inode'); os.close(fd); passed('real original flock identity')
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        for case in ('normal','prespawn','timeout','disk','artifact','hidden','audit','postflight','unknown'):
            r=root/case; r.mkdir(); flags=['-O'] if sys.flags.optimize else []
            with (r/'worker.log').open('w') as raw:
                p=subprocess.Popen([sys.executable,*flags,'-B',__file__,'--worker',case,str(r)],stdout=raw,stderr=raw)
                deadline=time.monotonic()+7
                while not (r/'ready').exists() and p.poll() is None and time.monotonic()<deadline: time.sleep(.02)
                check((r/'ready').exists(),'worker ready '+case)
                if case!='prespawn': check(subprocess.run(['/usr/bin/flock','-n',str(r/'lock'),'true']).returncode!=0,'lock busy '+case)
                parked=case in ('audit','postflight','unknown','disk','artifact','hidden')
                while p.poll() is None and time.monotonic()<deadline:
                    if parked and 'FAILED_LOCK_RETAINED' in (r/'worker.log').read_text(): break
                    time.sleep(.02)
                if parked:
                    check(p.poll() is None,'park alive '+case)
                    check('FAILED_LOCK_RETAINED' in (r/'worker.log').read_text(),'park evidence '+case)
                    check(subprocess.run(['/usr/bin/flock','-n',str(r/'lock'),'true']).returncode!=0,'park lock '+case)
                else: check(p.wait(timeout=2)==0,'worker exit '+case)
                pidfile=r/'child.pid'
                if pidfile.exists(): check(not Path('/proc/'+pidfile.read_text()).exists(),'owned child absent '+case)
                if case in ('timeout','disk','artifact','hidden'):
                    expected={'timeout':'NATIVE_900_SECOND_DEADLINE','disk':'DISK_FLOOR','artifact':'ARTIFACT_CAP','hidden':'RANK_RECEIPT_CAP'}[case]
                    check(expected in (r/'result').read_text(),'fault observed '+case)
                if case=='normal':
                    check((r/'result').read_text()=='125','orphan is failed phase')
                    check('"signal": 9' in (r/'events.jsonl').read_text(),'setsid TERM ignoring child KILL')
                check(unrelated.poll() is None,'unrelated survival '+case)
                if parked:
                    # No child remains; this is disposal of our tiny parked fixture only.
                    check(not Path('/proc/'+str(p.pid)+'/task/'+str(p.pid)+'/children').read_text().strip(),'fixture child empty')
                    p.kill(); p.wait(timeout=2)
                else:
                    events=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()]
                    check(any(e['event']=='release_ready' for e in events),'release proof '+case)
                passed('real process '+case)
    finally: unrelated.terminate(); unrelated.wait(timeout=2)
    # Tiny tensor oracle uses 3 output channels, never full model/width/native.
    probe=load('probe',BASE/'packet'/d.PROBE); h=probe.foundation(); torch=h.torch; torch.set_num_threads(1)
    class Tiny(torch.nn.Module):
        def forward(self,x): return torch.stack((x.float(),x.float()+1,x.float()*2),dim=-1)
    model=Tiny().eval(); baseline=probe.snapshot(h,model,lambda:None)
    count=0
    for bb,tt in probe.SHAPES:
        x=torch.tensor([[1+(j%2) for j in range(tt)] for i in range(bb)],dtype=torch.int64)
        # odd suffix length needs non-palindromic fixture
        x[:, -1]=2 if x[0,-1]==1 else 1
        y,case=probe.verify_shape(h,model,x,3,'cpu',lambda:None,baseline)
        expected=b''.join(struct.pack('<f',v) for row in x.tolist() for token in row for v in (token,token+1,token*2))
        check(hashlib.sha256(expected).hexdigest()==case['calls'][0]['output']['sha256'],'independent output bytes')
        expected_input=b''.join(struct.pack('<q',v) for row in x.tolist() for v in row)
        check(case['input']['sha256']==hashlib.sha256(expected_input).hexdigest(),'independent token hash'); count+=len(case['calls'])
    check(count==30,'30 calls')
    rejects(lambda:probe.exact(h,torch.tensor([0.]),torch.tensor([-0.]),'zero'),'signed zero')
    class Mutate(Tiny):
        def forward(self,x): x[0,0]=2; return super().forward(x)
    rejects(lambda:probe.verify_shape(h,Mutate().eval(),torch.ones((1,256),dtype=torch.int64),3,'cpu',lambda:None,{}),'input mutation')
    passed('tiny scalar packed-byte oracle 6 shapes 30 calls exactness and mutation')
    (root/'summary.json').write_text(json.dumps({'mode':mode,'checks':checks},indent=2))
    print('COMPLETE',len(checks))
if __name__=='__main__':
    if '--worker' in sys.argv: worker(sys.argv[-2],Path(sys.argv[-1]))
    else: main()
