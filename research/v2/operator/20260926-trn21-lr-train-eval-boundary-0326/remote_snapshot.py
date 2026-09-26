"""One bounded read-only snapshot; no host files or process state are changed."""
import datetime, hashlib, json, os, socket, stat, subprocess, time
from pathlib import Path
started = time.monotonic()
root = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-lr012-trn21-r1-packet')
label = 'fs-20260926-clock36-lr012-trn21-r1'
assert socket.gethostname() == 'ip-172-31-34-85'
known = {3152603,3152604,3152605,3152621,3152622,3152637,3152652,3152653,3152654,3152655,3153251,3153259,3153290}
ps = subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'], capture_output=True, text=True, timeout=10)
assert ps.returncode == 0, ps.stderr
rows = []
for line in ps.stdout.splitlines()[1:]:
    r = line.split(None,5)
    if len(r)==6:
        rows.append({'pid':int(r[0]),'ppid':int(r[1]),'pgid':int(r[2]),'sid':int(r[3]),'comm':r[4],'args':r[5]})
selected = known | {r['pid'] for r in rows if str(root) in r['args'] or label in r['args']}
for _ in range(12):
    new = selected | {r['pid'] for r in rows if r['ppid'] in selected or r['pgid'] in selected or r['sid'] in selected}
    if new == selected: break
    selected = new
whitelist = {'RANK','LOCAL_RANK','WORLD_SIZE','NEURON_LOGICAL_NC_CONFIG','NEURON_COMPETITION_R1_SEED','NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS','NEURON_CC_FLAGS','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','NEURON_COMPILE_CACHE_URL','TORCH_NEURONX_DEBUG_DIR'}
processes = []
for pid in sorted(selected):
    p = Path('/proc')/str(pid)
    try:
        before=(p/'stat').read_text(); tail=before[before.rfind(')')+2:].split()
        argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0'); argv=[x for x in argv if x]
        env={}
        for x in (p/'environ').read_bytes().split(b'\0'):
            k,sep,v=x.partition(b'=')
            if sep and k.decode(errors='replace') in whitelist: env[k.decode()]=v.decode(errors='replace')
        after=(p/'stat').read_text(); later=after[after.rfind(')')+2:].split()
        processes.append({'pid':pid,'exists':True,'state':tail[0],'ppid':int(tail[1]),'pgid':int(tail[2]),'sid':int(tail[3]),'starttime':int(tail[19]),'identity_stable':tail[19]==later[19],'cwd':os.readlink(p/'cwd'),'argv':argv,'environment_whitelist':env})
    except FileNotFoundError:
        processes.append({'pid':pid,'exists':False})
files={};total=0
names=['operator-launch.json','training-launch.json','training.log','run-exit-code.txt','train/rehearsal.json','train/steps.jsonl','train/stdout.log','train/stderr.log','checkpoint.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval/stdout.log','eval/stderr.log','eval-exit-code.txt','terminal.json','outer-flock-exit.json','under-lock-idle.json','eval-under-lock-idle.json','dispatch.log']
for rank in range(4):
    names.append(f'metrics-training/graph_stats_rank_{rank}.csv')
for name in names:
    p=root/name
    try:
        st=p.lstat()
        if not stat.S_ISREG(st.st_mode): files[name]={'refused':'NOT_REGULAR'};continue
        limit=1024*1024
        if total+min(st.st_size,limit)>6*1024*1024:files[name]={'refused':'TOTAL_BOUND'};continue
        with p.open('rb') as f:
            if st.st_size>limit:f.seek(-limit,2)
            raw=f.read(limit)
        total+=len(raw)
        files[name]={'bytes_at_stat':st.st_size,'captured_bytes':len(raw),'capture':'complete_at_observation' if st.st_size<=limit else 'tail','sha256_captured':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}
    except FileNotFoundError:pass
cp=root/'train/out/final.pt'
checkpoint=None
if cp.is_file():
    s=cp.stat();checkpoint={'path':str(cp),'bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'sha256_source':'archived checkpoint.json / train rehearsal; no rehash during live eval'}
lock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock');lk=lock.stat()
lockrows=[]
for line in Path('/proc/locks').read_text().splitlines():
    for field in line.split():
        bits=field.split(':')
        if len(bits)==3:
            try:
                if (int(bits[0],16),int(bits[1],16),int(bits[2]))==(os.major(lk.st_dev),os.minor(lk.st_dev),lk.st_ino):lockrows.append(line)
            except ValueError:pass
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)
try:devices=json.loads(ls.stdout)
except ValueError:devices=None
v=os.statvfs('/')
optional={}
optional_start=time.monotonic()
for path in P['completed_parent_paths']:
    try:
        p=Path(path);s=p.lstat()
        if time.monotonic()-optional_start>15 or not stat.S_ISREG(s.st_mode) or s.st_size>1024*1024:
            optional[path]={'refused':'BOUND_OR_NOT_REGULAR'};continue
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as f:
            a=os.fstat(f.fileno());data=f.read(1024*1024+1);z=os.fstat(f.fileno())
        stable=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
        optional[path]={'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'stable_during_read':stable,'sha256':hashlib.sha256(data).hexdigest() if stable else None}
    except FileNotFoundError:optional[path]={'missing':True}
cache_exists={p:Path(p).is_dir() for p in P['completed_parent_cache_paths']}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','label':label,'root':str(root),'processes':processes,'known_identity_floor':sorted(known),'selected_ps_rows':[r for r in rows if r['pid'] in selected],'files':files,'captured_file_bytes':total,'checkpoint':checkpoint,'ps_rc':ps.returncode,'ps_stderr':ps.stderr,'neuron_ls_rc':ls.returncode,'neuron_ls':devices,'neuron_ls_stderr':ls.stderr,'lock':{'device':lk.st_dev,'inode':lk.st_ino,'held':bool(lockrows),'proc_lock_rows':lockrows},'free_bytes':v.f_bavail*v.f_frsize,'retired_queue_pids':{'2983496':Path('/proc/2983496').exists()},'completed_parent_metadata_only':optional,'completed_parent_cache_exists_only':cache_exists,'no_changes':True,'elapsed_seconds':time.monotonic()-started}))
