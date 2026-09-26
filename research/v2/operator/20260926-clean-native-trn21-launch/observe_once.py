"""One bounded read-only snapshot of the assigned detached diagnostic."""
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_one_use',HERE/'execute_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
con=m.load('connector',m.C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py','7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899')
body=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess
from pathlib import Path
root=Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1')
if socket.gethostname()!='ip-172-31-34-85':raise RuntimeError('HOST')
files={}
names=['operator-dispatch.json','staging-complete.json','staging-preflight.json','in-lock-preflight.json','preflight-idle.json','fresh-caches.json','diagnostic-launch.json','diagnostic.log','child-exit.json','terminal.json','outer-flock-exit.json','outer-flock.log','dispatch-launch.log','resource-progress.json','lifetime-events.jsonl','postflight-idle.json','final-resources.json','post-inputs.json']
for name in names:
 p=root/name
 if not p.exists():continue
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise RuntimeError('NONREGULAR:'+name)
 with p.open('rb') as f:raw=f.read(4*1024*1024+1)
 if len(raw)>4*1024*1024:files[name]={'bytes':s.st_size,'capture':'OVERSIZE_NOT_CAPTURED'};continue
 files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace'),'capture':'complete_at_observation'}
rows={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'stat').read_text();a=raw.rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  rows[int(p.name)]={'pid':int(p.name),'ppid':int(a[1]),'pgid':int(a[2]),'sid':int(a[3]),'starttime':int(a[19]),'state':a[0],'cmdline':cmd}
 except (OSError,ValueError):pass
owned={3202396}
owned.update(pid for pid,r in rows.items() if str(root) in r['cmdline'])
for _ in range(20):
 more={pid for pid,r in rows.items() if r['ppid'] in owned}
 if more<=owned:break
 owned.update(more)
procs=[]
keys={'RANK','LOCAL_RANK','WORLD_SIZE','NEURON_LOGICAL_NC_CONFIG','NEURON_CC_FLAGS','TRN_EXP_LABEL','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'}
for pid in sorted(owned):
 if pid not in rows:procs.append({'pid':pid,'exists':False});continue
 r=rows[pid];r['exists']=True
 try:
  env={}
  for line in Path('/proc',str(pid),'environ').read_bytes().split(b'\0'):
   k,sep,v=line.partition(b'=')
   if sep and k.decode(errors='replace') in keys:env[k.decode()]=v.decode(errors='replace')
  r['environment_whitelist']=env;r['cwd']=os.readlink('/proc/'+str(pid)+'/cwd')
 except OSError:r['observation_race']=True
 procs.append(r)
lock='/home/ubuntu/.cache/trainium_freshstart/device0.lock';fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);st=os.fstat(fd)
if (st.st_dev,st.st_ino)!=(66305,596823):raise RuntimeError('LOCK_IDENTITY')
free=False
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:pass
os.close(fd)
n=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)
v=os.statvfs('/')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'files':files,'processes':procs,'lock':{'device':st.st_dev,'inode':st.st_ino,'free':free},'neuron_ls':{'returncode':n.returncode,'stdout':n.stdout,'stderr':n.stderr},'free_disk_bytes':v.f_bavail*v.f_frsize,'retired_queue_pid_exists':Path('/proc/2983496').exists(),'no_host_writes':True}))
'''.encode()
m.require(con.SENSITIVE.search(body) is None,'BODY_SENSITIVE')
argv,*_=con.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -'
r=subprocess.run(argv,input=body,capture_output=True,timeout=45)
m.require(con.SENSITIVE.search(r.stdout+r.stderr) is None,'OUTPUT_SENSITIVE')
obj={'utc':con.utc(),'source_sha256':m.sha(body),'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
m.save('first-observation.json',obj)
o=obj['remote'] or {}
print(json.dumps({'utc':o.get('utc'),'returncode':r.returncode,'lock':o.get('lock'),'processes':o.get('processes'),'files':list(o.get('files',{})),'free_disk_bytes':o.get('free_disk_bytes'),'diagnostic_tail':o.get('files',{}).get('diagnostic.log',{}).get('text','')[-4000:],'terminal':o.get('files',{}).get('terminal.json'),'stderr':obj['stderr']},indent=2))
