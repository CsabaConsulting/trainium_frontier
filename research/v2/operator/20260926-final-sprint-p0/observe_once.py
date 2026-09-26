"""One bounded read-only actual-job observation; never launches or retries."""
import concurrent.futures, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
spec=importlib.util.spec_from_file_location('strict_connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
b=P['binding'];root=Path(b['root']);assert socket.gethostname()==b['hostname']
start=time.monotonic();deadline=start+20
def ident(pid):
 text=Path('/proc/'+str(pid)+'/stat').read_text();rest=text[text.rfind(')')+2:].split()
 return {'pid':pid,'state':rest[0],'ppid':int(rest[1]),'pgid':int(rest[2]),'sid':int(rest[3]),'start_ticks':int(rest[19])}
census=[];matched={};errors=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  row=ident(int(p.name));census.append(row)
  raw=(p/'cmdline').read_bytes();argv=raw.rstrip(b'\0').decode(errors='replace').split('\0')
  if any(b['root'] in arg for arg in argv):matched[row['pid']]={'identity_before':row,'argv':argv}
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert time.monotonic()<deadline,'PROCESS_CENSUS_BOUND'
changed=True
while changed:
 changed=False
 for row in census:
  if row['pid'] not in matched and row['ppid'] in matched:
   try:argv=Path('/proc/'+str(row['pid'])+'/cmdline').read_bytes().rstrip(b'\0').decode(errors='replace').split('\0')
   except (FileNotFoundError,PermissionError):continue
   matched[row['pid']]={'identity_before':row,'argv':argv};changed=True
keys={'RANK','LOCAL_RANK','WORLD_SIZE','NEURON_LOGICAL_NC_CONFIG','NEURON_CC_FLAGS','NEURON_COMPILE_CACHE_URL','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR','TRN_EXP_LABEL','NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS','NEURON_COMPETITION_R1_SEED'}
for pid,row in matched.items():
 p=Path('/proc/'+str(pid))
 try:
  row['cwd']=os.readlink(p/'cwd');env={}
  for field in (p/'environ').read_bytes().split(b'\0'):
   k,sep,v=field.partition(b'=')
   if sep and k.decode(errors='replace') in keys:env[k.decode()]=v.decode(errors='replace')
  row['environment_whitelist']=env;row['identity_after']=ident(pid);row['identity_stable']=row['identity_before']==row['identity_after']
  locks=[]
  for e in (p/'fd').iterdir():
   try:
    s=e.stat()
    if [s.st_dev,s.st_ino]==b['lock_identity']:locks.append({'fd':int(e.name),'fdinfo':(p/'fdinfo'/e.name).read_text()})
   except (FileNotFoundError,PermissionError):continue
  row['original_lock_fds']=locks
 except (FileNotFoundError,PermissionError,ProcessLookupError) as exc:row['observation_race']=type(exc).__name__
files={}
for name in ('operator-preflight.json','operator-launch.json','training-launch.json','under-lock-idle.json','training-preflight-fuser.json','terminal.json','outer-flock-exit.json','lifetime-events.jsonl','training.log','dispatch.log','outer-flock.log','train/rehearsal.json'):
 p=root/name
 try:
  s=p.lstat();assert stat.S_ISREG(s.st_mode)
  with p.open('rb') as f:
   if s.st_size>65536:f.seek(-65536,2)
   raw=f.read(65536)
  files[name]={'bytes_on_host':s.st_size,'captured_bytes':len(raw),'tail_only':s.st_size>65536,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}
 except FileNotFoundError:files[name]={'absent':True}
train_source=root/'train/train.py'
source={'exists':train_source.is_file()}
if source['exists']:source.update(sha256=hashlib.sha256(train_source.read_bytes()).hexdigest(),bytes=train_source.stat().st_size)
fd=os.open(b['lock_path'],os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd);assert [s.st_dev,s.st_ino]==b['lock_identity'];held=None
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False;fcntl.flock(fd,fcntl.LOCK_UN)
except BlockingIOError:held=True
os.close(fd)
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=12)
v=os.statvfs('/')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':b['host'],'label':b['label'],'root':b['root'],'processes':list(matched.values()),'census':census,'files':files,'actual_train_source':source,'original_lock':{'device':s.st_dev,'inode':s.st_ino,'held':held},'neuron_ls':{'returncode':ls.returncode,'stdout':ls.stdout,'stderr':ls.stderr},'free_bytes':v.f_bavail*v.f_frsize,'elapsed_seconds':time.monotonic()-start,'read_only':True}))
'''
def one(host):
 out=HERE/(host+'-live.json');assert not out.exists()
 b=json.loads((HERE/(host+'-bound-packet.json')).read_text())['binding']
 wire=('P='+repr({'binding':b})+'\n'+body).encode();assert not m.SENSITIVE.search(wire)
 argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
 p=subprocess.run(argv,input=wire,capture_output=True,timeout=45);assert not m.SENSITIVE.search(p.stdout+p.stderr)
 receipt={'returncode':p.returncode,'stderr':p.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(p.stdout) if p.returncode==0 else None,'stdout_if_error':p.stdout.decode() if p.returncode else None}
 with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 r=receipt['remote']
 workers=[{'pid':row['identity_before']['pid'],'rank':row.get('environment_whitelist',{}).get('RANK'),'argv':row['argv']} for row in r['processes'] if row.get('environment_whitelist',{}).get('RANK') is not None] if r else []
 print(json.dumps({'host':host,'returncode':p.returncode,'utc':r['utc'] if r else None,'workers':workers,'original_lock':r['original_lock'] if r else None,'free_bytes':r['free_bytes'] if r else None,'training_launch':r['files']['training-launch.json'] if r else None,'terminal':r['files']['terminal.json'] if r else None,'error':p.stderr.decode(),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))
