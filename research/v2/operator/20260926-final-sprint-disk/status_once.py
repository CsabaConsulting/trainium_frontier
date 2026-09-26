"""Bounded, lock-held, read-only status; no launches or remote writes."""
import concurrent.futures, datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
connector = REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(connector.read_bytes()).hexdigest() == '7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec = importlib.util.spec_from_file_location('strict_connector', connector)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
body = r'''
import datetime,fcntl,json,os,socket,stat,subprocess,time
from pathlib import Path
started=time.monotonic()
assert socket.gethostname()==P['hostname'] and os.getuid()==1000
fd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW)
ls=os.fstat(fd); assert (ls.st_dev,ls.st_ino)==(66305,P['inode'])
locked=False
try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
except BlockingIOError: pass
def run(argv,timeout=8):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 return {'argv':argv,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
ps=run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,lstart,comm,args'])
assert ps['exit_code']==0,ps['stderr']
terms=('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','preflight_dryrun.py','compare_checkpoints.py','guardian.py','neuronx-cc','walrus_driver','prepare.py eval-public')
jobs=[line for line in ps['stdout'].splitlines() if any(x in line for x in terms)]
nls=run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],12)
devices=json.loads(nls['stdout']) if nls['exit_code']==0 else None
fuser=run(['/usr/bin/fuser','/dev/neuron0'])
stops=[]
for root in P['roots']:
 for suffix in ('research/v2/STOP','research/agentic/STOP','STOP'):
  p=Path(root)/suffix
  if p.exists() or p.is_symlink():stops.append(str(p))
v=os.statvfs('/');mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':P['host'],'hostname':socket.gethostname(),'original_lock':{'device':ls.st_dev,'inode':ls.st_ino,'exclusive_acquired':locked},'ps_job_rows':jobs,'ps_sha256':__import__('hashlib').sha256(ps['stdout'].encode()).hexdigest(),'neuron_ls':nls,'devices':devices,'fuser':fuser,'stop_paths':stops,'free_bytes':v.f_bavail*v.f_frsize,'mem_available_bytes':mem,'no_host_writes':True,'elapsed_seconds':time.monotonic()-started}
if locked:fcntl.flock(fd,fcntl.LOCK_UN)
os.close(fd)
print(json.dumps(r))
'''
hosts={
 'trn21':{'hostname':'ip-172-31-34-85','inode':596823,'roots':['/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2']},
 'trn22':{'hostname':'ip-172-31-32-253','inode':832589,'roots':['/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1','/home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1']}}
def one(host):
 out=HERE/(host+'-status.json');assert not out.exists()
 state=m.local_state();assert not state['local_stops_present']
 wire=('P='+repr(dict(host=host,**hosts[host]))+'\n'+body).encode();assert not m.SENSITIVE.search(wire)
 argv,display,identity,transport_sha=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
 p=subprocess.run(argv,input=wire,capture_output=True,timeout=45)
 assert not m.SENSITIVE.search(p.stdout+p.stderr)
 receipt={'controller':state,'payload_sha256':hashlib.sha256(wire).hexdigest(),'transport_sha256':transport_sha,'exit_code':p.returncode,'stderr':p.stderr.decode(),'remote':json.loads(p.stdout) if p.returncode==0 else None,'stdout_if_error':p.stdout.decode() if p.returncode else None}
 with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 r=receipt['remote']
 print(json.dumps({'host':host,'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'status':r if r else receipt}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,hosts))
