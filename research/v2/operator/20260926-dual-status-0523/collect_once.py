"""Two assigned bounded read-only observations; no host writes or launch."""
import concurrent.futures, datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWN=HERE.parents[3]
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn21.json').exists() and not (HERE/'trn22.json').exists(),'LOCAL_RECEIPT_EXISTS'
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=HERE.parent/'20260926-trn22-confirmed36-clean-cold-full-r2'
body22=(HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/remote_snapshot.py').read_text()
known=set();prior_pins=[]
for path in [folder/'trn22.json',folder/'progress/boundary-0509.json']:
 raw=path.read_bytes();prior_pins.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()});o=json.loads(raw)['remote']
 known.update(p['pid'] for p in o['processes'] if p.get('exists'))
 receipt=o['files'].get('lifetime-events.jsonl')
 if receipt and receipt['capture']=='complete_at_observation':
  for line in receipt['text'].splitlines():
   e=json.loads(line)
   if 'identity' in e:known.add(e['identity']['pid'])
assert {3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958}<=known
body22=body22.replace('fs-20260926-confirmed36-clean-cold-trn22-r2','fs-20260926-confirmed36-clean-coldfull-trn22-r2').replace('known = {3276148}','known = '+repr(known))
body22=body22.replace("'dispatch.log']","'dispatch.log','lifetime-events.jsonl','training-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json']")
body22='P='+repr({'completed_parent_paths':[],'completed_parent_cache_paths':[]})+'\n'+body22
body21=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
started=time.monotonic();assert socket.gethostname()=='ip-172-31-34-85' and os.getuid()==1000
project=Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2')
source=project/'out/fs-20260926-arch36a8-startup-clock-trn21-r2-full-packet/train/train.py'
def meta(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
lock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock');fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);lk=os.fstat(fd);assert (lk.st_dev,lk.st_ino)==(66305,596823)
lock_free=False
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_free=True
except BlockingIOError:pass
checks={}
def command(name,argv,limit=12):
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=limit)
  assert len(r.stdout.encode())+len(r.stderr.encode())<=512*1024,'COMMAND_OUTPUT_BOUND'
  checks[name]={'command':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
  return r
 except (subprocess.TimeoutExpired,AssertionError) as exc:checks[name]={'command':argv,'error':str(exc)};return None
ps=command('ps',['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'])
jobs=None if ps is None or ps.returncode else [l for l in ps.stdout.splitlines() if any(k in l for k in ('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
# The complete ps output is used only for classification; retain job rows, not unrelated argv.
if 'ps' in checks and 'stdout' in checks['ps']:checks['ps']['stdout']='[omitted unrelated process argv; matching workload rows retained separately]'
ls=command('neuron_ls',['/opt/aws/neuron/bin/neuron-ls','--json-output'],20)
try:devices=json.loads(ls.stdout) if ls and ls.returncode==0 else None
except ValueError:devices=None
command('project_head',['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(project),'rev-parse','HEAD'])
command('project_status',['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(project),'status','--porcelain=v1','--untracked-files=normal'])
source_record={'path':str(source),'expected_sha256':'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'}
try:
 for p in reversed((source,*source.parents)):assert not stat.S_ISLNK(p.lstat().st_mode),'SOURCE_SYMLINK_COMPONENT'
 a=source.lstat();assert stat.S_ISREG(a.st_mode) and 0<a.st_size<=1024**2,'SOURCE_BOUND'
 f=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(f,'rb') as stream:
  b=os.fstat(stream.fileno());raw=stream.read(1024**2+1);z=os.fstat(stream.fileno())
 after=source.lstat();stable=meta(a)==meta(b)==meta(z)==meta(after)
 source_record.update(before=meta(a),after=meta(after),stable=stable,sha256=hashlib.sha256(raw).hexdigest())
 source_record['matches_expected']=stable and source_record['sha256']==source_record['expected_sha256']
except (OSError,AssertionError) as exc:source_record['error']=str(exc)
paths={}
for name in ('/home/ubuntu/k2-f0f2-trn21-r1-packet','/home/ubuntu/k2-f0f2-trn21-r1-output'):
 try:
  s=Path(name).lstat();paths[name]={'exists':True,'identity':meta(s),'symlink':stat.S_ISLNK(s.st_mode)}
 except FileNotFoundError:paths[name]={'exists':False}
memory={}
for line in Path('/proc/meminfo').read_text().splitlines():
 key,value=line.split(':',1)
 if key in ('MemAvailable','MemTotal'):memory[key+'_bytes']=int(value.split()[0])*1024
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
if lock_free:fcntl.flock(fd,fcntl.LOCK_UN)
os.close(fd)
status=checks.get('project_status',{})
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','project':str(project),'checks':checks,'project_clean':status.get('returncode')==0 and status.get('stdout')=='','project_clean_known':status.get('returncode')==0,'source':source_record,'path_existence':paths,'lock':{'device':lk.st_dev,'inode':lk.st_ino,'free_before_observation':lock_free,'held_for_readonly_checks':lock_free},'job_rows':jobs,'neuron_ls':devices,'device_idle':devices is not None and len(devices)==1 and devices[0].get('neuron_processes')==[],'free_bytes':free,'memory':memory,'no_host_writes':True,'no_reservation':True,'no_launch':True,'elapsed_seconds':time.monotonic()-started}))
'''
def collect(host,body):
 wire=body.encode();assert not m.SENSITIVE.search(wire)
 argv,*_=m.load_transport(host);argv[-1]='exec python3 -I -B -'
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
 assert not m.SENSITIVE.search(r.stdout+r.stderr)
 result={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'remote_source_sha256':hashlib.sha256(wire).hexdigest(),'prior_trn22_pins':prior_pins if host=='trn22' else [],'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
 p=HERE/(host+'.json')
 with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 if r.returncode:return {'host':host,'returncode':r.returncode,'error':r.stderr.decode(),'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 o=result['remote']
 if host=='trn22':
  rows=o['files'].get('training.log',{}).get('text','').splitlines()
  summary={'host':host,'utc':o['utc'],'workers':[{'pid':x['pid'],'starttime':x['starttime'],'rank':x['environment_whitelist']['RANK']} for x in o['processes'] if x.get('environment_whitelist',{}).get('RANK') is not None],'lock':o['lock'],'free_bytes':o['free_bytes'],'training_tail':rows[-7:],'terminal':o['files'].get('terminal.json'),'eval_tail':o['files'].get('fresh-public20m.log',{}).get('text','').splitlines()[-8:]}
 else:summary=o
 summary['receipt_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();return summary
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(collect,'trn22',body22),pool.submit(collect,'trn21',body21)]
 for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result(),indent=2),flush=True)
