"""One authorized pidfd-only stale guardian release after fresh idle proof."""
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;C=Path('/home/poojith-devan/Desktop/Trainium')
prior=HERE.parent/'20260926-trn21-guardian-release-precheck/trn21.json'
if hashlib.sha256(prior.read_bytes()).hexdigest()!='c3c2718dd5d175c03c2609a1c61aeb575a6cc48b0671620b15c64701540a6f95':raise RuntimeError('INDEPENDENT_PRECHECK_PIN')
accepted=json.loads(prior.read_bytes())['remote']
known={pid:v['expected_starttime'] for pid,v in accepted['observed_identities'].items()}
if len(known)!=43:raise RuntimeError('43_IDENTITY_SET')
body=('KNOWN='+repr(known)+'\n').encode()+r'''
import datetime,fcntl,hashlib,json,os,select,signal,socket,stat,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1')
LOCK='/home/ubuntu/.cache/trainium_freshstart/device0.lock';GUARD=3202398
CHAIN={3202396:94436551,3202397:94436553,3202398:94436554}
out={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','independently_accepted_precheck_sha256':'c3c2718dd5d175c03c2609a1c61aeb575a6cc48b0671620b15c64701540a6f95','signal_count':0,'lock_file_modified':False,'artifacts_deleted':False,'no_job_launched':True};pfd=None
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def identity(pid):
 try:
  raw=Path('/proc',str(pid),'stat').read_text();f=raw.rsplit(')',1)[1].split()
  return {'pid':pid,'comm':raw.split('(',1)[1].rsplit(')',1)[0],'state':f[0],'ppid':int(f[1]),'pgid':int(f[2]),'sid':int(f[3]),'starttime':int(f[19])}
 except FileNotFoundError:return None
def processes():
 census=[];work=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  row=identity(int(p.name))
  if row is None:continue
  census.append(row)
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except FileNotFoundError:continue
  if any(k in cmd for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public','probe.py')):work.append({**row,'cmdline':cmd})
 return census,work
def checks(phase):
 records={}
 for name,argv in [('ps',['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm']),('neuron_ls',['/opt/aws/neuron/bin/neuron-ls','--json-output']),('fuser',['/usr/bin/fuser','/dev/neuron0'])]:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=12);records[name]={'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 out[phase+'_checks']=records
 require(records['ps']['returncode']==0,'PS_FAILED')
 require(records['neuron_ls']['returncode']==0,'NEURON_LS_FAILED');ds=json.loads(records['neuron_ls']['stdout']);require(isinstance(ds,list) and len(ds)==1 and ds[0].get('neuron_device')==0 and ds[0].get('neuron_processes')==[],'DEVICE_BUSY_OR_UNKNOWN')
 require(records['fuser']['returncode']==1 and not records['fuser']['stdout'].strip() and not records['fuser']['stderr'].strip(),'DEVICE_HOLDER')
def children():
 result={}
 for p in Path('/proc',str(GUARD),'task').iterdir():result[p.name]=(p/'children').read_text()
 return result
def exact_chain():
 rows={pid:identity(pid) for pid in CHAIN}
 require(all(rows[p] is not None and rows[p]['starttime']==tick for p,tick in CHAIN.items()),'CHAIN_IDENTITY_DRIFT')
 require(rows[GUARD]['ppid']==3202397 and rows[3202397]['ppid']==3202396 and rows[3202396]['ppid']==1,'EXACT_PARENT_CHAIN')
 expected=['/usr/bin/python3','-I','-B',str(ROOT/'host_driver.py')]
 actual=Path('/proc',str(GUARD),'cmdline').read_bytes().split(b'\0');actual=[v.decode() for v in actual if v]
 require(actual==expected,'GUARDIAN_EXACT_ARGV')
 require(hashlib.sha256((ROOT/'host_driver.py').read_bytes()).hexdigest()=='4a062c97107e3afd9f0061824074e9f151644e204d2957dd28ddfd6b372e98f7','GUARDIAN_SOURCE_PIN')
 children_now=children();require(all(not value.strip() for value in children_now.values()),'GUARDIAN_HAS_CHILDREN')
 original=Path(LOCK).lstat();require(stat.S_ISREG(original.st_mode) and (original.st_dev,original.st_ino)==(66305,596823),'ORIGINAL_LOCK_CHANGED')
 held=os.stat('/proc/3202398/fd/3');require((held.st_dev,held.st_ino)==(66305,596823),'GUARDIAN_INHERITED_LOCK_CHANGED')
 fdinfo=Path('/proc/3202398/fdinfo/3').read_text();require('FLOCK' in fdinfo and 'WRITE' in fdinfo,'GUARDIAN_LOCK_NOT_HELD')
 return {'chain':rows,'guardian_argv':actual,'children':children_now,'original_lock':[original.st_dev,original.st_ino],'guardian_fd3_info':fdinfo}
try:
 require(socket.gethostname()=='ip-172-31-34-85' and os.getuid()==1000,'HOST_UID')
 out['identity_before']=exact_chain();checks('before')
 census,work=processes();out['census_before']=census;out['workload_before']=work
 require(all(r['pid'] in (3202397,3202398) and r['starttime']==CHAIN[r['pid']] for r in work),'OTHER_WORKLOAD')
 for pid,tick in KNOWN.items():
  if int(pid) in CHAIN:continue
  r=identity(int(pid));require(r is None or r['starttime']!=tick,'OLD_JOB_IDENTITY_STILL_ALIVE:'+pid)
 require(not any(r['ppid']==GUARD for r in census),'GUARDIAN_DESCENDANT_PRESENT')
 pfd=os.pidfd_open(GUARD,0)
 out['identity_after_pidfd_open']=exact_chain()
 require(not select.select([pfd],[],[],0)[0],'GUARDIAN_ALREADY_EXITED')
 out['signal_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 signal.pidfd_send_signal(pfd,signal.SIGKILL,None,0);out['signal_count']=1;out['signal']='SIGKILL';out['signal_target_pid']=GUARD;out['signal_target_starttime']=94436554;out['pidfd_bound']=True
 limit=time.monotonic()+15
 while time.monotonic()<limit:
  remaining=[p for p,t in CHAIN.items() if (identity(p) or {}).get('starttime')==t]
  if not remaining:break
  time.sleep(.1)
 out['expected_chain_after']={str(p):identity(p) for p in CHAIN}
 require(all((identity(p) or {}).get('starttime')!=tick for p,tick in CHAIN.items()),'EXPECTED_WRAPPER_EXIT_UNRESOLVED')
 checks('after');census,work=processes();out['census_after']=census;out['workload_after']=work;require(not work,'POST_RELEASE_WORKLOAD')
 out['observed43_after']={pid:{'expected_starttime':tick,'actual':identity(int(pid))} for pid,tick in KNOWN.items()}
 require(all(v['actual'] is None or v['actual']['starttime']!=v['expected_starttime'] for v in out['observed43_after'].values()),'OBSERVED_IDENTITY_SURVIVES')
 fd=os.open(LOCK,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  lk=os.fstat(fd);require((lk.st_dev,lk.st_ino)==(66305,596823),'LOCK_POST_IDENTITY');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);out['original_lock_after']={'device':lk.st_dev,'inode':lk.st_ino,'exclusive_acquisition_succeeded':True};fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)
 exitfile=ROOT/'outer-flock-exit.json'
 if exitfile.exists():
  raw=exitfile.read_bytes();require(len(raw)<65536,'EXIT_RECEIPT_BOUND');out['natural_outer_exit_receipt']={'text':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 v=os.statvfs('/');out['free_disk_bytes']=v.f_bavail*v.f_frsize
 out['status']='STALE_GUARDIAN_RELEASED_ORIGINAL_LOCK_FREE_DEVICE_IDLE'
except BaseException as error:
 out['status']='FAILED_OR_UNRESOLVED_NO_RETRY';out['error']=type(error).__name__+': '+str(error);out['traceback']=traceback.format_exc()
finally:
 if pfd is not None:os.close(pfd)
 out['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(out),flush=True)
raise SystemExit(0 if out['status']=='STALE_GUARDIAN_RELEASED_ORIGINAL_LOCK_FREE_DEVICE_IDLE' else 2)
'''.encode()
connector=C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
if hashlib.sha256(connector.read_bytes()).hexdigest()!='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899':raise RuntimeError('CONNECTOR_PIN')
spec=importlib.util.spec_from_file_location('connector',connector);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if m.SENSITIVE.search(body):raise RuntimeError('BODY_SENSITIVE')
argv,*_=m.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -S -'
started=m.utc();r=subprocess.run(argv,input=body,capture_output=True,timeout=60)
if m.SENSITIVE.search(r.stdout+r.stderr):raise RuntimeError('OUTPUT_SENSITIVE')
result={'started_utc':started,'ended_utc':m.utc(),'returncode':r.returncode,'source_sha256':hashlib.sha256(body).hexdigest(),'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
dest=HERE/'trn21.json'
with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
o=result['remote'] or {}
print(json.dumps({'status':o.get('status'),'utc':o.get('ended_utc'),'signal_utc':o.get('signal_utc'),'signal_count':o.get('signal_count'),'chain_after':o.get('expected_chain_after'),'lock':o.get('original_lock_after'),'error':o.get('error'),'receipt_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
raise SystemExit(r.returncode)
