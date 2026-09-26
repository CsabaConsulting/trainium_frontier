"""One read-only failed cold-launch archive and explicit descendant census."""
import datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
dest=HERE/'completed';assert not dest.exists()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=r'''
import datetime,fcntl,hashlib,json,os,stat,subprocess,time
from pathlib import Path
root=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet')
assert (root/'terminal.json').is_file() and (root/'outer-flock-exit.json').is_file(),'NOT_TERMINAL'
prep=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py')
names=['operator-launch.json','preflight.json','under-lock-idle.json','CT_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','SUPERVISOR_ASSIGNMENT.md','MANIFEST.sha256.json','host_driver.py','idle_guard.py','dispatch_driver.py','package/train.py','package/launch-command.txt','package/provenance.json','dispatch.log','outer-flock-exit.json','outer-flock.log','terminal.json','training-launch.json','training.log','run-exit-code.txt','train/rehearsal.json','train/steps.jsonl','train/stdout.log','train/stderr.log']+[f'metrics-training/metric_table_graph_stats_rank{rank}.csv' for rank in range(4)]
identities={};checkpoints={};known=set(P['known_prior_identities']);files={};total=0
def filehash(path):
 s=path.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=1024**3,'PIN_FILE_BOUND'
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  a=os.fstat(f.fileno());h=hashlib.sha256()
  while True:
   buf=f.read(65536)
   if not buf:break
   h.update(buf)
  z=os.fstat(f.fileno())
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'INPUT_CHANGED'
 return {'bytes':a.st_size,'sha256':h.hexdigest(),'device':a.st_dev,'inode':a.st_ino,'mtime_ns':a.st_mtime_ns,'ctime_ns':a.st_ctime_ns,'stable':True}
for name in [root/'package/train.py',prep,prep.parent/'research/scratch/submission_rehearsal.py']:
 identities[str(name)]=filehash(name)
for name in dict.fromkeys(names):
 p=root/name
 if not p.exists():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=2*1024*1024,'RECEIPT_FILE_BOUND'
 raw=p.read_bytes();total+=len(raw);assert total<=16*1024*1024,'RECEIPT_TOTAL_BOUND'
 files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
cp=root/'train/out/final.pt'
checkpoint_after=filehash(cp) if cp.is_file() else None
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10);assert ps.returncode==0,ps.stderr
census=[];related=[];jobs=[]
for line in ps.stdout.splitlines()[1:]:
 r=line.split(None,5)
 if len(r)!=6:continue
 p=Path('/proc')/r[0];record={'pid':int(r[0]),'ppid':int(r[1]),'pgid':int(r[2]),'sid':int(r[3]),'comm':r[4]}
 try:
  text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split();record['starttime']=int(tail[19]);record['state']=tail[0]
 except FileNotFoundError:record['exited_during_census']=True
 census.append(record)
 if any(int(x) in known for x in r[:4]) or str(root) in r[5]:related.append(line)
 if any(k in r[5] for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public')):jobs.append(line)
pidrows=[{'pid':p,'exists':Path('/proc/'+str(p)).exists()} for p in sorted(known)]
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);devices=json.loads(ls.stdout) if ls.returncode==0 else None
nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes
fu=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10)
fd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW);lk=os.fstat(fd);assert (lk.st_dev,lk.st_ino)==(66305,832589)
lockrc=1
try:
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lockrc=0;fcntl.flock(fd,fcntl.LOCK_UN)
 except BlockingIOError:pass
finally:os.close(fd)
v=os.statvfs('/')
post={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_pids':sorted(known),'pid_rows':pidrows,'all_observed_pids_absent':all(not p['exists'] for p in pidrows),'complete_process_census':census,'related_rows':related,'busy_job_rows':jobs,'ps_rc':ps.returncode,'neuron_ls_rc':ls.returncode,'neuron_ls':devices,'neuron_ls_stderr':ls.stderr,'neuron_device_idle':ls.returncode==0 and len(devices)==1 and devices[0].get('neuron_processes')==[],'fuser_rc':fu.returncode,'fuser_stdout':fu.stdout,'fuser_stderr':fu.stderr,'lock_competitor_rc':lockrc,'lock_device':lk.st_dev,'lock_inode':lk.st_ino,'known_prior_identities':P['known_prior_identities'],'actual_identity_evidence':'Union of all actual identities captured in current r2 live/boundary receipts, with original PID/starttime/PPID/PGID/SID. Complete final census checks their session ancestry and global job/compiler names. Unobserved transient children are not invented.','no_changes':True}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'files':files,'raw_bytes':total,'checkpoint_records':checkpoints,'checkpoint_after_terminal':checkpoint_after,'retained_input_identities':identities,'postflight':post,'free_bytes':v.f_bavail*v.f_frsize,'no_checkpoint_copy':True,'no_changes':True}))
'''

identity_map={}
for path in [HERE/'trn22.json',*sorted((HERE/'progress').glob('*.json'))]:
 if not path.is_file():continue
 obj=json.loads(path.read_bytes());remote=obj.get('remote',obj)
 for process in remote.get('processes',[]):
  if process.get('exists') and process.get('starttime') is not None:
   pid=str(process['pid'])
   if pid in identity_map:assert identity_map[pid]['starttime']==process['starttime'],'PID_REUSE_IN_OBSERVATIONS'
   identity_map[pid]=process
assert {str(p) for p in (3276148,3276149,3276150,3276165,3276166,3276180,3276195,3276196,3276197,3276198)}<=set(identity_map),'MISSING_LIVE_R2_IDENTITIES'
payload={'known_prior_identities':{int(k):v for k,v in identity_map.items()}}

wire='P='+repr(payload)+'\n'+body
assert not m.SENSITIVE.search(wire.encode())
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=wire,text=True,capture_output=True,timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
if r.returncode:print(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));raise SystemExit(r.returncode)
o=json.loads(r.stdout);dest.mkdir(exist_ok=False)
for name,value in [('results-bundle.json',o),('explicit-process-postflight.json',o['postflight']),('retained-input-identities.json',o['retained_input_identities'])]:
 with (dest/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'utc':o['utc'],'file_count':len(o['files']),'raw_bytes':o['raw_bytes'],'checkpoints':o['checkpoint_records'],'checkpoint_after_terminal':o['checkpoint_after_terminal'],'all_observed_pids_absent':o['postflight']['all_observed_pids_absent'],'related_rows':o['postflight']['related_rows'],'census_count':len(o['postflight']['complete_process_census']),'lock_competitor_rc':o['postflight']['lock_competitor_rc'],'device_idle':o['postflight']['neuron_device_idle'],'disk_free':o['free_bytes'],'bundle_sha256':hashlib.sha256((dest/'results-bundle.json').read_bytes()).hexdigest()},indent=2))
