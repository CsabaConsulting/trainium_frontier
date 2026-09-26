"""One read-only terminal archive, exact retained input hashes and explicit census."""
import datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
dest=HERE/'completed';assert not dest.exists()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=r'''
import datetime,fcntl,hashlib,json,os,stat,subprocess,time
from pathlib import Path
root=Path('/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1')
assert (root/'terminal.json').is_file() and (root/'outer-flock-exit.json').is_file(),'NOT_TERMINAL'
prep=Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py')
names=['operator-launch.json','preflight.json','under-lock-idle.json','CPU_PLAN.disabled.json','EXPERIMENT_PROPOSAL.json','MANIFEST.sha256.json','SUPERVISOR_ASSIGNMENT.md','dispatch.log','outer-flock-exit.json','terminal.json','standard-compare.json','standard-compare.stderr','standard-compare-exit.json','standard-compare-launch.json','standard-compare-preflight.json','standard-compare-under-lock-idle.json']
identities={};checkpoints={};known={3176685,3176686,3176687,3176720,3176776};files={};total=0
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
for arm in ['reference','candidate']:
 names += [arm+'/'+f for f in ['launch.json','process.json','terminal.json','preflight.json','under-lock-idle.json','g2.json','g2.log','checkpoint-identity.json','request.json']]
 ci=json.loads((root/arm/'checkpoint-identity.json').read_text());p=Path(ci['path'])
 rel=p.relative_to(root/arm).parts;assert len(rel)==4 and rel[0]=='tmp' and rel[1].startswith('g2-') and rel[2:]==('out','final.pt'),'ACTUAL_CHECKPOINT_PATH'
 obs=filehash(p);assert obs['sha256']==ci['sha256'] and obs['bytes']==ci['bytes'],'CHECKPOINT_CHANGED'
 checkpoints[arm]={'record':ci,'observed':obs};identities[str(p)]=obs
 for f in ['request.json','g2.json','terminal.json','checkpoint-identity.json','train.py']:
  fp=root/arm/f;identities[str(fp)]=filehash(fp)
 g2dir=p.parent.parent
 children=list(g2dir.iterdir());assert len(children)<=32,'G2_DIR_BOUND'
 for f in children:
  if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.log','.txt','.stderr'):
   names.append(str(f.relative_to(root)))
for p in [prep,root/'preflight_dryrun.py',root/'standard-compare.json',root/'standard-compare-exit.json',root/'terminal.json']:identities[str(p)]=filehash(p)
assert len(identities)==17,'INPUT_SET_NOT_17'
for name in dict.fromkeys(names):
 p=root/name
 if not p.exists():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=2*1024*1024,'RECEIPT_FILE_BOUND'
 raw=p.read_bytes();total+=len(raw);assert total<=16*1024*1024,'RECEIPT_TOTAL_BOUND'
 files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
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
 if any(k in r[5] for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public')):jobs.append(line)
pidrows=[{'pid':p,'exists':Path('/proc/'+str(p)).exists()} for p in sorted(known)]
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);devices=json.loads(ls.stdout) if ls.returncode==0 else None
nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes
fu=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10)
fd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW);lk=os.fstat(fd);assert (lk.st_dev,lk.st_ino)==(66305,596823)
lockrc=1
try:
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lockrc=0;fcntl.flock(fd,fcntl.LOCK_UN)
 except BlockingIOError:pass
finally:os.close(fd)
v=os.statvfs('/')
post={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_pids':sorted(known),'pid_rows':pidrows,'all_observed_pids_absent':all(not p['exists'] for p in pidrows),'complete_process_census':census,'related_rows':related,'busy_job_rows':jobs,'ps_rc':ps.returncode,'neuron_ls_rc':ls.returncode,'neuron_ls':devices,'neuron_ls_stderr':ls.stderr,'neuron_device_idle':ls.returncode==0 and len(devices)==1 and devices[0].get('neuron_processes')==[],'fuser_rc':fu.returncode,'fuser_stdout':fu.stdout,'fuser_stderr':fu.stderr,'lock_competitor_rc':lockrc,'lock_device':lk.st_dev,'lock_inode':lk.st_ino,'short_lived_worker_identity_limitation':'Four-rank jobs both completed before first live observation. Their exact worker/torchrun PIDs were not captured; no invented PIDs. Full authenticated rank receipts and complete final PPID/PGID/SID census archived.','no_changes':True}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'files':files,'raw_bytes':total,'checkpoint_records':checkpoints,'retained_input_identities':identities,'postflight':post,'free_bytes':v.f_bavail*v.f_frsize,'no_checkpoint_copy':True,'no_changes':True}))
'''
assert not m.SENSITIVE.search(body.encode())
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=body,text=True,capture_output=True,timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
if r.returncode:print(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));raise SystemExit(r.returncode)
o=json.loads(r.stdout);dest.mkdir(exist_ok=False)
for name,value in [('results-bundle.json',o),('explicit-process-postflight.json',o['postflight']),('retained-input-identities.json',o['retained_input_identities'])]:
 with (dest/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'utc':o['utc'],'file_count':len(o['files']),'raw_bytes':o['raw_bytes'],'checkpoints':o['checkpoint_records'],'all_observed_pids_absent':o['postflight']['all_observed_pids_absent'],'related_rows':o['postflight']['related_rows'],'census_count':len(o['postflight']['complete_process_census']),'lock_competitor_rc':o['postflight']['lock_competitor_rc'],'device_idle':o['postflight']['neuron_device_idle'],'disk_free':o['free_bytes'],'bundle_sha256':hashlib.sha256((dest/'results-bundle.json').read_bytes()).hexdigest()},indent=2))
