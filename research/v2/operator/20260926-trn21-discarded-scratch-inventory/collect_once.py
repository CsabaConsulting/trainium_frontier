"""One bounded metadata-only inventory; no cleanup or file-content reads."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'inventory.json').exists(), 'LOCAL_RECEIPT_EXISTS'
base = '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/'
targets = [
 {'path': base+'fs-20260926-k2-symmetric-consumer-trn21-r1-packet/caches', 'label': 'fs-20260926-k2-symmetric-consumer-trn21-r1', 'reason': 'Failed, parked synthetic candidate; private caches only, not its archived source/receipts.', 'binding': '20260926-k2-symmetric-native-trn21'},
 {'path': base+'fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches', 'label': 'fs-20260926-ordinary-fp32-subnormal-trn21-d1', 'reason': 'Completed failed ordinary diagnostic; private scratch caches only; retain source/results.', 'binding': '20260926-ordinary-embedding-subnormal-control'},
 {'path': base+'fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt', 'label': 'fs-20260926-basebcbd-initial-cache-trn21-r1-full', 'reason': 'Explicit DISCARD in ledger and DECISIONS; metadata only, deletion still needs separate authority.', 'binding': '20260926-base-cache-full-trn21'},
]
bindings = []
for item in targets:
 p=REPO/'research/v2/proposals'/item['binding']/'NATIVE_BINDING.disabled.json'
 raw=p.read_bytes(); b=json.loads(raw)
 assert b['host']=='trn21' and b['label']==item['label'] and item['path'].startswith(b['root']+'/')
 bindings.append({'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest()})
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=r'''
import datetime,fcntl,json,os,socket,stat,subprocess,time
from pathlib import Path
assert socket.gethostname()=='ip-172-31-34-85' and os.getuid()==1000
started=time.monotonic();deadline=started+25
fd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW)
lk=os.fstat(fd);assert (lk.st_dev,lk.st_ino)==(66305,596823)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=8);assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
assert not jobs,'BUSY_HOST'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=12);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0].get('neuron_processes')==[]
records=[];global_seen=set();total_entries=0
def identity(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
for item in P['targets']:
 row=dict(item);root=Path(item['path']);entries=0;logical=allocated=unique=0;files=dirs=links=0;seen=set();issues=[];directory_records=[]
 try:
  for p in reversed((root,*root.parents)):
   s=p.lstat()
   if stat.S_ISLNK(s.st_mode):raise RuntimeError('SYMLINK_COMPONENT:'+str(p))
  before=root.lstat();row['identity_before']=identity(before)
  assert before.st_dev==66305 and before.st_uid==1000,'OWNER_OR_DEVICE'
  pending=[root]
  while pending:
   if time.monotonic()>deadline or total_entries>=50000:issues.append('TIME_OR_ENTRY_BOUND');break
   p=pending.pop();s=p.lstat();entries+=1;total_entries+=1
   if s.st_dev!=66305 or s.st_uid!=1000:issues.append('OWNER_OR_DEVICE:'+str(p));continue
   key=(s.st_dev,s.st_ino)
   if key not in seen:logical+=s.st_size;allocated+=s.st_blocks*512;seen.add(key)
   if key not in global_seen:unique+=s.st_blocks*512;global_seen.add(key)
   if stat.S_ISLNK(s.st_mode):links+=1;issues.append('SYMLINK_NOT_FOLLOWED:'+str(p));continue
   if stat.S_ISREG(s.st_mode):files+=1
   elif stat.S_ISDIR(s.st_mode):
    dirs+=1;directory_records.append((p,s.st_ino,s.st_mtime_ns,s.st_ctime_ns))
    with os.scandir(p) as scan:
     for ent in scan:
      if time.monotonic()>deadline or total_entries+len(pending)>=50000:issues.append('TIME_OR_ENTRY_BOUND');break
      pending.append(Path(ent.path))
   else:issues.append('SPECIAL_FILE:'+str(p))
  for p,ino,mtime,ctime in directory_records:
   after=p.lstat()
   if (after.st_ino,after.st_mtime_ns,after.st_ctime_ns)!=(ino,mtime,ctime):issues.append('DIRECTORY_CHANGED:'+str(p))
  after=root.lstat();row['identity_after']=identity(after)
  row.update(entries=entries,regular_files=files,directories=dirs,symlinks=links,logical_bytes=logical,allocated_bytes=allocated,unique_allocated_bytes_across_selected=unique,status='COMPLETE_METADATA_ONLY' if not issues else 'UNKNOWN_OR_PARTIAL',issues=issues)
 except FileNotFoundError:row.update(status='MISSING_OR_CHANGED',issues=issues)
 except (OSError,RuntimeError,AssertionError) as exc:row.update(status='REFUSED',error=str(exc))
 records.append(row)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','free_bytes':free,'targets':records,'total_entries':total_entries,'elapsed_seconds':time.monotonic()-started,'original_lock':{'device':lk.st_dev,'inode':lk.st_ino,'held_during_inventory':True},'busy_job_rows':jobs,'neuron_ls':devices,'bounds':{'seconds':25,'entries':50000,'selected_paths':3},'file_contents_read':False,'deletion_authorized':False,'no_host_writes':True,'protected_shared_cache_and_best_candidate_paths_excluded':True}))
'''
wire=('P='+repr({'targets':targets})+'\n'+body).encode()
assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=45)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'payload_sha256':hashlib.sha256(wire).hexdigest(),'source_bindings':bindings,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with (HERE/'inventory.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
print(json.dumps(receipt['remote'],indent=2))
