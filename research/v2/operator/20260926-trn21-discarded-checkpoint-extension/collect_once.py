"""Two explicitly assigned discarded checkpoints, metadata and bounded hashes only."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'inventory.json').exists(),'LOCAL_RECEIPT_EXISTS'
targets=[
 {'path':'/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt','expected_bytes':739322371,'expected_sha256':'6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b','verdict':'DISCARD','provenance':'research/v2/analysis/20260925-a1-negative-a2-invalid-quality.md'},
 {'path':'/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/fs-20260925-fp-c5c-matrixlr0175-trn21-r1/out/final.pt','expected_bytes':None,'expected_sha256':'f722109658e944dc6fcb27fc0e838759f07733e4b116d1bf98e8ce1e61870187','verdict':'DISCARD','provenance':'research/v2/verification/20260925-matrixlr0175-result-review/REVIEW.md'}
]
provenance=[{'path':str(REPO/x['provenance']),'sha256':hashlib.sha256((REPO/x['provenance']).read_bytes()).hexdigest()} for x in targets]
body=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
assert socket.gethostname()=='ip-172-31-34-85' and os.getuid()==1000
started=time.monotonic();deadline=started+45
fd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW)
lk=os.fstat(fd);assert (lk.st_dev,lk.st_ino)==(66305,596823)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=8);assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
assert not jobs,'BUSY_HOST'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=12);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0].get('neuron_processes')==[]
def identity(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
records=[]
for item in P['targets']:
 row=dict(item);p=Path(item['path'])
 try:
  for parent in reversed((p,*p.parents)):
   if stat.S_ISLNK(parent.lstat().st_mode):raise RuntimeError('SYMLINK_COMPONENT:'+str(parent))
  before=p.lstat();row['identity_before']=identity(before)
  assert stat.S_ISREG(before.st_mode) and before.st_dev==66305 and before.st_uid==1000,'TYPE_OWNER_DEVICE'
  assert 0<before.st_size<=1024**3,'PER_FILE_SIZE_BOUND'
  if item['expected_bytes'] is not None:assert before.st_size==item['expected_bytes'],'ARCHIVED_SIZE_MISMATCH'
  streamfd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  with os.fdopen(streamfd,'rb') as stream:
   opened=os.fstat(stream.fileno());assert identity(opened)==identity(before),'OPEN_IDENTITY_CHANGED'
   h=hashlib.sha256();read=0
   while True:
    assert time.monotonic()<deadline,'HASH_TIME_BOUND'
    block=stream.read(65536)
    if not block:break
    read+=len(block);assert read<=1024**3,'STREAM_SIZE_BOUND';h.update(block)
   final=os.fstat(stream.fileno())
  after=p.lstat();row['identity_after']=identity(after)
  assert identity(before)==identity(final)==identity(after),'METADATA_CHANGED_DURING_HASH'
  digest=h.hexdigest();row.update(bytes_read=read,sha256=digest,stable_metadata=True,hash_matches_archive=digest==item['expected_sha256'],status='EXACT_ARCHIVED_DISCARD_CHECKPOINT' if digest==item['expected_sha256'] else 'ARCHIVED_HASH_MISMATCH')
 except (OSError,RuntimeError,AssertionError) as exc:row.update(status='UNKNOWN_OR_REFUSED',error=str(exc))
 records.append(row)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','free_bytes':free,'targets':records,'elapsed_seconds':time.monotonic()-started,'original_lock':{'device':lk.st_dev,'inode':lk.st_ino,'held_during_inventory':True},'busy_job_rows':jobs,'neuron_ls':devices,'bounds':{'seconds':45,'selected_files':2,'bytes_per_file':1024**3,'stream_buffer_bytes':65536},'checkpoint_contents_transferred':False,'deletion_authorized':False,'no_host_writes':True,'shared_caches_inspected':False}))
'''
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
wire=('P='+repr({'targets':targets})+'\n'+body).encode();assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60);assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'payload_sha256':hashlib.sha256(wire).hexdigest(),'provenance':provenance,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with (HERE/'inventory.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
print(json.dumps(receipt['remote'],indent=2))
