"""One human-approved, exact three-file unlink. No directory/cache cleanup or job launch."""
import ast
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
AUTH = REPO/'research/v2/verification/20260926-trn21-three-discarded-checkpoints-cleanup-authorization.md'
AUTH_SHA = '3cf0200e455ee58b57433e02af6784f4a98b0524c013a178b1cb93a55813b2c2'
auth = AUTH.read_bytes()
assert hashlib.sha256(auth).hexdigest() == AUTH_SHA, 'AUTH_DRIFT'
inventory_paths = [HERE.parent/'20260926-trn21-discarded-checkpoint-extension/inventory.json', HERE.parent/'20260926-trn21-discarded-scratch-inventory/inventory.json']
inventory_hashes = ['681309082b771e488efdd2e97d8cc5a2394e826ecff7a741e75bda31626cc541','122ac00bd9718ce89fc5c23174bd5c4dfcbdc6e7117e859a0fa50517b5a785b1']
inventory = {}
for path, expected in zip(inventory_paths, inventory_hashes):
    b = path.read_bytes(); assert hashlib.sha256(b).hexdigest() == expected
    inventory.update({v['path']:v['identity_before'] for v in json.loads(b)['remote']['targets']})
literal = [
    ('/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt',739322371,'6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b'),
    ('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/fs-20260925-fp-c5c-matrixlr0175-trn21-r1/out/final.pt',469834849,'f722109658e944dc6fcb27fc0e838759f07733e4b116d1bf98e8ce1e61870187'),
    ('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-basebcbd-initial-cache-trn21-r1-full-packet/train/out/final.pt',469834913,'edd9adca6f6f782f7ccb220f567ffed7177291a65f3e28ec269b38403e1a47ec'),
]
targets = []
for path, size, digest in literal:
    assert path.encode() in auth and digest.encode() in auth and str(size).encode() in auth
    expected = inventory[path]
    assert expected['bytes'] == size and expected['device'] == 66305 and expected['uid'] == 1000 and expected['nlink'] == 1
    targets.append({'path':path,'expected_bytes':size,'expected_sha256':digest,'expected_identity':expected})
body = r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time,traceback
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def free():
 v=os.statvfs('/');return v.f_bavail*v.f_frsize
def emit(event,**details):print(json.dumps({'utc':utc(),'event':event,**details}),flush=True)
started=time.monotonic();deadline=started+45;lockfd=None;held=False;opened=[]
result={'started_utc':utc(),'host':'trn21','authorization_sha256':P['authorization_sha256'],'targets':[],'operations':[],'training_launched':False,'cache_or_directory_deletion':False}
def idle():
 ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=8)
 if ps.returncode:raise RuntimeError('PS_FAILED:'+ps.stderr)
 busy=[l for l in ps.stdout.splitlines()[1:] if any(k in l for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
 if busy:raise RuntimeError('BUSY_HOST:'+repr(busy))
 nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=12)
 if nls.returncode:raise RuntimeError('NEURON_LS_FAILED:'+nls.stderr)
 devices=json.loads(nls.stdout)
 if not (isinstance(devices,list) and len(devices)==1 and devices[0].get('neuron_device')==0 and devices[0].get('neuron_processes')==[]):raise RuntimeError('DEVICE_NOT_IDLE_OR_UNKNOWN')
 nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'))
 if not nodes:raise RuntimeError('NO_DEVICE_NODES')
 fu=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=8)
 if fu.returncode!=1 or fu.stdout.strip() or fu.stderr.strip():raise RuntimeError('DEVICE_HOLDER_OR_FUSER_ERROR:'+repr((fu.returncode,fu.stdout,fu.stderr)))
 return {'utc':utc(),'ps_rc':0,'ps_row_count':len(ps.stdout.splitlines())-1,'busy':busy,'ps_sha256':hashlib.sha256(ps.stdout.encode()).hexdigest(),'neuron_ls':devices,'neuron_ls_rc':0,'neuron_ls_stderr':nls.stderr,'fuser_rc':fu.returncode,'fuser_stdout':fu.stdout,'fuser_stderr':fu.stderr}
def parent_nofollow(path):
 parts=Path(path).parts
 if parts[0]!='/' or '..' in parts:raise RuntimeError('NONEXACT_PATH')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in parts[1:-1]:
   nextfd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
   os.close(fd);fd=nextfd
  return fd,parts[-1]
 except BaseException:
  os.close(fd);raise
try:
 if socket.gethostname()!='ip-172-31-34-85' or os.getuid()!=1000:raise RuntimeError('WRONG_HOST_OR_UID')
 lockfd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDONLY|os.O_NOFOLLOW)
 lk=os.fstat(lockfd)
 if (lk.st_dev,lk.st_ino)!=(66305,596823) or not stat.S_ISREG(lk.st_mode):raise RuntimeError('ORIGINAL_LOCK_IDENTITY_MISMATCH')
 fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
 result['original_lock']={'device':lk.st_dev,'inode':lk.st_ino,'held_through_operation':True}
 result['preflight']=idle();result['free_before']=free()
 for item in P['targets']:
  if time.monotonic()>deadline:raise RuntimeError('PREVALIDATION_TIME_BOUND')
  pfd,name=parent_nofollow(item['path']);streamfd=None
  try:
   streamfd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=pfd)
   before=os.fstat(streamfd)
   if not stat.S_ISREG(before.st_mode) or identity(before)!=item['expected_identity']:raise RuntimeError('ARCHIVED_IDENTITY_MISMATCH:'+item['path'])
   h=hashlib.sha256();count=0
   while True:
    if time.monotonic()>deadline:raise RuntimeError('HASH_TIME_BOUND')
    chunk=os.read(streamfd,65536)
    if not chunk:break
    count+=len(chunk)
    if count>item['expected_bytes']:raise RuntimeError('SIZE_CHANGED')
    h.update(chunk)
   after=os.fstat(streamfd);named=os.stat(name,dir_fd=pfd,follow_symlinks=False)
   if identity(before)!=identity(after) or identity(after)!=identity(named):raise RuntimeError('HASH_IDENTITY_CHANGED')
   if count!=item['expected_bytes'] or h.hexdigest()!=item['expected_sha256']:raise RuntimeError('ARCHIVED_HASH_MISMATCH:'+item['path'])
   record={**item,'identity_before':identity(before),'identity_after_hash':identity(after),'bytes_read':count,'sha256':h.hexdigest(),'verified':True}
   result['targets'].append(record);opened.append({'parent_fd':pfd,'name':name,'file_fd':streamfd,'record':record})
   pfd=None;streamfd=None
  finally:
   if streamfd is not None:os.close(streamfd)
   if pfd is not None:os.close(pfd)
 if len(opened)!=3:raise RuntimeError('NOT_ALL_THREE_VERIFIED')
 # Reauthenticate all original names and descriptors before the first unlink.
 for entry in opened:
  r=entry['record'];pfd,name=parent_nofollow(r['path'])
  try:
   if identity(os.stat(name,dir_fd=pfd,follow_symlinks=False))!=r['expected_identity']:raise RuntimeError('PREUNLINK_PATH_DRIFT')
   if (os.fstat(pfd).st_dev,os.fstat(pfd).st_ino)!=(os.fstat(entry['parent_fd']).st_dev,os.fstat(entry['parent_fd']).st_ino):raise RuntimeError('PARENT_PATH_DRIFT')
  finally:os.close(pfd)
 result['all_three_verified_before_any_unlink']=True
 emit('ALL_THREE_VERIFIED',targets=result['targets'],free_before=result['free_before'])
 for entry in opened:
  r=entry['record']
  if identity(os.fstat(entry['file_fd']))!=r['expected_identity'] or identity(os.stat(entry['name'],dir_fd=entry['parent_fd'],follow_symlinks=False))!=r['expected_identity']:raise RuntimeError('IMMEDIATE_PREUNLINK_DRIFT')
  os.unlink(entry['name'],dir_fd=entry['parent_fd'])
  op={'utc':utc(),'operation':'unlink_exact_regular_file','path':r['path'],'device':r['expected_identity']['device'],'inode':r['expected_identity']['inode']}
  result['operations'].append(op);emit('UNLINK_COMPLETED',**op)
  os.close(entry['file_fd']);entry['file_fd']=None
 for entry in opened:
  try:os.stat(entry['name'],dir_fd=entry['parent_fd'],follow_symlinks=False)
  except FileNotFoundError:entry['record']['absent_after']=True
  else:raise RuntimeError('TARGET_REAPPEARED')
 result['postflight']=idle();result['free_after']=free();result['free_increase']=result['free_after']-result['free_before'];result['status']='THREE_APPROVED_FILES_REMOVED'
except BaseException as exc:
 result['status']='FAILED_OR_PARTIAL_STOPPED';result['error_type']=type(exc).__name__;result['error']=str(exc);result['traceback']=traceback.format_exc();result['free_after_error']=free()
finally:
 for entry in opened:
  if entry['file_fd'] is not None:os.close(entry['file_fd'])
  os.close(entry['parent_fd'])
 result['ended_utc']=utc();result['elapsed_seconds']=time.monotonic()-started
 if held:fcntl.flock(lockfd,fcntl.LOCK_UN)
 if lockfd is not None:os.close(lockfd)
 emit('FINAL',result=result)
raise SystemExit(0 if result['status']=='THREE_APPROVED_FILES_REMOVED' else 2)
'''
ast.parse(body)
payload = {'authorization_sha256':AUTH_SHA,'targets':targets}
wire = ('P='+repr(payload)+'\n'+body).encode()
spec = importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
assert len(targets)==3 and sum(x['expected_identity']['allocated_bytes'] for x in targets)==1679007744
if sys.argv[1:]==['--local-check']:
    print(json.dumps({'status':'LOCAL_CHECK_PASS','targets':len(targets),'allocated_bytes':1679007744,'payload_sha256':hashlib.sha256(wire).hexdigest(),'authorization_sha256':AUTH_SHA}));raise SystemExit(0)
assert sys.argv[1:]==['--execute'],'REQUIRE_EXPLICIT_EXECUTE'
assert not (HERE/'attempt.json').exists() and not (HERE/'receipt.json').exists(),'ONE_USE_ALREADY_CONSUMED'
with (HERE/'attempt.json').open('x') as f:json.dump({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'authorization_sha256':AUTH_SHA,'targets':targets},f,indent=2);f.write('\n')
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
events=[json.loads(x) for x in r.stdout.decode().splitlines() if x.strip()]
receipt={'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stderr':r.stderr.decode(),'events':events,'payload_sha256':hashlib.sha256(wire).hexdigest(),'authorization_sha256':AUTH_SHA}
with (HERE/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2));raise SystemExit(r.returncode)
