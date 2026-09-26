"""Remove only six inventoried private cache children of discarded scored 3e79."""
import ast, datetime, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
inv_path=HERE/'trn22-3e79-cache-inventory.json';raw=inv_path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a96c8d8665981f1c7faaf7d5cc1f609f86a373a43810c011493e0a4df855fe68'
inv=json.loads(raw)['remote'];targets=inv['targets']
root='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches'
assert [t['path'] for t in targets]==[root+'/'+x for x in ('compiler','neff','local','hlo','inductor','debug')]
assert all(t['status']=='COMPLETE' and not t['issues'] for t in targets)
auth=(REPO/'workflow/FINAL_SPRINT_2026-09-26.txt').read_bytes();assert b'discarded, already-scored runs' in auth
tree=ast.parse((HERE/'status_once.py').read_text())
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
head=body[:body.index('r={')]
tail=r'''
import shutil
assert locked and not jobs and len(devices)==1 and devices[0].get('neuron_processes')==[]
assert fuser['exit_code']==1 and not fuser['stdout'].strip() and not fuser['stderr'].strip() and not stops
assert shutil.rmtree.avoids_symlink_attacks
def ident(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'nlink':s.st_nlink,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def free():
 v=os.statvfs('/');return v.f_bavail*v.f_frsize
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(event,**kw):print(json.dumps({'event':event,'utc':utc(),**kw}),flush=True)
deadline=time.monotonic()+30
for target in P['targets']:
 path=Path(target['path'])
 for p in reversed((path,*path.parents)):
  if stat.S_ISLNK(p.lstat().st_mode):raise RuntimeError('SYMLINK_COMPONENT:'+str(p))
 assert ident(path.lstat())==target['identity_before'],'ROOT_IDENTITY_DRIFT'
 actual={};pending=[path]
 while pending:
  assert time.monotonic()<=deadline,'CENSUS_TIME_BOUND'
  p=pending.pop();s=p.lstat();rel=str(p.relative_to(path))
  actual[rel]=ident(s)
  assert s.st_dev==66305 and s.st_uid==1000,'OWNER_DEVICE_DRIFT'
  assert stat.S_ISDIR(s.st_mode) or (stat.S_ISREG(s.st_mode) and s.st_nlink==1),'TYPE_OR_HARDLINK_DRIFT'
  if stat.S_ISDIR(s.st_mode):
   with os.scandir(p) as it:pending.extend(Path(e.path) for e in it)
 expected={e['relative_path']:{k:v for k,v in e.items() if k!='relative_path'} for e in target['entries']}
 assert actual==expected,'CACHE_CENSUS_DRIFT'
# Reject a live process reference to a private retired cache, even if its name
# does not match training. Readlink only; never traverse its target.
references=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 links=[proc/x for x in ('cwd','root','exe')]
 try:links.extend((proc/'fd').iterdir())
 except (PermissionError,FileNotFoundError):pass
 for link in links:
  try:value=os.readlink(link)
  except (PermissionError,FileNotFoundError,ProcessLookupError,OSError):continue
  if value==P['root'] or value.startswith(P['root']+'/'):references.append({'pid':int(proc.name),'link':str(link),'target':value})
assert not references,'LIVE_CACHE_REFERENCE'
before=free();emit('ALL_SIX_PRIVATE_CACHES_VERIFIED',free_before=before,paths=[t['path'] for t in P['targets']],allocated_bytes=sum(t['allocated_bytes'] for t in P['targets']))
pfd=os.open(P['root'],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
removed=[]
try:
 for target in P['targets']:
  name=Path(target['path']).name
  assert ident(os.stat(name,dir_fd=pfd,follow_symlinks=False))==target['identity_before'],'PREDELETE_DRIFT'
  shutil.rmtree(name,dir_fd=pfd)
  try:os.stat(name,dir_fd=pfd,follow_symlinks=False)
  except FileNotFoundError:pass
  else:raise RuntimeError('TARGET_REMAINS')
  event={'path':target['path'],'allocated_bytes':target['allocated_bytes'],'root_inode':target['identity_before']['inode']}
  removed.append(event);emit('REMOVED_PRIVATE_CACHE',**event)
finally:os.close(pfd)
postps=run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'])
assert postps['exit_code']==0 and not any(k in line for line in postps['stdout'].splitlines() for k in terms)
postls=run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],12);assert postls['exit_code']==0 and json.loads(postls['stdout'])[0].get('neuron_processes')==[]
after=free()
emit('FINAL',result={'status':'SIX_DISCARDED_SCORED_PRIVATE_CACHES_REMOVED','host':'trn22','free_before':before,'free_after':after,'free_increase':after-before,'removed':removed,'original_lock':{'device':ls.st_dev,'inode':ls.st_ino,'held_through_cleanup':True},'no_checkpoint_or_source_or_runlog_deletion':True,'post_neuron_ls':postls,'elapsed_seconds':time.monotonic()-started})
fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
'''
body=head+tail;ast.parse(body)
payload={'host':'trn22','hostname':'ip-172-31-32-253','inode':832589,'roots':['/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1'],'root':root,'targets':targets}
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('strict_connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert not m.SENSITIVE.search(wire)
if sys.argv[1:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_CHECK_PASS','payload_sha256':hashlib.sha256(wire).hexdigest(),'paths':[t['path'] for t in targets],'allocated_bytes':sum(t['allocated_bytes'] for t in targets)}));raise SystemExit(0)
assert sys.argv[1:]==['--execute']
out=HERE/'trn22-3e79-cache-cleanup.json';attempt=HERE/'trn22-3e79-cache-cleanup-attempt.json'
assert not out.exists() and not attempt.exists()
state=m.local_state();assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
with attempt.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'controller':state,'inventory_sha256':hashlib.sha256(raw).hexdigest(),'authorization_sha256':hashlib.sha256(auth).hexdigest()},f,indent=2);f.write('\n')
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
p=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(p.stdout+p.stderr)
events=[json.loads(l) for l in p.stdout.splitlines() if l.strip()]
with out.open('x') as f:json.dump({'returncode':p.returncode,'stderr':p.stderr.decode(),'events':events,'payload_sha256':hashlib.sha256(wire).hexdigest()},f,indent=2);f.write('\n')
print(json.dumps({'returncode':p.returncode,'stderr':p.stderr.decode(),'final':events[-1] if events else None,'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
raise SystemExit(p.returncode)
