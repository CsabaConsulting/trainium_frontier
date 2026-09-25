import base64,datetime,hashlib,json,os,signal,stat,subprocess,time
from pathlib import Path
ROOT='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch36a8-startup-clock-trn22-r1-full-packet'
DEBUG='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch36a8-startup-clock-cold-trn22-r1-packet/train/caches/debug/neuronx-cc'
EXPECTED_WORKERS={3191056,3191057,3191058,3191059}
MAX_DIRECTORIES=1
EXCLUDED_DIRS=["/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch36a8-startup-clock-cold-trn22-r1-packet/train/caches/debug/neuronx-cc/5818832ea9b1141bb19fe350b46c050b"]
started=time.monotonic()
out={'schema':'live-current-clock-compiler-read-v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':ROOT,'debug_root':DEBUG,'status':'UNKNOWN','directories':[],'read_bytes':0,'no_host_writes':True,'no_sdk_import':True}
class Refuse(Exception):pass
class Deadline(BaseException):pass
def alarm(*a):raise Deadline('TIME_BOUND_29_SECONDS')
signal.signal(signal.SIGALRM,alarm);signal.alarm(29)
opened=[]
def ident(s):return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'mode':s.st_mode}
def ancestors(pid,rows):
 seen=set()
 while pid in rows and pid not in seen:
  seen.add(pid)
  if pid in EXPECTED_WORKERS:return pid
  pid=rows[pid]['ppid']
 return None
def nofollow_dir(path):
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY);opened.append(fd)
 for part in Path(path).parts[1:]:
  fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);opened.append(fd)
 return fd
try:
 if os.uname().nodename!='ip-172-31-32-253':raise Refuse('HOST_MISMATCH')
 ps=subprocess.run(['/usr/bin/ps','-eo','pid=,ppid=,pgid=,sid=,comm=,args='],capture_output=True,text=True,timeout=10)
 if ps.returncode or ps.stderr:raise Refuse('PS_FAILED:'+ps.stderr)
 out['ps_sha256']=hashlib.sha256(ps.stdout.encode()).hexdigest()
 rows={}
 for line in ps.stdout.splitlines():
  x=line.split(None,5)
  if len(x)!=6:raise Refuse('PS_PARSE')
  rows[int(x[0])]={'ppid':int(x[1]),'pgid':int(x[2]),'sid':int(x[3]),'comm':x[4],'argv':x[5]}
 for pid in EXPECTED_WORKERS:
  p=Path('/proc')/str(pid)
  if pid not in rows or os.readlink(p/'cwd')!=ROOT+'/train' or '--startup-adjusted-schedule' not in rows[pid]['argv']:raise Refuse('RECORDED_WORKER_IDENTITY_NOT_LIVE')
 out['worker_pids']=sorted(EXPECTED_WORKERS)
 selected={}
 for pid,row in rows.items():
  if 'neuronx-cc' not in row['comm'] and 'walrus' not in row['comm']:continue
  parent=ancestors(pid,rows)
  if parent is None:continue
  proc=Path('/proc')/str(pid)
  try:
   cwd=os.readlink(proc/'cwd')
   args=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
  except FileNotFoundError:continue
  p=Path(cwd)
  if str(p.parent)!=DEBUG or not p.name or cwd in EXCLUDED_DIRS:continue
  selected.setdefault(cwd,[]).append({'pid':pid,'worker_ancestor':parent,'ppid':row['ppid'],'pgid':row['pgid'],'sid':row['sid'],'argv':args,'cwd':cwd})
 if not selected:raise Refuse('NO_UNAMBIGUOUS_CURRENT_COMPILER_DIRECTORY')
 for dirname,provenance in list(selected.items())[:MAX_DIRECTORIES]:
  rec={'directory':dirname,'compiler_processes':provenance,'files':[]};out['directories'].append(rec)
  dfd=nofollow_dir(dirname)
  for fn in ('module.mlir','log-neuron-cc.txt'):
   r={'name':fn};rec['files'].append(r);fd=None
   try:
    before=os.stat(fn,dir_fd=dfd,follow_symlinks=False);r['before']=ident(before)
    if not stat.S_ISREG(before.st_mode):raise Refuse('NOT_REGULAR_OR_SYMLINK')
    if fn=='module.mlir' and before.st_size>4*1024**2:
     r['after']=ident(os.stat(fn,dir_fd=dfd,follow_symlinks=False));r['status']='OVERSIZE';continue
    fd=os.open(fn,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=dfd)
    if ident(os.fstat(fd))!=ident(before):raise Refuse('OPEN_RACE')
    limit=before.st_size if fn=='module.mlir' else min(65536,before.st_size)
    chunks=[];got=0
    while got<limit:
     b=os.read(fd,min(65536,limit-got))
     if not b:raise Refuse('UNEXPECTED_EOF')
     got+=len(b);out['read_bytes']+=len(b)
     if out['read_bytes']>9*1024**2:raise Refuse('BYTE_LIMIT')
     chunks.append(b)
    raw=b''.join(chunks)
    after=os.fstat(fd);path_after=os.stat(fn,dir_fd=dfd,follow_symlinks=False)
    r['after']=ident(after);r['path_after']=ident(path_after)
    r['bytes_retained']=len(raw);r['sha256']=hashlib.sha256(raw).hexdigest();r['base64']=base64.b64encode(raw).decode()
    r['hash_scope']='entire_module.mlir' if fn=='module.mlir' else 'first_at_most_65536_bytes_only'
    r['status']='READ_STABLE' if ident(before)==ident(after)==ident(path_after) else 'UNSTABLE'
   except (OSError,Refuse) as e:r['status']='MISSING_OR_REFUSED';r['error']=type(e).__name__+': '+str(e)
   finally:
    if fd is not None:os.close(fd)
  rec['provenance_pids_still_present']=[{'pid':p['pid'],'present':Path('/proc/'+str(p['pid'])).exists()} for p in provenance]
 out['status']='OBSERVED'
except BaseException as e:out['error']=type(e).__name__+': '+str(e)
finally:
 signal.alarm(0)
 for fd in reversed(opened):os.close(fd)
 out['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out['elapsed_seconds']=time.monotonic()-started
 out['interpretation']='Exact existing compiler input only; principal-graph mapping awaits closed graph_stats. No cache-hit or path-causality inference.'
 print(json.dumps(out))
