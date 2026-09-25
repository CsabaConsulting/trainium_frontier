import datetime,fcntl,hashlib,importlib.util,json,os,signal,stat,subprocess,time
from pathlib import Path
started=time.monotonic()
out={'schema':'bounded-s1-three-directory-existing-file-observation-v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'REFUSED','files':[],'bytes_read':0,'no_host_files_written':True,'no_neff_contents_retained':True,'limits':{'seconds':30,'read_bytes':60*1024**2,'retained_json_bytes':16*1024**2}}
class Bound(Exception):pass
class TimeBound(BaseException):pass
def deadline(*a):raise TimeBound('READ_TIME_LIMIT')
signal.signal(signal.SIGALRM,deadline);signal.alarm(29)
lockfd=None;opened=[]
def ss(st):return {'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'mode':st.st_mode}
def readpart(fd,n):
 if out['bytes_read']+n>60*1024**2:raise Bound('GLOBAL_READ_LIMIT')
 b=os.read(fd,n);out['bytes_read']+=len(b);return b
def dirfd_nofollow(path):
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY);opened.append(fd)
 for part in Path(path).parts[1:]:
  fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);opened.append(fd)
 return fd
try:
 if os.uname().nodename!='ip-172-31-32-253' or os.getuid()!=1000:raise Bound('HOST_UID_MISMATCH')
 lock='/home/ubuntu/.cache/trainium_freshstart/device0.lock'
 lockfd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
 ls=os.fstat(lockfd)
 if (ls.st_dev,ls.st_ino)!=(66305,832589) or not stat.S_ISREG(ls.st_mode):raise Bound('LOCK_IDENTITY_MISMATCH')
 fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out['lock']={'path':lock,'device':ls.st_dev,'inode':ls.st_ino,'fd':lockfd,'exclusive_held_through_all_reads':True}
 guard=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch36a8-startup-clock-cold-trn22-r1-packet/idle_guard.py')
 if guard.is_symlink() or hashlib.sha256(guard.read_bytes()).hexdigest()!='adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737':raise Bound('GUARD_HASH_MISMATCH')
 spec=importlib.util.spec_from_file_location('guard',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 out['idle_checks']={}
 g.check_idle(subprocess.run,os.getpid(),os.getppid(),out['idle_checks'])
 prefix='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-stablepath-trn22-r1-pair/s1/train/caches/debug/neuronx-cc'
 dirs=('96b199d519a331288b0a67881f092b12','ea30e57eba554980a991fca1b737d5ac','860bfc7a33ecceebfbabbde99c9f2e27')
 principals={'forward':'aa53f72d8a4c45dbc0ba1784f3536f8803525417f6de4d3f8fb82332aa2ef58e','backward':'d81a59fd782a752806c228735a21928133bf2d21c5d59344e2e0e94fb3db8c25'}
 parent=dirfd_nofollow(prefix)
 for dn in dirs:
  try:
   dfd=os.open(dn,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);opened.append(dfd)
  except OSError as e:
   out['files'].append({'directory':dn,'status':'DIRECTORY_UNAVAILABLE','error':type(e).__name__+': '+str(e)});continue
  for fn in ('module.mlir','module.neff','log-neuron-cc.txt'):
   r={'relative_path':dn+'/'+fn};out['files'].append(r);fd=None
   try:
    before=os.stat(fn,dir_fd=dfd,follow_symlinks=False);r['before']=ss(before)
    if not stat.S_ISREG(before.st_mode):raise Bound('NOT_REGULAR_OR_SYMLINK')
    fd=os.open(fn,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=dfd)
    if ss(os.fstat(fd))!=ss(before):raise Bound('OPEN_IDENTITY_RACE')
    limit=(4 if fn=='module.mlir' else 16)*1024**2
    if fn!='log-neuron-cc.txt' and before.st_size>limit:
     r['after']=ss(os.fstat(fd));r['path_after']=ss(os.stat(fn,dir_fd=dfd,follow_symlinks=False))
     if r['before']!=r['after'] or r['after']!=r['path_after']:raise Bound('OVERSIZE_IDENTITY_RACE')
     r['status']='OVERSIZE';continue
    if fn=='log-neuron-cc.txt':
     head=readpart(fd,min(65536,before.st_size))
     tail=b''
     if before.st_size>65536:
      os.lseek(fd,max(65536,before.st_size-65536),os.SEEK_SET)
      tail=readpart(fd,min(65536,before.st_size-65536))
     r['head_text']=head.decode(errors='replace');r['tail_text']=tail.decode(errors='replace')
     r['head_sha256']=hashlib.sha256(head).hexdigest();r['tail_sha256']=hashlib.sha256(tail).hexdigest()
     r['full_sha256']=hashlib.sha256(head).hexdigest() if before.st_size<=65536 else None
    else:
     h=hashlib.sha256();chunks=[];seen=0
     while seen<before.st_size:
      b=readpart(fd,min(65536,before.st_size-seen))
      if not b:raise Bound('UNEXPECTED_EOF')
      h.update(b);seen+=len(b)
      if fn=='module.mlir':chunks.append(b)
     r['sha256']=h.hexdigest();r['bytes_hashed']=seen
     if fn=='module.mlir':
      r['principal_matches']=[k for k,v in principals.items() if v==r['sha256']]
      r['text']=b''.join(chunks).decode()
    after=os.fstat(fd);path_after=os.stat(fn,dir_fd=dfd,follow_symlinks=False)
    r['after']=ss(after);r['path_after']=ss(path_after)
    if ss(before)!=ss(after) or ss(after)!=ss(path_after):raise Bound('READ_IDENTITY_RACE')
    r['status']='READ_STABLE'
    if len(json.dumps(out).encode())>15*1024**2:
     r.pop('text',None);r.pop('head_text',None);r.pop('tail_text',None);r['retained_content_omitted']='ARTIFACT_LIMIT'
   except (OSError,UnicodeError,Bound) as e:
    r['status']='MISSING' if isinstance(e,FileNotFoundError) else 'REFUSED'
    r['error']=type(e).__name__+': '+str(e)
   finally:
    if fd is not None:os.close(fd)
 if os.fstat(lockfd).st_ino!=832589:raise Bound('LOCK_CHANGED')
 out['status']='OBSERVED'
except BaseException as e:out['error']=type(e).__name__+': '+str(e)
finally:
 signal.alarm(0)
 out['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out['elapsed_seconds']=time.monotonic()-started
 for fd in reversed(opened):os.close(fd)
 if lockfd is not None:os.close(lockfd)
 out['interpretation']='Existing input identity only; absence now does not prove earlier absence. Original S1 refusal and no-S2 remain.'
 serialized=json.dumps(out)
 if len(serialized.encode())>16*1024**2:raise RuntimeError('FINAL_ARTIFACT_LIMIT')
 print(serialized)
