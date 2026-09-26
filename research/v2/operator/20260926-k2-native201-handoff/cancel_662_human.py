"""One pidfd SIGTERM to the exact guardian; its existing cleanup retains flock."""
import ast,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
packet=json.loads((HERE.parent/'20260926-final-sprint-successors/trn21-localconstants-native201-bound-packet.json').read_bytes())
b=packet['binding']
assert b['label']=='fs-20260926-k2-localconstants-native201-trn21-r1'
body=r'''
import datetime,hashlib,json,os,signal,socket,stat,time
signal.alarm(20)
def ident(pid):
 s=open('/proc/'+str(pid)+'/stat').read().rsplit(')',1)[1].split()
 return {'pid':pid,'state':s[0],'ppid':int(s[1]),'pgid':int(s[2]),'sid':int(s[3]),'start_ticks':int(s[19])}
def argv(pid):return open('/proc/'+str(pid)+'/cmdline','rb').read().rstrip(b'\0').decode().split('\0')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'HUMAN_STOP_COMPILE_COST','label':P['label'],'signal_sent':False}
pidfd=None
try:
 assert socket.gethostname()=='ip-172-31-34-85','HOSTNAME'
 expected=['/usr/bin/python3','-I','-B',P['root']+'/host_driver.py']
 before=ident(3280863);assert before['start_ticks']==95693587 and before['ppid']==3280862,'GUARDIAN_IDENTITY'
 assert argv(3280863)==expected,'GUARDIAN_ARGV'
 parent=ident(3280862);assert parent['start_ticks']==95693587 and parent['ppid']==3280861,'FLOCK_IDENTITY'
 assert argv(3280862)==['/usr/bin/flock','-n',P['lock']]+expected,'FLOCK_ARGV'
 out['guardian_before']=before;out['flock_before']=parent
 for name,want in P['pins'].items():
  fd=os.open(P['root']+'/'+name,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   s=os.fstat(fd);assert stat.S_ISREG(s.st_mode) and s.st_size<=1048576,'PIN_FILE_TYPE_SIZE'
   data=os.read(fd,s.st_size+1);assert hashlib.sha256(data).hexdigest()==want,'PIN_DRIFT:'+name
  finally:os.close(fd)
 lock=os.lstat(P['lock']);assert stat.S_ISREG(lock.st_mode) and (lock.st_dev,lock.st_ino)==(66305,596823),'LOCK_INODE'
 lockfds=[]
 for name in os.listdir('/proc/3280863/fd'):
  try:
   s=os.stat('/proc/3280863/fd/'+name)
   if (s.st_dev,s.st_ino)==(66305,596823):
    info=open('/proc/3280863/fdinfo/'+name).read()
    assert any(x.startswith('lock:') and 'FLOCK' in x and 'WRITE' in x for x in info.splitlines()),'LOCK_NOT_HELD'
    lockfds.append(int(name))
  except FileNotFoundError:continue
 assert lockfds,'GUARDIAN_LOCK_FD_MISSING';out['original_lock_fds']=lockfds
 pidfd=os.pidfd_open(3280863)
 after=ident(3280863);assert after['start_ticks']==before['start_ticks'] and after['ppid']==before['ppid'] and argv(3280863)==expected,'IDENTITY_AFTER_PIDFD'
 out['guardian_after_pidfd']=after;out['signal_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 signal.pidfd_send_signal(pidfd,signal.SIGTERM)
 out['signal_sent']=True;out['signal']='SIGTERM';out['status']='EXACT_GUARDIAN_CANCELLATION_REQUESTED'
 out['cleanup']='Existing reviewed Guardian.quiesce handles verified child pidfds, TERM60/KILL10 and ECHILD plus independent idle before original-lock release; no external group/name signal.'
except Exception as exc:out['status']='REFUSED_OR_ERROR';out['error']=repr(exc)
finally:
 if pidfd is not None:os.close(pidfd)
print(json.dumps(out))
'''
payload={'root':b['root'],'label':b['label'],'lock':b['lock_path'],
 'pins':{k:packet['files'][k]['sha256'] for k in ('host_driver.py','lifetime_guard.py','package/train.py')}}
cp=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(cp.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',cp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
wire=('P='+repr(payload)+'\n'+body).encode();ast.parse(wire);assert not m.SENSITIVE.search(wire)
out=HERE/'human-stop-signal.json';assert not out.exists()
argv_,*_=m.load_transport('trn21');argv_[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv_,input=wire,capture_output=True,timeout=35);assert not m.SENSITIVE.search(r.stdout+r.stderr)
d={'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None,'payload_sha256':hashlib.sha256(wire).hexdigest()}
with out.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d))
