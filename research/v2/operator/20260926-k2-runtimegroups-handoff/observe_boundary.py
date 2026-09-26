"""One lightweight read-only live boundary bracket; no exact-memory timestamp claim."""
import hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert len(sys.argv)==2 and re.fullmatch('[a-z0-9-]+',sys.argv[1])
out=HERE/('trn22-boundary-'+sys.argv[1]+'.json');assert not out.exists()
packet=json.loads((HERE.parent/'20260926-final-sprint-successors/trn22-runtimegroups-native201-bound-packet.json').read_text())
b=packet['binding'];assert b['host']=='trn22' and b['lock_identity']==[66305,832589]
body=r'''
import datetime,hashlib,json,os,socket,stat,time
from pathlib import Path
assert socket.gethostname()==b['hostname']
root=Path(b['root']);start=time.monotonic();files={}
before=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name,cap in [('train/steps.jsonl',16384),('train/rehearsal.json',131072),('training.log',16384),('terminal.json',8192),('outer-flock-exit.json',8192)]:
 p=root/name
 try:
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 except FileNotFoundError:files[name]={'absent':True};continue
 try:
  s=os.fstat(fd);assert stat.S_ISREG(s.st_mode)
  if s.st_size>cap:os.lseek(fd,-cap,os.SEEK_END)
  raw=os.read(fd,cap);after=os.fstat(fd)
  files[name]={'size_before':s.st_size,'size_after':after.st_size,'mtime_ns':after.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'tail_only':s.st_size>cap,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}
 finally:os.close(fd)
steps=files['train/steps.jsonl'];indices=None
if 'text' in steps and not steps['tail_only']:
 complete=steps['text'].splitlines()
 if steps['text'] and not steps['text'].endswith('\n'):complete=complete[:-1]
 indices=[int(x) for x in complete]
 assert indices==list(range(len(indices))),'NONSEQUENTIAL_LIVE_BOUNDARIES'
lock=Path(b['lock_path']).lstat();assert stat.S_ISREG(lock.st_mode) and [lock.st_dev,lock.st_ino]==b['lock_identity']
guardian={}
try:
 raw=Path('/proc/3399167/stat').read_text();v=raw[raw.rfind(')')+2:].split()
 guardian={'pid':3399167,'state':v[0],'ppid':int(v[1]),'pgid':int(v[2]),'sid':int(v[3]),'start_ticks':int(v[19])}
 assert guardian['start_ticks']==103271087,'GUARDIAN_IDENTITY_CHANGED'
 guardian['original_fd3_identity']=[Path('/proc/3399167/fd/3').stat().st_dev,Path('/proc/3399167/fd/3').stat().st_ino]
 guardian['fdinfo']=Path('/proc/3399167/fdinfo/3').read_text()
except FileNotFoundError:guardian['absent_or_exit_race']=True
print(json.dumps({'read_only':True,'label':b['label'],'utc_before':before,'utc_after':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_after':time.monotonic(),'elapsed_seconds':time.monotonic()-start,'files':files,'complete_live_boundary_count':len(indices) if indices is not None else None,'fifth_complete':len(indices)>=5 if indices is not None else None,'guardian':guardian,'exact_fifth_seconds_available_only_in_terminal_rehearsal':True}))
'''
wire=('b='+repr(b)+'\n'+body).encode()
connector=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(connector.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',connector);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=20);assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'returncode':r.returncode,'stderr':r.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'returncode':r.returncode,'error':receipt['stderr'],'remote':receipt['remote'],'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
