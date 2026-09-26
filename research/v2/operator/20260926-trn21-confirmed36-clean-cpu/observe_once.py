"""One bounded read-only CPU pair observation; names a unique local receipt."""
import datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert len(sys.argv)==2 and sys.argv[1].isalnum(),'UNIQUE_RECEIPT_NAME'
dest=HERE/'launch'/('observation-'+sys.argv[1]+'.json');assert not dest.exists()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=r'''
import datetime,json,os,stat,subprocess
from pathlib import Path
root=Path('/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1')
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0,ps.stderr
rows=[]
for line in ps.stdout.splitlines()[1:]:
 p=line.split(None,4)
 if len(p)==5:rows.append({'pid':int(p[0]),'ppid':int(p[1]),'pgid':int(p[2]),'sid':int(p[3]),'argv':p[4]})
ids={3176685}|{r['pid'] for r in rows if str(root) in r['argv']}
for _ in range(12):
 new=ids|{r['pid'] for r in rows if r['ppid'] in ids or r['pgid'] in ids or r['sid'] in ids}
 if new==ids:break
 ids=new
procs=[]
for r in rows:
 if r['pid'] not in ids:continue
 try:
  p=Path('/proc')/str(r['pid']);s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()
  r.update(starttime=int(t[19]),state=t[0],cwd=os.readlink(p/'cwd'))
  env={}
  for x in (p/'environ').read_bytes().split(b'\0'):
   k,sep,v=x.partition(b'=')
   if sep and k in (b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'OMP_NUM_THREADS',b'TRN_EXP_LABEL'):env[k.decode()]=v.decode()
  r['environment_whitelist']=env;procs.append(r)
 except FileNotFoundError:r['exited_during_observation']=True;procs.append(r)
names=['operator-launch.json','preflight.json','under-lock-idle.json','dispatch.log','outer-flock-exit.json','terminal.json','standard-compare.json','standard-compare.stderr','standard-compare-exit.json']
for arm in ['reference','candidate']:
 names += [arm+'/'+f for f in ['launch.json','process.json','terminal.json','preflight.json','under-lock-idle.json','g2.json','g2.log','checkpoint-identity.json']]
files={};total=0
for name in names:
 p=root/name
 if p.is_file():
  s=p.lstat()
  if not stat.S_ISREG(s.st_mode):continue
  if s.st_size>1024*1024 or total+s.st_size>6*1024*1024:files[name]={'refused':'BOUND'};continue
  raw=p.read_bytes();total+=len(raw);files[name]={'text':raw.decode(errors='replace'),'bytes':len(raw),'sha256':__import__('hashlib').sha256(raw).hexdigest()}
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)
lk=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock').stat()
locks=[line for line in Path('/proc/locks').read_text().splitlines() if any(field.endswith(':596823') for field in line.split())]
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'processes':procs,'selected_pids':sorted(ids),'files':files,'neuron_ls_rc':ls.returncode,'neuron_ls':json.loads(ls.stdout) if ls.returncode==0 else None,'neuron_ls_stderr':ls.stderr,'lock':{'device':lk.st_dev,'inode':lk.st_ino,'held':bool(locks),'rows':locks},'no_changes':True}))
'''
assert not m.SENSITIVE.search(body.encode())
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=body,text=True,capture_output=True,timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stderr':r.stderr,'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout if r.returncode else None}
with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
o=receipt['remote'];print(json.dumps({'utc':o['utc'],'processes':o['processes'],'lock':o['lock'],'terminal':o['files'].get('terminal.json'),'reference_terminal':o['files'].get('reference/terminal.json'),'candidate_terminal':o['files'].get('candidate/terminal.json'),'receipt':str(dest),'receipt_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
