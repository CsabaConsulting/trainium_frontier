"""One nofollow metadata census of six explicitly retired 3e79 cache children."""
import ast, datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
root='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches'
labels=['fs-20260926-arch3e79-primed-trn22-r2-full','fs-20260926-arch3e79-lr012-trn22-r1-full']
rows={r['label']:r for r in (json.loads(l) for l in (REPO/'research/v2/ledger.jsonl').read_text().splitlines())}
for label in labels:
 r=rows[label];assert r['host']=='trn22' and r['status']=='evaluated' and r['public_20m'] is not None
 assert r['env']['NEURON_COMPILE_CACHE_DIR']==root+'/compiler'
tree=ast.parse((HERE/'status_once.py').read_text())
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
head=body[:body.index('r={')]
tail=r'''
assert locked and not jobs and devices and len(devices)==1 and devices[0].get('neuron_processes')==[]
assert fuser['exit_code']==1 and not fuser['stdout'].strip() and not fuser['stderr'].strip()
assert not stops
def ident(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'nlink':s.st_nlink,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
deadline=time.monotonic()+30;total=0;records=[]
for target in P['targets']:
 row={'path':target};path=Path(target)
 try:
  for p in reversed((path,*path.parents)):
   if stat.S_ISLNK(p.lstat().st_mode):raise RuntimeError('SYMLINK_COMPONENT:'+str(p))
  before=path.lstat();row['identity_before']=ident(before)
  assert stat.S_ISDIR(before.st_mode) and before.st_dev==66305 and before.st_uid==1000
  pending=[path];entries=[];allocated=logical=0;issues=[];seen=set()
  while pending:
   if time.monotonic()>deadline or total>=100000:raise RuntimeError('CENSUS_BOUND')
   p=pending.pop();s=p.lstat();total+=1
   entry={'relative_path':str(p.relative_to(path)),**ident(s)}
   if s.st_dev!=66305 or s.st_uid!=1000:issues.append('OWNER_DEVICE:'+str(p))
   k=(s.st_dev,s.st_ino)
   if k not in seen:allocated+=s.st_blocks*512;logical+=s.st_size;seen.add(k)
   if stat.S_ISDIR(s.st_mode):
    with os.scandir(p) as it:pending.extend(Path(x.path) for x in it)
   elif stat.S_ISREG(s.st_mode):
    if s.st_nlink!=1:issues.append('MULTIPLE_HARD_LINKS:'+str(p))
   elif stat.S_ISLNK(s.st_mode):entry['link_text']=os.readlink(p)
   else:issues.append('SPECIAL_FILE:'+str(p))
   entries.append(entry)
  after=path.lstat();assert ident(before)==ident(after),'ROOT_CHANGED'
  row.update(status='COMPLETE',allocated_bytes=allocated,logical_bytes=logical,entries=entries,issues=issues,identity_after=ident(after))
 except FileNotFoundError:row['status']='ABSENT_OR_CHANGED'
 except Exception as exc:row.update(status='REFUSED_OR_PARTIAL',error=str(exc))
 records.append(row)
v=os.statvfs('/')
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn22','original_lock':{'device':ls.st_dev,'inode':ls.st_ino,'exclusive_acquired':locked},'ps_job_rows':jobs,'neuron_ls':nls,'fuser':fuser,'stop_paths':stops,'free_bytes':v.f_bavail*v.f_frsize,'targets':records,'total_entries':total,'no_host_writes':True,'file_contents_read':False,'elapsed_seconds':time.monotonic()-started}
fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
print(json.dumps(r))
'''
body=head+tail;ast.parse(body)
payload={'host':'trn22','hostname':'ip-172-31-32-253','inode':832589,'roots':['/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1'],'targets':[root+'/'+x for x in ('compiler','neff','local','hlo','inductor','debug')]}
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('strict_connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
out=HERE/'trn22-3e79-cache-inventory.json';assert not out.exists()
state=m.local_state();assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
p=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(p.stdout+p.stderr)
receipt={'returncode':p.returncode,'stderr':p.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(p.stdout) if p.returncode==0 else None,'stdout_if_error':p.stdout.decode() if p.returncode else None}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
r=receipt['remote']
print(json.dumps({'returncode':p.returncode,'utc':r['utc'] if r else None,'free_bytes':r['free_bytes'] if r else None,'targets':[{k:v for k,v in t.items() if k!='entries'} for t in r['targets']] if r else receipt,'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
raise SystemExit(p.returncode)
