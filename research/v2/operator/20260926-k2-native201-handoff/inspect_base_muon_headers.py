"""One small read-only capture in the completed clean-cold base's private caches."""
import ast,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
root='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet'
body=r'''
import datetime,hashlib,json,os,signal,stat
signal.alarm(30)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'root':P['root'],'source_sha256':P['source_sha256'],'inventories':{},'headers':{},'errors':[]}
budget=524288
def meta(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def entries(p,limit):
 try:
  s=os.lstat(p)
  if not stat.S_ISDIR(s.st_mode):return {'metadata':meta(s),'not_regular_directory':True},[]
  names=[]
  with os.scandir(p) as it:
   for e in it:
    if len(names)>=limit:return {'metadata':meta(s),'limit_exceeded':True},[]
    names.append(e.name)
  names.sort();rows=[]
  for n in names:
   try:rows.append({'name':n,**meta(os.lstat(p+'/'+n))})
   except OSError as e:rows.append({'name':n,'error':repr(e)})
  return {'metadata':meta(s),'entries':rows},rows
 except OSError as e:return {'error':repr(e)},[]
def sample(p,prefix,tail=0):
 global budget
 if budget<prefix+tail:return {'read_budget_exhausted':True}
 try:
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   s=os.fstat(fd)
   if not stat.S_ISREG(s.st_mode):return {'metadata':meta(s),'not_regular':True}
   a=os.read(fd,min(prefix,s.st_size));b=b''
   if tail and s.st_size>len(a):os.lseek(fd,max(len(a),s.st_size-tail),os.SEEK_SET);b=os.read(fd,tail)
   z=os.fstat(fd);budget-=len(a)+len(b)
   return {'before':meta(s),'after':meta(z),'full_file':len(a)==s.st_size,'prefix_sha256':hashlib.sha256(a).hexdigest(),'prefix_text':a.decode(errors='replace'),'tail_sha256':hashlib.sha256(b).hexdigest(),'tail_text':b.decode(errors='replace')}
  finally:os.close(fd)
 except OSError as e:return {'error':repr(e)}
for rel in ('train/caches/hlo','train/caches/neff','train/caches/debug/neuronx-cc'):
 p=P['root']+'/'+rel;report,rows=entries(p,128);out['inventories'][rel]=report
 if rel.endswith('neuronx-cc'):
  ds=[r for r in rows if 'mode' in r and stat.S_ISDIR(r['mode'])]
  if len(ds)>32:out['errors'].append('DEBUG_DIRECTORY_LIMIT_32_EXCEEDED');continue
  for d in ds:
   sub=rel+'/'+d['name'];rep,fs=entries(P['root']+'/'+sub,32);out['inventories'][sub]=rep
   present={r['name'] for r in fs if 'mode' in r and stat.S_ISREG(r['mode'])}
   for name in ('module.mlir','log-neuron-cc.txt'):
    if name in present:out['headers'][sub+'/'+name]=sample(P['root']+'/'+sub+'/'+name,4096,4096 if name.endswith('.txt') else 0)
 else:
  for r in rows:
   if 'mode' in r and stat.S_ISREG(r['mode']) and any(k in r['name'] for k in P['keys']):
    out['headers'][rel+'/'+r['name']]=sample(p+'/'+r['name'],8192)
out['captured_bytes']=524288-budget
print(json.dumps(out))
'''
payload={'root':root,'source_sha256':'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43',
 'keys':['561e61ab','ea28caef','50cab471','3d178ec2','aee6ba1f','6a741cc4','acae4190','c778b3a6']}
wire=('P='+repr(payload)+'\n'+body).encode();ast.parse(wire)
cp=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(cp.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',cp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
out=HERE/'base-cold-muon-cache-headers.json';assert not out.exists()
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=45);assert not m.SENSITIVE.search(r.stdout+r.stderr)
d={'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None,'payload_sha256':hashlib.sha256(wire).hexdigest()}
with out.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
o=d['remote'] or {};print(json.dumps({'returncode':r.returncode,'utc':o.get('utc'),'captured_bytes':o.get('captured_bytes'),'inventories':len(o.get('inventories',{})),'headers':len(o.get('headers',{})),'errors':o.get('errors'),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
