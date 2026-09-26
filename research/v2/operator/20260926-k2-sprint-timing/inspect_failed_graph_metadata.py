"""One targeted read-only inventory of the failed a318 debug/HLO roots."""
import ast
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
path = REPO / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec = importlib.util.spec_from_file_location('connector', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bound = json.loads((HERE.parent / '20260926-final-sprint-successors/trn22-indexrepair-bound-packet.json').read_text())['binding']
assert bound['source_sha256'] == 'a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71'
roots = [bound['train_environment'][key] for key in ('TORCH_NEURONX_DEBUG_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR')]
body = r'''
import datetime,hashlib,json,os,signal,stat,time
signal.alarm(30)
started=time.monotonic()
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn22','source_sha256':SOURCE,'read_only':True,'sdk_imported':False,'roots':[],'captured_files':[],'limitations':['At most two directory levels,128 entries/root; no symlink following or broad cache traversal.','Existing files are historical failed-job artifacts, not proof of current production compilation or semantic graph role.']}
for root in ROOTS:
 entry={'root':root,'entries':[],'status':'OBSERVED'};result['roots'].append(entry)
 try:
  s=os.lstat(root)
  entry['root_stat']={'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'size':s.st_size,'mtime_ns':s.st_mtime_ns}
  if not stat.S_ISDIR(s.st_mode):entry['status']='NOT_REGULAR_DIRECTORY';continue
  pending=[(root,0)]; count=0
  while pending and count<128 and time.monotonic()-started<25:
   directory,depth=pending.pop(0)
   with os.scandir(directory) as it:
    for item in it:
     count+=1
     if count>128:entry['status']='ENTRY_LIMIT';break
     q=item.stat(follow_symlinks=False)
     row={'path':item.path,'device':q.st_dev,'inode':q.st_ino,'mode':q.st_mode,'bytes':q.st_size,'mtime_ns':q.st_mtime_ns}
     entry['entries'].append(row)
     if stat.S_ISDIR(q.st_mode) and depth<1:pending.append((item.path,depth+1))
     if stat.S_ISREG(q.st_mode) and item.name.endswith(('.hlo','.mlir')) and q.st_size<=4*1024*1024 and len(result['captured_files'])<2:
      fd=os.open(item.path,os.O_RDONLY|os.O_NOFOLLOW)
      try:
       a=os.fstat(fd);assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns),'RACED_BEFORE_READ'
       chunks=[];left=q.st_size+1
       while left:
        data=os.read(fd,min(left,65536))
        if not data:break
        chunks.append(data);left-=len(data)
       raw=b''.join(chunks);z=os.fstat(fd)
       assert len(raw)==q.st_size and (z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns),'RACED_READ'
       # Prefix text only; the full byte digest/size is recorded, not a copied giant graph.
       result['captured_files'].append({'path':item.path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text_prefix':raw[:65536].decode('utf-8',errors='replace'),'prefix_bytes':min(len(raw),65536),'full_file_retrieved':len(raw)<=65536})
      finally:os.close(fd)
 except Exception as exc:entry['status']='ERROR';entry['error']=repr(exc)
result['elapsed_seconds']=time.monotonic()-started
print(json.dumps(result))
'''
wire = ('ROOTS=' + repr(roots) + '\nSOURCE=' + repr(bound['source_sha256']) + '\n' + body).encode()
ast.parse(wire)
assert not m.SENSITIVE.search(wire)
out = HERE / 'failed-a318-debug-hlo-metadata.json'
assert not out.exists()
argv, *_ = m.load_transport('trn22')
argv[-1] = 'exec /usr/bin/python3 -I -B -S -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, capture_output=True, timeout=45)
assert not m.SENSITIVE.search(r.stdout + r.stderr)
receipt = {'started_utc':started,'returncode':r.returncode,'stderr':r.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with out.open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps({'returncode':r.returncode,'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'roots':[{'root':x['root'],'status':x['status'],'entries':len(x['entries'])} for x in receipt['remote']['roots']] if receipt['remote'] else None,'captured_files':[{k:v for k,v in x.items() if k!='text_prefix'} for x in receipt['remote']['captured_files']] if receipt['remote'] else None,'error':receipt['stderr']}))
