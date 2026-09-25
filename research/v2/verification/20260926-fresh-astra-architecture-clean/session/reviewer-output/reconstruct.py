from pathlib import Path
import hashlib,json,re
p=Path('/packet'); w=Path('/work'); m=json.loads((p/'INPUT_MANIFEST.json').read_text())
results=[]
for e in m['included']:
 b=(p/e['path']).read_bytes(); actual=hashlib.sha256(b).hexdigest()
 results.append(dict(path=e['path'],bytes=len(b),sha256=actual,ok=len(b)==e['bytes'] and actual==e['sha256']))
 assert results[-1]['ok'],e
for filename,versions in [('vehicle-to-candidate.diff',('vehicle','candidate')),('scored-reference.diff',(None,'scored_reference'))]:
 lines=(p/filename).read_bytes().splitlines(keepends=True); old=[];new=[]; active=False
 for line in lines:
  if line.startswith(b'@@ '):
   assert not active,'Expected single full-context hunk'
   active=True; continue
  if not active: continue
  if line.startswith(b' '): old.append(line[1:]);new.append(line[1:])
  elif line.startswith(b'-'): old.append(line[1:])
  elif line.startswith(b'+'): new.append(line[1:])
  else: raise ValueError(repr(line))
 for name,parts in zip(versions,(old,new)):
  if name is None: continue
  b=b''.join(parts); digest=hashlib.sha256(b).hexdigest(); assert digest==m['source_sha256'][name],name
  (w/(name+'.py')).write_bytes(b)
  results.append(dict(path=name+'.py',bytes=len(b),lines=len(parts),sha256=digest,ok=True))
(w/'authentication.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
