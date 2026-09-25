import ast, difflib, hashlib, json, pathlib, shutil
P=pathlib.Path('/packet'); W=pathlib.Path('/work')
manifest=json.loads((P/'INPUT_MANIFEST.json').read_text())
checks=[]
for ent in manifest['included']:
    raw=(P/ent['path']).read_bytes()
    checks.append(dict(path=ent['path'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),ok=len(raw)==ent['bytes'] and hashlib.sha256(raw).hexdigest()==ent['sha256']))
    dst=W/'packet_copy'/ent['path']; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(raw)
shutil.copyfile(P/'INPUT_MANIFEST.json', W/'packet_copy/INPUT_MANIFEST.json')
checks.append(dict(path='INPUT_MANIFEST.json',sha256=hashlib.sha256((P/'INPUT_MANIFEST.json').read_bytes()).hexdigest(),note='self-hash independently recorded; not in manifest'))
lines=(P/'baseline-to-candidate.diff').read_text().splitlines(keepends=True)
assert lines[2]=='@@ -1,4948 +1,4962 @@\n'
for role, markers, count in [('baseline',' -',4948),('candidate',' +',4962)]:
    body=''.join(line[1:] for line in lines[3:] if line[0] in markers)
    assert len(body.splitlines())==count
    sha=hashlib.sha256(body.encode()).hexdigest()
    assert sha==manifest['source_sha256'][role]
    (W/f'{role}_train.py').write_text(body)
    tree=ast.parse(body)
    index=[dict(kind=type(n).__name__,name=getattr(n,'name',None),start=n.lineno,end=n.end_lineno) for n in tree.body]
    (W/f'{role}_source_index.json').write_text(json.dumps(index,indent=2)+'\n')
    checks.append(dict(path=f'reconstructed/{role}_train.py',bytes=len(body.encode()),lines=count,sha256=sha,ok=True))
assert all(c.get('ok',True) for c in checks)
(W/'input_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
a=(W/'baseline_train.py').read_text().splitlines(True); b=(W/'candidate_train.py').read_text().splitlines(True)
(W/'focused.diff').write_text(''.join(difflib.unified_diff(a,b,fromfile='baseline_train.py',tofile='candidate_train.py',n=15)))
print(json.dumps(checks[-3:],indent=2)); print((W/'focused.diff').read_text())
