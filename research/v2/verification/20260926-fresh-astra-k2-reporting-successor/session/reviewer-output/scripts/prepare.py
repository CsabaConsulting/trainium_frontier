import hashlib, json, pathlib, re, sys
p = pathlib.Path('/packet')
w = pathlib.Path('/work')
manifest = json.loads((p/'INPUT_MANIFEST.json').read_text())
results = []
for item in manifest['included']:
    data = (p/item['path']).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    ok = len(data) == item['bytes'] and got == item['sha256']
    results.append(dict(path=item['path'], bytes=len(data), sha256=got, expected=item, exact=ok))
    if not ok:
        raise RuntimeError('packet hash mismatch: '+item['path'])
for source, diff in [('check_native.py','checker.diff'), ('consumer_nki.py','consumer_nki.py.diff'), ('reference.py','reference.py.diff')]:
    lines = (p/diff).read_text().splitlines(keepends=True)
    if len([x for x in lines if x.startswith('@@')]) != 1:
        raise RuntimeError('expected full single hunk')
    h = next(i for i,x in enumerate(lines) if x.startswith('@@'))
    counts = re.fullmatch(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', lines[h])
    old = ''.join(x[1:] for x in lines[h+1:] if x[0] in ' -')
    new = ''.join(x[1:] for x in lines[h+1:] if x[0] in ' +')
    if len(old.splitlines()) != int(counts[2]) or len(new.splitlines()) != int(counts[4]):
        raise RuntimeError('hunk counts mismatch')
    digest = hashlib.sha256(new.encode()).hexdigest()
    if digest != manifest['candidate_source_sha256'][source]:
        raise RuntimeError('candidate hash mismatch')
    (w/'source'/source).write_text(new)
    results.append(dict(path='source/'+source, sha256=digest, exact=True, lines=len(new.splitlines())))
    if source == 'check_native.py':
        digest = hashlib.sha256(old.encode()).hexdigest()
        if digest != manifest['baseline_checker_sha256']:
            raise RuntimeError('baseline hash mismatch')
        (w/'source'/'baseline_check_native.py').write_text(old)
        results.append(dict(path='source/baseline_check_native.py', sha256=digest, exact=True))
(w/'raw'/'provenance.json').write_text(json.dumps(dict(status='PASS', checks=results), indent=2)+'\n')
print(json.dumps(dict(status='PASS', authenticated=len(results), python=sys.version)))
