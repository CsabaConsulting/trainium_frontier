"""Reconstruct only the supplied full-context diffs; authenticate all inputs."""
import ast
import difflib
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path('/packet')
OUT = pathlib.Path('/work')
manifest = json.loads((ROOT / 'INPUT_MANIFEST.json').read_text())
records = []
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        data = p.read_bytes()
        record = dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        claimed = next((x for x in manifest['included'] if x['path'] == str(p.relative_to(ROOT))), None)
        if claimed:
            record['manifest_match'] = record['bytes'] == claimed['bytes'] and record['sha256'] == claimed['sha256']
            assert record['manifest_match'], record
        records.append(record)

def unpack(path):
    lines = path.read_text().splitlines(keepends=True)
    assert len([x for x in lines if x.startswith('@@')]) == 1
    hunk = next(i for i, line in enumerate(lines) if line.startswith('@@'))
    counts = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', lines[hunk])
    old, new = [], []
    for line in lines[hunk + 1:]:
        assert line[0] in ' +-', repr(line)
        if line[0] in ' -': old.append(line[1:])
        if line[0] in ' +': new.append(line[1:])
    assert len(old) == int(counts[2]) and len(new) == int(counts[4])
    return ''.join(old), ''.join(new)

baseline, candidate = unpack(ROOT / 'baseline-to-candidate.diff')
empty, reference = unpack(ROOT / 'schedule-reference.diff')
assert empty == ''
sources = {'baseline': baseline, 'candidate': candidate, 'schedule_reference': reference}
for name, source in sources.items():
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert digest == manifest['source_sha256'][name], (name, digest)
    ast.parse(source)
    (OUT / 'source' / (name + '.py')).write_text(source)
    records.append(dict(source=name, sha256=digest, lines=len(source.splitlines()), manifest_match=True))
(OUT / 'artifacts' / 'input_hashes.json').write_text(json.dumps(records, indent=2) + '\n')
(OUT / 'artifacts' / 'candidate.diff').write_text(''.join(difflib.unified_diff(baseline.splitlines(True), candidate.splitlines(True), fromfile='baseline.py', tofile='candidate.py', n=6)))
(OUT / 'artifacts' / 'reference_vs_candidate.diff').write_text(''.join(difflib.unified_diff(reference.splitlines(True), candidate.splitlines(True), fromfile='schedule_reference.py', tofile='candidate.py', n=4)))
print(json.dumps(records, indent=2))
