"""Reconstruct complete sources using only the full-context packet diff."""
import hashlib
import json
import re
from pathlib import Path

PACKET = Path('/packet')
OUT = Path('/work')
manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())
checks = []
for item in manifest['included']:
    data = (PACKET / item['path']).read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    assert len(data) == item['bytes'], item['path']
    assert observed == item['sha256'], item['path']
    checks.append(dict(path=item['path'], bytes=len(data), sha256=observed))

lines = (PACKET / 'candidate.diff').read_bytes().splitlines(keepends=True)
hunks = [i for i, line in enumerate(lines) if line.startswith(b'@@ ')]
assert len(hunks) == 1
start = hunks[0]
match = re.fullmatch(rb'@@ -1,(\d+) \+1,(\d+) @@\n', lines[start])
assert match is not None
old, new, changes = [], [], []
for line in lines[start + 1:]:
    assert line[:1] in (b' ', b'+', b'-'), repr(line)
    if line[:1] != b'+':
        old.append(line[1:])
    if line[:1] != b'-':
        new.append(line[1:])
    if line[:1] in (b'+', b'-'):
        changes.append(dict(base_line=len(old), candidate_line=len(new), text=line.decode().rstrip('\n')))
assert (len(old), len(new)) == tuple(map(int, match.groups()))
expected = {
    'base': '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3',
    'candidate': 'bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522',
}
sources = []
for name, content in [('base', old), ('candidate', new)]:
    data = b''.join(content)
    observed = hashlib.sha256(data).hexdigest()
    assert observed == expected[name], name
    (OUT / f'{name}.py').write_bytes(data)
    sources.append(dict(path=f'{name}.py', lines=len(content), bytes=len(data), sha256=observed))
result = dict(status='PASS', packet_files=checks, sources=sources, changes=changes)
(OUT / 'authentication.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
