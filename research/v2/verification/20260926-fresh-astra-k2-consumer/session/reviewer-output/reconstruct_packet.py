#!/usr/bin/env python3
"""Authenticate supplied bytes and reconstruct only complete new-file hunks."""
import hashlib
import json
from pathlib import Path
import re

packet = Path('/packet')
work = Path('/work')
manifest = json.loads((packet / 'INPUT_MANIFEST.json').read_text())
audit = {'candidate_commit': manifest['candidate_commit'], 'inputs': [], 'reconstructed': []}
for entry in manifest['included']:
    data = (packet / entry['path']).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    record = dict(entry, actual_bytes=len(data), actual_sha256=got,
                  verified=len(data) == entry['bytes'] and got == entry['sha256'])
    audit['inputs'].append(record)
    if not record['verified']:
        raise RuntimeError(record)
text = (packet / 'candidate.diff').read_text()
chunks = re.split(r'(?=^diff --git )', text, flags=re.M)
for chunk in filter(None, chunks):
    lines = chunk.splitlines(keepends=True)
    m = re.fullmatch(r'diff --git a/(\S+) b/(\S+)\n', lines[0])
    if not m or m[1] != m[2] or '..' in Path(m[1]).parts:
        raise RuntimeError('unsafe or nonmatching path')
    if lines[1] != 'new file mode 100644\n' or lines[3] != '--- /dev/null\n':
        raise RuntimeError('not a complete new file')
    index = re.fullmatch(r'index 0{40}\.\.([a-f0-9]{40})\n', lines[2])
    hunk = re.fullmatch(r'@@ -0,0 \+1,(\d+) @@\n', lines[5])
    body = lines[6:]
    if not index or not hunk or len(body) != int(hunk[1]) or any(not x.startswith('+') for x in body):
        raise RuntimeError('unsupported or incomplete hunk')
    data = ''.join(x[1:] for x in body).encode()
    git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if git_blob != index[1]:
        raise RuntimeError('diff blob mismatch')
    destination = work / 'reconstructed' / m[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as f:
        f.write(data)
    audit['reconstructed'].append({'path': str(destination.relative_to(work)), 'bytes': len(data),
                                  'lines': len(body), 'sha256': hashlib.sha256(data).hexdigest(),
                                  'git_blob_sha1': git_blob, 'matches_diff_blob': True})
audit['scope'] = 'Internal packet integrity; no external authentication of commit or prior execution'
(work / 'packet_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
