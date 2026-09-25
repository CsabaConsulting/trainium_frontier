"""Reconstruct only the supplied full-context diff and verify input identities."""
import hashlib
import json
import re
from pathlib import Path

PACKET = Path('/packet')
WORK = Path('/work')
EXPECTED = {
    'original': '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5',
    'candidate': '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c',
}

def main():
    manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())
    checked = []
    for item in manifest['files']:
        data = (PACKET / item['path']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == item['sha256'], item['path']
        assert len(data) == item['bytes'], item['path']
        checked.append(dict(path=item['path'], bytes=len(data), sha256=digest))
    manifest_data = (PACKET / 'INPUT_MANIFEST.json').read_bytes()
    checked.append(dict(path='INPUT_MANIFEST.json', bytes=len(manifest_data),
                        sha256=hashlib.sha256(manifest_data).hexdigest()))
    assert set(p.relative_to(PACKET).as_posix() for p in PACKET.rglob('*') if p.is_file()) == set(i['path'] for i in checked)
    lines = (PACKET / 'candidate.diff').read_bytes().splitlines(keepends=True)
    assert lines[0] == b'diff --git a/workflow/proposals/research-v2/train.py b/workflow/proposals/research-v2/train.py\n'
    assert re.fullmatch(rb'@@ -1,4944 \+1,4948 @@\n', lines[4])
    sources = {'original': [], 'candidate': []}
    additions = []
    removals = []
    for line in lines[5:]:
        assert line[:1] in (b' ', b'+', b'-'), repr(line)
        if line[:1] in (b' ', b'-'):
            sources['original'].append(line[1:])
        if line[:1] in (b' ', b'+'):
            sources['candidate'].append(line[1:])
        if line[:1] == b'+':
            additions.append({'candidate_line': len(sources['candidate']), 'text': line[1:].decode().rstrip('\n')})
        if line[:1] == b'-':
            removals.append(line[1:].decode().rstrip('\n'))
    identities = {}
    for name, parts in sources.items():
        data = b''.join(parts)
        digest = hashlib.sha256(data).hexdigest()
        assert digest == EXPECTED[name], (name, digest)
        (WORK / (name + '_train.py')).write_bytes(data)
        identities[name] = dict(sha256=digest, bytes=len(data), lines=len(parts))
    result = dict(status='PASS', input_files=checked, sources=identities, additions=additions, removals=removals)
    (WORK / 'input_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
