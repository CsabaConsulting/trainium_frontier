#!/usr/bin/env python3
"""Reconstruct both files from this packet's single full-context unified diff."""
import hashlib
import json
from pathlib import Path
import re

PACKET = Path('/packet')
WORK = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())
    checks = []
    for entry in manifest['included']:
        data = (PACKET / entry['path']).read_bytes()
        check = dict(path=entry['path'], bytes=len(data), sha256=sha256(data))
        check['matches_manifest'] = (len(data) == entry['bytes'] and
                                     check['sha256'] == entry['sha256'])
        assert check['matches_manifest'], check
        checks.append(check)
    lines = (PACKET / 'kernel-full-context.diff').read_bytes().splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines) if line.startswith(b'@@ ')]
    assert len(headers) == 1
    h = headers[0]
    match = re.fullmatch(rb'@@ -1,(\d+) \+1,(\d+) @@\n', lines[h])
    assert match
    old, new = [], []
    for line in lines[h + 1:]:
        assert line[:1] in (b' ', b'-', b'+'), repr(line)
        if line[:1] in (b' ', b'-'):
            old.append(line[1:])
        if line[:1] in (b' ', b'+'):
            new.append(line[1:])
    assert [len(old), len(new)] == [int(x) for x in match.groups()]
    sources = []
    for name, lineset in [('baseline', old), ('candidate', new)]:
        data = b''.join(lineset)
        digest = sha256(data)
        assert digest == manifest['source_sha256'][name], (name, digest)
        destination = WORK / (name + '_consumer_nki.py')
        destination.write_bytes(data)
        sources.append(dict(source=name, artifact=destination.name,
                            lines=len(lineset), bytes=len(data), sha256=digest,
                            matches_manifest=True))
    result = dict(status='PASS', packet_files=checks, reconstructed=sources,
                  manifest_sha256=sha256((PACKET / 'INPUT_MANIFEST.json').read_bytes()),
                  provenance_limit='Hashes establish packet consistency, not external authenticity.')
    (WORK / 'reconstruction.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
