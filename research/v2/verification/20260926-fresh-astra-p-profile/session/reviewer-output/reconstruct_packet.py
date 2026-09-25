"""Reconstruct the two supplied whole-file additions and verify packet hashes."""
import hashlib
import json
import re
from pathlib import Path

PACKET = Path('/packet')
WORK = Path('/work')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reconstruct(name, destination, expected):
    lines = (PACKET / name).read_bytes().splitlines(keepends=True)
    hunks = [i for i, line in enumerate(lines) if line.startswith(b'@@ ')]
    assert len(hunks) == 1, (name, hunks)
    header = re.fullmatch(rb'@@ -0,0 \+1,(\d+) @@\n', lines[hunks[0]])
    assert header, name
    added = lines[hunks[0] + 1:]
    assert len(added) == int(header[1]), name
    assert all(line.startswith(b'+') for line in added), name
    content = b''.join(line[1:] for line in added)
    actual = digest(content)
    assert actual == expected, (name, actual, expected)
    (WORK / destination).write_bytes(content)
    return dict(diff=name, reconstructed=destination, bytes=len(content),
                lines=len(added), sha256=actual, expected_sha256=expected, match=True)


def main():
    manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())
    files = []
    for entry in manifest['files']:
        data = (PACKET / entry['path']).read_bytes()
        actual = digest(data)
        assert actual == entry['sha256'] and len(data) == entry['bytes'], entry['path']
        files.append(dict(path=entry['path'], bytes=len(data), sha256=actual, match=True))
    results = dict(manifest_files=files, reconstructed=[
        reconstruct('wrapper.diff', 'profile_actual_loop.py',
                    'eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d'),
        reconstruct('unchanged-recursive-source-context.diff', 'recursive6.py',
                    '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'),
    ])
    (WORK / 'hash_verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
