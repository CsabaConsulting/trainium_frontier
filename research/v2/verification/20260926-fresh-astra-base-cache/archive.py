#!/usr/bin/env python3
"""Archive a finished review verbatim, never start/retry or influence it."""
import hashlib
import json
import pathlib
import shutil

RUN = pathlib.Path('/tmp/trainium-fresh-base-cache-review-20260926.O9in3e')
PACKET = pathlib.Path('/tmp/trainium-fresh-base-cache-packet-20260926-cpuonly')
DEST = pathlib.Path(__file__).resolve().parent
EXPECTED = '56172b63b560d7a437a2fbf601b175be7b872e2e2e265adce7302daa03956fae'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not (RUN / 'terminal.json').is_file():
        raise RuntimeError('Session has not exited; do not archive partial success')
    if digest(PACKET / 'INPUT_MANIFEST.json') != EXPECTED:
        raise RuntimeError('Input manifest drift')
    for row in json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())['included']:
        path = PACKET / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RuntimeError('Input payload drift')
    for source, target in ((PACKET, DEST / 'packet'), (RUN / 'output', DEST / 'session/reviewer-output')):
        for file in source.rglob('*'):
            if file.is_symlink():
                raise RuntimeError('Refuse output symlink')
        shutil.copytree(source, target)
    for name in ('operate.py', 'preflight.json', 'prompt-input.json', 'prompt-input.stderr',
                 'launch.json', 'terminal.json', 'events.jsonl', 'stderr.log'):
        shutil.copyfile(RUN / name, DEST / 'session' / name)
    files = []
    for file in sorted(DEST.rglob('*')):
        if file.is_file() and file.name != 'ARCHIVE_MANIFEST.json':
            files.append({'path': str(file.relative_to(DEST)), 'bytes': file.stat().st_size, 'sha256': digest(file)})
    result = {'input_manifest_sha256': EXPECTED, 'verbatim_session_archive': True, 'files': files}
    (DEST / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'archived_files': len(files), 'bytes': sum(f['bytes'] for f in files),
                      'manifest_sha256': digest(DEST / 'ARCHIVE_MANIFEST.json')}))


if __name__ == '__main__':
    main()
