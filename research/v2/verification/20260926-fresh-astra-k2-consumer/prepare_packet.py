#!/usr/bin/env python3
"""Mechanical frozen input-only packet; no review conclusions are passed."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium-k2-resident-consumer-20260926')
COMMIT = '077481d0476cebfd79ddc751f8f63e61c1fbf82f'
PREFIX = 'research/v2/lanes/k2-resident-consumer/'
PACKET = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-consumer-packet-20260926.'))
RUN = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-consumer-review-20260926.'))


def main():
    (PACKET / 'raw').mkdir()
    (RUN / 'output').mkdir()
    diff = subprocess.check_output(['git', 'diff', COMMIT + '^', COMMIT, '--full-index', '--unified=100000', '--',
                                   PREFIX + 'consumer_nki.py', PREFIX + 'reference.py', PREFIX + 'check_native.py'], cwd=REPO)
    (PACKET / 'candidate.diff').write_bytes(diff)
    shutil.copyfile(HERE / 'lane-spec.md', PACKET / 'lane-spec.md')
    for name in ('checks-normal.json', 'checks-optimized.json'):
        (PACKET / 'raw' / name).write_bytes(subprocess.check_output(['git', 'show', COMMIT + ':' + PREFIX + name], cwd=REPO))
    included = []
    for file in sorted(PACKET.rglob('*')):
        if file.is_file():
            encoded = file.read_bytes()
            included.append(dict(path=str(file.relative_to(PACKET)), bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest()))
    manifest = dict(candidate_commit=COMMIT, included=included, prior_history=False,
                    shared_branch=False, provider_session_started=False,
                    excluded=['implementer notes', 'README', 'CHECKS.md', 'test_reference.py', 'prior reviews', 'production repository'])
    (PACKET / 'INPUT_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    digest = hashlib.sha256((PACKET / 'INPUT_MANIFEST.json').read_bytes()).hexdigest()
    template = (HERE.parent / '20260926-fresh-astra-base-cache/session/operate.py').read_text()
    template = template.replace('/tmp/trainium-fresh-base-cache-packet-20260926-cpuonly', str(PACKET))
    template = template.replace('56172b63b560d7a437a2fbf601b175be7b872e2e2e265adce7302daa03956fae', digest)
    (RUN / 'operate.py').write_text(template)
    receipt = dict(packet=str(PACKET), run=str(RUN), input_manifest_sha256=digest,
                   operation_template='unchanged reviewed base-cache operate.py except packet path and manifest pin',
                   fresh_session_started=False, candidate_commit=COMMIT)
    (HERE / 'PREPARED.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
