#!/usr/bin/env python3
"""Generate only frozen source diffs, neutral spec and raw receipts for a new review."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium-architecture-scored-clean-20260926')
COMMIT = 'bd9e33eee671ee489a1201485faf748bf0a35dea'
REFERENCE = 'b4d83b5db66971f3c348837f04f53cba65aa6d33'
PREFIX = 'workflow/candidates/architecture3e79-116188/'

def git(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    if (HERE / 'PREPARED.json').exists():
        raise RuntimeError('Existing prepared review; do not make duplicate session')
    packet = Path(tempfile.mkdtemp(prefix='trainium-fresh-architecture-clean-packet-20260926.'))
    run = Path(tempfile.mkdtemp(prefix='trainium-fresh-architecture-clean-review-20260926.'))
    (packet / 'raw').mkdir()
    (run / 'output').mkdir()
    candidate = git(COMMIT, PREFIX + 'train.py')
    vehicle = git(COMMIT, 'submissions/dense6x1024_direct_review_matrixlr015/train.py')
    reference = git(REFERENCE, 'workflow/proposals/research-v2/train.py')
    hashes = dict(candidate=digest(candidate), vehicle=digest(vehicle), scored_reference=digest(reference))
    if hashes != {'candidate': '3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f', 'vehicle': '116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d', 'scored_reference': '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'}:
        raise RuntimeError('Frozen source hash mismatch')
    diff = ''.join(difflib.unified_diff(vehicle.decode().splitlines(True), candidate.decode().splitlines(True), fromfile='vehicle/train.py', tofile='candidate/train.py', n=100000))
    (packet / 'vehicle-to-candidate.diff').write_text(diff)
    reference_diff = ''.join(difflib.unified_diff([], reference.decode().splitlines(True), fromfile='/dev/null', tofile='scored-reference/train.py', n=100000))
    (packet / 'scored-reference.diff').write_text(reference_diff)
    shutil.copyfile(HERE / 'lane-spec.md', packet / 'lane-spec.md')
    for name in ('tiny-g2-research.json', 'tiny-g2-clean.json', 'tiny-lockstep.json', 'tiny-checkpoint-checks.json'):
        (packet / 'raw' / name).write_bytes(git(COMMIT, PREFIX + name))
    included = []
    for path in sorted(packet.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            included.append(dict(path=str(path.relative_to(packet)), bytes=len(data), sha256=digest(data)))
    manifest = dict(candidate_commit=COMMIT, reference_commit=REFERENCE, source_sha256=hashes,
                    included=included, prior_history=False, shared_branch=False,
                    excluded=['implementer notes', 'IMPLEMENTATION.md', 'OPERATOR_RECIPE.md', 'test_port.py', 'check_tiny_checkpoints.py', 'source test conclusions', 'prior reviews', 'production repository', 'checkpoint files'])
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    (packet / 'INPUT_MANIFEST.json').write_bytes(encoded)
    pin = digest(encoded)
    template = (HERE.parent / '20260926-fresh-astra-k2-consumer/session/operate.py').read_text()
    template = template.replace('/tmp/trainium-fresh-k2-consumer-packet-20260926.00ofr9wh', str(packet))
    template = template.replace('5e0dde0832d3031b56ea4143a705808e1bd62326674799837b3444e1af440561', pin)
    (run / 'operate.py').write_text(template)
    result = dict(packet=str(packet), run=str(run), input_manifest_sha256=pin, candidate_commit=COMMIT,
                  fresh_session_started=False, operation_changes=['packet path', 'manifest pin'])
    (HERE / 'PREPARED.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
