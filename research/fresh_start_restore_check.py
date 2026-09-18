#!/usr/bin/env python3
"""Download a GitHub artifact and verify a complete restore without extraction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from fresh_start_artifact_backup import digest
from fresh_start_backup import ROOT, atomic_json, desktop_keyring_environment, now


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',required=True)
    args=ap.parse_args()
    assert all(c.isalnum() or c in '-_.' for c in args.id)
    receipt=json.loads((ROOT/'research/run-logs'/f'fs-artifact-backup-{args.id}.json').read_text())
    assert receipt['all_github_digests_match']
    desktop_keyring_environment()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory=ROOT/'out/fresh-start-artifact-backup/restore-checks'/args.id/stamp
    directory.mkdir(parents=True,exist_ok=False)
    for asset in receipt['assets']:
        subprocess.run(['gh','release','download',receipt['tag'],'--repo',receipt['repository'],
                        '--pattern',asset['name'],'--dir',str(directory)],check=True,timeout=900)
        path=directory/asset['name']
        assert path.stat().st_size==asset['bytes'] and digest(path)==asset['sha256']
    manifests=list(directory.glob('*--manifest--*.json'));assert len(manifests)==1
    manifest=json.loads(manifests[0].read_text())
    chunks=[directory/c['name'] for c in manifest['chunks']]
    if len(chunks)==1:
        payload=chunks[0]
    else:
        payload=directory/'joined-payload'
        with payload.open('xb') as output:
            for part in chunks:
                with part.open('rb') as source:shutil.copyfileobj(source,output,length=8*1024**2)
    assert payload.stat().st_size==manifest['payload_bytes']
    assert digest(payload)==manifest['payload_sha256']
    expected={row['path']:row for row in manifest['source_files']}
    verified=set()
    if manifest['payload_encoding']=='tar.gz':
        with tarfile.open(payload,'r:gz') as archive:
            for member in archive:
                path=PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert path.parts[0]==manifest['entry']['id']
                if member.isdir():continue
                relative=str(path.relative_to(manifest['entry']['id']))
                row=expected[relative]
                if member.issym():
                    assert member.linkname==row['symlink']
                else:
                    assert member.isfile() or member.islnk()
                    with archive.extractfile(member) as stream:
                        actual=hashlib.file_digest(stream,'sha256').hexdigest()
                    assert actual==row['sha256'],relative
                verified.add(relative)
    else:
        assert len(expected)==1
        key,row=next(iter(expected.items()))
        assert digest(payload)==row['sha256'] and payload.stat().st_size==row['bytes']
        verified.add(key)
    assert verified==set(expected)
    result=dict(checked_at=now(),artifact=args.id,passed=True,
                downloaded_assets=len(receipt['assets']),source_files_verified=len(verified),
                payload_sha256=manifest['payload_sha256'],payload_bytes=manifest['payload_bytes'],
                files_extracted_to_filesystem=False,local_download=str(directory.relative_to(ROOT)),
                github_receipt=f'research/run-logs/fs-artifact-backup-{args.id}.json')
    output=ROOT/'research/run-logs'/f'fs-artifact-restore-{args.id}-{stamp}.json'
    atomic_json(output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
