#!/usr/bin/env python3
"""Back up preserved weights/profiles as immutable, digest-verified draft assets.

Uses the user's existing GitHub authentication. No assets are replaced or deleted;
completed uploads are reusable after interruption. Actual artifact bytes, source
inventory and chunk hashes are retained separately from ordinary Git history.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

from fresh_start_backup import ROOT, atomic_json, desktop_keyring_environment, now


CHUNK_BYTES = 1536 * 1024**2  # Every asset is strictly below GitHub's 2 GiB limit.
CONFIG = ROOT / 'research/artifact-backups.json'
DIRECTORY = ROOT / 'out/fresh-start-artifact-backup'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_path(relative):
    source = (ROOT / relative).resolve()
    allowed = [ROOT / 'checkpoints', ROOT / 'out/fresh-start-profiles',
               ROOT.parent / 'trainium_frontier_backups/2026-09-18']
    if not any(source.is_relative_to(base.resolve()) for base in allowed):
        raise ValueError('Artifact outside the authorized source directories: ' + str(source))
    return source


def discover(config):
    entries = list(config['priority_entries'])
    seen = {str(source_path(e['source'])) for e in entries}
    candidates = []
    for p in sorted((ROOT / 'checkpoints').glob('*/final.pt')):
        candidates.append(dict(id=p.parent.name, source=str(p.relative_to(ROOT)), kind='checkpoint'))
    for receipt in sorted((ROOT / 'research/run-logs').glob('fs-*.preservation.json')):
        try:
            row = json.loads(receipt.read_text())
        except (ValueError, OSError):
            continue
        for key in ('local', 'local_path', 'local_root', 'local_base', 'local_directory'):
            value = row.get(key)
            if not isinstance(value, str):
                continue
            p = Path(value)
            p = p if p.is_absolute() else ROOT / p
            if p.is_dir() and p.resolve().is_relative_to((ROOT / 'out/fresh-start-profiles').resolve()):
                candidates.append(dict(id=p.name,source=str(p.relative_to(ROOT)),kind='raw_profile',
                                       preservation_receipt=str(receipt.relative_to(ROOT))))
    for entry in candidates:
        source = str(source_path(entry['source']))
        if source not in seen:
            entries.append(entry)
            seen.add(source)
    ids = [e['id'] for e in entries]
    assert len(set(ids)) == len(ids), 'Artifact IDs must be unique'
    for entry in entries:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', entry['id'])
    return entries


def source_stats(source):
    paths = sorted(source.rglob('*')) if source.is_dir() else [source]
    result = []
    for p in paths:
        relative = str(p.relative_to(source)) if source.is_dir() else source.name
        if p.is_symlink():
            result.append(dict(path=relative,symlink=os.readlink(p)))
        elif p.is_file():
            s = p.stat()
            result.append(dict(path=relative,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    return result


def split_asset(source, directory, name, chunk_bytes=CHUNK_BYTES):
    """Stream large payloads into numbered chunks; no whole-file memory allocation."""
    chunks = []
    if source.stat().st_size <= chunk_bytes:
        sha = digest(source)
        suffix = '.tar.gz' if source.name.endswith('.tar.gz') else source.suffix
        target = directory / f'{name}--{sha[:16]}{suffix}'
        if not target.exists():
            os.link(source, target)
        chunks.append(dict(name=target.name,bytes=target.stat().st_size,sha256=sha))
    else:
        with source.open('rb') as input_file:
            number = 0
            while True:
                data = input_file.read(min(chunk_bytes, 8 * 1024**2))
                if not data:
                    break
                part = directory / f'chunk-{number:04d}.partial'
                sha = hashlib.sha256()
                total = 0
                with part.open('wb') as output:
                    while data:
                        output.write(data)
                        sha.update(data)
                        total += len(data)
                        if total == chunk_bytes:
                            break
                        data = input_file.read(min(chunk_bytes-total, 8 * 1024**2))
                target = directory / f'{name}--part{number:04d}--{sha.hexdigest()[:16]}.bin'
                if target.exists():
                    assert digest(target) == sha.hexdigest()
                    part.unlink()  # Only this invocation's verified duplicate temporary chunk.
                else:
                    part.rename(target)
                chunks.append(dict(name=target.name,bytes=total,sha256=sha.hexdigest()))
                number += 1
    return chunks


def prepare(entry, directory=DIRECTORY):
    source = source_path(entry['source'])
    assert source.exists(), f'Missing artifact: {source}'
    before = source_stats(source)
    signature = hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
    folder = directory / entry['id'] / signature[:16]
    folder.mkdir(parents=True,exist_ok=True)
    path = folder / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if all((folder / c['name']).is_file() and (folder / c['name']).stat().st_size == c['bytes']
               for c in manifest['chunks']):
            return folder, manifest
    files = []
    for row in before:
        item = dict(row)
        if 'bytes' in item:
            p = source / row['path'] if source.is_dir() else source
            item['sha256'] = digest(p)
        files.append(item)
    if source.is_dir():
        payload = folder / 'payload.tar.gz'
        temporary = folder / 'payload.tar.gz.partial'
        with tarfile.open(temporary,'w:gz',compresslevel=1,dereference=False) as archive:
            archive.add(source,arcname=entry['id'])
        temporary.rename(payload)
        encoding = 'tar.gz'
    else:
        payload = source
        encoding = 'original_file'
    chunks = split_asset(payload,folder,entry['id'])
    assert source_stats(source) == before, 'Source changed while preparing the backup'
    manifest = dict(schema_version=1,prepared_at=now(),entry=entry,source_stat_signature=signature,
                    source_files=files,payload_encoding=encoding,payload_bytes=payload.stat().st_size,
                    payload_sha256=digest(payload),chunks=chunks,
                    restore='Concatenate chunks in listed order, verify payload SHA256, then extract tar.gz or restore the original file. Verify every source file against this inventory.')
    atomic_json(path,manifest)
    return folder,manifest


def gh(*args, timeout=120):
    result = subprocess.run(['gh',*args],cwd=ROOT,capture_output=True,text=True,timeout=timeout,
                            env=dict(os.environ,GH_PROMPT_DISABLED='1',GIT_TERMINAL_PROMPT='0'))
    if result.returncode:
        raise RuntimeError(f'gh {args[0]}: exit{result.returncode}: {result.stderr[-3000:]}')
    return result.stdout


def release(config):
    endpoint = 'repos/' + config['repository'] + '/releases'
    rows = json.loads(gh('api',endpoint+'?per_page=100'))
    matches = [r for r in rows if r['tag_name'] == config['tag']]
    if not matches:
        gh('release','create',config['tag'],'--repo',config['repository'],'--draft',
           '--target',config['target_branch'],'--title','Trainium research recovery — 2026-09-18',
           '--notes-file',str(ROOT/'research/artifact-backup-release-notes.md'))
        # Creation can precede visibility in the release-list API by a moment.
        for attempt in range(5):
            rows = json.loads(gh('api',endpoint+'?per_page=100'))
            matches = [r for r in rows if r['tag_name'] == config['tag']]
            if matches:
                break
            time.sleep(2)
    assert len(matches) == 1 and matches[0]['draft'], 'Expected one draft backup release'
    return matches[0]


def assets(config, release_id):
    # The installed gh predates --slurp; explicit REST pagination works on it.
    result = {}
    for page in range(1, 12):
        items = json.loads(gh('api',f"repos/{config['repository']}/releases/{release_id}/assets?per_page=100&page={page}"))
        result.update({item['name']:item for item in items})
        if len(items) < 100:
            return result
    raise RuntimeError('Release asset pagination exceeded the documented limit')


def upload(config, rel, folder, manifest):
    inventory = assets(config,rel['id'])
    manifest_file = folder / 'manifest.json'
    sha = digest(manifest_file)
    named_manifest = folder / f"{manifest['entry']['id']}--manifest--{sha[:16]}.json"
    if not named_manifest.exists():
        os.link(manifest_file,named_manifest)
    required = [*manifest['chunks'],dict(name=named_manifest.name,bytes=named_manifest.stat().st_size,sha256=sha)]
    verified = []
    for item in required:
        name = item['name']
        if name not in inventory:
            assert digest(folder/name) == item['sha256'], 'Local upload payload changed: '+name
            print(json.dumps(dict(event='uploading',artifact=manifest['entry']['id'],asset=name,bytes=item['bytes'])),flush=True)
            gh('release','upload',config['tag'],str(folder/name),'--repo',config['repository'],timeout=1800)
            inventory = assets(config,rel['id'])
        actual = inventory[name]
        assert actual['state'] == 'uploaded' and actual['size'] == item['bytes'], 'Incomplete upload: '+name
        assert actual.get('digest') == 'sha256:'+item['sha256'], 'GitHub digest mismatch/missing: '+name
        verified.append(dict(**item,github_asset_id=actual['id'],github_digest=actual['digest'],
                             download_url=actual['browser_download_url']))
    result = dict(verified_at=now(),repository=config['repository'],release_id=rel['id'],
                  tag=config['tag'],draft=True,entry=manifest['entry'],
                  source_stat_signature=manifest['source_stat_signature'],
                  payload_sha256=manifest['payload_sha256'],payload_bytes=manifest['payload_bytes'],
                  all_github_digests_match=True,assets=verified)
    atomic_json(ROOT/'research/run-logs'/f"fs-artifact-backup-{manifest['entry']['id']}.json",result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--max-items',type=int,default=0)
    args=ap.parse_args()
    desktop_keyring_environment()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (DIRECTORY/'backup.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Artifact backup already running; skipped',flush=True)
            return
        config=json.loads(CONFIG.read_text())
        entries=discover(config)
        if args.max_items: entries=entries[:args.max_items]
        state=dict(started_at=now(),planned=len(entries),completed=[],errors=[])
        rel=None
        for entry in entries:
            state['current']=entry['id'];atomic_json(DIRECTORY/'latest.json',state)
            try:
                print(json.dumps(dict(event='preparing',artifact=entry['id'],source=entry['source'])),flush=True)
                folder,manifest=prepare(entry)
                if not args.prepare_only:
                    if rel is None: rel=release(config)
                    result=upload(config,rel,folder,manifest)
                    state['completed'].append(dict(id=entry['id'],verified=True,bytes=result['payload_bytes']))
                else: state['completed'].append(dict(id=entry['id'],prepared=True,verified=False))
            except Exception as error:
                state['errors'].append(dict(id=entry['id'],error=str(error)))
                print(json.dumps(dict(event='failed',artifact=entry['id'],error=str(error))),flush=True)
            atomic_json(DIRECTORY/'latest.json',state)
        state.update(completed_at=now(),current=None)
        atomic_json(DIRECTORY/'latest.json',state)
        print(json.dumps(state),flush=True)
        if state['errors']:raise SystemExit(1)


if __name__=='__main__':
    main()
