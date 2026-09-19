#!/usr/bin/env python3
"""Preserve old NKI compiler diagnostics before retiring their remote copies.

The capture and retirement each hold the experiment's host lock. Only ordinary
files in idle, same-UID /tmp/nki_* directories are eligible. Shared NEFF caches,
source, checkpoints, profiles, SDK installations and the latest compile remain.
Every original file is verified inside the local archive; retirement additionally
requires current GitHub asset digests and a second remote content verification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh_start_sync import HOSTS, ROOT, SSH
from fresh_start_backup import atomic_json, desktop_keyring_environment


REMOTE = r'''
import fcntl,hashlib,io,json,os,pathlib,sys,tarfile
m=json.load(sys.stdin);base=pathlib.Path(m['remote'])
sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,utc_now,write_json
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp(p):
 s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,uid=s.st_uid)
def allowed(p):
 return p.parent==pathlib.Path('/tmp') and p.name.startswith('nki_') and p.is_dir() and not p.is_symlink() and p.stat().st_uid==os.getuid()
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text())
 assert state['label']==m['expected_previous_label'] and state['remote']==str(base)
 post=json.loads(pathlib.Path(state['postflight']).read_text())
 assert post['command_returncode']==0
 assert all(f=='less than8GiB disk space available' for f in post['failures'])
 before=snapshot(base,post['completed_at'])
 assert all(f=='less than8GiB disk space available' for f in health_failures(before,(os.getpid(),os.getppid())))
 logs=base/'research/run-logs'
 if m['action']=='capture':
  roots=[];files=[];total=0
  for root in sorted(pathlib.Path('/tmp').glob('nki_*'),key=lambda p:p.lstat().st_mtime_ns):
   if not allowed(root):continue
   paths=sorted(root.rglob('*'))
   if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in paths):continue
   regular=[p for p in paths if p.is_file()]
   if not regular or any(p.stat().st_mtime_ns>=m['cutoff_ns'] or p.stat().st_uid!=os.getuid() for p in regular):continue
   rows=[]
   for p in regular:
    old=stamp(p);sha=digest(p);assert old==stamp(p)
    rows.append(dict(path=str(p.relative_to('/tmp')),sha256=sha,**old))
   roots.append(str(root));files.extend(rows);total+=sum(r['bytes'] for r in rows)
   if total>=m['target_bytes']:break
  assert m['target_bytes']<=total<=m['target_bytes']+1024**3,(total,m['target_bytes'])
  inventory=dict(**m,roots=roots,files=files,total_source_bytes=total,host_lock_held=True,
                 source_preflight=before,captured_at=utc_now(),original_postflight=state['postflight'])
  raw=(json.dumps(inventory,indent=2)+'\n').encode()
  inventory_sha=hashlib.sha256(raw).hexdigest()
  write_json(logs/(m['label']+'.inventory.json'),inventory)
  with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1,dereference=True) as archive:
   info=tarfile.TarInfo('inventory.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
   for row in files:
    p=pathlib.Path('/tmp')/row['path'];assert stamp(p)=={k:row[k] for k in stamp(p)}
    archive.add(p,arcname=row['path'],recursive=False)
  for row in files:
   p=pathlib.Path('/tmp')/row['path'];assert stamp(p)=={k:row[k] for k in stamp(p)}
  write_json(logs/(m['label']+'.capture.json'),dict(label=m['label'],completed_at=utc_now(),
             files=len(files),total_source_bytes=total,inventory_sha256=inventory_sha,removed=False))
 elif m['action']=='retire':
  inventory=json.loads((logs/(m['label']+'.inventory.json')).read_text())
  assert hashlib.sha256((json.dumps(inventory,indent=2)+'\n').encode()).hexdigest()==m['inventory_sha256']
  assert inventory['expected_previous_label']==m['expected_previous_label']
  expected={row['path']:row for row in inventory['files']};found=set()
  for root_text in inventory['roots']:
   root=pathlib.Path(root_text);assert allowed(root)
   for p in root.rglob('*'):
    assert not p.is_symlink() and (p.is_file() or p.is_dir())
    if not p.is_file():continue
    rel=str(p.relative_to('/tmp'));assert rel in expected
    row=expected[rel];assert stamp(p)=={k:row[k] for k in stamp(p)}
    assert digest(p)==row['sha256'];found.add(rel)
  assert found==set(expected)
  result=dict(**m,started_at=utc_now(),host_lock_held=True,remote_hashes_reverified=True,
              total_source_bytes=inventory['total_source_bytes'],files_to_retire=len(expected),
              free_before=before['disk']['free_bytes'],original_postflight=state['postflight'])
  write_json(logs/(m['label']+'.retire-intent.json'),result)
  for root_index,root_text in enumerate(inventory['roots']):
   root=pathlib.Path(root_text)
   for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    assert not p.is_symlink()
    if p.is_file():
     row=expected[str(p.relative_to('/tmp'))]
     assert stamp(p)=={k:row[k] for k in stamp(p)};p.unlink()
    else:p.rmdir()
   root.rmdir()
   if (root_index+1)%256==0:
    write_json(logs/(m['label']+'.retired-'+str(root_index+1).zfill(6)+'.json'),
               dict(label=m['label'],completed_at=utc_now(),retired_roots=inventory['roots'][:root_index+1]))
  after=snapshot(base,post['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
  result.update(completed_at=utc_now(),free_after=after['disk']['free_bytes'],post_health=after,
                host_clean=not failures,failures=failures,retired_roots=inventory['roots'])
  write_json(logs/(m['label']+'.json'),result)
  print(json.dumps(result));assert result['host_clean'],failures
 else:raise ValueError(m['action'])
'''


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_archive(path):
    with tarfile.open(path, 'r|gz') as archive:
        first = archive.next()
        assert first.name == 'inventory.json' and first.isfile()
        raw = archive.extractfile(first).read()
        inventory = json.loads(raw)
        expected = {row['path']: row for row in inventory['files']}
        seen = set()
        for member in archive:
            if member is first:
                continue
            assert member.isfile() and member.name in expected and member.name not in seen
            row = expected[member.name]
            assert member.size == row['bytes']
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
            seen.add(member.name)
        assert seen == set(expected)
    return inventory, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'retire'))
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert request['host'] in HOSTS and re.fullmatch(r'fs-[A-Za-z0-9_.-]+', request['label'])
    assert re.fullmatch(r'/home/ubuntu/trainium_freshstart_20260918_wave[0-9]+', request['remote'])
    assert 1024**3 <= request['target_bytes'] <= 24 * 1024**3
    destination = ROOT / 'out/fresh-start-profiles' / request['label']
    logs = ROOT / 'research/run-logs'
    request = dict(request, action=args.action, maintenance_source_sha256=digest(Path(__file__)))
    if args.action == 'capture':
        destination.mkdir(parents=True, exist_ok=False)
        archive = destination / 'scratch.tar.gz'
        with archive.with_suffix('.partial').open('xb') as stream:
            result = subprocess.run([*SSH, HOSTS[request['host']], 'python3 -c ' + shlex.quote(REMOTE)],
                                    input=json.dumps(request).encode(), stdout=stream,
                                    stderr=subprocess.PIPE, timeout=1800)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        archive.with_suffix('.partial').rename(archive)
        print('Archive received; verifying every original file inside it.', flush=True)
        inventory, inventory_sha = verify_archive(archive)
        atomic_json(destination / 'inventory.json', inventory)
        receipt = dict(label=request['label'], host=request['host'],
                       local_directory=str(destination.relative_to(ROOT)),
                       archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                       inventory_sha256=inventory_sha, original_files=len(inventory['files']),
                       original_bytes=inventory['total_source_bytes'],
                       every_archived_source_hash_verified=True, completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(logs / (request['label'] + '.preservation.json'), receipt)
        print(json.dumps(receipt), flush=True)
    else:
        desktop_keyring_environment()
        receipt = json.loads((logs / (request['label'] + '.preservation.json')).read_text())
        assert receipt['every_archived_source_hash_verified']
        assert digest(destination / 'scratch.tar.gz') == receipt['archive_sha256']
        cloud = json.loads((logs / ('fs-artifact-backup-' + request['label'] + '.json')).read_text())
        assert cloud['all_github_digests_match'] and cloud['entry']['source'] == str(destination.relative_to(ROOT))
        assets = []
        manifest = None
        for asset in cloud['assets']:
            current = json.loads(subprocess.check_output([
                'gh', 'api', f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
            ], text=True, timeout=45))
            assert (current['name'], current['size'], current['digest'], current['state']) == (
                asset['name'], asset['bytes'], 'sha256:' + asset['sha256'], 'uploaded')
            assets.append({k: current[k] for k in ('id', 'name', 'size', 'digest')})
            if '--manifest--' in asset['name']:
                matches = list((ROOT / 'out/fresh-start-artifact-backup' / request['label']).rglob(asset['name']))
                assert len(matches) == 1 and digest(matches[0]) == asset['sha256']
                manifest = json.loads(matches[0].read_text())
        assert manifest and manifest['entry'] == cloud['entry']
        for row in manifest['source_files']:
            p = destination / row['path']
            assert p.resolve().is_relative_to(destination.resolve()) and not p.is_symlink()
            assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
        request.update(inventory_sha256=receipt['inventory_sha256'], cloud_assets_reverified=assets,
                       local_archive_sha256=receipt['archive_sha256'], local_archive_hash_reverified=True)
        result = subprocess.run([*SSH, HOSTS[request['host']], 'python3 -c ' + shlex.quote(REMOTE)],
                                input=json.dumps(request), text=True, capture_output=True, timeout=300)
        if result.returncode:
            atomic_json(logs / (request['label'] + '.failure.json'), dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            raise RuntimeError(result.stderr)
        record = json.loads(result.stdout)
        atomic_json(logs / (request['label'] + '.json'), record)
        print(json.dumps({k: record[k] for k in ('label', 'free_before', 'free_after', 'host_clean', 'files_to_retire')}))


if __name__ == '__main__':
    main()
