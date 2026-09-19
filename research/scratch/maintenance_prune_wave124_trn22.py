"""One-off recovery of the wave124 disk reserve from verified older profiles.

This is intentionally tied to one completed job. Reverify every preserved file,
the cloud manifest's file inventory and all GitHub asset digests before touching
the remote. Remove only matching files under four explicit older profile/copy roots.
Retain the failed postflight, checkpoints, source, logs and compilation caches.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path('research').resolve()))
from fresh_start_sync import SSH, HOSTS


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


items = []
for local_name, cloud_name in (
        ('fs-w36-dense1024-fused-mlp-forward-device-r1', 'mlp-full-forward-wave36'),
        ('fs-w39-mlp-forward-remote-ingest-r2', 'fs-w39-mlp-forward-parquet-r2')):
    lp = Path('research/run-logs', local_name + '.preservation.json')
    cp = Path('research/run-logs', 'fs-artifact-backup-' + cloud_name + '.json')
    saved, cloud = json.loads(lp.read_text()), json.loads(cp.read_text())
    assert saved['host'] == 'trn22' and saved['all_remote_local_hashes_match']
    local, remote = saved['local'], saved['remote']
    assert cloud['all_github_digests_match']
    assert cloud['entry']['source'] == local
    assert cloud['entry'].get('preservation_receipt') in (None, str(lp))
    root = Path(local)
    assert root.resolve().is_relative_to(Path('out/fresh-start-profiles').resolve())
    known = {f['path']: (f['bytes'], f['sha256']) for f in saved['files']}
    for f in saved['files']:
        path = root / f['path']
        assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
        assert (path.stat().st_size, digest(path)) == known[f['path']]
    verified_assets = []
    verified_manifest = None
    for asset in cloud['assets']:
        latest = json.loads(subprocess.check_output([
            'gh', 'api', f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
        ], text=True, timeout=40))
        assert latest['name'] == asset['name'] and latest['size'] == asset['bytes']
        assert latest['state'] == 'uploaded' and latest['digest'] == 'sha256:' + asset['sha256']
        verified_assets.append({k: latest[k] for k in ('id', 'name', 'size', 'digest')})
        if '--manifest--' in asset['name']:
            copies = list(Path('out/fresh-start-artifact-backup', cloud_name).rglob(asset['name']))
            assert len(copies) == 1 and digest(copies[0]) == asset['sha256']
            manifest = json.loads(copies[0].read_text())
            assert manifest['entry'] == cloud['entry']
            cloud_files = {f['path']: (f['bytes'], f['sha256']) for f in manifest['source_files']}
            assert all(cloud_files.get(path) == identity for path, identity in known.items())
            verified_manifest = str(copies[0])
    assert verified_manifest is not None
    roots = [remote]
    if local_name.startswith('fs-w36-'):
        roots += ['/home/ubuntu/trainium_freshstart_20260918_wave39/out/fs-w39-mlp-forward-ingest-input-r1',
                  '/home/ubuntu/trainium_freshstart_20260918_wave39/out/fs-w39-mlp-forward-ingest-input-r2']
    items.append(dict(roots=roots, files=saved['files'], local_receipt=str(lp),
                      cloud_receipt=str(cp), local_all_hashes_reverified=True,
                      cloud_file_inventory_reverified=verified_manifest,
                      cloud_assets_reverified=verified_assets))

now = datetime.datetime.now(datetime.timezone.utc)
label = 'fs-maintenance-preserved-wave124-trn22-' + now.strftime('%Y%m%dT%H%M%SZ')
intent = dict(label=label, host='trn22', prepared_at=now.isoformat(), items=items,
              expected_previous_label='fs-w124-fp8-complete-mlp-component-r1')
Path('research/run-logs', label + '.intent.json').write_text(json.dumps(intent, indent=2) + '\n')
remote_code = r'''
import pathlib,hashlib,json,sys,fcntl,os
sys.path.insert(0,'/home/ubuntu/trainium_freshstart_20260918_wave124/research')
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,utc_now,write_json
m=json.load(sys.stdin)
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text())
 assert state['label']==m['expected_previous_label'],state
 previous=json.loads(pathlib.Path(state['postflight']).read_text())
 assert previous['command_returncode']==0
 assert previous['failures']==['less than8GiB disk space available'],previous['failures']
 base=pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave124')
 before=snapshot(base,previous['completed_at'])
 failures=health_failures(before,(os.getpid(),os.getppid()))
 assert all(f=='less than8GiB disk space available' for f in failures),failures
 approved={
  '/home/ubuntu/trainium_freshstart_20260918_wave36/out/fs-w36-dense1024-fused-mlp-forward-device-r1',
  '/home/ubuntu/trainium_freshstart_20260918_wave39/out/fs-w39-mlp-forward-ingest-input-r1',
  '/home/ubuntu/trainium_freshstart_20260918_wave39/out/fs-w39-mlp-forward-ingest-input-r2',
  '/home/ubuntu/trainium_freshstart_20260918_wave39/out/fs-w39-mlp-forward-parquet-r2'}
 assert {r for i in m['items'] for r in i['roots']}==approved
 remove=[];keep=[];missing=[]
 for item in m['items']:
  known={f['path']:f for f in item['files']}
  allowed={(f['bytes'],f['sha256']) for f in item['files']}
  for root_text in item['roots']:
   root=pathlib.Path(root_text)
   assert root.is_dir() and not root.is_symlink(),root_text
   paths=sorted(root.rglob('*'));assert not any(p.is_symlink() for p in paths)
   found={str(p.relative_to(root)) for p in paths if p.is_file()}
   missing.extend(str(root/p) for p in sorted(set(known)-found))
   for p in paths:
    if not p.is_file(): continue
    rel=str(p.relative_to(root))
    assert p.resolve().is_relative_to(root.resolve())
    stat=p.stat()
    with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if rel in known:
     f=known[rel];assert (stat.st_size,digest)==(f['bytes'],f['sha256']),str(p)
    if (stat.st_size,digest) in allowed:
     remove.append(dict(path=str(p),bytes=stat.st_size,sha256=digest,
                        inode=stat.st_ino,links=stat.st_nlink))
    else: keep.append(str(p))
 assert remove
 result=dict(**m,host_lock_held=True,remote_all_existing_hashes_reverified=True,
  free_before=before['disk']['free_bytes'],bytes_to_unlink=sum(f['bytes'] for f in remove),
  files_to_unlink=remove,unmatched_files_retained=keep,already_absent_files=missing,
  original_failed_postflight_retained=state['postflight'],started_at=utc_now())
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 for f in remove:
  p=pathlib.Path(f['path']);assert p.stat().st_ino==f['inode'] and not p.is_symlink();p.unlink()
 after=snapshot(base,previous['completed_at'])
 failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),removed=True,free_after=after['disk']['free_bytes'],
  post_health=after,host_clean=not failures,failures=failures)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result))
 assert result['host_clean'],failures
'''
r = subprocess.run([*SSH, HOSTS['trn22'], 'python3 -c ' + shlex.quote(remote_code)],
                   input=json.dumps(intent), text=True, capture_output=True, timeout=180)
if r.returncode:
    Path('research/run-logs', label + '.failure.json').write_text(json.dumps(
        dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    print(r.stderr[-4000:])
    r.check_returncode()
result = json.loads(r.stdout)
Path('research/run-logs', label + '.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('label', 'bytes_to_unlink', 'free_before',
                                      'free_after', 'host_clean', 'completed_at')}))
