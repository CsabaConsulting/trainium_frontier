"""One-off recovery of wave123 using the verified wave112 parquet copy.

Hash-check all local/remote files and reverify cloud assets under the host lock.
Keep all source, logs, checkpoints and shared compiler caches. The original
failed postflight remains; this script is tied to the exact previous job.
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

items = []
for local_name, cloud_name in [
    ('fs-w112-head-loss-cached-caller-remote-ingest-r2',
     'fs-w112-head-loss-cached-caller-remote-ingest-r2-parquet'),
]:
    lp = Path('research/run-logs', local_name + '.preservation.json')
    cp = Path('research/run-logs', 'fs-artifact-backup-' + cloud_name + '.json')
    data, cloud = json.loads(lp.read_text()), json.loads(cp.read_text())
    assert data['host'] == 'trn21' and data['all_remote_local_hashes_match']
    assert cloud['all_github_digests_match']
    assert cloud['entry']['source'] == data['local']
    assert cloud['entry']['preservation_receipt'] == str(lp)
    root = Path(data['local'])
    assert root.resolve().is_relative_to(Path('out/fresh-start-profiles').resolve())
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == {
        f['path'] for f in data['files']}
    for f in data['files']:
        p = root / f['path']
        assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
        assert p.stat().st_size == f['bytes']
        with p.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == f['sha256']
    verified = []
    for a in cloud['assets']:
        latest = json.loads(subprocess.check_output([
            'gh', 'api', f"repos/{cloud['repository']}/releases/assets/{a['github_asset_id']}"
        ], text=True, timeout=40))
        assert latest['name'] == a['name'] and latest['size'] == a['bytes']
        assert latest['state'] == 'uploaded' and latest['digest'] == 'sha256:' + a['sha256']
        verified.append({k: latest[k] for k in ['id', 'name', 'size', 'digest']})
    roots = [data['remote']]
    items.append(dict(roots=roots, files=data['files'], local_receipt=str(lp),
                      cloud_receipt=str(cp), local_all_hashes_reverified=True,
                      cloud_assets_reverified=verified))

now = datetime.datetime.now(datetime.timezone.utc)
label = 'fs-maintenance-preserved-wave123-trn21-' + now.strftime('%Y%m%dT%H%M%SZ')
intent = dict(label=label, host='trn21', items=items, prepared_at=now.isoformat(),
              expected_previous_label='fs-w123-head-loss-split-cached-phase-r1')
Path('research/run-logs', label + '.intent.json').write_text(json.dumps(intent, indent=2) + '\n')
remote_code = r'''
import pathlib,hashlib,json,sys,fcntl,os
sys.path.insert(0,'/home/ubuntu/trainium_freshstart_20260918_wave123/research')
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,utc_now,write_json
m=json.load(sys.stdin)
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text())
 assert state['label']==m['expected_previous_label'],state
 previous=json.loads(pathlib.Path(state['postflight']).read_text())
 assert previous['command_returncode']==0
 assert previous['failures'] in ([],['less than8GiB disk space available']),previous['failures']
 base=pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave123')
 before=snapshot(base,previous['completed_at'])
 failures=health_failures(before,(os.getpid(),os.getppid()))
 assert all(f=='less than8GiB disk space available' for f in failures),failures
 remove=[];keep=[]
 approved={'fs-w112-head-loss-cached-caller-remote-ingest-r2-parquet'}
 seen=set()
 for item in m['items']:
  known={f['path']:f for f in item['files']}
  allowed={(f['bytes'],f['sha256']) for f in item['files']}
  for root_text in item['roots']:
   root=pathlib.Path(root_text)
   assert root.parent==pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave112/out')
   assert root.name in approved and root_text not in seen
   seen.add(root_text)
   assert root.is_dir() and not root.is_symlink()
   paths=sorted(root.rglob('*'));assert not any(p.is_symlink() for p in paths)
   files=[p for p in paths if p.is_file()]
   found={str(p.relative_to(root)) for p in files}
   assert set(known)<=found,sorted(set(known)-found)
   for p in files:
    assert p.resolve().is_relative_to(root.resolve())
    stat=p.stat();rel=str(p.relative_to(root))
    with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if rel in known: assert (stat.st_size,digest)==(known[rel]['bytes'],known[rel]['sha256']),str(p)
    entry=dict(path=str(p),bytes=stat.st_size,sha256=digest,inode=stat.st_ino,links=stat.st_nlink)
    (remove if (stat.st_size,digest) in allowed else keep).append(entry)
 assert len(seen)==1 and remove
 result=dict(**m,host_lock_held=True,remote_hashes_reverified=True,
  free_before=before['disk']['free_bytes'],bytes_to_unlink=sum(f['bytes'] for f in remove),
  files_to_unlink=remove,unmatched_files_retained=keep,started_at=utc_now())
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
r = subprocess.run([*SSH, HOSTS['trn21'], 'python3 -c ' + shlex.quote(remote_code)],
                   input=json.dumps(intent), text=True, capture_output=True, timeout=180)
if r.returncode:
    Path('research/run-logs', label + '.failure.json').write_text(json.dumps(
        dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    print(r.stderr[-4000:])
    r.check_returncode()
result = json.loads(r.stdout)
Path('research/run-logs', label + '.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['label', 'bytes_to_unlink', 'free_before',
                                      'free_after', 'host_clean', 'completed_at']}))
