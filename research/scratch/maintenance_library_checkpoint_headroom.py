"""Retire four named old trn22 checkpoint copies after fresh three-way verification.

Full checkpoint bytes remain on the desktop and in verified GitHub assets.
The current best and cold replay are excluded. A host lock and clean previous
experiment are mandatory, and every selected hash passes before any unlink.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path('research').resolve()))
from fresh_start_artifact_backup import desktop_keyring_environment
from fresh_start_sync import SSH, HOSTS

SELECTION = (
    (12, 'fs-w12-dense1024-nki-norm-1800-r1'),
    (15, 'fs-w15-dense1024-nki-sdpa-only-1800-r1'),
    (18, 'fs-w18-dense1024-nki-norm-clipafter-seed43-1800-r1'),
    (63, 'fs-w63-dense1024-native-mlp-context1024-pack16-adam-1800-r1'),
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    desktop_keyring_environment()
    logs = Path('research/run-logs')
    best = json.loads(Path('research/fresh-start-best-public.json').read_text())
    selected = []
    for wave, label in SELECTION:
        receipt = logs / ('fs-artifact-backup-' + label + '.json')
        cloud = json.loads(receipt.read_text())
        local = Path('checkpoints', label, 'final.pt')
        assert cloud['all_github_digests_match']
        assert cloud['entry']['source'] == str(local) and cloud['entry']['kind'] == 'checkpoint'
        assert cloud['payload_sha256'] != best['checkpoint']['local_sha256']
        assert local.is_file() and not local.is_symlink()
        assert (local.stat().st_size, digest(local)) == (cloud['payload_bytes'], cloud['payload_sha256'])
        assets = []
        for asset in cloud['assets']:
            live = json.loads(subprocess.check_output([
                'gh', 'api', f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
            ], text=True, timeout=40))
            assert (live['name'], live['size'], live['digest'], live['state']) == (
                asset['name'], asset['bytes'], 'sha256:' + asset['sha256'], 'uploaded')
            assets.append({k: live[k] for k in ('id', 'name', 'size', 'digest')})
        assert len([a for a in assets if a['size'] == cloud['payload_bytes']
                    and a['digest'] == 'sha256:' + cloud['payload_sha256']]) == 1
        selected.append(dict(local=str(local), cloud_receipt=str(receipt),
            remote=f'/home/ubuntu/trainium_freshstart_20260918_wave{wave}/out/{label}/final.pt',
            bytes=cloud['payload_bytes'], sha256=cloud['payload_sha256'],
            local_hash_reverified=True, cloud_assets_reverified=assets))
    now = datetime.now(timezone.utc)
    request = dict(label='fs-maintenance-library-checkpoints-trn22-' + now.strftime('%Y%m%dT%H%M%SZ'),
        host='trn22', remote='/home/ubuntu/trainium_freshstart_20260918_wave170',
        expected_previous_label='fs-w170-nkilib-attention-timing-r1',
        expected_source='2cd50e043742ff327f40eda70e9e43632a29118d', selected=selected,
        current_best_excluded=True, cold_replay_excluded=True, prepared_at=now.isoformat(),
        script_sha256=digest(Path(__file__)))
    (logs / (request['label'] + '.intent.json')).write_text(json.dumps(request, indent=2) + '\n')
    code = r'''
import pathlib,sys,json,hashlib,fcntl,os
m=json.load(sys.stdin);base=pathlib.Path(m['remote']);sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,write_json,utc_now
allowed={
 '/home/ubuntu/trainium_freshstart_20260918_wave12/out/fs-w12-dense1024-nki-norm-1800-r1/final.pt',
 '/home/ubuntu/trainium_freshstart_20260918_wave15/out/fs-w15-dense1024-nki-sdpa-only-1800-r1/final.pt',
 '/home/ubuntu/trainium_freshstart_20260918_wave18/out/fs-w18-dense1024-nki-norm-clipafter-seed43-1800-r1/final.pt',
 '/home/ubuntu/trainium_freshstart_20260918_wave63/out/fs-w63-dense1024-native-mlp-context1024-pack16-adam-1800-r1/final.pt',
}
assert {x['remote'] for x in m['selected']}==allowed and len(m['selected'])==4
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text());assert state['label']==m['expected_previous_label']
 post=json.loads(pathlib.Path(state['postflight']).read_text())
 assert post['command_returncode']==0 and post['host_clean'] and post['source_commit']==m['expected_source']
 before=snapshot(base,post['completed_at']);failures=health_failures(before,(os.getpid(),os.getppid()))
 assert not failures,failures
 verified=[]
 for item in m['selected']:
  p=pathlib.Path(item['remote']);assert p.resolve()==p and p.is_file() and not p.is_symlink()
  st=p.stat()
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  current=p.stat();identity=lambda s:(s.st_ino,s.st_mtime_ns,s.st_size)
  assert identity(st)==identity(current) and (st.st_size,sha)==(item['bytes'],item['sha256'])
  verified.append((p,st))
 result=dict(**m,host_lock_held=True,remote_hashes_reverified=True,started_at=utc_now(),
             free_before=before['disk']['free_bytes'],removed=[])
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 for i,(p,st) in enumerate(verified):
  assert identity(p.stat())==identity(st) and not p.is_symlink()
  p.unlink();result['removed'].append(str(p))
  write_json(base/'research/run-logs'/(m['label']+'.progress-'+str(i+1)+'.json'),
             dict(removed=str(p),at=utc_now()))
 after=snapshot(base,post['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),post_health=after,host_clean=not failures,
               failures=failures,free_after=after['disk']['free_bytes'])
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result));assert result['host_clean'],failures
'''
    result = subprocess.run([*SSH, HOSTS['trn22'], 'python3 -c ' + shlex.quote(code)],
        input=json.dumps(request), text=True, capture_output=True, timeout=240)
    if result.returncode:
        (logs / (request['label'] + '.failure.json')).write_text(json.dumps(dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + '\n')
        print(result.stderr[-4000:])
        result.check_returncode()
    record = json.loads(result.stdout)
    (logs / (request['label'] + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('label', 'removed', 'free_before', 'free_after', 'host_clean')}))


if __name__ == '__main__':
    main()
