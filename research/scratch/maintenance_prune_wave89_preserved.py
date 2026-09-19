from pathlib import Path
import hashlib,json,subprocess,shlex,datetime,sys
sys.path.insert(0,str(Path('research').resolve()))
from fresh_start_sync import SSH,HOSTS
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
items=[]
for local_name,cloud_name in [
 ('fs-w89-native-owned-fp8-micro-device-profile-r1','fs-w89-native-owned-fp8-micro-device-profile-r1'),
 ('fs-w89-native-owned-fp8-caller-remote-ingest-r1','fs-w89-native-owned-fp8-caller-remote-ingest-r1-parquet')]:
 local_receipt=Path('research/run-logs')/(local_name+'.preservation.json')
 cloud_receipt=Path('research/run-logs')/('fs-artifact-backup-'+cloud_name+'.json')
 data=json.loads(local_receipt.read_text());cloud=json.loads(cloud_receipt.read_text())
 assert data['host']=='trn22' and data['all_remote_local_hashes_match']
 assert cloud['all_github_digests_match'] and cloud['entry']['source']==data['local']
 assert cloud['entry']['preservation_receipt']==str(local_receipt)
 root=Path(data['local']);assert root.resolve().is_relative_to(Path('out/fresh-start-profiles').resolve())
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 assert actual=={f['path'] for f in data['files']}
 for f in data['files']:
  path=root/f['path'];assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
  assert path.stat().st_size==f['bytes']
  with path.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==f['sha256']
 verified=[]
 for asset in cloud['assets']:
  latest=json.loads(subprocess.check_output(['gh','api',f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"],text=True,timeout=40))
  assert latest['name']==asset['name'] and latest['size']==asset['bytes']
  assert latest['state']=='uploaded' and latest['digest']=='sha256:'+asset['sha256']
  verified.append(dict(id=latest['id'],name=latest['name'],bytes=latest['size'],digest=latest['digest']))
 items.append(dict(remote=data['remote'],files=data['files'],local_receipt=str(local_receipt),cloud_receipt=str(cloud_receipt),local_all_hashes_reverified=True,cloud_assets_reverified=verified))
label='fs-maintenance-preserved-wave89-trn22-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
intent=dict(label=label,host='trn22',expected_previous_label='fs-w98-native-owned-context1024-pack32-phase-r1',items=items,prepared_at=now())
Path('research/run-logs',label+'.intent.json').write_text(json.dumps(intent,indent=2)+'\n')
remote_code=r'''
import pathlib,hashlib,json,sys,fcntl,shutil,os
sys.path.insert(0,'/home/ubuntu/trainium_freshstart_20260918_wave98/research')
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,utc_now,write_json
m=json.load(sys.stdin)
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text())
 assert state['label']==m['expected_previous_label'],state
 previous=json.loads(pathlib.Path(state['postflight']).read_text())
 assert previous['command_returncode']==0 and previous['host_clean']
 base=pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave98')
 before=snapshot(base,previous['completed_at'])
 failures=health_failures(before,(os.getpid(),os.getppid()))
 assert all(f=='less than8GiB disk space available' for f in failures),failures
 bytes_total=0
 for item in m['items']:
  root=pathlib.Path(item['remote'])
  assert root.parent==pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave89/out')
  assert root.name in ('fs-w89-native-owned-fp8-micro-device-profile-r1','fs-w89-native-owned-fp8-caller-remote-ingest-r1-parquet')
  assert root.is_dir() and not root.is_symlink()
  assert not any(p.is_symlink() for p in root.rglob('*'))
  assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}=={f['path'] for f in item['files']}
  for f in item['files']:
   path=root/f['path'];assert path.resolve().is_relative_to(root.resolve())
   assert path.stat().st_size==f['bytes']
   with path.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==f['sha256']
   bytes_total+=f['bytes']
 # All paths and hashes pass before any mutation. The active model/source,
 # compiler caches, checkpoints and local/cloud evidence remain untouched.
 result=dict(**m,host_lock_held=True,remote_all_hashes_reverified=True,free_before=before['disk']['free_bytes'],bytes_to_remove=bytes_total,started_at=utc_now())
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 for item in m['items']: shutil.rmtree(item['remote'])
 after=snapshot(base,previous['completed_at'])
 failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),removed=True,free_after=after['disk']['free_bytes'],post_health=after,host_clean=not failures,failures=failures)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result))
 assert result['host_clean'],failures
'''
r=subprocess.run([*SSH,HOSTS['trn22'],'python3 -c '+shlex.quote(remote_code)],input=json.dumps(intent),text=True,capture_output=True,timeout=180)
if r.returncode:
 Path('research/run-logs',label+'.failure.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 print(r.stderr[-4000:]);r.check_returncode()
result=json.loads(r.stdout)
Path('research/run-logs',label+'.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['label','bytes_to_remove','free_before','free_after','host_clean','completed_at']}))
