#!/usr/bin/env python3
"""Finish the three remaining retirements after a partial wave140 cleanup.

Every full checkpoint remains local and in a currently verified GitHub asset.
The exact prior experiment must have exited successfully; only a disk-reserve
health failure is recoverable here. An exclusive host lock excludes active
experiments. No source, logs, profiles or compilation caches are removed.
The first attempt retired two verified copies, then its second progress write
hit the audit writer's no-overwrite rule. Preserve that failure and its intent.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path("research").resolve()))
from fresh_start_sync import HOSTS, SSH


SELECTION = (
    ("fs-w4-w896-nki-rms-fp32-r1", 4, "fs-w4-w896-nki-rms-fp32-r1"),
    ("fs-w50-dense1024-native-mlp-pack8-1800-r1", 50, "fs-w50-dense1024-native-mlp-pack8-1800-r1"),
    ("fs-w65-dense1024-native-mlp-pack8-owned-1800-r1", 65, "fs-w65-dense1024-native-mlp-pack8-owned-1800-r1"),
    ("fs-w70-dense1024-context1024-pack16-owned-ce-1800-r1", 70, "fs-w70-dense1024-context1024-pack16-owned-ce-1800-r1"),
    ("best-dense1024-clipafter", 16, "fs-w16-dense1024-nki-norm-clipafter-1800-r1"),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    logs = Path("research/run-logs")
    best = json.loads(Path("research/fresh-start-best-public.json").read_text())
    selected = []
    # Exact recovery scope: the first two files were already retired by the
    # recorded attempt. Do not silently skip missing files in the remaining set.
    for artifact, wave, checkpoint_label in SELECTION[2:]:
        cloud = json.loads((logs / ("fs-artifact-backup-" + artifact + ".json")).read_text())
        local = Path("checkpoints", checkpoint_label, "final.pt")
        assert cloud["all_github_digests_match"] and cloud["entry"]["source"] == str(local)
        assert cloud["entry"]["kind"] in ("checkpoint", "best_checkpoint")
        assert cloud["payload_sha256"] != best["checkpoint"]["local_sha256"]
        assert not local.is_symlink() and local.is_file()
        assert (local.stat().st_size, digest(local)) == (cloud["payload_bytes"], cloud["payload_sha256"])
        assets = []
        for asset in cloud["assets"]:
            live = json.loads(subprocess.check_output([
                "gh", "api", f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
            ], text=True, timeout=45))
            assert (live["name"], live["size"], live["digest"], live["state"]) == (
                asset["name"], asset["bytes"], "sha256:" + asset["sha256"], "uploaded")
            assets.append({k: live[k] for k in ("id", "name", "size", "digest")})
        # These checkpoints are stored as original single-file assets. Match
        # the full payload directly, independently of the auxiliary manifest.
        full = [a for a in assets if a["size"] == cloud["payload_bytes"]
                and a["digest"] == "sha256:" + cloud["payload_sha256"]]
        assert len(full) == 1
        selected.append(dict(
            artifact_id=artifact, local_checkpoint=str(local),
            remote_checkpoint=f"/home/ubuntu/trainium_freshstart_20260918_wave{wave}/out/{checkpoint_label}/final.pt",
            bytes=cloud["payload_bytes"], sha256=cloud["payload_sha256"],
            cloud_assets_reverified=assets, local_hash_reverified=True))
    now = datetime.now(timezone.utc)
    request = dict(
        label="fs-maintenance-full-fp8-phase-headroom-" + now.strftime("%Y%m%dT%H%M%SZ"),
        host="trn21", remote="/home/ubuntu/trainium_freshstart_20260918_wave140",
        expected_previous_label="fs-w140-full-fp8-native-control-phase-r1",
        previous_partial_cleanup="fs-maintenance-full-fp8-phase-headroom-20260919T072852Z",
        previous_maintenance_sha256="8ffd369afa4397ec32d83b2dc53150e798bd931a057e75609aa11f871094ca12",
        selected=selected, current_best_excluded=True, prepared_at=now.isoformat(),
        maintenance_source_sha256=digest(Path(__file__)))
    (logs / (request["label"] + ".intent.json")).write_text(json.dumps(request, indent=2) + "\n")
    code = r'''
import sys,pathlib,json,fcntl,hashlib,os
m=json.load(sys.stdin);base=pathlib.Path(m['remote']);sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,write_json,utc_now
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(HOST_STATE.read_text())
 assert state['label']==m['expected_previous_label'] and state['remote']==str(base)
 previous=json.loads(pathlib.Path(state['postflight']).read_text())
 assert previous['command_returncode']==0
 assert all(f=='less than8GiB disk space available' for f in previous['failures'])
 before=snapshot(base,previous['completed_at'])
 failures=health_failures(before,(os.getpid(),os.getppid()))
 assert all(f=='less than8GiB disk space available' for f in failures),failures
 prior=json.loads((base/'research/run-logs'/(m['previous_partial_cleanup']+'.intent.json')).read_text())
 progress=json.loads((base/'research/run-logs'/(m['previous_partial_cleanup']+'.progress.json')).read_text())
 assert prior['maintenance_source_sha256']==m['previous_maintenance_sha256']
 assert prior['remote_hashes_reverified'] and prior['expected_previous_label']==m['expected_previous_label']
 old_selection=prior['selected'];assert len(old_selection)==5
 already_absent=[x['remote_checkpoint'] for x in old_selection[:2]]
 assert progress['removed']==already_absent[:1]
 assert all(not pathlib.Path(p).exists() for p in already_absent)
 assert [(x['remote_checkpoint'],x['sha256'],x['bytes']) for x in old_selection[2:]]==[(x['remote_checkpoint'],x['sha256'],x['bytes']) for x in m['selected']]
 verified=[]
 for item in m['selected']:
  p=pathlib.Path(item['remote_checkpoint'])
  assert p.resolve()==p and p.is_file() and not p.is_symlink()
  assert p.name=='final.pt' and p.parent.parent.name=='out'
  assert p.parents[3]==pathlib.Path('/home/ubuntu')
  assert p.parents[2].name.startswith('trainium_freshstart_20260918_wave')
  st=p.stat()
  with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  current=p.stat()
  assert (st.st_size,st.st_mtime_ns,st.st_ino)==(current.st_size,current.st_mtime_ns,current.st_ino)
  assert (st.st_size,sha)==(item['bytes'],item['sha256'])
  verified.append((p,st))
 # Verify the complete selected set before deleting its first redundant copy.
 result=dict(**m,started_at=utc_now(),host_lock_held=True,remote_hashes_reverified=True,
             free_before=before['disk']['free_bytes'],removed=[],
             previous_observed_absent=already_absent,
             previous_written_progress_removed=progress['removed'],
             original_experiment_postflight_retained=state['postflight'])
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 for p,st in verified:
  current=p.stat()
  assert (current.st_ino,current.st_mtime_ns,current.st_size)==(st.st_ino,st.st_mtime_ns,st.st_size)
  assert not p.is_symlink()
  p.unlink();result['removed'].append(str(p))
  write_json(base/'research/run-logs'/(m['label']+'.progress-'+str(len(result['removed'])).zfill(3)+'.json'),result)
 after=snapshot(base,previous['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),free_after=after['disk']['free_bytes'],post_health=after,
               host_clean=not failures,failures=failures)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result));assert result['host_clean'],failures
'''
    result = subprocess.run([*SSH, HOSTS["trn21"], "python3 -c " + shlex.quote(code)],
                            input=json.dumps(request), text=True, capture_output=True, timeout=180)
    if result.returncode:
        (logs / (request["label"] + ".failure.json")).write_text(json.dumps(dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + "\n")
        print(result.stderr)
        result.check_returncode()
    record = json.loads(result.stdout)
    (logs / (request["label"] + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("label", "removed", "free_before", "free_after", "host_clean")}))


if __name__ == "__main__":
    main()
