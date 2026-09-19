#!/usr/bin/env python3
"""Retire only the redundant trn21 checkpoint transfer after its replay.

The source98 checkpoint remains locally, on its original trn22 host and in the
verified GitHub release. Require the exact successful replay and the host lock.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path("research").resolve()))
from fresh_start_sync import HOSTS, SSH


def main():
    logs = Path("research/run-logs")
    copy = json.loads((logs / "fs-w127-submission-source98-checkpoint-copy.json").read_text())
    cloud = json.loads((logs / "fs-artifact-backup-fs-w98-dense1024-context1024-pack32-owned-ce-1800-r1.json").read_text())
    assert copy["host"] == "trn21" and copy["matched"] and cloud["all_github_digests_match"]
    assert (cloud["entry"]["source"], cloud["payload_bytes"], cloud["payload_sha256"]) == (
        copy["source"], copy["bytes"], copy["sha256"])
    local = Path(copy["source"])
    with local.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == copy["sha256"]
    assert local.stat().st_size == copy["bytes"]
    assets = []
    for asset in cloud["assets"]:
        live = json.loads(subprocess.check_output([
            "gh", "api", f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
        ], text=True, timeout=45))
        assert (live["name"], live["size"], live["digest"], live["state"]) == (
            asset["name"], asset["bytes"], "sha256:" + asset["sha256"], "uploaded")
        assets.append({k: live[k] for k in ("id", "name", "size", "digest")})
    now = datetime.datetime.now(datetime.timezone.utc)
    request = dict(
        label="fs-maintenance-submission-replay-copy-" + now.strftime("%Y%m%dT%H%M%SZ"),
        host="trn21", remote="/home/ubuntu/trainium_freshstart_20260918_wave129",
        expected_previous_label="fs-w129-submission-source98-public20m-r1",
        copy=copy, local_hash_reverified=True, cloud_assets_reverified=assets,
        prepared_at=now.isoformat())
    (logs / (request["label"] + ".intent.json")).write_text(json.dumps(request, indent=2) + "\n")
    code = r'''
import sys,pathlib,json,fcntl,hashlib,os
m=json.load(sys.stdin);base=pathlib.Path(m['remote']);sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,write_json,utc_now
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=json.loads(HOST_STATE.read_text())
 assert previous['label']==m['expected_previous_label'] and previous['host_clean']
 post_path=base/'research/run-logs'/(m['expected_previous_label']+'.postflight.json')
 assert previous['postflight']==str(post_path)
 post=json.loads(post_path.read_text())
 assert post['command_returncode']==0 and post['host_clean']
 before=snapshot(base,previous['completed_at'])
 assert not health_failures(before,(os.getpid(),os.getppid()))
 p=pathlib.Path(m['copy']['destination'])
 assert p==pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave127/out/submission-source98-final.pt')
 assert not p.is_symlink() and p.is_file()
 st=p.stat()
 with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 assert (st.st_size,sha)==(m['copy']['bytes'],m['copy']['sha256'])
 result=dict(**m,started_at=utc_now(),host_lock_held=True,remote_hash_reverified=True,
             free_before=before['disk']['free_bytes'],removed_path=str(p),removed_bytes=st.st_size)
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 assert p.stat().st_ino==st.st_ino and not p.is_symlink();p.unlink()
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
    print(json.dumps({k: record[k] for k in ("label", "free_before", "free_after", "host_clean")}))


if __name__ == "__main__":
    main()
