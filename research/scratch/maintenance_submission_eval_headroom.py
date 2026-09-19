#!/usr/bin/env python3
"""Retire one verified old raw-profile archive after the cold rehearsal.

This is tied to the completed source129 training job on trn22. The complete
archive remains local and in verified GitHub chunks. No checkpoint, source,
log or compilation cache is removed; the original training postflight remains.
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


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    logs = Path("research/run-logs")
    artifact = "dense1024-wave4-archived-trace"
    cloud = json.loads((logs / ("fs-artifact-backup-" + artifact + ".json")).read_text())
    local = Path("out/fresh-start-profiles/fs-w4-dense1024-device-micro-r2.tar.gz")
    assert cloud["all_github_digests_match"] and cloud["entry"]["source"] == str(local)
    assert cloud["entry"]["kind"] == "archived_raw_profile"
    assert not local.is_symlink() and local.is_file()
    assert (local.stat().st_size, digest(local)) == (cloud["payload_bytes"], cloud["payload_sha256"])
    assets, manifest = [], None
    for asset in cloud["assets"]:
        live = json.loads(subprocess.check_output([
            "gh", "api", f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
        ], text=True, timeout=45))
        assert (live["name"], live["size"], live["digest"], live["state"]) == (
            asset["name"], asset["bytes"], "sha256:" + asset["sha256"], "uploaded")
        assets.append({k: live[k] for k in ("id", "name", "size", "digest")})
        if "--manifest--" in asset["name"]:
            copies = list(Path("out/fresh-start-artifact-backup", artifact).rglob(asset["name"]))
            assert len(copies) == 1 and digest(copies[0]) == asset["sha256"]
            manifest = json.loads(copies[0].read_text())
    assert manifest is not None and manifest["entry"] == cloud["entry"]
    assert (manifest["payload_bytes"], manifest["payload_sha256"], manifest["payload_encoding"]) == (
        cloud["payload_bytes"], cloud["payload_sha256"], "original_file")
    assert {(x["name"], x["bytes"], x["sha256"]) for x in manifest["chunks"]} == {
        (x["name"], x["bytes"], x["sha256"]) for x in cloud["assets"] if "--manifest--" not in x["name"]}
    now = datetime.now(timezone.utc)
    request = dict(
        label="fs-maintenance-submission-eval-headroom-" + now.strftime("%Y%m%dT%H%M%SZ"),
        host="trn22", remote="/home/ubuntu/trainium_freshstart_20260918_wave129",
        expected_previous_label="fs-w129-submission-cold-rehearsal-1800-r1",
        local_archive=str(local), bytes=cloud["payload_bytes"], sha256=cloud["payload_sha256"],
        cloud_assets_reverified=assets, local_hash_reverified=True,
        cloud_chunk_manifest_verified=True, prepared_at=now.isoformat())
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
 p=pathlib.Path('/home/ubuntu/trainium_freshstart_20260918_wave4/out/fs-w4-dense1024-device-micro-r2.tar.gz')
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 st=p.stat()
 with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 assert (st.st_size,sha)==(m['bytes'],m['sha256'])
 result=dict(**m,started_at=utc_now(),host_lock_held=True,remote_hash_reverified=True,
             free_before=before['disk']['free_bytes'],removed_path=str(p),
             original_training_postflight_retained=state['postflight'])
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 assert p.stat().st_ino==st.st_ino and p.stat().st_mtime_ns==st.st_mtime_ns and not p.is_symlink()
 p.unlink()
 after=snapshot(base,previous['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),free_after=after['disk']['free_bytes'],post_health=after,
               host_clean=not failures,failures=failures,removed=True)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result));assert result['host_clean'],failures
'''
    result = subprocess.run([*SSH, HOSTS["trn22"], "python3 -c " + shlex.quote(code)],
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
