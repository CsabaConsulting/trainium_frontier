#!/usr/bin/env python3
"""Retire one remote checkpoint copy only after local and cloud verification.

This preserves the full checkpoint locally and in its verified GitHub release.
The current public-best checkpoint is excluded. A clean idle host, its exact
previous job, unchanged file metadata and all three SHA256 identities are
required. No source, logs, cache directories or other checkpoint is touched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from fresh_start_sync import HOSTS, ROOT, SSH


REMOTE_SCRIPT = r'''
import sys,pathlib,json,fcntl,hashlib,os
m=json.load(sys.stdin);base=pathlib.Path(m['remote']);sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,write_json,utc_now
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=json.loads(HOST_STATE.read_text())
 assert previous['label']==m['expected_previous_label'] and previous['host_clean']
 assert previous['remote']==str(base)
 post_path=base/'research/run-logs'/(previous['label']+'.postflight.json')
 assert previous['postflight']==str(post_path)
 post=json.loads(post_path.read_text());assert post['host_clean'] and post['command_returncode']==0
 before=snapshot(base,previous['completed_at'])
 assert not health_failures(before,(os.getpid(),os.getppid()))
 p=pathlib.Path(m['remote_checkpoint'])
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 assert p.name=='final.pt' and p.parent.name==m['artifact_id'] and p.parent.parent.name=='out'
 assert p.parents[3]==pathlib.Path('/home/ubuntu')
 assert p.parents[2].name.startswith('trainium_freshstart_20260918_wave')
 st=p.stat()
 with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 after_hash=p.stat()
 assert (st.st_size,st.st_mtime_ns,st.st_ino)==(after_hash.st_size,after_hash.st_mtime_ns,after_hash.st_ino)
 assert (st.st_size,sha)==(m['bytes'],m['sha256'])
 result=dict(**m,started_at=utc_now(),host_lock_held=True,remote_hash_reverified=True,
             free_before=before['disk']['free_bytes'],removed=False)
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 if m['apply']:
  assert p.stat().st_ino==st.st_ino and not p.is_symlink();p.unlink();result['removed']=True
 after=snapshot(base,previous['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),free_after=after['disk']['free_bytes'],post_health=after,
               host_clean=not failures,failures=failures)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result));assert result['host_clean'],failures
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", choices=tuple(HOSTS), required=True)
    ap.add_argument("--remote", type=Path, required=True)
    ap.add_argument("--expected-previous-label", required=True)
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--remote-checkpoint", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", args.artifact_id)
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", args.expected_previous_label)
    assert args.remote.parent == Path("/home/ubuntu")
    assert re.fullmatch(r"trainium_freshstart_20260918_wave[0-9]+", args.remote.name)
    logs = ROOT / "research/run-logs"
    cloud = json.loads((logs / ("fs-artifact-backup-" + args.artifact_id + ".json")).read_text())
    best = json.loads((ROOT / "research/fresh-start-best-public.json").read_text())
    assert cloud["all_github_digests_match"] and cloud["entry"]["id"] == args.artifact_id
    assert cloud["entry"]["kind"] == "checkpoint"
    assert cloud["payload_sha256"] != best["checkpoint"]["local_sha256"]
    local = ROOT / cloud["entry"]["source"]
    assert local == ROOT / "checkpoints" / args.artifact_id / "final.pt"
    assert local.resolve() == local and local.is_file() and not local.is_symlink()
    with local.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == cloud["payload_sha256"]
    assert local.stat().st_size == cloud["payload_bytes"]
    assets = []
    for asset in cloud["assets"]:
        live = json.loads(subprocess.check_output([
            "gh", "api", f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
        ], text=True, timeout=45))
        assert (live["name"], live["size"], live["digest"], live["state"]) == (
            asset["name"], asset["bytes"], "sha256:" + asset["sha256"], "uploaded")
        assets.append({k: live[k] for k in ("id", "name", "size", "digest")})
    now = datetime.now(timezone.utc)
    request = dict(
        label="fs-maintenance-preserved-checkpoint-" + args.host + "-" + now.strftime("%Y%m%dT%H%M%SZ"),
        host=args.host, remote=str(args.remote), expected_previous_label=args.expected_previous_label,
        artifact_id=args.artifact_id, remote_checkpoint=str(args.remote_checkpoint),
        local_checkpoint=str(local.relative_to(ROOT)), bytes=cloud["payload_bytes"],
        sha256=cloud["payload_sha256"], cloud_assets_reverified=assets,
        local_hash_reverified=True, current_best_excluded=True,
        prepared_at=now.isoformat(), apply=args.apply)
    (logs / (request["label"] + ".intent.json")).write_text(json.dumps(request, indent=2) + "\n")
    result = subprocess.run([*SSH, HOSTS[args.host], "python3 -c " + shlex.quote(REMOTE_SCRIPT)],
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
