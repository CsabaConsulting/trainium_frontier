#!/usr/bin/env python3
"""Retire exact, locally/cloud-preserved profile files before submission tests.

No checkpoints, source, logs, SDK packages or compilation caches are deleted.
This operation holds the normal device lock and rechecks every remote file hash.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path("research").resolve()))
from fresh_start_sync import SSH, HOSTS


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    host = sys.argv[1]
    assert host in HOSTS
    choices = {
        "trn21": [
            ("baseline-dense1024-wave16", "/home/ubuntu/trainium_freshstart_20260918_wave16/out/fs-w16-dense1024-nki-norm-device-micro-r1"),
            ("fs-w87-fp8-packed-weights-component-profile-r1", "/home/ubuntu/trainium_freshstart_20260918_wave87/out/fs-w87-fp8-packed-weights-component-profile-r1"),
        ],
        "trn22": [
            ("fs-w48-mlp-isolated-reference-r1", "/home/ubuntu/trainium_freshstart_20260918_wave48/out/fs-w48-mlp-isolated-reference-r1"),
            ("fs-w82-fp8-token-row-backward-component-profile-r1", "/home/ubuntu/trainium_freshstart_20260918_wave82/out/fs-w82-fp8-token-row-backward-component-profile-r1"),
            ("fs-w78-fp8-token-rows-component-profile-r1", "/home/ubuntu/trainium_freshstart_20260918_wave78/out/fs-w78-fp8-token-rows-component-profile-r1"),
        ],
    }
    items = []
    for name, remote in choices[host]:
        receipt = Path("research/run-logs", "fs-artifact-backup-" + name + ".json")
        cloud = json.loads(receipt.read_text())
        assert cloud["all_github_digests_match"]
        source = Path(cloud["entry"]["source"])
        assert source.resolve().is_relative_to(Path("out/fresh-start-profiles").resolve())
        assets, files = [], None
        for asset in cloud["assets"]:
            live = json.loads(subprocess.check_output([
                "gh", "api", f"repos/{cloud['repository']}/releases/assets/{asset['github_asset_id']}"
            ], text=True, timeout=45))
            assert (live["name"], live["size"], live["digest"], live["state"]) == (
                asset["name"], asset["bytes"], "sha256:" + asset["sha256"], "uploaded")
            assets.append({k: live[k] for k in ("id", "name", "size", "digest")})
            if "--manifest--" in asset["name"]:
                matches = list(Path("out/fresh-start-artifact-backup", name).rglob(asset["name"]))
                assert len(matches) == 1 and digest(matches[0]) == asset["sha256"]
                manifest = json.loads(matches[0].read_text())
                assert manifest["entry"] == cloud["entry"]
                files = manifest["source_files"]
        assert files
        for row in files:
            p = source / row["path"]
            assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(source.resolve())
            assert (p.stat().st_size, digest(p)) == (row["bytes"], row["sha256"])
        items.append(dict(root=remote, files=files, local_source=str(source),
                          cloud_receipt=str(receipt), cloud_assets_reverified=assets))
    now = datetime.datetime.now(datetime.timezone.utc)
    label = "fs-maintenance-submission-headroom-" + host + "-" + now.strftime("%Y%m%dT%H%M%SZ")
    base = "/home/ubuntu/trainium_freshstart_20260918_wave125"
    expected = "fs-w125-" + ("native" if host == "trn21" else "full") + "-complete-mlp-remote-ingest-r1"
    request = dict(label=label, host=host, remote=base, expected_previous_label=expected,
                   prepared_at=now.isoformat(), items=items)
    Path("research/run-logs", label + ".intent.json").write_text(json.dumps(request, indent=2) + "\n")
    code = r'''
import sys,pathlib,json,fcntl,hashlib,os
m=json.load(sys.stdin); base=pathlib.Path(m['remote']);sys.path.insert(0,str(base/'research'))
from fresh_start_host import HOST_LOCK,HOST_STATE,snapshot,health_failures,utc_now,write_json
with HOST_LOCK.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=json.loads(HOST_STATE.read_text());assert previous['label']==m['expected_previous_label'] and previous['host_clean']
 before=snapshot(base,previous['completed_at']);assert not health_failures(before,(os.getpid(),os.getppid()))
 remove=[];retained=[];missing=[]
 for item in m['items']:
  root=pathlib.Path(item['root']);assert root.is_dir() and not root.is_symlink()
  assert root.parent.name=='out' and root.parent.parent.parent==pathlib.Path('/home/ubuntu')
  assert root.parent.parent.name.startswith('trainium_freshstart_20260918_wave')
  known={f['path']:f for f in item['files']}
  for relative,row in known.items():
   p=root/relative;assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
   if not p.exists():missing.append(str(p));continue
   st=p.stat()
   with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
   assert (st.st_size,sha)==(row['bytes'],row['sha256']),str(p)
   remove.append(dict(path=str(p),bytes=st.st_size,sha256=sha,inode=st.st_ino))
  retained.extend(str(p) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in known)
 assert remove
 result=dict(**m,host_lock_held=True,local_and_cloud_hashes_reverified=True,remote_hashes_reverified=True,
             free_before=before['disk']['free_bytes'],files_to_unlink=remove,unmatched_files_retained=retained,
             already_absent=missing,bytes_to_unlink=sum(f['bytes'] for f in remove),started_at=utc_now())
 write_json(base/'research/run-logs'/(m['label']+'.intent.json'),result)
 for row in remove:
  p=pathlib.Path(row['path']);assert p.stat().st_ino==row['inode'] and not p.is_symlink();p.unlink()
 after=snapshot(base,previous['completed_at']);failures=health_failures(after,(os.getpid(),os.getppid()))
 result.update(completed_at=utc_now(),free_after=after['disk']['free_bytes'],post_health=after,
               removed=True,host_clean=not failures,failures=failures)
 write_json(base/'research/run-logs'/(m['label']+'.json'),result)
 print(json.dumps(result));assert result['host_clean'],failures
'''
    result = subprocess.run([*SSH, HOSTS[host], "python3 -c " + shlex.quote(code)],
                            input=json.dumps(request), text=True, capture_output=True, timeout=180)
    if result.returncode:
        Path("research/run-logs", label + ".failure.json").write_text(json.dumps(dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + "\n")
        print(result.stderr)
        result.check_returncode()
    record = json.loads(result.stdout)
    Path("research/run-logs", label + ".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("label", "bytes_to_unlink", "free_before", "free_after", "host_clean")}))


if __name__ == "__main__":
    main()
