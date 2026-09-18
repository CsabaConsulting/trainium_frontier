#!/usr/bin/env python3
"""Retrieve a completed campaign checkpoint and record independent SHA256s.

Call only after the training runner has completed. Source and remote checkpoint
are read-only. A partial transfer never replaces a previously preserved file.
Heavy checkpoints stay under the existing ignored checkpoints/ directory.
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


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", choices=tuple(HOSTS), required=True)
    ap.add_argument("--remote", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--bwlimit", type=int, default=8192, help="rsync KiB/s")
    args = ap.parse_args()
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", args.label)
    assert re.fullmatch(r"[0-9a-f]{40}", args.commit)
    assert args.remote.parent == Path("/home/ubuntu")
    assert args.remote.name.startswith("trainium_freshstart_") and args.bwlimit > 0
    remote_file = args.remote / "out" / args.label / "final.pt"
    remote_script = "\n".join([
        "import hashlib,json,subprocess", "from pathlib import Path",
        f"root=Path({str(args.remote)!r})", f"p=Path({str(remote_file)!r})",
        "commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()",
        f"assert commit == {args.commit!r}",
        "before=p.stat()", "with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()",
        "after=p.stat()", "assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)",
        "print(json.dumps(dict(bytes=after.st_size,sha256=digest,commit=commit,mtime_ns=after.st_mtime_ns)))",
    ])
    remote = json.loads(subprocess.check_output(
        [*SSH, HOSTS[args.host], "python3 -c " + shlex.quote(remote_script)], text=True, timeout=90))
    destination = ROOT / "checkpoints" / args.label / "final.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.stat().st_size == remote["bytes"] and sha256(destination) == remote["sha256"], \
            "Refusing to replace a distinct locally preserved checkpoint"
    else:
        partial = destination.with_suffix(".pt.partial")
        subprocess.run(["rsync", "-a", "--partial", "--timeout=45", f"--bwlimit={args.bwlimit}",
                        "-e", shlex.join(SSH), f"{HOSTS[args.host]}:{remote_file}", str(partial)], check=True)
        assert partial.stat().st_size == remote["bytes"]
        assert sha256(partial) == remote["sha256"], "Transferred checkpoint hash mismatch"
        # Refuse an unexpected concurrent finalizer instead of overwriting it.
        import os
        os.link(partial, destination)
        partial.unlink()
    local_hash = sha256(destination)
    receipt = dict(label=args.label, verified_at_utc=datetime.now(timezone.utc).isoformat(),
                   source_commit=args.commit, local_path=str(destination.relative_to(ROOT)),
                   remote_host=args.host, remote_path=str(remote_file), bytes=remote["bytes"],
                   local_sha256=local_hash, remote_sha256=remote["sha256"],
                   remote_mtime_ns=remote["mtime_ns"], matched=(local_hash == remote["sha256"]))
    path = ROOT / "research/run-logs" / (args.label + ".preservation.json")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
