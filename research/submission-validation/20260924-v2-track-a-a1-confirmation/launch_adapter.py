#!/usr/bin/env python3
"""Bind one Track-A confirmation arm to the exact standard fresh-start launcher request."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed_bytes(head, path):
    actual = aio.regular_bytes(path)
    committed = subprocess.check_output(["git", "show", f"{head}:{path.relative_to(ROOT)}"], cwd=ROOT)
    if actual != committed:
        raise SystemExit(f"working file differs from HEAD: {path.relative_to(ROOT)}")
    return actual


def validate_checkout(arm, dry_run_origin_head):
    head = git("rev-parse", "HEAD")
    if git("symbolic-ref", "--short", "HEAD") != c.BRANCH:
        raise SystemExit("wrong branch")
    auth_path = HERE / f"{arm}-authorization.json"
    request_path = HERE / f"{arm}-request.json"
    auth_bytes = committed_bytes(head, auth_path)
    request_bytes = committed_bytes(head, request_path)
    auth = c.json_object(auth_bytes, "authorization", canonical=True)
    request = c.json_object(request_bytes, "request", canonical=True)
    core = auth.get("verified_core_commit", "")
    if not c.HEX40.fullmatch(core or "") or subprocess.run(["git", "merge-base", "--is-ancestor", core, head], cwd=ROOT).returncode:
        raise SystemExit("verified core is not an ancestor")
    allowed = {str(HERE.relative_to(ROOT) / f"{name}-{suffix}.json") for name in c.ARMS for suffix in ("authorization", "request")}
    if set(git("diff", "--name-only", core, head).splitlines()) != allowed:
        raise SystemExit("finalization path set mismatch")
    if git("diff", "--name-only", "HEAD") or git("diff", "--cached", "--name-only") or git("ls-files", "--others", "--exclude-standard"):
        raise SystemExit("checkout is not clean")
    if dry_run_origin_head is None:
        fields = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{c.BRANCH}"], cwd=ROOT, text=True).split()
        if fields != [head, f"refs/heads/{c.BRANCH}"]:
            raise SystemExit("origin branch not aligned to finalization HEAD")
    elif dry_run_origin_head != head:
        raise SystemExit("dry-run origin HEAD mismatch")
    authorization = c.validate_authorization(auth, request, arm, core)
    request_sha = authorization["request_sha256"]
    for relative, expected in request["source_sha256"].items():
        core_blob = subprocess.check_output(["git", "show", f"{core}:{relative}"], cwd=ROOT)
        head_blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if core_blob != head_blob or hashlib.sha256(core_blob).hexdigest() != expected:
            raise SystemExit(f"protected source mismatch: {relative}")
    return head, core, auth_bytes, request, request_sha


def validate_concrete(value, arm):
    spec = c.ARMS[arm]
    required = {
        "schema", "arm", "host", "expected_hostname", "remote", "expected_commit", "origin", "label",
        "purpose", "min_free_bytes", "fingerprint_paths", "expected_sha256", "environment", "command",
        "runner_owns_manifest", "non_overwrite", "verified_core_commit", "finalization_head",
        "request_path", "request_sha256", "authorization_path", "authorization_sha256",
    }
    if type(value) is not dict or set(value) != required or value["schema"] != "fresh-start-request-v2":
        raise SystemExit("concrete request schema mismatch")
    if (value["arm"] != arm or value["host"] != spec["host"] or value["expected_hostname"] != spec["hostname"]
            or value["remote"] != spec["remote"] or value["label"] != spec["label"]
            or value["expected_commit"] != value["finalization_head"]
            or value["min_free_bytes"] != c.MIN_FREE_BYTES or value["environment"] != c.ENVIRONMENT
            or value["command"] != c.expected_command(arm) or value["runner_owns_manifest"] is not False
            or value["non_overwrite"] is not True):
        raise SystemExit("concrete request identity mismatch")
    if value["fingerprint_paths"] != sorted(value["expected_sha256"]):
        raise SystemExit("concrete fingerprint map mismatch")


def durable(path, value):
    path = path.absolute()
    if path.parent != Path("/tmp") or path.name in {"", ".", ".."}:
        raise SystemExit("concrete request path rejected")
    parent = os.lstat(path.parent)
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise SystemExit("unsafe concrete request parent")
    data = c.canonical_json(value)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            offset = 0
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written <= 0:
                    raise SystemExit("short concrete request write")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(directory)
    finally:
        os.close(directory)
    if aio.regular_bytes(path) != data:
        raise SystemExit("concrete request persistence mismatch")
    return aio.digest(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=c.ARMS, required=True)
    parser.add_argument("--concrete-request", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-origin-head")
    args = parser.parse_args()
    if bool(args.dry_run_origin_head) != args.dry_run:
        raise SystemExit("dry-run origin override required exactly with --dry-run")
    head, core, auth_bytes, request, request_sha = validate_checkout(args.arm, args.dry_run_origin_head)
    authorization_sha = aio.digest(auth_bytes)
    concrete = c.build_concrete(request, authorization_sha, request_sha, head)
    validate_concrete(concrete, args.arm)
    expected = Path("/tmp") / f"{c.ARMS[args.arm]['label']}.request.json"
    target = args.concrete_request.absolute()
    if not args.dry_run and target != expected:
        raise SystemExit("non-dry-run concrete request path mismatch")
    concrete_sha = durable(target, concrete)
    print(json.dumps({
        "schema": "track-a-confirmation-launch-adapter-v1", "arm": args.arm, "verified_core_commit": core,
        "finalization_head": head, "authorization_sha256": authorization_sha,
        "request_sha256": request_sha, "concrete_request_sha256": concrete_sha,
        "concrete_request_path": str(target), "dry_run": args.dry_run,
    }, sort_keys=True))
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "research/fresh_start_launch.py"), str(target)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
