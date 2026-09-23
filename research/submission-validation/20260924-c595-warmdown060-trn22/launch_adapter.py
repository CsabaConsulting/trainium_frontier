#!/usr/bin/env python3
"""Materialize and optionally dispatch the warmdown request without Git self-reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BRANCH = "experiment/20260924-c595-warmdown060-trn22"
DRIVER = HERE / "retune_job.py"
AUTHORIZATION = HERE / "authorization.json"
TEMPLATE = HERE / "final-request.json"
LAUNCHER = ROOT / "research/fresh_start_launch.py"


def load_driver():
    source = regular_bytes(DRIVER)
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{DRIVER.relative_to(ROOT)}"], cwd=ROOT
    )
    if source != committed:
        raise SystemExit("driver differs from current HEAD before adapter import")
    module = types.ModuleType("warmdown060_retune_job_for_adapter")
    module.__file__ = str(DRIVER)
    exec(compile(source, str(DRIVER), "exec", dont_inherit=True), module.__dict__)
    return module


def git(*argv: str) -> str:
    return subprocess.check_output(["git", *argv], cwd=ROOT, text=True).strip()


def regular_bytes(path: Path) -> bytes:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"not a regular non-symlink file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
        if len(data) != opened.st_size:
            raise SystemExit(f"file changed while reading: {path}")
        return data
    finally:
        os.close(fd)


def committed_bytes(head: str, path: Path) -> bytes:
    relative = str(path.relative_to(ROOT))
    actual = regular_bytes(path)
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    if actual != committed:
        raise SystemExit(f"working file differs from HEAD: {relative}")
    return actual


def validate_standard_request(value: dict, driver) -> None:
    required = {
        "schema", "host", "expected_hostname", "remote", "expected_commit", "origin", "label",
        "purpose", "fingerprint_paths", "expected_sha256", "environment", "command",
        "runner_owns_manifest", "non_overwrite", "verified_core_commit", "finalization_head",
        "template_path", "template_sha256", "template_content", "authorization_path",
        "authorization_sha256",
    }
    if set(value) != required or value["schema"] != "fresh-start-request-v2":
        raise SystemExit("concrete request schema mismatch")
    if value["host"] not in {"trn21", "trn22"} or value["host"] != driver.HOST:
        raise SystemExit("concrete request host mismatch")
    if not re.fullmatch(r"fs-[A-Za-z0-9_.-]+", value["label"] or ""):
        raise SystemExit("concrete request label mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", value["expected_commit"] or ""):
        raise SystemExit("concrete expected commit malformed")
    remote = Path(value["remote"])
    if remote.parent != Path("/home/ubuntu") or not remote.name.startswith("trainium_freshstart_"):
        raise SystemExit("concrete remote path rejected")
    if value["command"][:1] != ["timeout"] or not all(isinstance(item, str) for item in value["command"]):
        raise SystemExit("bounded command missing")
    if value["environment"].get("NEURON_LOGICAL_NC_CONFIG") not in {"1", "2"}:
        raise SystemExit("LNC configuration missing")
    if "--optlevel=1" not in value["environment"].get("NEURON_CC_FLAGS", ""):
        raise SystemExit("compiler flags missing")
    if value["fingerprint_paths"] != sorted(value["expected_sha256"]):
        raise SystemExit("fingerprint map mismatch")
    if value["expected_commit"] != value["finalization_head"]:
        raise SystemExit("concrete finalization mismatch")
    rebuilt = driver.build_concrete_request(
        value["template_content"], value["authorization_sha256"],
        value["template_sha256"], value["finalization_head"],
    )
    if value != rebuilt:
        raise SystemExit("concrete request differs from immutable template projection")


def validate_checkout(driver, dry_run_origin_head: str | None) -> tuple[str, str, dict, str, dict, str]:
    head = git("rev-parse", "HEAD")
    parents = git("rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2:
        raise SystemExit("finalization HEAD must have one parent")
    core = parents[1]
    if git("symbolic-ref", "--short", "HEAD") != BRANCH:
        raise SystemExit("wrong branch")
    if set(git("diff", "--name-only", core, head).splitlines()) != {driver.AUTH_REL, driver.REQUEST_REL}:
        raise SystemExit("finalization child path mismatch")
    if git("diff", "--name-only", "HEAD") or git("diff", "--cached", "--name-only"):
        raise SystemExit("tracked checkout is dirty")
    if git("ls-files", "--others", "--exclude-standard"):
        raise SystemExit("checkout has untracked files")
    if dry_run_origin_head is None:
        fields = subprocess.check_output(
            ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], cwd=ROOT, text=True
        ).split()
        if fields != [head, f"refs/heads/{BRANCH}"]:
            raise SystemExit("origin branch is not aligned to finalization HEAD")
    elif dry_run_origin_head != head:
        raise SystemExit("dry-run origin head mismatch")
    auth_bytes = committed_bytes(head, AUTHORIZATION)
    template_bytes = committed_bytes(head, TEMPLATE)
    try:
        auth = json.loads(auth_bytes)
        template = json.loads(template_bytes)
    except json.JSONDecodeError as error:
        raise SystemExit("authorization/template JSON invalid") from error
    if auth_bytes != driver.canonical_json(auth) or template_bytes != driver.canonical_json(template):
        raise SystemExit("authorization/template is not canonical JSON")
    template_sha = driver.validate_authorization_request(auth, template, core)
    if hashlib.sha256(template_bytes).hexdigest() != template_sha:
        raise SystemExit("template hash mismatch")
    for relative, expected in template["source_sha256"].items():
        core_blob = subprocess.check_output(["git", "show", f"{core}:{relative}"], cwd=ROOT)
        head_blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if core_blob != head_blob or hashlib.sha256(core_blob).hexdigest() != expected:
            raise SystemExit(f"core source identity mismatch: {relative}")
    return head, core, auth, hashlib.sha256(auth_bytes).hexdigest(), template, template_sha


def durable_concrete(path: Path, value: dict, driver) -> str:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise SystemExit("concrete request path must be explicit and absolute")
    parent_info = os.lstat(path.parent)
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise SystemExit("unsafe concrete request parent")
    data = driver.canonical_json(value)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            offset = 0
            while offset < len(data):
                count = os.write(fd, data[offset:])
                if count <= 0:
                    raise SystemExit("short concrete request write")
                offset += count
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(directory)
    finally:
        os.close(directory)
    if regular_bytes(path) != data:
        raise SystemExit("concrete request persistence mismatch")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concrete-request", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-origin-head", help="Test-only origin observation; valid only with --dry-run")
    args = parser.parse_args()
    if bool(args.dry_run_origin_head) != args.dry_run:
        raise SystemExit("dry-run origin override requires --dry-run and is required for dry-run")
    driver = load_driver()
    head, core, _auth, authorization_sha, template, template_sha = validate_checkout(
        driver, args.dry_run_origin_head
    )
    concrete = driver.build_concrete_request(template, authorization_sha, template_sha, head)
    validate_standard_request(concrete, driver)
    concrete_path = args.concrete_request.resolve(strict=False)
    concrete_sha = durable_concrete(concrete_path, concrete, driver)
    receipt = {
        "schema": "c595-warmdown060-launch-adapter-v1",
        "verified_core_commit": core,
        "finalization_head": head,
        "adapter_sha256": hashlib.sha256(regular_bytes(Path(__file__))).hexdigest(),
        "authorization_sha256": authorization_sha,
        "template_sha256": template_sha,
        "concrete_request_path": str(concrete_path),
        "concrete_request_sha256": concrete_sha,
        "dry_run": args.dry_run,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if not args.dry_run:
        subprocess.run([sys.executable, str(LAUNCHER), str(concrete_path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
