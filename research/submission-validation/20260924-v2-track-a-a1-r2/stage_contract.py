#!/usr/bin/env python3
"""Fail closed unless an r2 bundle checkout is on its exact symbolic branch and tree."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=check,
    )


def validate_symbolic_checkout(repo, expected_branch, expected_head):
    repo = Path(repo).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise SystemExit("expected HEAD is not an exact commit literal")
    if git(repo, "cat-file", "-e", f"{expected_head}^{{commit}}", check=False).returncode:
        raise SystemExit("expected HEAD does not resolve to a commit")
    symbolic = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if symbolic.returncode or symbolic.stdout.strip() != expected_branch:
        raise SystemExit("detached or wrong symbolic branch")
    if git(repo, "rev-parse", "HEAD").stdout.strip() != expected_head:
        raise SystemExit("symbolic branch is not at expected HEAD")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout:
        raise SystemExit("staged checkout is not clean")
    return {"branch": expected_branch, "head": expected_head, "clean": True}


def validate_stage(repo, expected_head, request_path):
    repo = Path(repo).resolve()
    observed = validate_symbolic_checkout(repo, c.BRANCH, expected_head)
    request_path = Path(request_path).resolve()
    expected_path = repo / HERE.relative_to(ROOT) / f"{c.RETRY_ARM}-request.json"
    if request_path != expected_path:
        raise SystemExit("request path is not canonical")
    request_bytes = aio.regular_bytes(request_path)
    request = c.json_object(request_bytes, "r2 request", canonical=True)
    core = request.get("verified_core_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", core):
        raise SystemExit("verified core is malformed")
    if git(repo, "merge-base", "--is-ancestor", core, expected_head, check=False).returncode:
        raise SystemExit("verified core is not an ancestor of expected HEAD")
    c.validate_request(request, c.RETRY_ARM, core)
    allowed = sorted(str(HERE.relative_to(ROOT) / f"{c.RETRY_ARM}-{suffix}.json")
                     for suffix in ("authorization", "request"))
    changed = git(repo, "diff", "--name-only", core, expected_head).stdout.splitlines()
    if changed != allowed:
        raise SystemExit("finalization path set mismatch")
    for relative, expected_sha in request["source_sha256"].items():
        working = aio.regular_bytes(repo / relative)
        committed = subprocess.check_output(["git", "show", f"{expected_head}:{relative}"], cwd=repo)
        if working != committed or hashlib.sha256(working).hexdigest() != expected_sha:
            raise SystemExit(f"staged source/tree mismatch: {relative}")
    observed.update(request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                    verified_core_commit=core, protected_paths=len(request["source_sha256"]))
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    print(c.canonical_json(validate_stage(args.repo, args.expected_head, args.request)).decode(), end="")


if __name__ == "__main__":
    main()
