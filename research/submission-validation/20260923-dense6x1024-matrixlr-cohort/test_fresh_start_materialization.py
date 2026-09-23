#!/usr/bin/env python3
"""Exercise the exact remote authentication boundary without SSH or dispatch."""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
import fresh_start_launch  # noqa: E402

MARKER = "# CHECKOUT_AUTHENTICATION_BOUNDARY"
boundary, marker, _remainder = fresh_start_launch.REMOTE_SCRIPT.partition(MARKER)
assert marker
boundary += "\nprint('BOUNDARY_PASS')\n"
CLOSURE = [
    "research/fresh_start_launch.py",
    "research/fresh_start_sync.py",
    "research/fresh_start_guard.py",
    "research/fresh_start_host.py",
    "research/fresh_start_backup.py",
]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def request(repo):
    return {
        "remote": str(repo),
        "expected_commit": git(repo, "rev-parse", "HEAD"),
        "fingerprint_paths": CLOSURE,
        "expected_sha256": {
            path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
            for path in CLOSURE
        },
    }


def check(repo, should_pass):
    result = subprocess.run(
        [sys.executable, "-c", boundary], input=json.dumps(request(repo)),
        text=True, capture_output=True,
    )
    if should_pass:
        assert result.returncode == 0 and result.stdout.strip() == "BOUNDARY_PASS", result.stderr
    else:
        assert result.returncode != 0 and "sparse checkout" in result.stderr.lower(), result.stderr


with tempfile.TemporaryDirectory(prefix="fresh-start-materialization-") as temp:
    temp = pathlib.Path(temp)
    complete = temp / "complete"
    sparse = temp / "sparse"
    subprocess.run(["git", "clone", "--quiet", "--no-local", str(ROOT), str(complete)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", git(ROOT, "rev-parse", "HEAD")],
                   cwd=complete, check=True)
    check(complete, True)

    subprocess.run(["git", "clone", "--quiet", "--no-local", str(ROOT), str(sparse)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", git(ROOT, "rev-parse", "HEAD")],
                   cwd=sparse, check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=sparse, check=True)
    patterns = "/*\n!/research/fresh_start_sync.py\n"
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "/*", "!/research/fresh_start_sync.py"],
                   cwd=sparse, check=True)
    assert not (sparse / "research/fresh_start_sync.py").exists()
    # Build expected hashes from the exact commit, not the incomplete worktree.
    sparse_request = request(complete)
    sparse_request["remote"] = str(sparse)
    result = subprocess.run([sys.executable, "-c", boundary], input=json.dumps(sparse_request),
                            text=True, capture_output=True)
    assert result.returncode != 0 and "sparse checkout" in result.stderr.lower(), result.stderr

print("fresh_start complete/sparse materialization boundary: PASS (no dispatch)")
