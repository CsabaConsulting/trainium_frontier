"""Exact local execution-root gate for trn21 r10 read-only timing.

This entrypoint cannot authorize contact by itself. The collector still needs
its separate independent final review and Supervisor one-attempt assignment.
"""

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys


STAGE_ROOT = Path("/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root")
STAGE_HEAD = "7205abf5469edc616c4e4b58caee27bc1957bbd4"
STAGE_FILES = {
    "train.py": "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py":
        "6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f",
    "research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/remote_incremental.py":
        "47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b",
    "research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/REQUEST.HOLD.json":
        "4c48243b7a5bee3f5aa6333e8ec5e81e5d7d434d926cbce18af907bc5ddfdd7c",
}
GIT_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}


def _git(root, *args):
    result = subprocess.run(["/usr/bin/git", "-C", str(root), *args],
                            env=GIT_ENV, capture_output=True, timeout=10,
                            check=False)
    if result.returncode or result.stderr:
        raise RuntimeError("local execution-root Git check failed")
    return result.stdout.strip()


def _regular_sha256(path):
    named = os.lstat(path)
    if (path != path.resolve(strict=True) or
            not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
            named.st_nlink != 1 or named.st_size > 4 * 1024 * 1024):
        raise RuntimeError("local execution-root source metadata changed")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(named, field) for field in fields):
            raise RuntimeError("local execution-root source changed during open")
        digest = hashlib.sha256()
        count = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > 4 * 1024 * 1024:
                raise RuntimeError("local execution-root source exceeded byte bound")
            digest.update(block)
        after, named_after = os.fstat(fd), os.lstat(path)
        if (count != opened.st_size or
                any(getattr(other, field) != getattr(opened, field)
                    for other in (after, named_after) for field in fields)):
            raise RuntimeError("local execution-root source changed during read")
        return digest.hexdigest()
    finally:
        os.close(fd)


def check_execution_root(root=STAGE_ROOT, expected_head=STAGE_HEAD,
                         expected_files=STAGE_FILES):
    """Require the exact unaliased, clean c5c stage-r3 worktree before contact."""
    if (not isinstance(root, Path) or not root.is_absolute() or
            root != root.resolve(strict=True) or
            not stat.S_ISDIR(os.lstat(root).st_mode)):
        raise RuntimeError("local execution-root path is not exact and real")
    if Path.cwd() != root:
        raise RuntimeError("timing invocation must run from exact execution root")
    for suffix in ("research/agentic/STOP", "research/v2/STOP"):
        if os.path.lexists(root / suffix):
            raise RuntimeError("execution-root STOP present")
    if _git(root, "rev-parse", "--show-toplevel") != os.fsencode(root):
        raise RuntimeError("local execution-root Git top-level changed")
    if _git(root, "rev-parse", "HEAD") != expected_head.encode("ascii"):
        raise RuntimeError("local execution-root HEAD changed")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("local execution-root is dirty")
    for relative, expected in expected_files.items():
        if _regular_sha256(root / relative) != expected:
            raise RuntimeError("local execution-root source bytes changed: " + relative)
    if (_git(root, "rev-parse", "HEAD") != expected_head.encode("ascii") or
            _git(root, "status", "--porcelain=v1", "--untracked-files=all")):
        raise RuntimeError("local execution-root changed during pin check")
    for suffix in ("research/agentic/STOP", "research/v2/STOP"):
        if os.path.lexists(root / suffix):
            raise RuntimeError("execution-root STOP appeared during pin check")
    return {"execution_root": str(root), "head": expected_head,
            "source_sha256": dict(expected_files), "stops_absent": True,
            "clean": True}


def main():
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python is forbidden")
    if len(sys.argv) != 1:
        raise RuntimeError("no alternate timing arguments")
    check_execution_root()
    import collect_readonly as collector
    if collector.EXECUTION_ROOT != STAGE_ROOT or collector.STAGE_COMMIT != STAGE_HEAD:
        raise RuntimeError("collector execution-root pin changed")
    destination = collector.capture()
    print(str(destination))
    if collector._recorded_status(destination) != "READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
