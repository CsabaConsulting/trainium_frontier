"""Local execution-root gate for the reviewed read-only trn22 timing collector.

This file cannot authorize a host attempt. The collector still requires its
distinct independent final review and Supervisor assignment, both absent.
"""

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys


STAGE_ROOT = Path("/home/poojith-devan/Desktop/Trainium-trn22-r10-stage-r3-active-root")
STAGE_HEAD = "b3ec7406f0e7a3df6848698876e2106305dd283f"
STAGE_FILES = {
    "train.py": "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py":
        "a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda",
    "research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/remote_incremental.py":
        "da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f",
    "research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/REQUEST.HOLD.json":
        "48e3ad5d0bb1d077dbf8304e3ddfe7e3263892e522dc277f32812eea5c8e111a",
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


def check_execution_root(root=STAGE_ROOT, expected_head=STAGE_HEAD,
                         expected_files=STAGE_FILES):
    """Require one exact, clean, unaliased c5c stage worktree before contact."""
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
        path = root / relative
        if (not stat.S_ISREG(os.lstat(path).st_mode) or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise RuntimeError("local execution-root source bytes changed")
    if _git(root, "rev-parse", "HEAD") != expected_head.encode("ascii") or \
            _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("local execution-root changed during pin check")


def main():
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python is forbidden")
    if len(sys.argv) != 1:
        raise RuntimeError("no alternate timing arguments")
    check_execution_root()
    import operator_timing_readonly as collector
    collector.main()


if __name__ == "__main__":
    main()
