"""Archive declared evidence to git. Plain add/commit; optional plain push. Never force, never rewrite."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import config as C


class ArchiveError(RuntimeError):
    pass


class GitArchiver:
    def __init__(self, root: Path = C.ROOT, runner=None, push: bool = False, remote: str = "origin"):
        self.root = root
        self.push = push
        self.remote = remote
        self._run = runner or self._subprocess

    def _subprocess(self, argv: list[str]):
        p = subprocess.run(["git", "-C", str(self.root), *argv], text=True, capture_output=True, check=False)
        return p.returncode, p.stdout, p.stderr

    def _check_paths(self, paths: list[Path]) -> list[str]:
        rel: list[str] = []
        base = (self.root / "research" / "agentic").resolve()
        for p in paths:
            rp = p.resolve()
            try:
                rp.relative_to(base)
            except ValueError as exc:
                raise ArchiveError(f"refusing to archive outside research/agentic/: {p}") from exc
            if rp.name.endswith(".pem") or rp.name == ".env":
                raise ArchiveError("refusing to archive secrets")
            rel.append(str(rp.relative_to(self.root.resolve())))
        return rel

    def commit(self, paths: list[Path], message: str) -> str | None:
        rel = self._check_paths(paths)
        for flag in ("-f", "--force", "--amend", "--hard"):
            if flag in message.split():
                raise ArchiveError("suspicious commit message")
        rc, _, err = self._run(["add", "--", *rel])
        if rc:
            raise ArchiveError(f"git add failed: {err.strip()}")
        rc, out, _ = self._run(["diff", "--cached", "--quiet", "--", *rel])
        if rc == 0:
            return None  # nothing staged
        rc, _, err = self._run(["commit", "--only", "-m", message, "--", *rel])
        if rc:
            raise ArchiveError(f"git commit failed: {err.strip()}")
        rc, out, _ = self._run(["rev-parse", "HEAD"])
        sha = out.strip() if rc == 0 else None
        if self.push:
            rc, _, err = self._run(["push", self.remote, "HEAD"])
            if rc:
                raise ArchiveError(f"git push failed (no retry, no force): {err.strip()}")
        return sha

    def verify_commit(self, commit: str | None, required_files: list[Path]) -> dict:
        """Prove every required local byte is in the commit and, when enabled, on the remote."""
        if not commit:
            raise ArchiveError("archive verification requires a new commit")
        local_hashes: dict[str, str] = {}
        for path in required_files:
            if path.is_symlink() or not path.is_file():
                raise ArchiveError(f"required archive file missing/non-regular: {path}")
            rel = str(path.resolve().relative_to(self.root.resolve()))
            rc, expected, err = self._run(["hash-object", rel])
            if rc or not expected.strip():
                raise ArchiveError(f"cannot hash archived file {rel}: {err.strip()}")
            rc, actual, err = self._run(["rev-parse", f"{commit}:{rel}"])
            if rc:
                raise ArchiveError(f"archive commit lacks {rel}: {err.strip()}")
            if actual.strip() != expected.strip():
                raise ArchiveError(f"archive commit bytes differ for {rel}")
            local_hashes[rel] = expected.strip()
        remote_verified = False
        if self.push:
            rc, branch, err = self._run(["symbolic-ref", "--short", "HEAD"])
            if rc or not branch.strip():
                raise ArchiveError(f"cannot identify archive branch: {err.strip()}")
            rc, out, err = self._run(["ls-remote", self.remote, f"refs/heads/{branch.strip()}"])
            if rc or not out.strip() or out.split()[0] != commit:
                raise ArchiveError(f"configured remote does not contain archive commit: {err.strip()}")
            remote_verified = True
        return {"commit": commit, "local_verified": True, "remote_verified": remote_verified,
                "files": local_hashes}
