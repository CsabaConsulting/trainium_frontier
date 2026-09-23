"""Typed remote operator. The only path from this controller to a Trn2 host.

Every remote invocation is an argv list checked by :func:`guard_remote_argv` before it reaches
``ssh``. There is no ``run_shell``. The private key is referenced by path only; its mode is checked
with ``stat`` and its contents are never read.

Operations: ``inventory``, ``acquire_lock``/``release_lock``, ``stage``, ``launch``, ``poll``,
``collect``, and hash-verified ``prune_checkpoint``. Each returns a dataclass; each fails closed
with :class:`RemoteError`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from . import config as C
from .ledger import Ledger
from .manifest import RunManifest, assert_manifest_legal

LAUNCHER_SRC = Path(__file__).with_name("remote_launcher.py")
CACHE_CLEANUP_SRC = Path(__file__).with_name("remote_cache_cleanup.py")
BUSY_PROCESS_RE = re.compile(r"train\.py|phase_timer|reconcile|run_experiment\.py|neuronx-cc|torchrun")

ALLOWED_REMOTE_PROGRAMS = frozenset(
    {"git", "test", "mkdir", "rmdir", "rm", "cat", "tail", "sha256sum", "df", "ps", "fuser", "hostname",
     "stat", "ls", "true", C.REMOTE_PYTHON}
)


class RemoteError(RuntimeError):
    pass


class GuardError(RemoteError):
    pass


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Transport(Protocol):
    def run(self, host: str, argv: list[str], timeout: int = 60) -> CommandResult: ...
    def put(self, host: str, local: Path, remote: str, timeout: int = 300) -> CommandResult: ...
    def get(self, host: str, remote: str, local: Path, timeout: int = 300) -> CommandResult: ...


# ---- guards ------------------------------------------------------------------------------------


def check_key_path(path: Path = C.SSH_KEY_PATH) -> None:
    """Existence + mode 0600 via stat only. Never opens the file."""
    try:
        st = path.lstat()
    except FileNotFoundError as exc:
        raise RemoteError(f"SSH key missing at {path}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RemoteError("SSH key must be a regular file, not a symlink")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise RemoteError(f"SSH key mode must be 600, found {oct(stat.S_IMODE(st.st_mode))}")


def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip("/") + "/")


def guard_remote_argv(argv: list[str]) -> None:
    if not argv or not all(isinstance(a, str) and a for a in argv):
        raise GuardError("argv must be a non-empty list of non-empty strings")
    joined = " ".join(argv)
    for tok in C.REMOTE_FORBIDDEN_PATH_TOKENS:
        if tok in joined:
            raise GuardError(f"forbidden path token {tok!r}")
    for meta in ("|", ";", "&&", "||", ">", "<", "`", "$(", "\n", "\r"):
        if meta in joined:
            raise GuardError(f"shell metacharacter {meta!r}")
    if argv[0] not in ALLOWED_REMOTE_PROGRAMS:
        raise GuardError(f"program {argv[0]!r} not allowlisted")
    for a in argv[1:]:
        if a in C.REMOTE_FORBIDDEN_COMMAND_TOKENS:
            raise GuardError(f"forbidden token {a!r}")
    prog = argv[0]
    if prog == "git":
        _guard_git(argv)
    elif prog == C.REMOTE_PYTHON:
        launcher = (len(argv) == 3
                    and _under(argv[1], C.REMOTE_AUTOLOOP_DIR + "/launchers")
                    and argv[1].endswith("remote_launcher.py")
                    and _under(argv[2], C.REMOTE_AUTOLOOP_DIR + "/launchers"))
        cache_cleanup = (len(argv) == 3
                         and _under(argv[1], C.REMOTE_AUTOLOOP_DIR + "/cache-cleanup")
                         and argv[1].endswith("remote_cache_cleanup.py")
                         and _under(argv[2], C.REMOTE_AUTOLOOP_DIR + "/cache-cleanup")
                         and argv[2].endswith("cleanup.json"))
        if not launcher and not cache_cleanup:
            raise GuardError("remote python may only run the staged launcher or exact cache cleanup")
    elif prog in ("mkdir", "rmdir"):
        for a in argv[1:]:
            if a.startswith("-") and a != "-p":
                raise GuardError("mkdir/rmdir accept only -p")
            if not a.startswith("-") and not (_under(a, C.REMOTE_AUTOLOOP_DIR) or a == C.REMOTE_RUNS_DIR):
                raise GuardError(f"mkdir/rmdir outside autoloop scratch: {a}")
        if prog == "rmdir" and not all(_under(a, C.REMOTE_AUTOLOOP_DIR + "/locks") for a in argv[1:]):
            raise GuardError("rmdir only for lock directories")
    elif prog == "rm":
        if len(argv) != 3 or argv[1] != "--":
            raise GuardError("rm accepts exactly one path after --")
        path = argv[2]
        if ".." in path.split("/"):
            raise GuardError("rm path traversal forbidden")
        if not _under(path, C.REMOTE_RUNS_DIR) or "/out/" not in path or not path.endswith("/final.pt"):
            raise GuardError("rm may only prune a final.pt inside an autoloop run")
    elif prog == "fuser":
        if argv[1:] != ["/dev/neuron0"]:
            raise GuardError("fuser only on /dev/neuron0")
    elif prog in ("cat", "tail", "sha256sum", "stat", "ls", "test"):
        for a in argv[1:]:
            if not a.startswith("/"):
                continue  # option or literal (e.g. tail -n 40, stat -c %s)
            if not (_under(a, C.REMOTE_AUTOLOOP_DIR) or _under(a, C.REMOTE_RUNS_DIR) or _under(a, C.REMOTE_DATA_CACHE)
                    or a.startswith(C.REMOTE_HOME + "/workspace/native_venv")):
                raise GuardError(f"read outside allowed remote paths: {a}")


def _guard_git(argv: list[str]) -> None:
    sub = None
    if len(argv) >= 4 and argv[1] == "-C":
        target = argv[2]
        sub = argv[3]
        if not _under(target, C.REMOTE_RUNS_DIR):
            raise GuardError("git -C target must be a fresh autoloop run dir")
    elif len(argv) >= 2:
        sub = argv[1]
    if sub in C.REMOTE_GIT_FORBIDDEN_SUBCOMMANDS or sub not in C.REMOTE_GIT_ALLOWED_SUBCOMMANDS:
        raise GuardError(f"git subcommand {sub!r} not allowed remotely")
    if sub == "clone":
        # git clone --quiet <bundle> <run_dir>
        paths = [a for a in argv[2:] if not a.startswith("-")]
        if len(paths) != 2 or not _under(paths[0], C.REMOTE_AUTOLOOP_DIR + "/bundles") or not _under(paths[1], C.REMOTE_RUNS_DIR):
            raise GuardError("git clone only from the bundles dir into the runs dir")
    if sub == "checkout" and ("-f" in argv or "--force" in argv):
        raise GuardError("forced checkout forbidden")
    for a in argv:
        if a in ("-f", "--force", "--hard", "-fdx"):
            raise GuardError("forced git operation forbidden")


# ---- transport -----------------------------------------------------------------------------------


class SSHTransport:
    def __init__(self, key_path: Path = C.SSH_KEY_PATH, known_hosts_policy: str = "accept-new"):
        check_key_path(key_path)
        self.key_path = key_path
        self.policy = known_hosts_policy

    def _base(self, prog: str) -> list[str]:
        return [
            prog, "-i", str(self.key_path), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", f"StrictHostKeyChecking={self.policy}", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
        ]

    def run(self, host: str, argv: list[str], timeout: int = 60) -> CommandResult:
        if host not in C.PINNED_HOSTS.values():
            raise GuardError(f"host {host!r} is not pinned")
        guard_remote_argv(argv)
        cmd = self._base("ssh") + [host, "--", shlex.join(argv)]
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(argv, 124, exc.stdout or "", f"timeout after {timeout}s")
        return CommandResult(argv, p.returncode, p.stdout, p.stderr)

    def put(self, host: str, local: Path, remote: str, timeout: int = 300) -> CommandResult:
        if host not in C.PINNED_HOSTS.values():
            raise GuardError("host not pinned")
        if not (_under(remote, C.REMOTE_AUTOLOOP_DIR)):
            raise GuardError("put only into the autoloop scratch dir")
        cmd = self._base("scp") + ["-q", str(local), f"{host}:{remote}"]
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return CommandResult(["scp-put", remote], p.returncode, p.stdout, p.stderr)

    def get(self, host: str, remote: str, local: Path, timeout: int = 300) -> CommandResult:
        if host not in C.PINNED_HOSTS.values():
            raise GuardError("host not pinned")
        if not (_under(remote, C.REMOTE_AUTOLOOP_DIR) or _under(remote, C.REMOTE_RUNS_DIR)):
            raise GuardError("get only from autoloop dirs")
        for tok in C.REMOTE_FORBIDDEN_PATH_TOKENS:
            if tok in remote:
                raise GuardError("forbidden path token")
        local.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._base("scp") + ["-q", f"{host}:{remote}", str(local)]
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return CommandResult(["scp-get", remote], p.returncode, p.stdout, p.stderr)


# ---- local git (bundle creation) ---------------------------------------------------------------


class LocalGit:
    """Thin wrapper so tests can inject a fake. Only read operations plus temp-ref create/delete."""

    def __init__(self, root: Path = C.ROOT, runner=None, show_runner=None):
        self.root = root
        self._run = runner or self._subprocess
        self._show = show_runner or self._subprocess_show

    def _subprocess(self, argv: list[str]) -> CommandResult:
        p = subprocess.run(["git", "-C", str(self.root), *argv], text=True, capture_output=True, check=False)
        return CommandResult(argv, p.returncode, p.stdout, p.stderr)

    def _subprocess_show(self, commit: str, path: str) -> bytes | None:
        p = subprocess.run(["git", "-C", str(self.root), "show", f"{commit}:{path}"], capture_output=True, check=False)
        return None if p.returncode else p.stdout

    def commit_exists(self, commit: str) -> bool:
        return self._run(["cat-file", "-e", f"{commit}^{{commit}}"]).ok

    def file_at_commit(self, commit: str, path: str) -> bytes:
        data = self._show(commit, path)
        if data is None:
            raise RemoteError(f"git show {commit}:{path} failed")
        return data

    def parent_of(self, commit: str) -> str:
        result = self._run(["rev-parse", f"{commit}^"])
        if not result.ok or not re.fullmatch(r"[0-9a-f]{40}", result.stdout.strip()):
            raise RemoteError(f"cannot resolve parent of {commit}")
        return result.stdout.strip()

    def changed_paths(self, parent: str, commit: str) -> list[str]:
        result = self._run(["diff-tree", "--no-commit-id", "--name-only", "-r", parent, commit])
        if not result.ok:
            raise RemoteError(f"cannot inspect changed paths for {commit}")
        return [line for line in result.stdout.splitlines() if line]

    def make_bundle(self, commit: str, out: Path) -> Path:
        ref = f"refs/heads/autoloop-stage-{commit[:12]}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self._run(["update-ref", ref, commit]).ok:
            raise RemoteError("could not create temporary staging ref")
        try:
            r = self._run(["bundle", "create", str(out), ref])
            if not r.ok:
                raise RemoteError(f"git bundle create failed: {r.stderr.strip()}")
            v = self._run(["bundle", "verify", str(out)])
            if not v.ok:
                raise RemoteError(f"git bundle verify failed: {v.stderr.strip()}")
        finally:
            self._run(["update-ref", "-d", ref])
        return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- results -------------------------------------------------------------------------------------


@dataclass
class Inventory:
    box: str
    host: str
    hostname: str
    busy_processes: list[str]
    device_holders: str
    free_gib: float
    data_cache_present: bool
    remote_python_present: bool
    captured_at: str
    findings: list[str] = field(default_factory=list)

    @property
    def clean_and_idle(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["clean_and_idle"] = self.clean_and_idle
        return d


@dataclass
class StagedRun:
    box: str
    run_dir: str
    commit: str
    head: str
    train_py_sha256: str
    porcelain_empty: bool
    bundle_sha256: str
    staged_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LaunchReceipt:
    manifest_id: str
    pid: int
    launcher_dir: str
    launched_at: str
    launcher_sha256: str
    pre_launch_inventory: dict
    pre_launch_head: str
    process_group: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PollStatus:
    pid_alive: bool
    done: bool
    returncode: int | None
    log_tail: str
    polled_at: str
    group_members: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Collected:
    files: dict[str, dict]
    ledger_row: dict | None
    checkpoint_present: bool
    checkpoint_sha256: str | None
    checkpoint_bytes: int | None
    post_run_porcelain: list[str]
    post_run_dirt_explained: bool
    collected_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---- operator ------------------------------------------------------------------------------------


class RemoteOperator:
    def __init__(self, transport: Transport, local_git: LocalGit, cfg: C.LoopConfig, controller_id: str, clock=None):
        self.t = transport
        self.git = local_git
        self.cfg = cfg
        self.controller_id = controller_id
        self._now = clock or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def _host(self, box: str) -> str:
        if box not in C.PINNED_HOSTS:
            raise RemoteError(f"box {box!r} not pinned")
        return C.PINNED_HOSTS[box]

    def _run(self, box: str, argv: list[str], timeout: int = 60, must_ok: bool = True) -> CommandResult:
        r = self.t.run(self._host(box), argv, timeout=timeout)
        if must_ok and not r.ok:
            raise RemoteError(f"remote {argv[0]} failed rc={r.returncode}: {r.stderr.strip()[:500]}")
        return r

    # inventory --------------------------------------------------------------------------------

    def inventory(self, box: str) -> Inventory:
        findings: list[str] = []
        hostname = self._run(box, ["hostname"]).stdout.strip()
        ps = self._run(box, ["ps", "-eo", "pid,args"]).stdout
        busy = [ln.strip() for ln in ps.splitlines()[1:] if BUSY_PROCESS_RE.search(ln) and "remote_launcher" not in ln]
        if busy:
            findings.append(f"{len(busy)} training-related process(es) present")
        fuser = self._run(box, ["fuser", "/dev/neuron0"], must_ok=False)
        holders = (fuser.stdout + fuser.stderr).strip()
        if fuser.returncode not in (0, 1):
            findings.append("fuser unavailable; cannot prove device is free")
        elif holders and re.search(r"\d", holders.replace("/dev/neuron0:", "")):
            findings.append("a process holds /dev/neuron0")
        df = self._run(box, ["df", "-Pk", "/"]).stdout.splitlines()
        free_gib = 0.0
        try:
            free_kb = int(df[1].split()[3])
            free_gib = free_kb / (1024 ** 2)
        except (IndexError, ValueError):
            findings.append("could not parse df output")
        if free_gib < self.cfg.caps.min_free_gib:
            findings.append(f"free disk {free_gib:.1f} GiB below {self.cfg.caps.min_free_gib} GiB")
        cache_ok = self._run(box, ["test", "-d", C.REMOTE_DATA_CACHE], must_ok=False).ok
        if not cache_ok:
            findings.append("competition data cache dir missing")
        py_ok = self._run(box, ["test", "-x", C.REMOTE_PYTHON], must_ok=False).ok
        if not py_ok:
            findings.append("pinned native python missing")
        return Inventory(box, self._host(box), hostname, busy, holders, round(free_gib, 2), cache_ok, py_ok, self._now(), findings)

    # locks ------------------------------------------------------------------------------------

    def _lock_dir(self, box: str) -> str:
        return f"{C.REMOTE_AUTOLOOP_DIR}/locks/{box}.lock"

    def acquire_lock(self, box: str) -> bool:
        self._run(box, ["mkdir", "-p", f"{C.REMOTE_AUTOLOOP_DIR}/locks"])
        r = self._run(box, ["mkdir", self._lock_dir(box)], must_ok=False)
        if not r.ok:
            return False
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"owner": self.controller_id, "at": self._now()}, fh)
            tmp = Path(fh.name)
        try:
            self.t.put(self._host(box), tmp, f"{C.REMOTE_AUTOLOOP_DIR}/locks/{box}.owner")
        finally:
            tmp.unlink(missing_ok=True)
        return True

    def lock_owner(self, box: str) -> str | None:
        r = self._run(box, ["cat", f"{C.REMOTE_AUTOLOOP_DIR}/locks/{box}.owner"], must_ok=False)
        if not r.ok:
            return None
        try:
            return json.loads(r.stdout).get("owner")
        except json.JSONDecodeError:
            return "unparsable"

    def release_lock(self, box: str) -> None:
        owner = self.lock_owner(box)
        if owner != self.controller_id:
            raise RemoteError(f"refusing to release lock owned by {owner!r}")
        self._run(box, ["rmdir", self._lock_dir(box)], must_ok=False)

    # staging ----------------------------------------------------------------------------------

    def stage(self, box: str, proposal_id: str, label: str, commit: str = C.AUTHORITATIVE_COMMIT) -> StagedRun:
        if not self.git.commit_exists(commit):
            raise RemoteError("requested verified commit is not present in the local repository")
        local_train = self.git.file_at_commit(commit, "train.py")
        expected_train_sha = sha256_bytes(local_train)

        bundle = C.BUNDLE_DIR / f"{commit[:12]}.bundle"
        if not bundle.is_file():
            self.git.make_bundle(commit, bundle)
        bundle_sha = sha256_file(bundle)

        run_dir = f"{C.REMOTE_RUNS_DIR}/{commit[:8]}--{proposal_id}--{label}"
        for tok in C.REMOTE_FORBIDDEN_PATH_TOKENS:
            if tok in run_dir:
                raise RemoteError("run dir name collides with a forbidden token")
        if self._run(box, ["test", "-e", run_dir], must_ok=False).ok:
            raise RemoteError(f"run dir already exists; a run dir is never reused: {run_dir}")

        self._run(box, ["mkdir", "-p", f"{C.REMOTE_AUTOLOOP_DIR}/bundles"])
        self._run(box, ["mkdir", "-p", C.REMOTE_RUNS_DIR])
        remote_bundle = f"{C.REMOTE_AUTOLOOP_DIR}/bundles/{commit[:12]}.bundle"
        remote_sha = self._remote_sha(box, remote_bundle)
        if remote_sha != bundle_sha:
            r = self.t.put(self._host(box), bundle, remote_bundle, timeout=900)
            if not r.ok:
                raise RemoteError(f"bundle upload failed: {r.stderr.strip()[:300]}")
            remote_sha = self._remote_sha(box, remote_bundle)
            if remote_sha != bundle_sha:
                raise RemoteError("bundle sha256 mismatch after upload")

        self._run(box, ["git", "clone", "--quiet", "--no-checkout", remote_bundle, run_dir], timeout=600)
        self._run(box, ["git", "-C", run_dir, "checkout", "--quiet", "--detach", commit], timeout=600)
        head = self._run(box, ["git", "-C", run_dir, "rev-parse", "HEAD"]).stdout.strip()
        if head != commit:
            raise RemoteError(f"staged HEAD {head} != {commit}")
        porcelain = self._run(box, ["git", "-C", run_dir, "status", "--porcelain"]).stdout.strip()
        if porcelain:
            raise RemoteError(f"staged tree not clean: {porcelain[:300]}")
        remote_train_sha = self._remote_sha(box, f"{run_dir}/train.py")
        if remote_train_sha != expected_train_sha:
            raise RemoteError("staged train.py hash differs from the authoritative commit")
        return StagedRun(box, run_dir, commit, head, remote_train_sha, True, bundle_sha, self._now())

    def _remote_sha(self, box: str, path: str) -> str | None:
        r = self._run(box, ["sha256sum", path], must_ok=False)
        if not r.ok:
            return None
        return r.stdout.split()[0] if r.stdout.split() else None

    # launch -----------------------------------------------------------------------------------

    def launch(self, manifest: RunManifest, staged: StagedRun) -> LaunchReceipt:
        assert_manifest_legal(manifest)
        if staged.run_dir != manifest.remote_run_dir or staged.box != manifest.box or staged.head != manifest.commit:
            raise RemoteError("manifest does not match the staged run directory")
        if (manifest.execution_mode == "exact-dense1024-cold-eval"
                and staged.train_py_sha256 != manifest.expected_train_sha256):
            raise RemoteError("staged train.py differs from the exact manifest hash")
        if self.lock_owner(manifest.box) != self.controller_id:
            raise RemoteError("box lock not held by this controller")
        if manifest.execution_mode == "exact-dense1024-cold-eval":
            for key in C.CACHE_ENV_KEYS:
                for env_key in (key, f"AUTOLOOP_EVAL_{key}"):
                    cache_path = manifest.env[env_key]
                    if self._run(manifest.box, ["test", "-e", cache_path], must_ok=False).ok:
                        raise RemoteError(f"cold cache path already exists: {env_key}")
        inv = self.inventory(manifest.box)
        if not inv.clean_and_idle:
            raise RemoteError(f"pre-launch inventory failed: {inv.findings}")
        head = self._run(manifest.box, ["git", "-C", staged.run_dir, "rev-parse", "HEAD"]).stdout.strip()
        if head != manifest.commit:
            raise RemoteError("HEAD changed since staging")
        if self._run(manifest.box, ["git", "-C", staged.run_dir, "status", "--porcelain"]).stdout.strip():
            raise RemoteError("tree dirty at launch time")

        launcher_dir = f"{C.REMOTE_AUTOLOOP_DIR}/launchers/{manifest.label}"
        if self._run(manifest.box, ["test", "-e", launcher_dir], must_ok=False).ok:
            raise RemoteError("launcher dir exists; this label was already launched")
        self._run(manifest.box, ["mkdir", "-p", launcher_dir])
        launcher_sha = sha256_file(LAUNCHER_SRC)
        remote_launcher = f"{launcher_dir}/remote_launcher.py"
        r = self.t.put(manifest.host, LAUNCHER_SRC, remote_launcher)
        if not r.ok or self._remote_sha(manifest.box, remote_launcher) != launcher_sha:
            raise RemoteError("launcher upload/verification failed")
        spec = {
            "cwd": manifest.remote_run_dir,
            "argv": manifest.argv,
            "env": manifest.env,
            "stdout": f"{launcher_dir}/runner.log",
            "pidfile": f"{launcher_dir}/pid",
            "donefile": f"{launcher_dir}/done.json",
            "expect_cwd_head": manifest.commit,
            "manifest_fingerprint": manifest.fingerprint(),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(spec, fh, indent=2)
            tmp = Path(fh.name)
        try:
            r = self.t.put(manifest.host, tmp, f"{launcher_dir}/launch.json")
        finally:
            tmp.unlink(missing_ok=True)
        if not r.ok:
            raise RemoteError("launch spec upload failed")
        out = self._run(manifest.box, [C.REMOTE_PYTHON, remote_launcher, f"{launcher_dir}/launch.json"], timeout=120, must_ok=False)
        try:
            data = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {}
        except json.JSONDecodeError:
            data = {}
        if not out.ok or "pid" not in data:
            raise RemoteError(f"launcher refused: {data.get('error') or out.stderr.strip()[:300]}")
        pid = int(data["pid"])
        alive = self._run(manifest.box, ["ps", "-p", str(pid), "-o", "pid="], must_ok=False).ok
        # A smoke run can fail before the follow-up SSH round trip completes.  A durable
        # done marker is still a successful *launch* and must be reconciled/collected, not
        # treated as an ambiguous launch that could be retried under the same state.
        done = self._run(manifest.box, ["test", "-f", f"{launcher_dir}/done.json"], must_ok=False).ok
        if not alive and not done:
            raise RemoteError("runner process not visible and no completion marker after launch")
        process_group = int(data.get("process_group", pid))
        if process_group <= 0:
            raise RemoteError("launcher returned an invalid process group")
        return LaunchReceipt(
            manifest.manifest_id, pid, launcher_dir, self._now(), launcher_sha,
            inv.to_dict(), head, process_group,
        )

    # poll / collect --------------------------------------------------------------------------

    def poll(self, manifest: RunManifest, receipt: LaunchReceipt) -> PollStatus:
        process_group = receipt.process_group or receipt.pid
        direct_alive = self._run(
            manifest.box, ["ps", "-p", str(receipt.pid), "-o", "pid="], must_ok=False
        ).ok
        group_members: list[dict] = ([{
            "pid": receipt.pid, "ppid": None, "pgid": process_group,
            "args": "runner (direct pid observation)",
        }] if direct_alive else [])
        if not direct_alive:
            processes = self._run(
                manifest.box, ["ps", "-eo", "pid=,ppid=,pgid=,args="], must_ok=False
            )
            if not processes.ok:
                raise RemoteError("cannot prove launched process group state")
            for line in processes.stdout.splitlines():
                fields = line.strip().split(None, 3)
                if len(fields) < 3:
                    continue
                try:
                    pid, ppid, pgid = (int(fields[0]), int(fields[1]), int(fields[2]))
                except ValueError:
                    continue
                if pgid == process_group:
                    group_members.append({
                        "pid": pid, "ppid": ppid, "pgid": pgid,
                        "args": fields[3] if len(fields) == 4 else "",
                    })
        alive = bool(group_members)
        done_r = self._run(manifest.box, ["cat", f"{receipt.launcher_dir}/done.json"], must_ok=False)
        rc = None
        done = False
        if done_r.ok:
            try:
                rc = int(json.loads(done_r.stdout)["returncode"])
                done = True
            except (ValueError, KeyError, json.JSONDecodeError):
                done = False
        tail = self._run(manifest.box, ["tail", "-n", "40", f"{receipt.launcher_dir}/runner.log"], must_ok=False).stdout
        return PollStatus(alive, done, rc, tail[-6000:], self._now(), group_members)

    def collect_failure_evidence(
        self,
        manifest: RunManifest,
        receipt: LaunchReceipt,
        status: PollStatus,
        local_dir: Path,
        reason: str,
    ) -> dict:
        """Best-effort final evidence for a quiescent failed run.

        Missing run artifacts are represented explicitly instead of turning a failed run into an
        unhandled collection exception.  This method refuses to seal evidence while any member of
        the launch process group remains visible.
        """
        if status.pid_alive or status.group_members:
            raise RemoteError("refusing final failure evidence while process group is alive")
        local_dir.mkdir(parents=True, exist_ok=True)
        wanted = {
            "run_log": manifest.remote_log,
            "run_launch_receipt": (
                f"{manifest.remote_run_dir}/research/run-logs/{manifest.label}.launch.json"
            ),
            "ledger": manifest.remote_ledger,
            "runner_stdout": f"{receipt.launcher_dir}/runner.log",
            "done": f"{receipt.launcher_dir}/done.json",
            "launch_spec": f"{receipt.launcher_dir}/launch.json",
        }
        if manifest.execution_mode == "exact-dense1024-cold-eval":
            wanted["cold_public_eval_log"] = (
                f"{manifest.remote_run_dir}/research/run-logs/{manifest.label}.eval.log"
            )
        files: dict[str, dict] = {}
        errors: list[str] = []
        for name, remote in wanted.items():
            try:
                remote_sha = self._remote_sha(manifest.box, remote)
                if remote_sha is None:
                    files[name] = {"remote": remote, "available": False}
                    continue
                local = local_dir / Path(remote).name
                if local.exists():
                    if local.is_symlink() or not local.is_file() or sha256_file(local) != remote_sha:
                        raise RemoteError(f"existing failure artifact differs: {local}")
                else:
                    fetched = self.t.get(manifest.host, remote, local)
                    if not fetched.ok or sha256_file(local) != remote_sha:
                        raise RemoteError(f"failure-evidence retrieval failed for {remote}")
                files[name] = {
                    "remote": remote, "local": str(local), "available": True,
                    "sha256": remote_sha, "bytes": local.stat().st_size,
                }
            except Exception as exc:  # evidence must retain partial collection failures
                files[name] = {"remote": remote, "available": False, "error": str(exc)[:500]}
                errors.append(f"{name}: {exc}"[:1000])
        ledger_row = None
        ledger_meta = files.get("ledger", {})
        if ledger_meta.get("available"):
            try:
                ledger_row = Ledger.load(Path(ledger_meta["local"])).latest_by_label(manifest.label)
            except Exception as exc:
                errors.append(f"ledger parse: {exc}"[:1000])
        try:
            postflight = self.inventory(manifest.box).to_dict()
        except Exception as exc:
            postflight = {"available": False, "error": str(exc)[:500]}
            errors.append(f"postflight: {exc}"[:1000])
        checkpoint_sha = self._remote_sha(manifest.box, manifest.remote_checkpoint)
        return {
            "schema_version": 1,
            "outcome": "failed",
            "final": True,
            "reason": reason,
            "manifest_id": manifest.manifest_id,
            "label": manifest.label,
            "box": manifest.box,
            "manifest": manifest.to_dict(),
            "launch_receipt": receipt.to_dict(),
            "status": status.to_dict(),
            "preflight": receipt.pre_launch_inventory,
            "postflight": postflight,
            "process": {
                "pid": receipt.pid, "process_group": receipt.process_group or receipt.pid,
                "quiescent": True, "remaining_members": [],
            },
            "files": files,
            "available_result": {
                "ledger_row": ledger_row,
                "checkpoint_present": checkpoint_sha is not None,
                "checkpoint_sha256": checkpoint_sha,
                "returncode": status.returncode,
            },
            "errors": errors,
            "collected_at": self._now(),
        }

    def collect(self, manifest: RunManifest, receipt: LaunchReceipt, local_dir: Path) -> Collected:
        local_dir.mkdir(parents=True, exist_ok=True)
        wanted = {
            "run_log": manifest.remote_log,
            "run_launch_receipt": f"{manifest.remote_run_dir}/research/run-logs/{manifest.label}.launch.json",
            "ledger": manifest.remote_ledger,
            "runner_stdout": f"{receipt.launcher_dir}/runner.log",
            "done": f"{receipt.launcher_dir}/done.json",
            "launch_spec": f"{receipt.launcher_dir}/launch.json",
        }
        if manifest.execution_mode == "exact-dense1024-cold-eval":
            wanted["cold_public_eval_log"] = (
                f"{manifest.remote_run_dir}/research/run-logs/{manifest.label}.eval.log"
            )
        files: dict[str, dict] = {}
        for name, remote in wanted.items():
            remote_sha = self._remote_sha(manifest.box, remote)
            if remote_sha is None:
                raise RemoteError(f"declared artifact missing on host: {remote}")
            local = local_dir / Path(remote).name
            r = self.t.get(manifest.host, remote, local)
            if not r.ok:
                raise RemoteError(f"retrieval failed for {remote}")
            local_sha = sha256_file(local)
            if local_sha != remote_sha:
                raise RemoteError(f"sha256 mismatch for {remote}")
            files[name] = {"remote": remote, "local": str(local), "sha256": local_sha, "bytes": local.stat().st_size}
        ledger = Ledger.load(local_dir / "experiment-log.jsonl")
        row = ledger.latest_by_label(manifest.label)
        ckpt_sha = self._remote_sha(manifest.box, manifest.remote_checkpoint)
        ckpt_bytes = None
        if ckpt_sha:
            st = self._run(manifest.box, ["stat", "-c", "%s", manifest.remote_checkpoint], must_ok=False)
            ckpt_bytes = int(st.stdout.strip()) if st.ok and st.stdout.strip().isdigit() else None
        porcelain = self._run(
            manifest.box,
            ["git", "-C", manifest.remote_run_dir, "status", "--porcelain", "--untracked-files=all"],
        ).stdout.splitlines()
        allowed_dirt = {
            "research/experiment-log.jsonl",
            f"research/run-logs/{manifest.label}.launch.json",
            f"research/run-logs/{manifest.label}.log",
        }
        if manifest.execution_mode == "exact-dense1024-cold-eval":
            allowed_dirt.add(f"research/run-logs/{manifest.label}.eval.log")
        # Porcelain v1 is two status columns, one space, then the repository-relative path.
        # Requesting every untracked file is important: accepting a collapsed run-logs/
        # directory would hide an arbitrary extra artifact beneath it.
        observed_dirt = {line[3:] for line in porcelain if len(line) >= 4 and line.strip()}
        explained = len(observed_dirt) == len(porcelain) and observed_dirt.issubset(allowed_dirt)
        return Collected(files, row, ckpt_sha is not None, ckpt_sha, ckpt_bytes, porcelain, explained, self._now())

    def prune_checkpoint(
        self,
        manifest: RunManifest,
        expected_sha256: str,
        *,
        authorization_commit: str | None = None,
    ) -> dict:
        """Delete only a completed run's checkpoint after proving its preserved hash matches."""
        expected_path = (
            f"{manifest.remote_run_dir}/out/final.pt"
            if manifest.execution_mode == "exact-dense1024-cold-eval"
            else f"{manifest.remote_run_dir}/out/{manifest.label}/final.pt"
        )
        if manifest.remote_checkpoint != expected_path:
            raise RemoteError("checkpoint path does not match the frozen manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise RemoteError("expected checkpoint sha256 is invalid")
        actual = self._remote_sha(manifest.box, manifest.remote_checkpoint)
        if actual is None:
            if not authorization_commit or not re.fullmatch(r"[0-9a-f]{40}", authorization_commit):
                raise RemoteError("checkpoint absent without a verified deletion-intent commit")
            return {
                "box": manifest.box,
                "label": manifest.label,
                "remote_checkpoint": manifest.remote_checkpoint,
                "sha256": expected_sha256,
                "pruned_at": self._now(),
                "reconciled_after_restart": True,
                "authorization_commit": authorization_commit,
            }
        if actual != expected_sha256:
            raise RemoteError(f"refusing checkpoint prune: remote hash {actual!r} != preserved hash")
        self._run(manifest.box, ["rm", "--", manifest.remote_checkpoint])
        if self._remote_sha(manifest.box, manifest.remote_checkpoint) is not None:
            raise RemoteError("checkpoint still exists after prune")
        return {
            "box": manifest.box,
            "label": manifest.label,
            "remote_checkpoint": manifest.remote_checkpoint,
            "sha256": expected_sha256,
            "pruned_at": self._now(),
            "reconciled_after_restart": False,
            "authorization_commit": authorization_commit,
        }

    def confirm_checkpoint_pruned(self, manifest: RunManifest, receipt: dict) -> None:
        if (receipt.get("box") != manifest.box or receipt.get("label") != manifest.label
                or receipt.get("remote_checkpoint") != manifest.remote_checkpoint
                or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", "")))
                or not re.fullmatch(
                    r"[0-9a-f]{40}", str(receipt.get("authorization_commit", ""))
                )):
            raise RemoteError("persisted checkpoint-prune receipt does not match manifest")
        if self._remote_sha(manifest.box, manifest.remote_checkpoint) is not None:
            raise RemoteError("persisted prune receipt exists but checkpoint is present")

    def cleanup_compile_caches(
        self,
        manifest: RunManifest,
        local_receipt: Path,
        *,
        authorization_commit: str | None = None,
    ) -> dict:
        """Delete only the exact label-owned cold caches after archived evidence is remote-safe."""
        if manifest.execution_mode != "exact-dense1024-cold-eval":
            raise RemoteError("compile-cache cleanup is restricted to exact dense1024 runs")
        cache_root = f"{C.REMOTE_AUTOLOOP_DIR}/caches/{manifest.label}"
        expected_paths = []
        for group in ("train", "eval"):
            for key in C.CACHE_ENV_KEYS:
                expected_paths.append(f"{cache_root}/{group}/{key.lower()}")
        observed = [manifest.env[key] for key in C.CACHE_ENV_KEYS]
        observed += [manifest.env[f"AUTOLOOP_EVAL_{key}"] for key in C.CACHE_ENV_KEYS]
        if observed != expected_paths[:5] + expected_paths[5:]:
            raise RemoteError("manifest cache paths differ from exact cleanup schema")
        cleanup_dir = f"{C.REMOTE_AUTOLOOP_DIR}/cache-cleanup/{manifest.label}"
        remote_script = f"{cleanup_dir}/remote_cache_cleanup.py"
        script_sha = sha256_file(CACHE_CLEANUP_SRC)
        receipt_path = f"{cleanup_dir}/receipt.json"
        spec = {
            "label": manifest.label, "cache_root": cache_root,
            "cache_paths": expected_paths, "receipt_path": receipt_path,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(spec, fh, sort_keys=True)
            tmp = Path(fh.name)
        try:
            spec_sha = sha256_file(tmp)
            cleanup_exists = self._run(
                manifest.box, ["test", "-e", cleanup_dir], must_ok=False
            ).ok
            remote_receipt_sha = self._remote_sha(manifest.box, receipt_path)
            if remote_receipt_sha is None and not self._run(
                    manifest.box, ["test", "-e", cache_root], must_ok=False).ok:
                if not authorization_commit or not re.fullmatch(r"[0-9a-f]{40}", authorization_commit):
                    raise RemoteError("cache absent without a verified deletion-intent commit")
                receipt = {
                    "label": manifest.label, "cache_root": cache_root,
                    "cache_paths": expected_paths, "absent_after": True,
                    "cleaned_at": self._now(), "reconciled_after_restart": True,
                    "authorization_commit": authorization_commit,
                }
                local_receipt.parent.mkdir(parents=True, exist_ok=True)
                if local_receipt.exists():
                    raise RemoteError("refusing to overwrite local cache-cleanup receipt")
                local_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                return receipt
            if not cleanup_exists:
                self._run(manifest.box, ["mkdir", "-p", cleanup_dir])
                put = self.t.put(manifest.host, CACHE_CLEANUP_SRC, remote_script)
                if not put.ok or self._remote_sha(manifest.box, remote_script) != script_sha:
                    raise RemoteError("cache-cleanup helper upload/hash verification failed")
                put = self.t.put(manifest.host, tmp, f"{cleanup_dir}/cleanup.json")
                if (not put.ok or self._remote_sha(
                        manifest.box, f"{cleanup_dir}/cleanup.json") != spec_sha):
                    raise RemoteError("cache-cleanup spec upload/hash verification failed")
            elif remote_receipt_sha is None:
                if (self._remote_sha(manifest.box, remote_script) != script_sha
                        or self._remote_sha(manifest.box, f"{cleanup_dir}/cleanup.json") != spec_sha):
                    raise RemoteError("existing cache-cleanup transaction bytes changed")
        finally:
            tmp.unlink(missing_ok=True)
        if self._remote_sha(manifest.box, receipt_path) is None:
            result = self._run(
                manifest.box,
                [C.REMOTE_PYTHON, remote_script, f"{cleanup_dir}/cleanup.json"],
                timeout=600, must_ok=False,
            )
            if not result.ok:
                raise RemoteError(f"cache cleanup failed: {result.stderr.strip()[:300]}")
        remote_sha = self._remote_sha(manifest.box, receipt_path)
        if remote_sha is None:
            raise RemoteError("cache-cleanup receipt missing")
        local_receipt.parent.mkdir(parents=True, exist_ok=True)
        get = self.t.get(manifest.host, receipt_path, local_receipt)
        if not get.ok or sha256_file(local_receipt) != remote_sha:
            raise RemoteError("cache-cleanup receipt retrieval/hash mismatch")
        receipt = json.loads(local_receipt.read_text())
        if (receipt.get("label") != manifest.label or receipt.get("cache_root") != cache_root
                or receipt.get("cache_paths") != expected_paths or receipt.get("absent_after") is not True):
            raise RemoteError("cache-cleanup receipt schema mismatch")
        receipt.setdefault("reconciled_after_restart", False)
        receipt.setdefault("authorization_commit", authorization_commit)
        receipt.setdefault("remote_receipt_sha256", remote_sha)
        local_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    def confirm_compile_caches_clean(self, manifest: RunManifest, receipt: dict) -> None:
        cache_root = f"{C.REMOTE_AUTOLOOP_DIR}/caches/{manifest.label}"
        expected_paths = [
            f"{cache_root}/{group}/{key.lower()}"
            for group in ("train", "eval") for key in C.CACHE_ENV_KEYS
        ]
        if (receipt.get("label") != manifest.label or receipt.get("cache_root") != cache_root
                or receipt.get("cache_paths") != expected_paths
                or receipt.get("absent_after") is not True
                or not re.fullmatch(
                    r"[0-9a-f]{40}", str(receipt.get("authorization_commit", ""))
                )):
            raise RemoteError("persisted compile-cache cleanup receipt does not match manifest")
        if self._run(manifest.box, ["test", "-e", cache_root], must_ok=False).ok:
            raise RemoteError("persisted cache cleanup receipt exists but cache root is present")
