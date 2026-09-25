"""Disabled trn22 e65->r10 r2 source stage; no standalone host entry point.

Only a separately reviewed, one-shot local wrapper may invoke ``perform``.
Target creation is exclusive. A failed post-open gate retains even an empty
provisional bundle and consumes the one-attempt marker; no host unlink or
other cleanup is authorized by this disabled package. All partials remain.
The accepted e65 linked worktree is read-only throughout.
"""

from contextlib import contextmanager
import datetime
import errno
import fcntl
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import stat
import subprocess
import time


HOST = "ip-172-31-32-253"
UID = GID = 1000
DEV = 66305
ROOT = Path("/home/ubuntu")
SOURCE = ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
TARGET = ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r2"
BUNDLE = ROOT / "trainium-c5c-e65-to-r10-20260925.trn22-stage-r2.bundle"
LABEL = "fs-20260925-trn22-c5c-a0-r10-incremental-stage-r2"
PRIOR_R1_TARGET = ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1"
PRIOR_R1_BUNDLE = ROOT / "trainium-c5c-e65-to-r10-20260925.trn22-stage-r1.bundle"
BASE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TARGET_COMMIT = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"
DELTA_COUNT = 15
BUNDLE_BYTES = 109800
BUNDLE_SHA256 = "b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e"
PRE_FREE = 13_177_973_911
POST_FREE = 10_737_418_240
MIN_INODES = 10_000
MAX_FINAL_CHECK_AGE_SECONDS = 1.0
MAX_PREOPEN_ATTEMPTS = 3
SOURCE_INO = 2432738
SOURCE_MODE = 0o775
SOURCE_NLINK = 4
GIT_MARKER_INO = 2432740
GIT_MARKER_MODE = 0o664
GIT_MARKER_BYTES = 148
GIT_MARKER_SHA256 = "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b"
GIT_DIR = (ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1"
           / ".git/worktrees/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2")
GIT_DIR_INO = 2432736
GIT_DIR_MODE = 0o775
LOCK_CACHE = ROOT / ".cache"
LOCK_DIR = LOCK_CACHE / "trainium_freshstart"
LOCK = LOCK_DIR / "device0.lock"
LOCK_INODES = (295736, 832588, 832589)
LOCK_MODES = (0o700, 0o775, 0o664)
CHECKPOINT = (ROOT / "trainium_freshstart_20260924_current_base_control_trn22_r1"
              / "out/fs-20260924-current-base-control-trn22-r1-work/train/out/final.pt")
CHECKPOINT_INO = 2417091
CHECKPOINT_BYTES = 469831009
CHECKPOINT_SHA256 = "1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b"
CACHE_ROOT = (ROOT / "trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1"
              / "out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train")
CACHE_INODES = {"neff": 2430420, "local": 2430421, "hlo": 2430422,
                "inductor": 2430423, "compiler": 2430424, "debug": 2430425}
BASE_FILES = {
    "train.py": "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
}
TARGET_FILES = {
    **BASE_FILES,
    "research/run_experiment.py": "88243a74f0f080929f93d0b88ddedbe6f5351ad37bf272229d9ae6d26efd5d2d",
    "research/fresh_start_sync.py": "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c",
    "research/fresh_start_backup.py": "c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3",
    "research/fresh_start_guard.py": "df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc",
    "research/fresh_start_host.py": "b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3",
    "research/fresh_start_launch.py": "1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7",
    "research/fresh_start_monitor_interlock.py": "f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d",
    "research/v2/disk-monitor/root_free_guard_monitor.py": "91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60",
    "research/v2/disk-monitor/paired_root_free_guard_monitor.py": "9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344",
}
STOP_PATHS = [ROOT / "research/agentic/STOP", ROOT / "research/v2/STOP",
              ROOT / "trainium_freshstart_20260918/research/agentic/STOP",
              ROOT / "trainium_freshstart_20260918/research/v2/STOP"]
for _name in ("r1", "r2", "r3", "r4", "r9_stage_r1", "r10_stage_r1",
              "r10_stage_r2"):
    _checkout = ROOT / ("trainium_freshstart_20260925_v2_c5c_a0_control_trn22_" + _name)
    STOP_PATHS += [_checkout / "research/agentic/STOP", _checkout / "research/v2/STOP"]
PROCESS_MARKERS = ("train.py", "torchrun", "eval-public", "prepare.py",
                   "run_experiment.py", "fresh_start_launch", "fresh_start_guard",
                   "root_free_guard_monitor", "paired_root_free_guard_monitor",
                   "run_job.py", "preflight_dryrun", "contained.py", "g2-probe",
                   "phase_timer", "reconcile", "neuronx-cc")
GIT_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "file",
           "GIT_OPTIONAL_LOCKS": "0"}


class StageGateError(RuntimeError):
    """Internal refusal with a bounded, nonsecret diagnostic category."""

    def __init__(self, reason, failure_code="OTHER_GATE"):
        super().__init__(reason)
        self.failure_code = failure_code


class PreOpenFreshnessExpired(StageGateError):
    def __init__(self, reason):
        super().__init__(reason, "PREOPEN_TIMING_EXPIRED")


def _failure_code(reason):
    # The reason is never returned to the controller in a remote HOLD.
    if "STOP" in reason:
        return "STOP_GATE"
    if "label" in reason:
        return "LABEL_GATE"
    if "Neuron" in reason or "process" in reason or "holder" in reason:
        return "ACTIVITY_GATE"
    if "floor" in reason or "space" in reason or "inode" in reason:
        return "SPACE_GATE"
    if "bundle" in reason or "bytes" in reason or "hash" in reason:
        return "DATA_GATE"
    if ("source" in reason or "checkout" in reason or "host" in reason or
            "identity" in reason or "mount" in reason or "UID" in reason):
        return "IDENTITY_GATE"
    return "OTHER_GATE"


def _fail(reason):
    raise StageGateError(reason, _failure_code(reason))


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _meta(path, kind=None, *, uid=UID, gid=GID, inode=None, mode=None,
          nlink=None):
    info = os.lstat(path)
    expected_kind = (stat.S_ISDIR if kind == "dir" else stat.S_ISREG)
    if (not expected_kind(info.st_mode) or info.st_dev != DEV or
            info.st_uid != uid or info.st_gid != gid or
            (inode is not None and info.st_ino != inode) or
            (mode is not None and stat.S_IMODE(info.st_mode) != mode) or
            (nlink is not None and info.st_nlink != nlink)):
        _fail("type/device/owner/inode/mode/link drift: " + str(path))
    return {"device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "gid": info.st_gid,
            "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
            "bytes": info.st_size}


def _hash_regular(path, *, expected_bytes=None, expected_sha=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        first = os.fstat(fd)
        named = os.lstat(path)
        if (not stat.S_ISREG(first.st_mode) or first.st_nlink != 1 or
                (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino)):
            _fail("unsafe file during no-follow hash: " + str(path))
        digest, count = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 4 * 1024 * 1024)
            if not block:
                break
            count += len(block)
            if expected_bytes is not None and count > expected_bytes:
                _fail("file grew past expected bytes: " + str(path))
            digest.update(block)
        final = os.fstat(fd)
        named_after = os.lstat(path)
        if ((first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) !=
                (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) or
                (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) !=
                (named_after.st_dev, named_after.st_ino, named_after.st_size,
                 named_after.st_mtime_ns) or
                (expected_bytes is not None and count != expected_bytes) or
                (expected_sha is not None and digest.hexdigest() != expected_sha)):
            _fail("file bytes/hash changed: " + str(path))
        return {"device": first.st_dev, "inode": first.st_ino,
                "bytes": count, "sha256": digest.hexdigest()}
    finally:
        os.close(fd)


def _acl_absent(path, *, directory):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    flags |= os.O_DIRECTORY if directory else os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        for name in (("system.posix_acl_access", "system.posix_acl_default")
                     if directory else ("system.posix_acl_access",)):
            try:
                os.getxattr(fd, name)
            except OSError as error:
                if error.errno != errno.ENODATA:
                    _fail("unknown ACL at " + str(path))
            else:
                _fail("POSIX ACL present at " + str(path))
    finally:
        os.close(fd)


def _group_isolation():
    account = pwd.getpwnam("ubuntu")
    group = grp.getgrgid(GID)
    primary = [(item.pw_name, item.pw_uid) for item in pwd.getpwall()
               if item.pw_gid == GID]
    if ((account.pw_uid, account.pw_gid) != (UID, GID) or group.gr_mem or
            primary != [("ubuntu", UID)]):
        _fail("GID 1000 local/NSS isolation not established")
    return {"primary_gid_1000": primary, "explicit_members": []}


def _safe_chain(path):
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        _fail("Git path escapes expected home")
    parts = [Path("/"), Path("/home"), ROOT]
    relative = path.relative_to(ROOT)
    current = ROOT
    for part in relative.parts:
        current = current / part
        parts.append(current)
    out = {}
    for item in parts:
        info = os.lstat(item)
        if not stat.S_ISDIR(info.st_mode) or info.st_dev != DEV:
            _fail("Git/parent ancestry has symlink or mount: " + str(item))
        mode = stat.S_IMODE(info.st_mode)
        if item in (Path("/"), Path("/home")):
            if info.st_uid != 0 or mode & 0o022:
                _fail("unsafe root/home ancestry")
        elif item == ROOT:
            if (info.st_uid, info.st_gid, info.st_ino, mode) != (UID, GID, 295711, 0o750):
                _fail("/home/ubuntu ancestry changed")
        elif (info.st_uid, info.st_gid) != (UID, GID) or mode & 0o002 or mode & 0o7000:
            _fail("Git ancestor owner/mode unsafe")
        _acl_absent(item, directory=True)
        out[str(item)] = {"inode": info.st_ino, "mode": mode,
                          "uid": info.st_uid, "gid": info.st_gid}
    return out


def _mount_identity():
    raw = Path("/proc/self/mountinfo").read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        _fail("mountinfo missing or oversized")
    roots = []
    for line in raw.decode("utf-8", "surrogateescape").splitlines():
        left, sep, right = line.partition(" - ")
        a, b = left.split(), right.split()
        if not sep or len(a) < 6 or len(b) < 2:
            _fail("malformed mountinfo")
        point = a[4]
        mount_root = a[3]
        if point == "/":
            roots.append((a[0], a[2], mount_root, b[0], b[1]))
        if (point == str(ROOT) or point.startswith(str(ROOT) + "/") or
                (a[2] == "259:1" and
                 (mount_root == str(SOURCE) or mount_root.startswith(str(SOURCE) + "/") or
                  mount_root == str(LOCK_CACHE) or
                  mount_root.startswith(str(LOCK_CACHE) + "/")))):
            _fail("relevant home/source/lock mount alias observed")
    if len(roots) != 1 or roots[0][1:] != ("259:1", "/", "ext4", "/dev/root"):
        _fail("root ext4 mount identity differs")
    return {"root": roots[0], "sha256": hashlib.sha256(raw).hexdigest(),
            "namespace": os.readlink("/proc/self/ns/mnt")}


def _git(repo, *args, timeout=120):
    result = subprocess.run(["/usr/bin/git", "-c", "core.fsmonitor=false",
                             "-c", "submodule.recurse=false", "-C", str(repo), *args],
                            env=GIT_ENV, capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode != 0:
        _fail("Git command failed: " + " ".join(args))
    return result.stdout.strip()


def _source():
    source = _meta(SOURCE, "dir", inode=SOURCE_INO, mode=SOURCE_MODE,
                   nlink=SOURCE_NLINK)
    _group_isolation()
    chain = _safe_chain(SOURCE)
    marker = _meta(SOURCE / ".git", "file", inode=GIT_MARKER_INO,
                   mode=GIT_MARKER_MODE, nlink=1)
    if marker["bytes"] != GIT_MARKER_BYTES:
        _fail("linked-worktree pointer length differs")
    _acl_absent(SOURCE / ".git", directory=False)
    _hash_regular(SOURCE / ".git", expected_bytes=GIT_MARKER_BYTES,
                  expected_sha=GIT_MARKER_SHA256)
    gitdir = _meta(GIT_DIR, "dir", inode=GIT_DIR_INO, mode=GIT_DIR_MODE)
    git_chain = _safe_chain(GIT_DIR)
    resolved = _git(SOURCE, "rev-parse", "--absolute-git-dir")
    if resolved != str(GIT_DIR):
        _fail("linked worktree resolved Git directory differs")
    common = Path(_git(SOURCE, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    common_chain = _safe_chain(common)
    if (_git(SOURCE, "rev-parse", "HEAD") != BASE_COMMIT or
            _git(SOURCE, "status", "--porcelain=v1", "--untracked-files=all",
                 "--ignore-submodules=all")):
        _fail("accepted e65 checkout is not exact and clean")
    for relative, digest in BASE_FILES.items():
        _hash_regular(SOURCE / relative, expected_sha=digest)
    return {"source": source, "pointer": marker, "gitdir": gitdir,
            "source_chain": chain, "git_chain": git_chain,
            "common_chain": common_chain, "common_gitdir": str(common),
            "commit": BASE_COMMIT, "science_sha256": BASE_FILES}


@contextmanager
def _host_lock():
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fds = []
    try:
        fds.append(os.open(LOCK_CACHE, flags))
        fds.append(os.open("trainium_freshstart", flags, dir_fd=fds[0]))
        if not stat.S_ISREG(os.stat("device0.lock", dir_fd=fds[1],
                                  follow_symlinks=False).st_mode):
            _fail("host lock is not regular")
        fds.append(os.open("device0.lock", os.O_RDONLY | os.O_NONBLOCK |
                           os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fds[1]))
        out = {}
        for path, fd, ino, mode, directory in zip(
                (LOCK_CACHE, LOCK_DIR, LOCK), fds, LOCK_INODES, LOCK_MODES,
                (True, True, False)):
            opened, named = os.fstat(fd), os.lstat(path)
            if ((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink) !=
                    (named.st_dev, named.st_ino, named.st_mode, named.st_nlink) or
                    opened.st_dev != DEV or opened.st_ino != ino or
                    (opened.st_uid, opened.st_gid) != (UID, GID) or
                    stat.S_IMODE(opened.st_mode) != mode or
                    (not directory and (opened.st_nlink != 1 or opened.st_size != 0))):
                _fail("existing launcher-compatible lock changed")
            _acl_absent(path, directory=directory)
            out[str(path)] = {"device": opened.st_dev, "inode": opened.st_ino,
                              "mode": mode, "nlink": opened.st_nlink}
        fcntl.flock(fds[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield out
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _protected():
    checkpoint = _meta(CHECKPOINT, "file", inode=CHECKPOINT_INO, nlink=1)
    if checkpoint["bytes"] != CHECKPOINT_BYTES:
        _fail("retained checkpoint length changed")
    _hash_regular(CHECKPOINT, expected_bytes=CHECKPOINT_BYTES,
                  expected_sha=CHECKPOINT_SHA256)
    caches = {}
    for name, inode in CACHE_INODES.items():
        caches[name] = _meta(CACHE_ROOT / name, "dir", inode=inode)
    return {"checkpoint": checkpoint, "checkpoint_sha256": CHECKPOINT_SHA256,
            "caches": caches}


def _proc_ancestry():
    own = set()
    pid = os.getpid()
    while pid > 1 and pid not in own:
        own.add(pid)
        try:
            text = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
            pid = int(text.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return own


def _stop_and_process():
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("host/source/target STOP is present")
    own, hits = _proc_ancestry(), []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            words = Path("/proc", entry.name, "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            _fail("process table has unreadable cmdline")
        if any(marker.encode() in word for marker in PROCESS_MARKERS for word in words):
            hits.append(int(entry.name))
    if hits:
        _fail("matched train/eval/guard/process activity")
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("host/source/target STOP appeared during process scan")
    return {"stops_absent": True, "matched_processes": []}


def _activity():
    _stop_and_process()
    fuser = subprocess.run(["/usr/bin/fuser", "/dev/neuron0"],
                           capture_output=True, timeout=20, check=False)
    if fuser.returncode != 1 or fuser.stdout or fuser.stderr.strip():
        _fail("Neuron device holder or unknown fuser result")
    neuron = subprocess.run(["neuron-ls", "-j", "-a"], env={"PATH":
                            "/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin"},
                            capture_output=True, timeout=20, check=False)
    if neuron.returncode != 0 or neuron.stderr.strip():
        _fail("neuron-ls failed or emitted a diagnostic")
    devices = json.loads(neuron.stdout)
    if not isinstance(devices, list):
        _fail("neuron-ls JSON is not a list")
    for device in devices:
        if not isinstance(device, dict):
            _fail("neuron-ls device record malformed")
        for process in device.get("neuron_processes", []):
            if not isinstance(process, dict) or not str(process.get("command", "")).startswith("neuron-ls"):
                _fail("Neuron process active or malformed")
    _stop_and_process()  # The slow commands may have admitted a new STOP.
    return {"stops_absent": True, "matched_processes": [],
            "neuron_holder_absent": True}


def _label_scan():
    scan = subprocess.run(["/usr/bin/find", str(ROOT), "-xdev", "-name",
                           "*" + LABEL + "*", "-print"], capture_output=True,
                          timeout=45, check=False)
    if scan.returncode != 0 or scan.stderr.strip() or scan.stdout.strip():
        _fail("stage attempt label already appears or scan failed")
    return {"label": LABEL, "matches": [], "scan_utc": _utc()}


def _final_no_write_continuity():
    """Return monotonic samples under flock for checks at the actual open.

    No finite userspace check prevents a hostile same-UID/root actor racing
    after the observations. The inherited one-second bound is unmeasured on
    trn22 and may refuse every live attempt; this package remains disabled.
    """
    # Preserve r2's first activity gate so a late label/STOP arriving during
    # its slow fuser/neuron-ls phase is caught by the subsequent broad scans.
    _activity()
    scan_started = time.monotonic()
    _label_scan()
    _label_scan()  # Detect a label introduced after the first traversal.
    label_end = time.monotonic()
    label_scan_duration = label_end - scan_started
    scan_expired = not 0 <= label_scan_duration <= MAX_FINAL_CHECK_AGE_SECONDS
    # Persistent process/Neuron activity introduced during either broad scan
    # must be observed before the first write, not after reconstruction.
    _activity()
    activity_end = time.monotonic()
    root_space = _space(PRE_FREE)
    space_end = time.monotonic()
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("STOP appeared during final continuity checks")
    if (os.path.lexists(BUNDLE) or os.path.lexists(TARGET) or
            os.path.lexists(PRIOR_R1_BUNDLE) or os.path.lexists(PRIOR_R1_TARGET)):
        _fail("one-shot stage target appeared during final checks")
    samples = {"activity_at": activity_end, "label_at": label_end,
               "space_at": space_end, "root_space_before_open": root_space,
               "label_scan_duration_seconds": label_scan_duration}
    # Finish every activity/label/space/STOP gate before treating a slow
    # traversal as retryable. A non-timing refusal is never retried.
    if scan_expired:
        raise PreOpenFreshnessExpired("final label scan duration too old for first O_EXCL")
    _freshness_evidence(samples, time.monotonic(), pre_open=True)
    return samples


def _freshness_evidence(samples, now, *, pre_open=False):
    ages = {"activity_age_seconds": now - samples["activity_at"],
            "label_age_seconds": now - samples["label_at"],
            "space_age_seconds": now - samples["space_at"]}
    if any(not 0 <= age <= MAX_FINAL_CHECK_AGE_SECONDS for age in ages.values()):
        if pre_open:
            raise PreOpenFreshnessExpired(
                "final activity/label/space proof too old before first O_EXCL")
        _fail("final activity/label/space proof too old at O_EXCL return")
    return {**ages, "double_label_scan": True,
            "post_label_activity_check": True,
            "label_scan_duration_seconds": samples["label_scan_duration_seconds"],
            "root_space_before_open": samples["root_space_before_open"]}


def _space(minimum):
    info = os.statvfs("/")
    free = info.f_bavail * info.f_frsize
    inodes = info.f_favail
    if free < minimum or free < 6_979_321_856 or inodes < MIN_INODES:
        _fail("root byte/inode floor failed")
    return {"root_free_bytes": free, "root_free_inodes": inodes}


def _process_identity():
    text = Path("/proc/self/stat").read_text(encoding="ascii")
    return {"pid": os.getpid(), "start_ticks": int(text.rsplit(")", 1)[1].split()[19])}


def _probe(lock, *, minimum=PRE_FREE):
    if socket.gethostname() != HOST or os.geteuid() != UID:
        _fail("wrong trn22 host or UID")
    identity_before = _space(minimum)
    source = _source()
    mount = _mount_identity()
    protected = _protected()
    activity = _activity()
    label = _label_scan()
    # The slow final scan is followed by a fresh live activity/STOP check.
    _activity()
    identity_after = _space(minimum)
    if os.path.lexists(PRIOR_R1_BUNDLE) or os.path.lexists(PRIOR_R1_TARGET):
        _fail("failed r1 source-stage artifact appeared")
    return {"schema": "trn22-r10-incremental-full-readonly-v1",
            "utc": _utc(), "process_identity": _process_identity(),
            "hostname": HOST, "uid": UID,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "mount": mount, "lock": lock, "source": source,
            "protected": protected, "activity": activity, "label": label,
            "identity_before": identity_before,
            "identity_after": identity_after,
            "bundle_exists": os.path.lexists(BUNDLE),
            "target_exists": os.path.lexists(TARGET)}


def _stable(prior, current):
    if not isinstance(prior, dict) or prior.get("schema") != current["schema"]:
        _fail("missing exact preflight binding")
    for key in ("hostname", "uid", "boot_id", "mount", "lock", "source", "protected"):
        if prior.get(key) != current[key]:
            _fail("preflight-to-stage identity drift: " + key)
    if prior.get("process_identity") == current["process_identity"]:
        _fail("preflight was not a distinct SSH process")
    stamp = datetime.datetime.fromisoformat(prior["utc"])
    age = (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds()
    if stamp.tzinfo is None or not -5 <= age <= 180:
        _fail("fresh preflight has expired")


def _checked_bundle():
    info = _meta(BUNDLE, "file", mode=0o600, nlink=1)
    if info["bytes"] != BUNDLE_BYTES:
        _fail("bundle length changed")
    digest = _hash_regular(BUNDLE, expected_bytes=BUNDLE_BYTES,
                           expected_sha=BUNDLE_SHA256)
    return {**info, "sha256": digest["sha256"]}


def _checked_target():
    target = _meta(TARGET, "dir", mode=0o700)
    if _git(TARGET, "rev-parse", "HEAD") != TARGET_COMMIT:
        _fail("new checkout HEAD differs")
    if _git(TARGET, "status", "--porcelain=v1", "--untracked-files=all",
            "--ignore-submodules=all"):
        _fail("new r10 checkout is dirty")
    if _git(TARGET, "remote"):
        _fail("new r10 checkout unexpectedly has remotes")
    if os.path.lexists(TARGET / ".git/objects/info/alternates"):
        _fail("new r10 checkout depends on source object alternates")
    if _git(TARGET, "rev-list", "--count", BASE_COMMIT + ".." + TARGET_COMMIT) != str(DELTA_COUNT):
        _fail("new checkout delta count differs")
    _git(TARGET, "fsck", "--full", "--strict", "--no-reflogs")
    files = {}
    for relative, digest in TARGET_FILES.items():
        files[relative] = _hash_regular(TARGET / relative, expected_sha=digest)
    return {"identity": target, "commit": TARGET_COMMIT,
            "clean": True, "files": files}


def _write_bundle(data):
    if (not isinstance(data, bytes) or len(data) != BUNDLE_BYTES or
            hashlib.sha256(data).hexdigest() != BUNDLE_SHA256):
        _fail("incoming exact bundle bytes/hash mismatch")
    parent = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if (os.path.lexists(BUNDLE) or os.path.lexists(TARGET) or
                os.path.lexists(PRIOR_R1_BUNDLE) or os.path.lexists(PRIOR_R1_TARGET)):
            _fail("one-shot stage target already exists")
        # Only timing expiry before O_EXCL may recheck. Every pass reruns the
        # complete final activity/double-label/space/STOP sequence under the
        # held host lock. Once O_EXCL is attempted, there is never a retry.
        for preopen_attempt in range(1, MAX_PREOPEN_ATTEMPTS + 1):
            try:
                samples = _final_no_write_continuity()
                _freshness_evidence(samples, time.monotonic(), pre_open=True)
            except PreOpenFreshnessExpired:
                if preopen_attempt == MAX_PREOPEN_ATTEMPTS:
                    raise PreOpenFreshnessExpired(
                        "pre-open timing retry limit exhausted")
                continue
            open_started = time.monotonic()
            fd = os.open(BUNDLE.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            break
        try:
            # os.open can be delayed after the preceding freshness check.
            # Observe the actual return boundary before any payload. If any
            # subsequent gate fails, retain the empty inode; cleanup requires
            # a separate human decision and this attempt cannot be retried.
            open_returned = time.monotonic()
            final_gate = _freshness_evidence(samples, open_returned)
            open_elapsed = open_returned - open_started
            if not 0 <= open_elapsed <= MAX_FINAL_CHECK_AGE_SECONDS:
                _fail("exclusive open call exceeded final freshness bound")
            # A persistent change in the helper-to-O_EXCL interval is
            # checked again while the provisional inode is still empty.
            _label_scan()
            post_open_label = time.monotonic()
            _activity()
            post_open_activity = time.monotonic()
            root_space_after_open = _space(PRE_FREE)
            post_open_space = time.monotonic()
            if (any(os.path.lexists(path) for path in STOP_PATHS) or
                    os.path.lexists(TARGET) or
                    os.path.lexists(PRIOR_R1_BUNDLE) or
                    os.path.lexists(PRIOR_R1_TARGET)):
                _fail("STOP or checkout appeared after bundle O_EXCL")
            now = time.monotonic()
            post_open_ages = {
                "post_open_label_age_seconds": now - post_open_label,
                "post_open_activity_age_seconds": now - post_open_activity,
                "post_open_space_age_seconds": now - post_open_space}
            if any(not 0 <= age <= MAX_FINAL_CHECK_AGE_SECONDS
                   for age in post_open_ages.values()):
                _fail("post-open label/activity/space proof too old for payload")
            final_gate.update(post_open_ages)
            final_gate["open_call_elapsed_seconds"] = open_elapsed
            final_gate["preopen_attempts"] = preopen_attempt
            final_gate["post_open_label_and_activity_check"] = True
            final_gate["root_space_after_open"] = root_space_after_open
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
        finally:
            # Even an empty provisional inode is evidence. Never unlink or
            # rename it here; make its directory entry as durable as possible
            # on both refusal and success before a checkout can be created.
            try:
                os.fsync(fd)
            finally:
                try:
                    os.close(fd)
                finally:
                    os.fsync(parent)
        _checked_bundle()
        os.mkdir(TARGET.name, 0o700, dir_fd=parent)
        os.fsync(parent)
        return final_gate
    finally:
        os.close(parent)


def _reconstruct():
    # Git fetch uses the accepted linked worktree read-only. It creates a
    # separate object database, no alternates/hardlinks and no patch in place.
    _git(TARGET, "init", "-q", "--initial-branch=stage")
    _git(TARGET, "fetch", "-q", "--no-tags", "--no-recurse-submodules",
         "file://" + str(SOURCE), "HEAD")
    if _git(TARGET, "cat-file", "-t", BASE_COMMIT) != "commit":
        _fail("accepted e65 base object not present in new checkout")
    _git(TARGET, "fetch", "-q", "--no-tags", "--no-recurse-submodules",
         str(BUNDLE), "HEAD")
    _git(TARGET, "checkout", "-q", "--detach", TARGET_COMMIT)
    return _checked_target()


def perform(phase, prior=None, data=b""):
    """One remote phase. A trusted future wrapper must authenticate authority.

    Current package supplies no dispatch path. Preflight, write, readback and
    postflight must be separate SSH processes; postflight is unconditional.
    """
    if phase not in {"preflight", "write", "readback", "postflight"}:
        _fail("unknown remote phase")
    with _host_lock() as lock:
        snapshot = _probe(lock, minimum=PRE_FREE if phase in ("preflight", "write") else POST_FREE)
        if phase == "preflight":
            if snapshot["bundle_exists"] or snapshot["target_exists"]:
                _fail("one-shot bundle/target already exists")
            return {"phase": phase, "snapshot": snapshot,
                    "scope": "SOURCE_STAGE_ONLY"}
        if phase == "write":
            _stable(prior, snapshot)
            if snapshot["bundle_exists"] or snapshot["target_exists"]:
                _fail("one-shot bundle/target already exists")
            # Post-label activity/disk checks and both pre-/post-open age
            # gates are inside _write_bundle under the held flock. A failed
            # post-open gate retains even its empty provisional bundle.
            final_gate = _write_bundle(data)
            target = _reconstruct()
            source_after = _source()
            if source_after != snapshot["source"]:
                _fail("accepted source drifted during reconstruction")
            _activity()
            _space(POST_FREE)
            return {"phase": phase, "snapshot": snapshot,
                    "final_no_write_continuity": final_gate,
                    "bundle": _checked_bundle(), "target": target,
                    "source_after": source_after,
                    "scope": "SOURCE_STAGE_ONLY_NO_NATIVE_AUTHORITY"}
        if phase == "readback":
            if prior is None or prior.get("phase") != "write":
                _fail("readback requires a write receipt")
            previous = prior.get("snapshot", {})
            if previous.get("process_identity") == snapshot["process_identity"]:
                _fail("readback is not a distinct SSH process")
            for key in ("hostname", "uid", "boot_id", "mount", "lock", "source", "protected"):
                if previous.get(key) != snapshot[key]:
                    _fail("readback source/host/lock/protected identity drift")
            return {"phase": phase, "snapshot": snapshot,
                    "bundle": _checked_bundle(), "target": _checked_target(),
                    "scope": "SOURCE_STAGE_READBACK_ONLY"}
        return {"phase": phase, "snapshot": snapshot,
                "bundle_exists": os.path.lexists(BUNDLE),
                "target_exists": os.path.lexists(TARGET),
                "scope": "UNCONDITIONAL_READONLY_POSTFLIGHT"}


if __name__ == "__main__":
    raise SystemExit("HOLD: no enabled request, independent review, or assignment")
