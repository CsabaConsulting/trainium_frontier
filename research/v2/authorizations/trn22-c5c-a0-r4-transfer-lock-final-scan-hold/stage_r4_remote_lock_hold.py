"""Disabled trn22 r4 cooperative-lock/final-label transfer repair.

This source is a new HOLD-only successor, not a dispatchable replacement for
the archived r3 writer. The existing launcher cooperates through device0.lock;
non-cooperating manual processes cannot be atomically excluded. No extraction,
checkout, deletion, rename, retry, or training path exists here.
"""

import base64
import binascii
import datetime
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import time


SCHEMA = "trn22-c5c-a0-r4-transfer-lock-final-scan-hold-v1"
HOST = "ip-172-31-32-253"
UID = 1000
ROOT = "/home/ubuntu"
MOUNT_SOURCE = "/dev/root"
MOUNT_MAJOR_MINOR = "259:1"
HISTORICAL_DEVICE_NAME = "/dev/nvme0n1p1"
TARGET_NAME = "trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz"
TARGET = ROOT + "/" + TARGET_NAME
CHECKOUT = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4"
MONITOR = ROOT + "/trainium_a0_disk_monitor_20260925_r4"
OBSERVER = ROOT + "/trainium_a0_boundary_observer_20260925_r1"
LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r4"
BYTES = 104154875
SHA256 = "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
WRITE_BINDING_SCHEMA = "trn22-r4-preflight-lock-write-binding-hold-v1"
HOST_LOCK = ROOT + "/.cache/trainium_freshstart/device0.lock"
LOCK_NAME = "device0.lock"
MAX_FINAL_SCAN_TO_OPEN_NS = 1_000_000_000
STABLE_IDENTITY_KEYS = ("hostname", "uid", "boot_id", "mount_namespace",
                        "mount", "block_aliases", "ancestry")
PRE_FREE = 13177973911
POST_FREE = 10737418240
MIN_INODES = 10000
CHUNK = 4 * 1024 * 1024
CHECKPOINT = (ROOT + "/trainium_freshstart_20260924_current_base_control_trn22_r1"
              "/out/fs-20260924-current-base-control-trn22-r1-work/train/out/final.pt")
CHECKPOINT_BYTES = 469831009
CHECKPOINT_SHA256 = "1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b"
CACHE_ROOT = (ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1"
              "/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train")
CACHE_INODES = {"neff": 2430420, "local": 2430421, "hlo": 2430422,
                "inductor": 2430423, "compiler": 2430424, "debug": 2430425}
STOP_PATHS = [
    ROOT + "/research/agentic/STOP", ROOT + "/research/v2/STOP",
    ROOT + "/trainium_freshstart_20260918/research/agentic/STOP",
    ROOT + "/trainium_freshstart_20260918/research/v2/STOP",
]
for _revision in ("r1", "r2", "r3"):
    _root = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_" + _revision
    STOP_PATHS.extend((_root + "/research/agentic/STOP", _root + "/research/v2/STOP"))
PROCESS_MARKERS = ("train.py", "eval-public", "torchrun", "preflight_dryrun.py",
                   "contained.py", "root_free_guard_monitor.py", "paired_root_free_guard_monitor.py",
                   "fresh_start_launch.py", "run_experiment.py", "prepare.py", "phase_timer",
                   "reconcile", "run_job.py", "fresh_start_guard.py",
                   "fresh_start_monitor_interlock.py", "a0_boundary_observer.py",
                   "g2-probe", "g2_preflight", "neuronx-cc")


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _all_stop_paths():
    return STOP_PATHS + [CHECKOUT + "/research/agentic/STOP",
                         CHECKOUT + "/research/v2/STOP"]


def _decode_write_binding(token):
    if not isinstance(token, str) or len(token) > 65536:
        raise RuntimeError("missing or oversized exact preflight binding")
    try:
        raw = base64.b64decode(token.encode("ascii"), altchars=b"-_", validate=True)
        if len(raw) > 49152:
            raise RuntimeError("oversized decoded preflight binding")
        binding = json.loads(raw)
    except (UnicodeError, ValueError, binascii.Error) as error:
        raise RuntimeError("malformed exact preflight binding") from error
    if not isinstance(binding, dict) or raw != _canonical_bytes(binding):
        raise RuntimeError("noncanonical exact preflight binding")
    return binding


def _require_preflight_binding(binding, live_identity, live_lock):
    required = {"schema", "source_commit", "target", "capsule_sha256",
                "preflight_utc", "preflight_process_identity",
                "preflight_receipt_sha256", "identity_pin", "lock_identity_pin"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise RuntimeError("incomplete exact preflight binding")
    if (binding["schema"] != WRITE_BINDING_SCHEMA or
            binding["source_commit"] != SOURCE_COMMIT or
            binding["target"] != TARGET or
            binding["capsule_sha256"] != SHA256 or
            not isinstance(binding["preflight_receipt_sha256"], str) or
            re.fullmatch(r"[0-9a-f]{64}", binding["preflight_receipt_sha256"]) is None or
            not isinstance(binding["preflight_process_identity"], dict) or
            set(binding["preflight_process_identity"]) != {"pid", "start_ticks"} or
            not all(isinstance(binding["preflight_process_identity"][key], int) and
                    binding["preflight_process_identity"][key] > 0
                    for key in ("pid", "start_ticks")) or
            not isinstance(binding["identity_pin"], dict) or
            set(binding["identity_pin"]) != set(STABLE_IDENTITY_KEYS) or
            binding["identity_pin"] != {key: live_identity[key]
                                         for key in STABLE_IDENTITY_KEYS} or
            not isinstance(binding["lock_identity_pin"], dict) or
            binding["lock_identity_pin"] != live_lock):
        raise RuntimeError("preflight-to-write host/boot/mount/lock identity mismatch")
    try:
        preflight_time = datetime.datetime.fromisoformat(binding["preflight_utc"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("malformed preflight time") from error
    if preflight_time.tzinfo is None:
        raise RuntimeError("preflight time has no timezone")
    age = (datetime.datetime.now(datetime.timezone.utc) - preflight_time).total_seconds()
    if not -5 <= age <= 180:
        raise RuntimeError("preflight expired before remote write")
    if binding["preflight_process_identity"] == _process_identity():
        raise RuntimeError("write did not enter a distinct SSH process")
    return hashlib.sha256(_canonical_bytes(binding)).hexdigest()


def _mount():
    matches = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            before, sep, after = line.partition(" - ")
            if not sep:
                raise RuntimeError("malformed mountinfo")
            left, right = before.split(), after.split()
            if len(left) < 5 or len(right) < 2:
                raise RuntimeError("malformed mountinfo fields")
            point = left[4]
            if point == ROOT or ROOT.startswith(point.rstrip("/") + "/"):
                matches.append((len(point), {
                    "mount_id": left[0], "major_minor": left[2], "point": point,
                    "filesystem": right[0], "source": right[1],
                }))
    if not matches:
        raise RuntimeError("no mount covers /home/ubuntu")
    selected = max(matches, key=lambda pair: pair[0])[1]
    if (selected["filesystem"] != "ext4" or selected["source"] != MOUNT_SOURCE or
            selected["major_minor"] != MOUNT_MAJOR_MINOR):
        raise RuntimeError("persistent root ext4 source does not match frozen target")
    return selected


def _block_aliases(mount):
    """Read-only kernel-device reconciliation of both historical root names."""
    major, minor = (int(value) for value in mount["major_minor"].split(":"))
    expected_rdev = os.makedev(major, minor)
    names = {}
    for path in (MOUNT_SOURCE, HISTORICAL_DEVICE_NAME):
        try:
            named = os.lstat(path)
        except FileNotFoundError:
            if path != MOUNT_SOURCE:
                raise RuntimeError("historical root block-device name is absent")
            names[path] = {"exists": False, "mount_source_string_only": True}
            continue
        resolved = os.stat(path)
        if not stat.S_ISBLK(resolved.st_mode) or resolved.st_rdev != expected_rdev:
            raise RuntimeError("root-name alias resolves to a different block device")
        names[path] = {"exists": True,
                       "name_type": "symlink" if stat.S_ISLNK(named.st_mode) else
                                    "block" if stat.S_ISBLK(named.st_mode) else "other",
                       "name_device": named.st_dev, "name_inode": named.st_ino,
                       "resolved_device": resolved.st_dev,
                       "resolved_inode": resolved.st_ino,
                       "resolved_rdev_major_minor": "%d:%d" %
                                                    (os.major(resolved.st_rdev), os.minor(resolved.st_rdev))}
    sysfs = "/sys/dev/block/" + mount["major_minor"]
    link = os.readlink(sysfs)
    real = os.path.realpath(sysfs)
    if not real.startswith("/sys/devices/") or not real.endswith("/nvme0n1p1"):
        raise RuntimeError("sysfs root block-device binding is unexpected")
    return {"mount_major_minor": mount["major_minor"],
            "historical_names": names, "sysfs_link": link,
            "sysfs_realpath": real}


def _open_parent():
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    root_fd = os.open("/", flags)
    try:
        root_info = os.fstat(root_fd)
        home_fd = os.open("home", flags, dir_fd=root_fd)
        try:
            home_info = os.fstat(home_fd)
            parent_fd = os.open("ubuntu", flags, dir_fd=home_fd)
            parent_info = os.fstat(parent_fd)
            if (root_info.st_uid != 0 or home_info.st_uid != 0 or
                    parent_info.st_uid != UID or
                    any(not stat.S_ISDIR(item.st_mode) or
                        stat.S_IMODE(item.st_mode) & 0o022
                        for item in (root_info, home_info, parent_info))):
                os.close(parent_fd)
                raise RuntimeError("unsafe root/home/ubuntu ancestry")
            named = os.lstat(ROOT)
            if (named.st_dev, named.st_ino) != (parent_info.st_dev, parent_info.st_ino):
                os.close(parent_fd)
                raise RuntimeError("/home/ubuntu parent inode changed")
            return parent_fd, {
                "root": _dir_identity(root_info), "home": _dir_identity(home_info),
                "ubuntu": _dir_identity(parent_info),
            }
        finally:
            os.close(home_fd)
    finally:
        os.close(root_fd)


def _meta(info):
    kind = ("directory" if stat.S_ISDIR(info.st_mode) else
            "regular" if stat.S_ISREG(info.st_mode) else
            "symlink" if stat.S_ISLNK(info.st_mode) else "other")
    return {"exists": True, "type": kind, "device": info.st_dev,
            "inode": info.st_ino, "uid": info.st_uid, "gid": info.st_gid,
            "mode": oct(stat.S_IMODE(info.st_mode)), "nlink": info.st_nlink,
            "bytes": info.st_size}


def _dir_identity(info):
    return {"type": "directory", "device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "gid": info.st_gid,
            "mode": oct(stat.S_IMODE(info.st_mode))}


def _lock_identity(parent_fd, parts):
    """Authenticate the existing cooperative lock path without modifying it."""
    cache_fd, lock_dir_fd, lock_fd = parts
    parent_device = os.fstat(parent_fd).st_dev
    cache = os.fstat(cache_fd)
    cache_named = os.stat(".cache", dir_fd=parent_fd, follow_symlinks=False)
    lock_dir = os.fstat(lock_dir_fd)
    lock_dir_named = os.stat("trainium_freshstart", dir_fd=cache_fd,
                             follow_symlinks=False)
    for opened, named in ((cache, cache_named), (lock_dir, lock_dir_named)):
        if (not stat.S_ISDIR(opened.st_mode) or
                _dir_identity(opened) != _dir_identity(named) or
                opened.st_uid != UID or opened.st_dev != parent_device or
                stat.S_IMODE(opened.st_mode) & 0o022):
            raise RuntimeError("existing cooperative-lock ancestry is unsafe or changed")
    opened = os.fstat(lock_fd)
    named = os.stat(LOCK_NAME, dir_fd=lock_dir_fd, follow_symlinks=False)
    if (not stat.S_ISREG(opened.st_mode) or _meta(opened) != _meta(named) or
            opened.st_uid != UID or opened.st_dev != parent_device or
            opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) & 0o002):
        raise RuntimeError("existing cooperative lock is absent, replaced or unsafe")
    return {"path": HOST_LOCK, "cache_directory": _dir_identity(cache),
            "lock_directory": _dir_identity(lock_dir), "file": _meta(opened)}


def _open_existing_lock(parent_fd):
    """No O_CREAT, no writes and no symlink traversal at any lock component."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parts = []
    try:
        parts.append(os.open(".cache", flags, dir_fd=parent_fd))
        parts.append(os.open("trainium_freshstart", flags, dir_fd=parts[0]))
        parts.append(os.open(LOCK_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=parts[1]))
        return tuple(parts), _lock_identity(parent_fd, parts)
    except BaseException:
        for fd in reversed(parts):
            os.close(fd)
        raise


def _close_lock(parts):
    for fd in reversed(parts):
        os.close(fd)


def _inspect_existing_lock(parent_fd):
    parts, record = _open_existing_lock(parent_fd)
    try:
        return record
    finally:
        _close_lock(parts)


def _identity(parent_fd, ancestry):
    if socket.gethostname() != HOST or os.geteuid() != UID:
        raise RuntimeError("wrong host or UID")
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot = stream.read().strip()
    if not boot:
        raise RuntimeError("missing boot ID")
    parent = os.fstat(parent_fd)
    if _dir_identity(parent) != ancestry["ubuntu"]:
        raise RuntimeError("stage parent identity changed")
    fs = os.statvfs(ROOT)
    mount = _mount()
    return {"hostname": socket.gethostname(), "uid": os.geteuid(),
            "boot_id": boot, "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "mount": mount, "block_aliases": _block_aliases(mount),
            "ancestry": ancestry,
            "root_free_bytes": fs.f_bavail * fs.f_frsize,
            "root_free_inodes": fs.f_favail}


def _stable(before, after):
    for key in ("hostname", "uid", "boot_id", "mount_namespace", "mount",
                "block_aliases", "ancestry"):
        if before[key] != after[key]:
            raise RuntimeError("host/boot/mount/parent drift: " + key)


def _process_identity():
    with open("/proc/self/stat", encoding="ascii") as stream:
        raw = stream.read()
    return {"pid": os.getpid(), "start_ticks": int(raw.rsplit(")", 1)[1].split()[19])}


def _safe_path(path):
    if not path.startswith("/") or path == "/" or "//" in path or "/../" in path:
        raise RuntimeError("invalid fixed inventory path")
    parts = path.strip("/").split("/")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for index, part in enumerate(parts[:-1]):
            if part in ("", ".", ".."):
                raise RuntimeError("bad inventory ancestor")
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except FileNotFoundError:
                return {"exists": False}
            except NotADirectoryError as error:
                # An old Git worktree may have a regular .git file rather
                # than a .git directory. Classify that one known form; never
                # follow a symlink or silently hide another bad ancestor.
                if not (path.endswith("/.git/HEAD") and part == ".git" and
                        index == len(parts) - 2):
                    raise RuntimeError("non-directory inventory ancestor") from error
                named = os.stat(part, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISREG(named.st_mode):
                    raise RuntimeError("old .git ancestor is not a regular file") from error
                blocker_fd = os.open(part, os.O_RDONLY | os.O_NONBLOCK |
                                     os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                try:
                    opened = os.fstat(blocker_fd)
                    linked = os.stat(part, dir_fd=fd, follow_symlinks=False)
                    if (not stat.S_ISREG(opened.st_mode) or
                            _meta(opened) != _meta(named) or
                            _meta(opened) != _meta(linked)):
                        raise RuntimeError("old .git file changed during inventory")
                    return {"exists": False, "file_backed_git": _meta(opened)}
                finally:
                    os.close(blocker_fd)
            os.close(fd)
            fd = child
        try:
            return _meta(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False))
        except FileNotFoundError:
            return {"exists": False}
    finally:
        os.close(fd)


def _safe_open_regular(path):
    """Open every ancestor by descriptor with O_NOFOLLOW, never following links."""
    if not path.startswith("/") or "//" in path or "/../" in path:
        raise RuntimeError("invalid fixed regular path")
    parts = path.strip("/").split("/")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                            os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                       dir_fd=fd)
    finally:
        os.close(fd)


def _digest_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest, count = hashlib.sha256(), 0
    while True:
        block = os.read(fd, CHUNK)
        if not block:
            return count, digest.hexdigest()
        count += len(block)
        digest.update(block)


def _checked_target(fd, parent_fd):
    opened = os.fstat(fd)
    linked = os.stat(TARGET_NAME, dir_fd=parent_fd, follow_symlinks=False)
    if (_meta(opened) != _meta(linked) or not stat.S_ISREG(opened.st_mode) or
            opened.st_nlink != 1 or opened.st_uid != UID or
            stat.S_IMODE(opened.st_mode) != 0o600 or
            opened.st_dev != os.fstat(parent_fd).st_dev):
        raise RuntimeError("archive type, inode, ownership or mode mismatch")
    return _meta(opened)


def _old_paths():
    result = []
    for revision in ("r1", "r2", "r3"):
        checkout = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_" + revision
        label = "fs-20260925-v2-c5c-a0-control-trn22-" + revision
        result.extend((checkout, checkout + "/out", checkout + "/out/" + label,
                       checkout + "/out/" + label + "-work",
                       checkout + "/train.py", checkout + "/.git",
                       checkout + "/.git/HEAD",
                       checkout + "/out/" + label + "/caches/train",
                       checkout + "/out/" + label + "/train/out/final.pt",
                       checkout + "/out/" + label + "-work/caches",
                       checkout + "/out/" + label + "-work/caches/train",
                       checkout + "/out/" + label + "-work/train/out/final.pt"))
    result.extend((
        "/dev/shm/trainium-v2-g2-20260925-trn22-r3",
        ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1",
        ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out",
        CACHE_ROOT, CHECKPOINT,
    ))
    return result


def _old_inventory(parent_fd):
    paths = {path: _safe_path(path) for path in _old_paths()}
    prefix = "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_"
    dynamic = {}
    for entry in os.scandir(parent_fd):
        if entry.name.startswith(prefix):
            dynamic[ROOT + "/" + entry.name] = _safe_path(ROOT + "/" + entry.name)
    return {"paths": paths, "top_level_matching_sources": dynamic,
            "catalog_complete": len(paths) == len(_old_paths())}


def _protected():
    checkpoint = _safe_path(CHECKPOINT)
    if checkpoint.get("type") != "regular":
        raise RuntimeError("protected checkpoint absent or not regular")
    fd = _safe_open_regular(CHECKPOINT)
    try:
        opened = _meta(os.fstat(fd))
        count, digest = _digest_fd(fd)
        if opened != checkpoint or _meta(os.fstat(fd)) != opened:
            raise RuntimeError("protected checkpoint changed during read")
        checkpoint = {**checkpoint, "read_bytes": count, "sha256": digest}
    finally:
        os.close(fd)
    caches = {name: _safe_path(CACHE_ROOT + "/" + name) for name in CACHE_INODES}
    return {"checkpoint": checkpoint, "cache_root": _safe_path(CACHE_ROOT),
            "six_caches": caches}


def _processes():
    ancestor = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestor:
        ancestor.add(pid)
        try:
            with open("/proc/%d/stat" % pid, encoding="ascii") as stream:
                pid = int(stream.read().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, PermissionError):
            break
    matches, unreadable = [], 0
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) in ancestor:
            continue
        path = "/proc/" + name
        try:
            with open(path + "/cmdline", "rb") as stream:
                raw = stream.read(65537)
        except FileNotFoundError:
            continue
        except PermissionError:
            unreadable += 1
            continue
        if len(raw) > 65536:
            unreadable += 1
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        hits = [marker for marker in PROCESS_MARKERS if marker in command]
        if hits:
            matches.append({"pid": int(name), "markers": hits,
                            "argv_sha256": hashlib.sha256(raw).hexdigest()})
    return {"matched": sorted(matches, key=lambda item: item["pid"]),
            "unreadable_cmdlines": unreadable}


def _fuser(path):
    if not _safe_path(path)["exists"]:
        return {"path_exists": False, "rc": None, "pids": []}
    result = subprocess.run(["fuser", path], capture_output=True, timeout=20, check=False)
    pids = sorted({int(part) for part in re.findall(rb"\b\d+\b", result.stdout)})
    return {"path_exists": True, "rc": result.returncode, "pids": pids,
            "stderr_empty": not bool(result.stderr.strip()),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}


def _neuron_ls():
    result = subprocess.run(["neuron-ls", "-j", "-a"], capture_output=True,
                            timeout=20, check=False)
    try:
        parsed = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    valid = isinstance(parsed, list) and all(isinstance(device, dict) for device in parsed)
    non_diagnostic = []
    if valid:
        for device in parsed:
            groups = device.get("neuron_processes")
            if not isinstance(groups, list):
                valid = False
                break
            for process in groups:
                if not isinstance(process, dict):
                    valid = False
                    break
                if not str(process.get("command", "")).startswith("neuron-ls"):
                    non_diagnostic.append(process.get("pid"))
    return {"rc": result.returncode, "json_valid": parsed is not None,
            "schema_valid": valid, "device_count": len(parsed) if isinstance(parsed, list) else 0,
            "non_diagnostic_pids": non_diagnostic,
            "stderr_empty": not bool(result.stderr.strip()),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}


def _label_scan():
    result = subprocess.run(["find", ROOT, "-xdev", "-name", "*" + LABEL + "*", "-print"],
                            capture_output=True, timeout=45, check=False)
    paths = result.stdout.decode("utf-8", "replace").splitlines()
    return {"rc": result.returncode, "matches": paths[:20],
            "match_count": len(paths), "truncated": len(paths) > 20,
            "stderr_empty": not bool(result.stderr.strip()),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest()}


def _require_empty_label(scan):
    if (not isinstance(scan, dict) or scan.get("rc") != 0 or
            scan.get("match_count") != 0 or scan.get("matches") != [] or
            scan.get("truncated") is not False or scan.get("stderr_empty") is not True or
            not isinstance(scan.get("stdout_sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", scan["stdout_sha256"]) is None):
        raise RuntimeError("science label appeared or final scan is uncertain")


def _final_label_observation():
    started_utc = _utc()
    started_ns = time.monotonic_ns()
    scan = _label_scan()
    completed_ns = time.monotonic_ns()
    completed_utc = _utc()
    _require_empty_label(scan)
    if completed_ns < started_ns:
        raise RuntimeError("final label scan monotonic clock went backwards")
    return {"started_utc": started_utc, "completed_utc": completed_utc,
            "started_monotonic_ns": started_ns,
            "completed_monotonic_ns": completed_ns, "scan": scan}


def probe():
    parent_fd, ancestry = _open_parent()
    try:
        before = _identity(parent_fd, ancestry)
        host_lock_identity = _inspect_existing_lock(parent_fd)
        old = _old_inventory(parent_fd)
        targets = {path: _safe_path(path) for path in (TARGET, CHECKOUT, MONITOR,
                   CHECKOUT + "/out", CHECKOUT + "/out/" + LABEL,
                   CHECKOUT + "/out/" + LABEL + "-work",
                   CHECKOUT + "/research/run-logs/" + LABEL + ".log",
                   CHECKOUT + "/research/run-logs/" + LABEL + ".steps.jsonl",
                   CHECKOUT + "/research/run-logs/" + LABEL + ".process.json",
                   CHECKOUT + "/research/run-logs/" + LABEL + ".postflight.json",
                   OBSERVER + "/" + LABEL + ".observer.json",
                   OBSERVER + "/" + LABEL + ".observer.stderr.log")}
        stops = {path: _safe_path(path) for path in _all_stop_paths()}
        protected = _protected()
        checkpoint_fuser = _fuser(CHECKPOINT)
        processes = _processes()
        lock = _fuser(ROOT + "/.cache/trainium_freshstart/device0.lock")
        neuron0 = _fuser("/dev/neuron0")
        neuron_ls = _neuron_ls()
        label_scan = _label_scan()
        after = _identity(parent_fd, ancestry)
        _stable(before, after)
        if _inspect_existing_lock(parent_fd) != host_lock_identity:
            raise RuntimeError("cooperative lock path changed during read-only probe")
        return {"schema": SCHEMA, "operation": "probe", "utc": _utc(),
                "process_identity": _process_identity(), "identity_before": before,
                "identity_after": after, "old_inventory": old, "targets": targets,
                "host_lock_identity": host_lock_identity,
                "stops": stops, "protected": protected, "processes": processes,
                "checkpoint_fuser": checkpoint_fuser, "science_label_scan": label_scan,
                "shared_lock_fuser": lock, "neuron0_fuser": neuron0,
                "neuron_ls": neuron_ls}
    finally:
        os.close(parent_fd)


def _transfer_receipt(operation, before, after, metadata, read_sha, streamed_sha=None,
                      prewrite_evidence=None):
    _stable(before, after)
    if metadata["bytes"] != BYTES or read_sha != SHA256:
        raise RuntimeError("capsule bytes or SHA mismatch")
    receipt = {"schema": SCHEMA, "operation": operation,
            "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY", "utc": _utc(),
            "process_identity": _process_identity(), "path": TARGET,
            "expected_bytes": BYTES, "expected_sha256": SHA256,
            "streamed_sha256": streamed_sha, "readback_sha256": read_sha,
            "target_stat": metadata, "identity_before": before,
            "identity_after": after}
    if prewrite_evidence is not None:
        receipt["prewrite_evidence"] = prewrite_evidence
    return receipt


def _require_no_stops_or_processes():
    stops = {path: _safe_path(path) for path in _all_stop_paths()}
    if len(stops) != len(_all_stop_paths()) or any(
            item.get("exists") is not False for item in stops.values()):
        raise RuntimeError("live host STOP appeared before archive creation")
    processes = _processes()
    if (processes.get("matched") != [] or
            processes.get("unreadable_cmdlines") != 0):
        raise RuntimeError("live host process busy or uncertain before archive creation")
    return {"stops": stops, "processes": processes}


def _live_prewrite_safety(*, owned_lock=None):
    """Rerun live host gates after preflight, before the sole O_EXCL create."""
    safety = _require_no_stops_or_processes()
    # The label scan can be comparatively slow. Put it before the quick
    # holder/Neuron checks so their result is as close to O_EXCL as possible.
    label = _label_scan()
    checkpoint_fuser = _fuser(CHECKPOINT)
    if owned_lock is None:
        lock = _fuser(HOST_LOCK)
        owned_record = None
    else:
        parent_fd, parts, expected = owned_lock
        owned_record = _lock_identity(parent_fd, parts)
        if owned_record != expected:
            raise RuntimeError("cooperative lock identity changed while held")
        lock = None
    neuron0 = _fuser("/dev/neuron0")
    neuron_ls = _neuron_ls()
    if (checkpoint_fuser.get("path_exists") is not True or
            checkpoint_fuser.get("rc") != 1 or
            checkpoint_fuser.get("pids") != [] or
            checkpoint_fuser.get("stderr_empty") is not True or
            (lock is not None and (lock.get("pids") != [] or
             (lock.get("path_exists") and lock.get("rc") != 1) or
             (lock.get("path_exists") and lock.get("stderr_empty") is not True))) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            neuron_ls.get("rc") != 0 or neuron_ls.get("stderr_empty") is not True or
            neuron_ls.get("json_valid") is not True or
            neuron_ls.get("schema_valid") is not True or
            neuron_ls.get("device_count", 0) < 1 or
            neuron_ls.get("non_diagnostic_pids") != [] or
            label.get("rc") != 0 or label.get("match_count") != 0 or
            label.get("matches") != [] or label.get("truncated") is not False or
            label.get("stderr_empty") is not True):
        raise RuntimeError("live host checkpoint, lock, Neuron or label is busy/uncertain")
    return {**safety, "checkpoint_fuser": checkpoint_fuser,
            "shared_lock_fuser": lock, "cooperative_lock": owned_record,
            "neuron0_fuser": neuron0,
            "neuron_ls": neuron_ls, "science_label_scan": label}


def write(input_stream, binding):
    parent_fd, ancestry = _open_parent()
    lock_parts = None
    try:
        before = _identity(parent_fd, ancestry)
        lock_pin = _inspect_existing_lock(parent_fd)
        binding_sha = _require_preflight_binding(binding, before, lock_pin)
        if before["root_free_bytes"] < PRE_FREE or before["root_free_inodes"] < MIN_INODES:
            raise RuntimeError("fresh persistent root free-space/inode gate failed")
        if _safe_path(TARGET)["exists"]:
            raise FileExistsError("one-shot target already exists; no overwrite or retry")
        live_safety = _live_prewrite_safety()
        before_create = _identity(parent_fd, ancestry)
        _stable(before, before_create)
        _require_preflight_binding(binding, before_create, lock_pin)
        if (before_create["root_free_bytes"] < PRE_FREE or
                before_create["root_free_inodes"] < MIN_INODES):
            raise RuntimeError("persistent root space/inode gate changed before archive creation")
        if _safe_path(TARGET)["exists"]:
            raise FileExistsError("one-shot target appeared before exclusive create")
        # This is the exact existing file used by the frozen 446b launcher.
        # Never create, truncate, rewrite, delete or replace it. An exclusive
        # nonblocking flock excludes that cooperative launcher while held.
        lock_parts, lock_record = _open_existing_lock(parent_fd)
        if lock_record != lock_pin:
            raise RuntimeError("cooperative lock drifted after preflight")
        try:
            fcntl.flock(lock_parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("cooperative host lock is already held") from error
        lock_acquired_ns = time.monotonic_ns()
        if _lock_identity(parent_fd, lock_parts) != lock_pin:
            raise RuntimeError("cooperative lock path changed after acquisition")
        # Recheck all live gates while holding the lock. fuser on the lock
        # would see our own FD, so this check authenticates our held inode
        # instead; checkpoint/Neuron/STOP/process/label checks still run.
        final_live_safety = _live_prewrite_safety(
            owned_lock=(parent_fd, lock_parts, lock_pin))
        last_safety = _require_no_stops_or_processes()
        last_lock = _lock_identity(parent_fd, lock_parts)
        if last_lock != lock_pin:
            raise RuntimeError("cooperative lock path changed before final label scan")
        # The broad label scan is deliberately the LAST potentially slow
        # operation before O_EXCL. This closes the r3 late-label gap for
        # cooperating launchers, but cannot atomically exclude manual work.
        final_label = _final_label_observation()
        target_open_attempt_ns = time.monotonic_ns()
        if (target_open_attempt_ns - final_label["completed_monotonic_ns"] >
                MAX_FINAL_SCAN_TO_OPEN_NS):
            raise RuntimeError("final label observation expired before O_EXCL")
        fd = os.open(TARGET_NAME, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        try:
            target_open_completed_ns = time.monotonic_ns()
            if (target_open_completed_ns - target_open_attempt_ns >
                    MAX_FINAL_SCAN_TO_OPEN_NS):
                raise RuntimeError("O_EXCL followed an expired final label scan; partial target retained")
            lock_after_create = _lock_identity(parent_fd, lock_parts)
            if lock_after_create != lock_pin:
                raise RuntimeError("cooperative lock path changed at O_EXCL; partial target retained")
            streamed, count = hashlib.sha256(), 0
            while True:
                block = input_stream.read(CHUNK)
                if not block:
                    break
                count += len(block)
                if count > BYTES:
                    raise RuntimeError("too many source bytes; partial target retained")
                streamed.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise RuntimeError("short write; partial target retained")
                    view = view[written:]
            if count != BYTES or streamed.hexdigest() != SHA256:
                raise RuntimeError("source mismatch; partial target retained")
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            os.fsync(parent_fd)
            metadata = _checked_target(fd, parent_fd)
            read_bytes, read_sha = _digest_fd(fd)
            if read_bytes != BYTES or _checked_target(fd, parent_fd) != metadata:
                raise RuntimeError("same-FD readback mismatch; target retained")
            after = _identity(parent_fd, ancestry)
            if after["root_free_bytes"] < POST_FREE:
                raise RuntimeError("post-write free-space floor failed; target retained")
            lock_after_write = _lock_identity(parent_fd, lock_parts)
            if lock_after_write != lock_pin:
                raise RuntimeError("cooperative lock path changed during write; target retained")
            evidence = {"preflight_binding_sha256": binding_sha,
                        "preflight_receipt_sha256": binding["preflight_receipt_sha256"],
                        "preflight_lock_identity": lock_pin,
                        "cooperative_lock_acquired_monotonic_ns": lock_acquired_ns,
                        "cooperative_lock_after_create": lock_after_create,
                        "cooperative_lock_after_write": lock_after_write,
                        "live_prewrite_safety": live_safety,
                        "last_live_precreate_safety": final_live_safety,
                        "last_stop_process_check": last_safety,
                        "final_label_observation": final_label,
                        "target_open_attempt_monotonic_ns": target_open_attempt_ns,
                        "target_open_completed_monotonic_ns": target_open_completed_ns}
            return _transfer_receipt("write", before, after, metadata, read_sha,
                                     streamed.hexdigest(), prewrite_evidence=evidence)
        finally:
            os.close(fd)
    finally:
        if lock_parts is not None:
            _close_lock(lock_parts)
        os.close(parent_fd)


def verify():
    parent_fd, ancestry = _open_parent()
    try:
        before = _identity(parent_fd, ancestry)
        fd = os.open(TARGET_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=parent_fd)
        try:
            metadata = _checked_target(fd, parent_fd)
            count, digest = _digest_fd(fd)
            if count != BYTES or _checked_target(fd, parent_fd) != metadata:
                raise RuntimeError("read-only target changed")
            after = _identity(parent_fd, ancestry)
            if after["root_free_bytes"] < POST_FREE:
                raise RuntimeError("post-stage free-space floor failed")
            return _transfer_receipt("verify", before, after, metadata, digest)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def main():
    raise SystemExit("DISPATCH HOLD: trn22 lock/final-label repair is not an enabled request")


if __name__ == "__main__":
    main()
