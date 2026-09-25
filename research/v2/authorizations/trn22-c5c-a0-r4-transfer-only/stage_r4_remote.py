"""Fixed trn22 r4 persistent-ext4 capsule probe/writer/reader.

This source is a HOLD-only proposal. An independent review and a separate,
one-shot Supervisor transfer-only assignment are prerequisites to execution.
There is no extraction, checkout, deletion, rename, retry, or training path.
"""

import datetime
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys


SCHEMA = "trn22-c5c-a0-r4-transfer-only-v1"
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
        for part in parts[:-1]:
            if part in ("", ".", ".."):
                raise RuntimeError("bad inventory ancestor")
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except FileNotFoundError:
                return {"exists": False}
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
                       checkout + "/train.py", checkout + "/.git/HEAD",
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


def probe():
    parent_fd, ancestry = _open_parent()
    try:
        before = _identity(parent_fd, ancestry)
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
        stops = {path: _safe_path(path) for path in STOP_PATHS + [
            CHECKOUT + "/research/agentic/STOP", CHECKOUT + "/research/v2/STOP"]}
        protected = _protected()
        checkpoint_fuser = _fuser(CHECKPOINT)
        processes = _processes()
        lock = _fuser(ROOT + "/.cache/trainium_freshstart/device0.lock")
        neuron0 = _fuser("/dev/neuron0")
        neuron_ls = _neuron_ls()
        label_scan = _label_scan()
        after = _identity(parent_fd, ancestry)
        _stable(before, after)
        return {"schema": SCHEMA, "operation": "probe", "utc": _utc(),
                "process_identity": _process_identity(), "identity_before": before,
                "identity_after": after, "old_inventory": old, "targets": targets,
                "stops": stops, "protected": protected, "processes": processes,
                "checkpoint_fuser": checkpoint_fuser, "science_label_scan": label_scan,
                "shared_lock_fuser": lock, "neuron0_fuser": neuron0,
                "neuron_ls": neuron_ls}
    finally:
        os.close(parent_fd)


def _transfer_receipt(operation, before, after, metadata, read_sha, streamed_sha=None):
    _stable(before, after)
    if metadata["bytes"] != BYTES or read_sha != SHA256:
        raise RuntimeError("capsule bytes or SHA mismatch")
    return {"schema": SCHEMA, "operation": operation,
            "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY", "utc": _utc(),
            "process_identity": _process_identity(), "path": TARGET,
            "expected_bytes": BYTES, "expected_sha256": SHA256,
            "streamed_sha256": streamed_sha, "readback_sha256": read_sha,
            "target_stat": metadata, "identity_before": before,
            "identity_after": after}


def write(input_stream):
    parent_fd, ancestry = _open_parent()
    try:
        before = _identity(parent_fd, ancestry)
        if before["root_free_bytes"] < PRE_FREE or before["root_free_inodes"] < MIN_INODES:
            raise RuntimeError("fresh persistent root free-space/inode gate failed")
        if _safe_path(TARGET)["exists"]:
            raise FileExistsError("one-shot target already exists; no overwrite or retry")
        fd = os.open(TARGET_NAME, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        try:
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
            return _transfer_receipt("write", before, after, metadata, read_sha,
                                     streamed.hexdigest())
        finally:
            os.close(fd)
    finally:
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
    if len(sys.argv) != 2 or sys.argv[1] not in ("probe", "write", "verify"):
        raise SystemExit("fixed reviewed operation required: probe|write|verify")
    operation = sys.argv[1]
    result = (probe() if operation == "probe" else
              write(sys.stdin.buffer) if operation == "write" else verify())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
