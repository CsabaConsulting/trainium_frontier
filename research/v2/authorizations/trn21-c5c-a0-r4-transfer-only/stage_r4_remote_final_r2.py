"""Final r2 trn21 r4 transfer-only remote payload, pending independent review.

This is a separate successor to the held final payload. It adds a late
science-label scan and a complete activity recheck after that scan, before
O_EXCL. The disabled and held final files stay unchanged.
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


SCHEMA = "trn21-c5c-a0-r4-transfer-only-v1"
HOST = "ip-172-31-34-85"
UID = 1000
ROOT = "/home/ubuntu"
MOUNT_SOURCE = "/dev/root"
MOUNT_MAJOR_MINOR = "259:1"
HISTORICAL_DEVICE_NAME = "/dev/nvme0n1p1"
TARGET_NAME = "trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz"
TARGET = ROOT + "/" + TARGET_NAME
CHECKOUT = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r4"
E65_CHECKOUT = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
MONITOR = ROOT + "/trainium_a0_disk_monitor_20260925_trn21_r4"
OBSERVER = ROOT + "/trainium_a0_boundary_observer_20260925_trn21_r4"
LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r4"
E65_LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r1"
BYTES = 104154875
SHA256 = "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a"
PRE_FREE = 13177973911
POST_FREE = 10737418240
MIN_INODES = 10000
CHUNK = 4 * 1024 * 1024
CHECKPOINTS = {
    (ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1"
     "/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/sandbox/out/final.pt"):
    (469834849, "aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535", 2947631),
    (ROOT + "/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2"
     "/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt"):
    (469831009, "5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9", 2938565),
}
CACHE_ROOT = (ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1"
              "/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train")
CACHE_ROOT_INODE = 2945950
CACHE_INODES = {"neff": 2945951, "local": 2945952, "hlo": 2945953,
                "inductor": 2945954, "compiler": 2945955, "debug": 2945956}
E65_CHECKOUT_INODE = 3168319
E65_SOURCE = {
    ".git/HEAD": (41, "bf830a9e8a8e9c4c4bd24938bfc4ab7ff0945d5f8550339d9e1c2928ac6f98f6", 3168322),
    "train.py": (316834, "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586", 3174933),
    "prepare.py": (35467, "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb", 3168417),
}
E65_TRANSFER_FILES = {
    ROOT + "/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz":
    (104005787, "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264", 296045),
    ROOT + "/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py":
    (23550, "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8", 294946),
    ROOT + "/trainium-v2-g2-contained-20260925-trn21-stage-r2.py":
    (9185, "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d", 296647),
}
STOP_PATHS = [
    ROOT + "/research/agentic/STOP", ROOT + "/research/v2/STOP",
    ROOT + "/trainium_freshstart_20260918/research/agentic/STOP",
    ROOT + "/trainium_freshstart_20260918/research/v2/STOP",
    E65_CHECKOUT + "/research/agentic/STOP", E65_CHECKOUT + "/research/v2/STOP",
]
for _revision in ("r1", "r2"):
    _root = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_" + _revision
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


def _identity_digest(identity):
    """Bind a later writer to the entire prior read-only host identity."""
    pinned = {key: identity[key] for key in (
        "hostname", "uid", "boot_id", "mount_namespace", "mount",
        "block_aliases", "ancestry")}
    return hashlib.sha256(json.dumps(pinned, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


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
            except NotADirectoryError as error:
                raise RuntimeError("HOLD: non-directory ancestor in fixed inventory: " + path) from error
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
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                                os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except NotADirectoryError as error:
                raise RuntimeError("HOLD: non-directory ancestor in protected file: " + path) from error
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
    result = [
        "/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz",
        ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1",
        ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r2",
        ROOT + "/trainium-v2-g2-20260925-trn21-r2",
        ROOT + "/trainium-v2-g2-20260925-trn21-stage-r2",
        E65_CHECKOUT, E65_CHECKOUT + "/.git/HEAD",
        E65_CHECKOUT + "/train.py", E65_CHECKOUT + "/prepare.py",
        E65_CHECKOUT + "/out", E65_CHECKOUT + "/out/" + E65_LABEL,
        E65_CHECKOUT + "/out/" + E65_LABEL + "-work",
        E65_CHECKOUT + "/research/run-logs/" + E65_LABEL + ".log",
        E65_CHECKOUT + "/research/run-logs/" + E65_LABEL + ".steps.jsonl",
        CACHE_ROOT,
    ]
    result.extend(E65_TRANSFER_FILES)
    result.extend(CHECKPOINTS)
    return result


def _old_inventory(parent_fd):
    paths = {path: _safe_path(path) for path in _old_paths()}
    prefix = "trainium_freshstart_20260925_v2_c5c_a0_control_trn21_"
    dynamic = {}
    for entry in os.scandir(parent_fd):
        if entry.name.startswith(prefix):
            dynamic[ROOT + "/" + entry.name] = _safe_path(ROOT + "/" + entry.name)
    return {"paths": paths, "top_level_matching_sources": dynamic,
            "catalog_complete": len(paths) == len(_old_paths())}


def _read_pinned_regular(path, size, digest, inode=None):
    observed = _safe_path(path)
    if observed.get("type") != "regular":
        raise RuntimeError("protected file absent or not regular: " + path)
    fd = _safe_open_regular(path)
    try:
        opened = _meta(os.fstat(fd))
        count, actual = _digest_fd(fd)
        if (opened != observed or _meta(os.fstat(fd)) != opened or
                opened["bytes"] != size or count != size or actual != digest or
                opened["device"] != 66305 or opened["uid"] != UID or
                opened["nlink"] != 1 or
                (inode is not None and opened["inode"] != inode)):
            raise RuntimeError("protected file identity or SHA changed: " + path)
        return {**observed, "read_bytes": count, "sha256": actual}
    finally:
        os.close(fd)


def _protected():
    checkpoints = {path: _read_pinned_regular(path, *pin)
                   for path, pin in CHECKPOINTS.items()}
    transferred = {path: _read_pinned_regular(path, *pin)
                   for path, pin in E65_TRANSFER_FILES.items()}
    e65_checkout = _safe_path(E65_CHECKOUT)
    if (e65_checkout.get("type") != "directory" or
            e65_checkout.get("device") != 66305 or
            e65_checkout.get("inode") != E65_CHECKOUT_INODE or
            e65_checkout.get("uid") != UID or
            e65_checkout.get("mode") != "0o700"):
        raise RuntimeError("accepted e65 checkout identity changed")
    e65_source = {name: _read_pinned_regular(E65_CHECKOUT + "/" + name, *pin)
                  for name, pin in E65_SOURCE.items()}
    caches = {name: _safe_path(CACHE_ROOT + "/" + name) for name in CACHE_INODES}
    return {"checkpoints": checkpoints, "e65_transfer_files": transferred,
            "e65_checkout": e65_checkout, "e65_source": e65_source,
            "cache_root": _safe_path(CACHE_ROOT), "six_caches": caches}


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
        targets = {path: _safe_path(path) for path in _target_paths()}
        stops = {path: _safe_path(path) for path in STOP_PATHS + [
            CHECKOUT + "/research/agentic/STOP", CHECKOUT + "/research/v2/STOP"]}
        protected = _protected()
        checkpoint_fusers = {path: _fuser(path) for path in CHECKPOINTS}
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
                "checkpoint_fusers": checkpoint_fusers, "science_label_scan": label_scan,
                "shared_lock_fuser": lock, "neuron0_fuser": neuron0,
                "neuron_ls": neuron_ls}
    finally:
        os.close(parent_fd)


def _prewrite_safety(snapshot):
    """Rerun the full host gate in the writer process before target creation."""
    if snapshot.get("schema") != SCHEMA or snapshot.get("operation") != "probe":
        raise RuntimeError("writer did not obtain its own full safety probe")
    before, after = snapshot["identity_before"], snapshot["identity_after"]
    _stable(before, after)
    if (before["root_free_bytes"] < PRE_FREE or after["root_free_bytes"] < PRE_FREE or
            before["root_free_inodes"] < MIN_INODES or
            after["root_free_inodes"] < MIN_INODES):
        raise RuntimeError("writer-process root space/inode gate failed")
    old = snapshot["old_inventory"]
    if old.get("catalog_complete") is not True or set(old["paths"]) != set(_old_paths()):
        raise RuntimeError("writer-process old-path catalog incomplete")
    if any(item.get("exists") not in (True, False) or
           (item.get("exists") and item.get("type") not in ("directory", "regular"))
           for item in old["paths"].values()):
        raise RuntimeError("writer-process old path unsafe")
    allowed = {ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_" + name
               for name in ("r1", "r2", "stage_r2", "r4")}
    if not set(old["top_level_matching_sources"]).issubset(allowed):
        raise RuntimeError("writer-process unknown old source root")
    if old["top_level_matching_sources"].get(E65_CHECKOUT) != old["paths"][E65_CHECKOUT]:
        raise RuntimeError("writer-process accepted e65 source omitted from dynamic inventory")
    if any(old["paths"].get(path) != item
           for path, item in old["top_level_matching_sources"].items()):
        raise RuntimeError("writer-process old source catalog mismatch")
    if old["paths"][E65_CHECKOUT + "/out/" + E65_LABEL].get("exists") is not False:
        raise RuntimeError("accepted e65 checkout already has science output")
    if (set(snapshot["targets"]) != set(_target_paths()) or
            any(item.get("exists") is not False for item in snapshot["targets"].values())):
        raise RuntimeError("writer-process proposed target already exists")
    if (set(snapshot["stops"]) != set(STOP_PATHS + [
            CHECKOUT + "/research/agentic/STOP", CHECKOUT + "/research/v2/STOP"]) or
            any(item.get("exists") is not False for item in snapshot["stops"].values())):
        raise RuntimeError("writer-process STOP gate failed")
    if (snapshot["processes"].get("matched") != [] or
            snapshot["processes"].get("unreadable_cmdlines") != 0):
        raise RuntimeError("writer-process activity gate failed")
    label = snapshot["science_label_scan"]
    if (label.get("rc") != 0 or label.get("matches") != [] or
            label.get("match_count") != 0 or label.get("truncated") is not False or
            label.get("stderr_empty") is not True):
        raise RuntimeError("writer-process science label gate failed")
    if set(snapshot["checkpoint_fusers"]) != set(CHECKPOINTS):
        raise RuntimeError("writer-process checkpoint-holder inventory incomplete")
    for item in snapshot["checkpoint_fusers"].values():
        if (item.get("path_exists") is not True or item.get("rc") != 1 or
                item.get("pids") != [] or item.get("stderr_empty") is not True):
            raise RuntimeError("writer-process protected checkpoint has a holder")
    lock, neuron0, devices = (snapshot["shared_lock_fuser"],
                              snapshot["neuron0_fuser"], snapshot["neuron_ls"])
    if (lock.get("pids") != [] or
            (lock.get("path_exists") and
             (lock.get("rc") != 1 or lock.get("stderr_empty") is not True)) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            devices.get("rc") != 0 or devices.get("stderr_empty") is not True or
            devices.get("json_valid") is not True or
            devices.get("schema_valid") is not True or
            devices.get("device_count", 0) < 1 or
            devices.get("non_diagnostic_pids") != []):
        raise RuntimeError("writer-process Neuron or shared-lock gate failed")
    # _protected() already rehashed both checkpoints, all three prior stage
    # files and the accepted e65 HEAD/train/prepare under no-follow opens.
    protected = snapshot["protected"]
    if (set(protected["checkpoints"]) != set(CHECKPOINTS) or
            set(protected["e65_transfer_files"]) !=
            set(E65_TRANSFER_FILES) or
            set(protected["e65_source"]) != set(E65_SOURCE)):
        raise RuntimeError("writer-process protected inventory incomplete")
    if (protected["e65_checkout"]["inode"] != E65_CHECKOUT_INODE or
            protected["e65_checkout"]["device"] != 66305 or
            protected["e65_checkout"]["mode"] != "0o700" or
            old["paths"][E65_CHECKOUT] != protected["e65_checkout"]):
        raise RuntimeError("writer-process accepted e65 checkout changed")
    for path, item in {**protected["checkpoints"],
                       **protected["e65_transfer_files"]}.items():
        if old["paths"][path] != {key: value for key, value in item.items()
                                  if key not in ("read_bytes", "sha256")}:
            raise RuntimeError("writer-process protected old file inventory raced")
    for name, item in protected["e65_source"].items():
        path = E65_CHECKOUT + "/" + name
        if old["paths"][path] != {key: value for key, value in item.items()
                                  if key not in ("read_bytes", "sha256")}:
            raise RuntimeError("writer-process accepted e65 source inventory raced")
    cache_root = protected["cache_root"]
    if (cache_root.get("type") != "directory" or cache_root.get("uid") != UID or
            cache_root.get("device") != 66305 or
            cache_root.get("inode") != CACHE_ROOT_INODE or
            old["paths"][CACHE_ROOT] != cache_root or
            set(protected["six_caches"]) != set(CACHE_INODES)):
        raise RuntimeError("writer-process protected warm-cache root changed")
    for name, inode in CACHE_INODES.items():
        item = protected["six_caches"][name]
        if (item.get("type") != "directory" or item.get("uid") != UID or
                item.get("device") != 66305 or item.get("inode") != inode):
            raise RuntimeError("writer-process protected warm-cache child changed")
    return snapshot


def _target_paths():
    return (TARGET, CHECKOUT, MONITOR, OBSERVER,
            CHECKOUT + "/out", CHECKOUT + "/out/" + LABEL,
            CHECKOUT + "/out/" + LABEL + "-work",
            CHECKOUT + "/research/run-logs/" + LABEL + ".log",
            CHECKOUT + "/research/run-logs/" + LABEL + ".steps.jsonl",
            CHECKOUT + "/research/run-logs/" + LABEL + ".process.json",
            CHECKOUT + "/research/run-logs/" + LABEL + ".postflight.json",
            OBSERVER + "/" + LABEL + ".observer.json",
            OBSERVER + "/" + LABEL + ".observer.stderr.log")


def _activity_gate_snapshot():
    """Record a complete activity/holder/target gate before target creation."""
    targets = {path: _safe_path(path) for path in _target_paths()}
    if any(item.get("exists") is not False for item in targets.values()):
        raise RuntimeError("future target appeared immediately before target creation")
    stops = {path: _safe_path(path) for path in STOP_PATHS + [
        CHECKOUT + "/research/agentic/STOP", CHECKOUT + "/research/v2/STOP"]}
    if any(item.get("exists") is not False for item in stops.values()):
        raise RuntimeError("host STOP appeared immediately before target creation")
    processes = _processes()
    if processes.get("matched") != [] or processes.get("unreadable_cmdlines") != 0:
        raise RuntimeError("host process appeared immediately before target creation")
    checkpoint_fusers = {path: _fuser(path) for path in CHECKPOINTS}
    for item in checkpoint_fusers.values():
        if (item.get("path_exists") is not True or item.get("rc") != 1 or
                item.get("pids") != [] or item.get("stderr_empty") is not True):
            raise RuntimeError("checkpoint holder appeared before target creation")
    lock, neuron0, devices = (_fuser(ROOT + "/.cache/trainium_freshstart/device0.lock"),
                              _fuser("/dev/neuron0"), _neuron_ls())
    if (lock.get("pids") != [] or
            (lock.get("path_exists") and
             (lock.get("rc") != 1 or lock.get("stderr_empty") is not True)) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            devices.get("rc") != 0 or devices.get("stderr_empty") is not True or
            devices.get("json_valid") is not True or
            devices.get("schema_valid") is not True or
            devices.get("device_count", 0) < 1 or
            devices.get("non_diagnostic_pids") != []):
        raise RuntimeError("Neuron or shared lock appeared before target creation")
    # The holder/Neuron commands can take time. Repeat the cheap gates at the
    # very end so a STOP/process/future target appearing during them cannot
    # slip through to the O_EXCL call.
    stops = {path: _safe_path(path) for path in STOP_PATHS + [
        CHECKOUT + "/research/agentic/STOP", CHECKOUT + "/research/v2/STOP"]}
    if any(item.get("exists") is not False for item in stops.values()):
        raise RuntimeError("host STOP appeared at final target-creation gate")
    processes = _processes()
    if processes.get("matched") != [] or processes.get("unreadable_cmdlines") != 0:
        raise RuntimeError("host process appeared at final target-creation gate")
    targets = {path: _safe_path(path) for path in _target_paths()}
    if any(item.get("exists") is not False for item in targets.values()):
        raise RuntimeError("future target appeared at final target-creation gate")
    return {"utc": _utc(), "targets": targets, "stops": stops, "processes": processes,
            "checkpoint_fusers": checkpoint_fusers,
            "shared_lock_fuser": lock, "neuron0_fuser": neuron0,
            "neuron_ls": devices}


def _immediate_activity_gate():
    """Retain the first and pre-label creation gates as distinct observations."""
    return _activity_gate_snapshot()


def _final_creation_gate():
    """Scan the label after the second gate, then recheck all live activity."""
    pre_label = _immediate_activity_gate()
    label_started = _utc()
    label = _label_scan()
    label_completed = _utc()
    if (not isinstance(label, dict) or label.get("rc") != 0 or
            label.get("matches") != [] or label.get("match_count") != 0 or
            label.get("truncated") is not False or
            label.get("stderr_empty") is not True or
            label.get("stdout_sha256") != hashlib.sha256(b"").hexdigest()):
        raise RuntimeError("science label appeared at final target-creation gate")
    # The find may take up to 45 seconds. Close its own STOP/process/
    # checkpoint/lock/Neuron/target gap with one more complete live gate.
    final = _activity_gate_snapshot()
    final["science_label_scan"] = {**label, "started_utc": label_started,
                                    "completed_utc": label_completed}
    final["pre_label_gate"] = pre_label
    return final


def _transfer_receipt(operation, before, after, metadata, read_sha,
                      streamed_sha=None, prewrite_probe=None,
                      preflight_identity_sha256=None, immediate_gate=None,
                      final_creation_gate=None):
    _stable(before, after)
    if metadata["bytes"] != BYTES or read_sha != SHA256:
        raise RuntimeError("capsule bytes or SHA mismatch")
    return {"schema": SCHEMA, "operation": operation,
            "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY", "utc": _utc(),
            "process_identity": _process_identity(), "path": TARGET,
            "expected_bytes": BYTES, "expected_sha256": SHA256,
            "streamed_sha256": streamed_sha, "readback_sha256": read_sha,
            "target_stat": metadata, "identity_before": before,
            "identity_after": after,
            "prewrite_probe": prewrite_probe,
            "preflight_identity_sha256": preflight_identity_sha256,
            "immediate_gate": immediate_gate,
            "final_creation_gate": final_creation_gate}


def write(input_stream, preflight_identity_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", preflight_identity_sha256):
        raise RuntimeError("missing exact preflight identity bridge")
    prewrite = _prewrite_safety(probe())
    if _identity_digest(prewrite["identity_after"]) != preflight_identity_sha256:
        raise RuntimeError("host identity changed after separate preflight")
    parent_fd, ancestry = _open_parent()
    try:
        before = _identity(parent_fd, ancestry)
        if _identity_digest(before) != preflight_identity_sha256:
            raise RuntimeError("host identity changed before one-shot target creation")
        if before["root_free_bytes"] < PRE_FREE or before["root_free_inodes"] < MIN_INODES:
            raise RuntimeError("fresh persistent root free-space/inode gate failed")
        immediate = _immediate_activity_gate()
        if _safe_path(TARGET)["exists"]:
            raise FileExistsError("one-shot target already exists; no overwrite or retry")
        # The second complete gate precedes a fresh science-label scan. A
        # third complete gate closes activity that begins during the scan.
        # Nothing except the exclusive open follows the combined gate.
        final_creation = _final_creation_gate()
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
                                     streamed.hexdigest(), prewrite,
                                     preflight_identity_sha256, immediate,
                                     final_creation)
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
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("probe", "write", "verify"):
        raise SystemExit("fixed reviewed operation required: probe|write <pin>|verify")
    operation = sys.argv[1]
    if (operation == "write") != (len(sys.argv) == 3):
        raise SystemExit("write alone requires a preflight identity pin")
    result = (probe() if operation == "probe" else
              write(sys.stdin.buffer, sys.argv[2]) if operation == "write" else verify())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
