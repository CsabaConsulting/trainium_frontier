"""Read-only, no-follow facts for the two exact A2 CPU-G2 tool paths.

Stream to ``python3 -B - trn22`` only under a separately reviewed Operator
request. No file, lock, marker, process, Neuron job, or AWS resource is changed.
The older pinned snapshot covers STOP/occupancy/disk and the pinned ACL probe
covers the linked Git directory; this adds bounded tool bytes and live RAM.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import stat
import sys


CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
TOOLS = {
    "g2": CHECKOUT + "/workflow/tools/preflight_dryrun.py",
    "contained": CHECKOUT + "/workflow/tools/contained.py",
}
HOSTNAME = "ip-172-31-32-253"
MAX_TOOL_BYTES = 2_000_000
OPEN_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
OPEN_REGULAR = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def kind(info: os.stat_result) -> str:
    if stat.S_ISREG(info.st_mode):
        return "regular"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    return "other"


def metadata(info: os.stat_result) -> dict:
    return {
        "type": kind(info), "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "gid": info.st_gid,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        "nlink": info.st_nlink, "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return all(getattr(first, key) == getattr(second, key) for key in (
        "st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
        "st_size", "st_mtime_ns", "st_ctime_ns",
    ))


def nofollow_regular(path: str) -> dict:
    """Hash only the fixed leaf beneath a no-follow directory-fd walk.

    Missing or non-regular paths are observations, not exceptions. All errors
    fail closed in the later collector decision; no path is followed by name.
    """
    if path not in TOOLS.values():
        raise ValueError("not a fixed tool path")
    components = path.strip("/").split("/")
    parent = os.open("/", OPEN_DIRECTORY)
    try:
        for part in components[:-1]:
            try:
                child = os.open(part, OPEN_DIRECTORY, dir_fd=parent)
            except FileNotFoundError:
                return {"exists": False, "reason": "missing_ancestor", "component": part}
            except (NotADirectoryError, OSError) as error:
                return {"exists": None, "reason": "unsafe_ancestor", "error": type(error).__name__}
            os.close(parent)
            parent = child
        leaf = components[-1]
        try:
            named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return {"exists": False, "reason": "missing_leaf"}
        except OSError as error:
            return {"exists": None, "reason": "leaf_stat_error", "error": type(error).__name__}
        result = {"exists": True, **metadata(named)}
        if not stat.S_ISREG(named.st_mode):
            return result
        if named.st_size > MAX_TOOL_BYTES:
            return {**result, "hash_error": "tool_exceeds_2m_bound"}
        try:
            fd = os.open(leaf, OPEN_REGULAR, dir_fd=parent)
        except OSError as error:
            return {**result, "hash_error": "open_" + type(error).__name__}
        try:
            opened = os.fstat(fd)
            if not same_identity(named, opened):
                return {**result, "hash_error": "identity_changed_on_open"}
            digest = hashlib.sha256()
            count = 0
            while count <= MAX_TOOL_BYTES:
                chunk = os.read(fd, min(65536, MAX_TOOL_BYTES + 1 - count))
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
            final_fd = os.fstat(fd)
            final_name = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (count != opened.st_size or not same_identity(opened, final_fd)
                    or not same_identity(opened, final_name)):
                return {**result, "hash_error": "identity_changed_during_read"}
            return {**result, "sha256": digest.hexdigest(), "read_bytes": count}
        except OSError as error:
            return {**result, "hash_error": "read_" + type(error).__name__}
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def mem_available() -> int | None:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB" and fields[1].isdecimal():
                    return int(fields[1]) * 1024
                return None
    return None


def main() -> int:
    if sys.argv[1:] != ["trn22"]:
        raise SystemExit("exact trn22 read-only target required")
    started = utc()
    if socket.gethostname() != HOSTNAME or os.geteuid() != 1000:
        print(json.dumps({"schema": "trn22-a2-toolpath-readonly-v1",
                          "identity_error": "hostname_or_uid_mismatch"}, sort_keys=True))
        return 2
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot_id = stream.read().strip()
    answer = {
        "schema": "trn22-a2-toolpath-readonly-v1",
        "scope": "point_in_time_tool_path_and_memory_feasibility_only",
        "started_utc": started, "ended_utc": None,
        "host": "trn22", "hostname": socket.gethostname(), "uid": os.geteuid(),
        "boot_id": boot_id, "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "root_device": os.stat("/").st_dev,
        "checkout_path": CHECKOUT,
        "checkout": nofollow_checkout(),
        "memory_available_bytes": mem_available(),
        "tools": {name: {"path": path, **nofollow_regular(path)}
                  for name, path in TOOLS.items()},
    }
    answer["ended_utc"] = utc()
    print(json.dumps(answer, sort_keys=True, separators=(",", ":")))
    return 0


def nofollow_checkout() -> dict:
    """Metadata for the fixed directory; no linked path is followed here."""
    parent = os.open("/", OPEN_DIRECTORY)
    try:
        for part in ("home", "ubuntu"):
            child = os.open(part, OPEN_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        leaf = CHECKOUT.rsplit("/", 1)[1]
        try:
            named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return {"exists": False}
        if not stat.S_ISDIR(named.st_mode):
            return {"exists": True, **metadata(named)}
        child = os.open(leaf, OPEN_DIRECTORY, dir_fd=parent)
        try:
            opened = os.fstat(child)
            return {"exists": True, **metadata(named),
                    "opened_same_inode": same_identity(named, opened)}
        finally:
            os.close(child)
    finally:
        os.close(parent)


if __name__ == "__main__":
    raise SystemExit(main())
