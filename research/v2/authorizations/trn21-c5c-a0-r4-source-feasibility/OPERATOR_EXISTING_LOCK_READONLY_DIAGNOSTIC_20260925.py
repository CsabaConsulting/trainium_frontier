"""One-off, read-only trn21 existing-lock metadata and flock diagnostic.

Run only by the Operator over the separately verified, strict-host-key SSH
transport. No file is created, written, chmodded, renamed, or removed here.
"""

import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
import stat
import time


DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
LOCK_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
LOCK_PATH = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def host_identity():
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot_id = stream.read().strip()
    return {"hostname": socket.gethostname(), "uid": os.geteuid(),
            "gid": os.getegid(), "supplementary_gids": os.getgroups(),
            "boot_id": boot_id, "mount_namespace": os.readlink("/proc/self/ns/mnt")}


def metadata(value):
    kind = ("directory" if stat.S_ISDIR(value.st_mode) else
            "regular" if stat.S_ISREG(value.st_mode) else
            "symlink" if stat.S_ISLNK(value.st_mode) else "other")
    return {"exists": True, "type": kind, "is_symlink": stat.S_ISLNK(value.st_mode),
            "uid": value.st_uid, "gid": value.st_gid,
            "mode": oct(stat.S_IMODE(value.st_mode)), "device": value.st_dev,
            "inode": value.st_ino, "nlink": value.st_nlink, "bytes": value.st_size,
            "atime_ns": value.st_atime_ns, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns}


def same_metadata(left, right):
    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
              "st_size", "st_atime_ns", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(left, key) == getattr(right, key) for key in fields)


def local_group_1000():
    """Report only local group/passwd membership facts; never raw file lines."""
    output = {"scope": "local /etc/group and /etc/passwd only", "gid": 1000,
              "group_records": [], "primary_gid_accounts": []}
    for filename, key in (("/etc/group", "group"), ("/etc/passwd", "passwd")):
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1048576:
                raise RuntimeError("unsafe or oversized local account database: " + filename)
            chunks = []
            count = 0
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                count += len(block)
                if count > 1048576:
                    raise RuntimeError("oversized local account database: " + filename)
                chunks.append(block)
            raw = b"".join(chunks)
            output[key + "_file"] = {"bytes": count,
                                     "sha256": hashlib.sha256(raw).hexdigest(),
                                     "uid": info.st_uid, "gid": info.st_gid,
                                     "mode": oct(stat.S_IMODE(info.st_mode))}
            for line in raw.decode("utf-8").splitlines():
                fields = line.split(":")
                if key == "group" and len(fields) >= 4 and fields[2] == "1000":
                    output["group_records"].append({"group_name": fields[0],
                                                      "gid": 1000,
                                                      "explicit_members":
                                                      [v for v in fields[3].split(",") if v]})
                if key == "passwd" and len(fields) >= 4 and fields[3] == "1000":
                    output["primary_gid_accounts"].append({"name": fields[0],
                                                            "uid": int(fields[2]),
                                                            "primary_gid": 1000})
        finally:
            os.close(fd)
    return output


result = {"schema": "trn21-existing-lock-readonly-flock-v1",
          "operation": "inspect_existing_lock_and_try_nonblocking_flock",
          "utc_started": utc(), "lock_path": LOCK_PATH,
          "components": {}, "flock": {"attempted": False, "acquired": False,
                                      "released": False}, "status": "HOLD"}
fds = []
held_fd = None
try:
    result["identity_before"] = host_identity()
    if (result["identity_before"]["hostname"] != "ip-172-31-34-85" or
            result["identity_before"]["uid"] != 1000):
        raise RuntimeError("wrong trn21 host or UID")
    result["local_group_1000"] = local_group_1000()
    root_fd = os.open("/", DIR_FLAGS)
    fds.append(root_fd)

    def checked_dir(parent_fd, name, path):
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        item = {"named": metadata(named)}
        result["components"][path] = item
        if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
            raise RuntimeError("non-directory or symlink component: " + path)
        fd = os.open(name, DIR_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(fd)
        item["opened"] = metadata(opened)
        if not same_metadata(named, opened):
            os.close(fd)
            raise RuntimeError("directory changed across no-follow open: " + path)
        fds.append(fd)
        return fd

    home_fd = checked_dir(root_fd, "home", "/home")
    ubuntu_fd = checked_dir(home_fd, "ubuntu", "/home/ubuntu")
    cache_fd = checked_dir(ubuntu_fd, ".cache", "/home/ubuntu/.cache")
    trainium_fd = checked_dir(cache_fd, "trainium_freshstart",
                              "/home/ubuntu/.cache/trainium_freshstart")

    named = os.stat("device0.lock", dir_fd=trainium_fd, follow_symlinks=False)
    item = {"named": metadata(named)}
    result["components"][LOCK_PATH] = item
    if (not stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode) or
            named.st_nlink != 1 or named.st_uid != 1000 or
            named.st_dev != os.fstat(ubuntu_fd).st_dev):
        raise RuntimeError("existing lock file has unsafe no-follow identity")
    if (os.fstat(cache_fd).st_uid != 1000 or
            stat.S_IMODE(os.fstat(cache_fd).st_mode) & 0o077 or
            os.fstat(trainium_fd).st_uid != 1000):
        raise RuntimeError("lock path is not below private owner-controlled cache")
    lock_fd = os.open("device0.lock", LOCK_FLAGS, dir_fd=trainium_fd)
    fds.append(lock_fd)
    opened = os.fstat(lock_fd)
    item["opened"] = metadata(opened)
    if not same_metadata(named, opened):
        raise RuntimeError("lock changed across read-only no-follow open")

    result["flock"]["attempted"] = True
    started_ns = time.monotonic_ns()
    acquired_ns = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired_ns = time.monotonic_ns()
        held_fd = lock_fd
        result["flock"]["acquired"] = True
    except OSError as error:
        result["flock"]["errno"] = error.errno
        result["flock"]["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if held_fd is not None:
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            result["flock"]["released"] = True
            result["flock"]["held_ns"] = time.monotonic_ns() - acquired_ns
            held_fd = None
        result["flock"]["attempt_duration_ns"] = time.monotonic_ns() - started_ns

    named_after = os.stat("device0.lock", dir_fd=trainium_fd, follow_symlinks=False)
    opened_after = os.fstat(lock_fd)
    item["named_after"] = metadata(named_after)
    item["opened_after"] = metadata(opened_after)
    item["unchanged"] = same_metadata(named, named_after) and same_metadata(named, opened_after)
    if not item["unchanged"]:
        raise RuntimeError("lock metadata changed during read-only check")
    result["status"] = ("READ_ONLY_NONBLOCKING_FLOCK_ACQUIRED_AND_RELEASED"
                        if result["flock"]["acquired"] and result["flock"]["released"]
                        else "HOLD_LOCK_NOT_ACQUIRED")
except BaseException as error:
    result["status"] = "HOLD"
    result["error"] = type(error).__name__ + ": " + str(error)
finally:
    if held_fd is not None:
        try:
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            result["flock"]["released_in_cleanup"] = True
        except BaseException as error:
            result["flock"]["cleanup_error"] = type(error).__name__ + ": " + str(error)
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError as error:
            result.setdefault("close_errors", []).append(type(error).__name__ + ": " + str(error))
    result["identity_after"] = host_identity()
    if result.get("identity_before") != result["identity_after"]:
        result["status"] = "HOLD"
        result["identity_drift"] = True
    result["utc_finished"] = utc()
print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
