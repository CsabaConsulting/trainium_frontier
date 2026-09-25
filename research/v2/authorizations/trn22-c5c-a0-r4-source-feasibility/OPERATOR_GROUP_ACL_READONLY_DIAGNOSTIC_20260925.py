"""One-off read-only trn22 lock-ancestry, group and POSIX-ACL diagnostic."""

import datetime as dt
import errno
import hashlib
import json
import os
import socket
import stat
import struct
import subprocess


DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
LOCK_PATH = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"
ACL_TAGS = {1: "USER_OBJ", 2: "USER", 4: "GROUP_OBJ", 8: "GROUP",
            16: "MASK", 32: "OTHER"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identity():
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot = stream.read().strip()
    return {"hostname": socket.gethostname(), "uid": os.geteuid(),
            "gid": os.getegid(), "supplementary_gids": os.getgroups(),
            "boot_id": boot, "mount_namespace": os.readlink("/proc/self/ns/mnt")}


def meta(info):
    kind = ("directory" if stat.S_ISDIR(info.st_mode) else
            "regular" if stat.S_ISREG(info.st_mode) else
            "symlink" if stat.S_ISLNK(info.st_mode) else "other")
    return {"type": kind, "is_symlink": stat.S_ISLNK(info.st_mode),
            "uid": info.st_uid, "gid": info.st_gid,
            "mode": oct(stat.S_IMODE(info.st_mode)), "device": info.st_dev,
            "inode": info.st_ino, "nlink": info.st_nlink, "bytes": info.st_size,
            "atime_ns": info.st_atime_ns, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns}


def same(left, right):
    keys = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(left, key) == getattr(right, key) for key in keys)


def acl_for_fd(fd, attribute):
    try:
        raw = os.getxattr(fd, attribute)
    except OSError as error:
        if error.errno == errno.ENODATA:
            return {"present": False, "errno": error.errno,
                    "meaning": "no extended POSIX ACL xattr on this inode"}
        return {"present": None, "errno": error.errno,
                "error": type(error).__name__ + ": " + str(error)}
    output = {"present": True, "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest(), "entries": []}
    if len(raw) < 4 or (len(raw) - 4) % 8 or struct.unpack_from("<I", raw)[0] != 2:
        output["parse_error"] = "unexpected POSIX ACL xattr version or length"
        return output
    for offset in range(4, len(raw), 8):
        tag, perms, qualifier = struct.unpack_from("<HHI", raw, offset)
        output["entries"].append({"tag": ACL_TAGS.get(tag, "UNKNOWN_" + str(tag)),
                                  "perm": oct(perms),
                                  "id": None if qualifier == 0xffffffff else qualifier})
    return output


def account_facts():
    result = {"scope": "local /etc/group and /etc/passwd plus bounded NSS lookup",
              "gid": 1000, "local_group_records": [],
              "local_primary_gid_accounts": []}
    for path, kind in (("/etc/group", "group"), ("/etc/passwd", "passwd")):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1048576:
                raise RuntimeError("unsafe or oversized account database: " + path)
            raw = bytearray()
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                raw.extend(block)
                if len(raw) > 1048576:
                    raise RuntimeError("oversized account database: " + path)
            result[kind + "_file"] = {"bytes": len(raw),
                                     "sha256": hashlib.sha256(raw).hexdigest()}
            for line in raw.decode("utf-8").splitlines():
                fields = line.split(":")
                if kind == "group" and len(fields) >= 4 and fields[2] == "1000":
                    result["local_group_records"].append({
                        "group_name": fields[0], "gid": 1000,
                        "explicit_members": [v for v in fields[3].split(",") if v]})
                if kind == "passwd" and len(fields) >= 4 and fields[3] == "1000":
                    result["local_primary_gid_accounts"].append({
                        "name": fields[0], "uid": int(fields[2]), "primary_gid": 1000})
        finally:
            os.close(fd)
    for key, argv in (("nss_group_1000", ["getent", "group", "1000"]),
                      ("nss_passwd_ubuntu", ["getent", "passwd", "ubuntu"])):
        try:
            completed = subprocess.run(argv, capture_output=True, timeout=10,
                                       check=False)
            item = {"rc": completed.returncode, "stdout_bytes": len(completed.stdout),
                    "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                    "stderr_bytes": len(completed.stderr),
                    "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest()}
            if len(completed.stdout) < 65536 and completed.returncode == 0:
                lines = completed.stdout.decode("utf-8").splitlines()
                if len(lines) == 1:
                    fields = lines[0].split(":")
                    if key == "nss_group_1000" and len(fields) >= 4:
                        item["parsed"] = {"group_name": fields[0], "gid": int(fields[2]),
                                          "explicit_members": [v for v in fields[3].split(",") if v]}
                    if key == "nss_passwd_ubuntu" and len(fields) >= 4:
                        item["parsed"] = {"name": fields[0], "uid": int(fields[2]),
                                          "primary_gid": int(fields[3])}
            result[key] = item
        except BaseException as error:
            result[key] = {"error": type(error).__name__ + ": " + str(error)}
    return result


def relevant_mounts():
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    entries = []
    for line in lines:
        head = line.split(" - ", 1)[0].split()
        if len(head) >= 5:
            mount_root, mount_point = head[3], head[4]
            if any(fragment in value for value in (mount_root, mount_point)
                   for fragment in ("/home/ubuntu", "/.cache", "trainium_freshstart")):
                entries.append({"major_minor": head[2], "mount_root": mount_root,
                                "mount_point": mount_point})
    return {"mount_count": len(lines), "relevant_entries": entries}


result = {"schema": "trn22-group-acl-readonly-v1", "utc_started": utc(),
          "path": LOCK_PATH, "components": {}, "status": "HOLD"}
fds = []
try:
    result["identity_before"] = identity()
    if (result["identity_before"]["hostname"] != "ip-172-31-32-253" or
            result["identity_before"]["uid"] != 1000):
        raise RuntimeError("wrong trn22 host or UID")
    result["account_facts"] = account_facts()
    result["mountinfo"] = relevant_mounts()
    root_fd = os.open("/", DIR_FLAGS)
    fds.append(root_fd)

    def checked(parent_fd, name, path, directory):
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        item = {"named": meta(named)}
        result["components"][path] = item
        if stat.S_ISLNK(named.st_mode) or (directory and not stat.S_ISDIR(named.st_mode)):
            raise RuntimeError("symlink or wrong type: " + path)
        if not directory and (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1):
            raise RuntimeError("nonregular or multiply linked lock file")
        fd = os.open(name, DIR_FLAGS if directory else FILE_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(fd)
        item["opened"] = meta(opened)
        if not same(named, opened):
            os.close(fd)
            raise RuntimeError("name/inode changed across no-follow open: " + path)
        item["access_acl"] = acl_for_fd(fd, "system.posix_acl_access")
        if directory:
            item["default_acl"] = acl_for_fd(fd, "system.posix_acl_default")
        fds.append(fd)
        return fd

    home_fd = checked(root_fd, "home", "/home", True)
    ubuntu_fd = checked(home_fd, "ubuntu", "/home/ubuntu", True)
    cache_fd = checked(ubuntu_fd, ".cache", "/home/ubuntu/.cache", True)
    host_fd = checked(cache_fd, "trainium_freshstart",
                      "/home/ubuntu/.cache/trainium_freshstart", True)
    lock_fd = checked(host_fd, "device0.lock", LOCK_PATH, False)
    for path, fd in (("/home/ubuntu", ubuntu_fd),
                     ("/home/ubuntu/.cache", cache_fd),
                     ("/home/ubuntu/.cache/trainium_freshstart", host_fd),
                     (LOCK_PATH, lock_fd)):
        result["components"][path]["opened_after"] = meta(os.fstat(fd))
        result["components"][path]["stable"] = same(
            os.stat(path, follow_symlinks=False), os.fstat(fd))
    result["status"] = "PASS_READ_ONLY_GROUP_ACL_OBSERVATION"
except BaseException as error:
    result["status"] = "HOLD"
    result["error"] = type(error).__name__ + ": " + str(error)
finally:
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError as error:
            result.setdefault("close_errors", []).append(type(error).__name__ + ": " + str(error))
    result["identity_after"] = identity()
    if result.get("identity_before") != result["identity_after"]:
        result["status"] = "HOLD"
        result["identity_drift"] = True
    result["utc_finished"] = utc()
print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
