"""Read-only trn22 e65-r2 Git pointer, path ACL, and local GID-1000 probe."""

from __future__ import annotations

import datetime as dt
import errno
import hashlib
import json
import os
import socket
import stat
import struct
import sys


ROOT = "/home/ubuntu"
CHECKOUT = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
GIT_POINTER = CHECKOUT + "/.git"
ACL_TAGS = {1: "user_obj", 2: "user", 4: "group_obj", 8: "group",
            16: "mask", 32: "other"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(path: str) -> dict:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {"exists": False}
    kind = (
        "directory" if stat.S_ISDIR(info.st_mode) else
        "regular" if stat.S_ISREG(info.st_mode) else
        "symlink" if stat.S_ISLNK(info.st_mode) else "other"
    )
    return {"exists": True, "type": kind, "device": info.st_dev,
            "inode": info.st_ino, "uid": info.st_uid, "gid": info.st_gid,
            "mode": format(stat.S_IMODE(info.st_mode), "04o"),
            "nlink": info.st_nlink, "bytes": info.st_size}


def parse_acl(raw: bytes) -> dict:
    answer = {"bytes": len(raw), "sha256": sha(raw), "valid": False, "entries": []}
    if len(raw) < 4 or (len(raw) - 4) % 8 != 0:
        return answer
    version = struct.unpack_from("<I", raw, 0)[0]
    if version != 2:
        answer["version"] = version
        return answer
    for position in range(4, len(raw), 8):
        tag, perm, ident = struct.unpack_from("<HHI", raw, position)
        answer["entries"].append({
            "tag": ACL_TAGS.get(tag, f"unknown_{tag}"),
            "id": None if ident == 0xFFFFFFFF else ident,
            "perm": "".join(symbol if perm & bit else "-"
                            for symbol, bit in (("r", 4), ("w", 2), ("x", 1))),
        })
    answer["valid"] = True
    return answer


def acl(path: str, name: str) -> dict:
    try:
        raw = os.getxattr(path, name, follow_symlinks=False)
    except OSError as error:
        if error.errno == errno.ENODATA:
            return {"present": False, "errno": "ENODATA"}
        return {"present": None, "errno": errno.errorcode.get(error.errno, "unknown")}
    return {"present": True, **parse_acl(raw)}


def entry(path: str) -> dict:
    result = {"path": path, "metadata": metadata(path)}
    if result["metadata"].get("exists"):
        result["access_acl"] = acl(path, "system.posix_acl_access")
        if result["metadata"].get("type") == "directory":
            result["default_acl"] = acl(path, "system.posix_acl_default")
    return result


def read_pointer() -> dict:
    info = metadata(GIT_POINTER)
    if info.get("type") != "regular" or info["bytes"] > 4096:
        return {"metadata": info, "error": "not_bounded_regular_pointer"}
    fd = os.open(GIT_POINTER, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            info["device"], info["inode"], info["bytes"]
        ):
            raise RuntimeError("pointer_identity_changed_on_open")
        raw = os.read(fd, 4097)
        after = os.fstat(fd)
        if len(raw) != info["bytes"] or (
            after.st_dev, after.st_ino, after.st_size
        ) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise RuntimeError("pointer_identity_changed_during_read")
    finally:
        os.close(fd)
    result = {"metadata": info, "sha256": sha(raw), "raw_bytes": len(raw)}
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        result["error"] = "non_utf8_pointer"
        return result
    if not text.startswith("gitdir: ") or "\n" in text:
        result["error"] = "invalid_gitdir_pointer_syntax"
        return result
    pointer = text[len("gitdir: "):]
    if not os.path.isabs(pointer):
        pointer = os.path.normpath(os.path.join(CHECKOUT, pointer))
    if not (pointer == ROOT or pointer.startswith(ROOT + "/")):
        result["error"] = "pointer_outside_scoped_home"
        return result
    resolved = os.path.realpath(pointer)
    if not (resolved == ROOT or resolved.startswith(ROOT + "/")):
        result["error"] = "resolved_gitdir_outside_scoped_home"
        return result
    result["pointer_path"] = pointer
    result["resolved_gitdir"] = resolved
    result["resolved_gitdir_metadata"] = metadata(resolved)
    return result


def path_ancestry(path: str) -> list[str]:
    if not (path == ROOT or path.startswith(ROOT + "/")):
        raise RuntimeError("path outside scoped home")
    current = path
    values = []
    while True:
        values.append(current)
        if current == "/":
            break
        current = os.path.dirname(current)
    return list(reversed(values))


def gid_1000_local_facts() -> dict:
    passwd_accounts = []
    with open("/etc/passwd", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split(":")
            if len(fields) >= 4 and fields[3] == "1000":
                passwd_accounts.append({"name": fields[0], "uid": fields[2]})
    group_records = []
    with open("/etc/group", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split(":")
            if len(fields) >= 4 and fields[2] == "1000":
                group_records.append({"name": fields[0],
                                      "explicit_members": [value for value in fields[3].split(",") if value]})
    other_primary = [item for item in passwd_accounts if item["name"] != "ubuntu"]
    other_explicit = [name for group in group_records
                      for name in group["explicit_members"] if name != "ubuntu"]
    return {
        "primary_gid_accounts": passwd_accounts,
        "gid_1000_group_records": group_records,
        "other_local_account_shares_gid_1000": bool(other_primary or other_explicit),
    }


def main() -> None:
    if sys.argv[1:] != ["trn22"]:
        raise SystemExit("trn22 only")
    started = utc()
    if socket.gethostname() != "ip-172-31-32-253" or os.geteuid() != 1000:
        print(json.dumps({"schema": "trn22-e65-r2-gitdir-acl-readonly-v1",
                          "utc": started, "identity_error": "hostname_or_uid_mismatch"},
                         sort_keys=True))
        raise SystemExit(2)
    pointer = read_pointer()
    paths = set(path_ancestry(CHECKOUT))
    paths.add(GIT_POINTER)
    if pointer.get("resolved_gitdir"):
        paths.update(path_ancestry(pointer["resolved_gitdir"]))
    result = {
        "schema": "trn22-e65-r2-gitdir-acl-readonly-v1",
        "started_utc": started, "ended_utc": utc(),
        "hostname": socket.gethostname(), "uid": os.geteuid(),
        "boot_id": open("/proc/sys/kernel/random/boot_id", encoding="ascii").read().strip(),
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "checkout": CHECKOUT, "pointer": pointer,
        "ancestry": [entry(path) for path in sorted(paths, key=lambda item: (item.count("/"), item))],
        "gid_1000_local_facts": gid_1000_local_facts(),
    }
    result["ended_utc"] = utc()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
