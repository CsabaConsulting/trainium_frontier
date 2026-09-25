"""Read-only diagnostic for trn22 root aliases and old A0 paths.

This supplements the unmodified feasibility probe after the repaired
transfer probe failed on an old checkout's file-backed ``.git`` parent.
It is planning evidence only and is not the transfer package's preflight.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import stat


ROOT = "/home/ubuntu"
SOURCE_PREFIX = "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_"
CHECKPOINT = (
    ROOT + "/trainium_freshstart_20260924_current_base_control_trn22_r1"
    "/out/fs-20260924-current-base-control-trn22-r1-work/train/out/final.pt"
)
CACHE_ROOT = (
    ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1"
    "/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train"
)


def kind(info: os.stat_result) -> str:
    return (
        "directory" if stat.S_ISDIR(info.st_mode) else
        "regular" if stat.S_ISREG(info.st_mode) else
        "symlink" if stat.S_ISLNK(info.st_mode) else
        "block" if stat.S_ISBLK(info.st_mode) else "other"
    )


def metadata(info: os.stat_result) -> dict:
    return {
        "type": kind(info), "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "mode": oct(stat.S_IMODE(info.st_mode)),
        "nlink": info.st_nlink, "bytes": info.st_size,
    }


def nofollow_path(path: str) -> dict:
    if not path.startswith("/") or path == "/" or "//" in path or "/../" in path:
        raise RuntimeError("bad fixed diagnostic path")
    parts = path.strip("/").split("/")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    traversed = ""
    try:
        for part in parts[:-1]:
            traversed += "/" + part
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
            except FileNotFoundError:
                return {"exists": False, "reason": "missing_ancestor", "ancestor": traversed}
            except NotADirectoryError:
                ancestor = os.stat(part, dir_fd=fd, follow_symlinks=False)
                return {
                    "exists": False,
                    "reason": "nondirectory_ancestor",
                    "ancestor": traversed,
                    "ancestor_metadata": metadata(ancestor),
                }
            os.close(fd)
            fd = child
        try:
            named = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return {"exists": False, "reason": "missing_leaf"}
        return {"exists": True, **metadata(named)}
    finally:
        os.close(fd)


def old_paths() -> list[str]:
    result = []
    for revision in ("r1", "r2", "r3"):
        checkout = ROOT + "/" + SOURCE_PREFIX + revision
        label = "fs-20260925-v2-c5c-a0-control-trn22-" + revision
        result.extend((
            checkout,
            checkout + "/out",
            checkout + "/out/" + label,
            checkout + "/out/" + label + "-work",
            checkout + "/train.py",
            checkout + "/.git",
            checkout + "/.git/HEAD",
            checkout + "/out/" + label + "/caches/train",
            checkout + "/out/" + label + "/train/out/final.pt",
            checkout + "/out/" + label + "-work/caches",
            checkout + "/out/" + label + "-work/caches/train",
            checkout + "/out/" + label + "-work/train/out/final.pt",
        ))
    result.extend((
        "/dev/shm/trainium-v2-g2-20260925-trn22-r3",
        ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1",
        ROOT + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out",
        CACHE_ROOT,
        CHECKPOINT,
    ))
    return result


def mount_root() -> dict:
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            before, separator, after = line.partition(" - ")
            if not separator:
                continue
            left, right = before.split(), after.split()
            if len(left) >= 5 and len(right) >= 2 and left[4] == "/":
                return {
                    "mount_id": left[0], "major_minor": left[2],
                    "point": left[4], "filesystem": right[0], "source": right[1],
                }
    raise RuntimeError("root mount missing")


def block_alias(path: str) -> dict:
    try:
        named = os.lstat(path)
    except FileNotFoundError:
        return {"exists": False}
    resolved = os.stat(path)
    return {
        "exists": True,
        "name": metadata(named),
        "resolved": metadata(resolved),
        "resolved_rdev_major_minor": (
            str(os.major(resolved.st_rdev)) + ":" + str(os.minor(resolved.st_rdev))
        ),
    }


def main() -> None:
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot_id = stream.read().strip()
    root = mount_root()
    sysfs = "/sys/dev/block/" + root["major_minor"]
    old = old_paths()
    dynamic = {
        ROOT + "/" + entry.name: nofollow_path(ROOT + "/" + entry.name)
        for entry in os.scandir(ROOT)
        if entry.name.startswith(SOURCE_PREFIX)
    }
    result = {
        "schema": "trn22-r4-readonly-alias-oldpath-diagnostic-v1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "euid": os.geteuid(),
        "boot_id": boot_id,
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "root_mount": root,
        "root_device": os.stat("/").st_dev,
        "ubuntu_parent": nofollow_path(ROOT),
        "block_aliases": {
            "/dev/root": block_alias("/dev/root"),
            "/dev/nvme0n1p1": block_alias("/dev/nvme0n1p1"),
            "sysfs_link": os.readlink(sysfs),
            "sysfs_realpath": os.path.realpath(sysfs),
        },
        "old_path_catalog_count": len(old),
        "old_paths": {path: nofollow_path(path) for path in old},
        "top_level_matching_sources": dynamic,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
