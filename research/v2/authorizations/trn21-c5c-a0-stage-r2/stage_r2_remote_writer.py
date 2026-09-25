"""Fixed trn21 ext4 source-stage writer/reader; HOLD until separate authority.

The Operator may pass these exact reviewed bytes to ``python3 -c`` in one SSH
process per operation. No path, expected hash, or expected byte count comes
from the remote command line. A failed write deliberately leaves its unique
partial target in place for diagnosis; this module never removes a file.
"""

import datetime
import hashlib
import json
import os
import socket
import stat
import sys


SCHEMA = "trn21-c5c-a0-stage-r2-transfer-v1"
HOST = "ip-172-31-34-85"
UID = 1000
ROOT = "/home/ubuntu"
ROOT_FREE_FLOOR = 10737418240
CHUNK = 4 * 1024 * 1024
ARTIFACTS = {
    "capsule": {
        "name": "trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz",
        "bytes": 104005787,
        "sha256": "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264",
    },
    "preflight": {
        "name": "trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py",
        "bytes": 23550,
        "sha256": "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8",
    },
    "contained": {
        "name": "trainium-v2-g2-contained-20260925-trn21-stage-r2.py",
        "bytes": 9185,
        "sha256": "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d",
    },
}


def _mount_for_home():
    matches = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            before, separator, after = line.partition(" - ")
            if not separator:
                raise RuntimeError("malformed mountinfo")
            left = before.split()
            right = after.split()
            if len(left) < 5 or len(right) < 2:
                raise RuntimeError("malformed mountinfo fields")
            mount_point = left[4]
            if mount_point == ROOT or ROOT.startswith(mount_point.rstrip("/") + "/"):
                matches.append((len(mount_point), {
                    "mount_id": left[0],
                    "major_minor": left[2],
                    "mount_point": mount_point,
                    "filesystem": right[0],
                    "source": right[1],
                }))
    if not matches:
        raise RuntimeError("no mountinfo entry covers stage root")
    mount = max(matches, key=lambda item: item[0])[1]
    if mount["filesystem"] != "ext4":
        raise RuntimeError("stage root is not on persistent ext4")
    return mount


def _identity(parent_fd):
    parent = os.fstat(parent_fd)
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != UID:
        raise RuntimeError("unexpected stage parent owner/type")
    home_ancestor = os.lstat("/home")
    named_parent = os.lstat(ROOT)
    if (not stat.S_ISDIR(home_ancestor.st_mode) or
            not stat.S_ISDIR(named_parent.st_mode) or
            (named_parent.st_dev, named_parent.st_ino) != (parent.st_dev, parent.st_ino) or
            stat.S_IMODE(parent.st_mode) & 0o022):
        raise RuntimeError("stage parent path, inode or permissions are unsafe")
    if socket.gethostname() != HOST or os.geteuid() != UID:
        raise RuntimeError("wrong host or run UID")
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot_id = stream.read().strip()
    if not boot_id:
        raise RuntimeError("missing boot ID")
    namespace = os.readlink("/proc/self/ns/mnt")
    mount = _mount_for_home()
    filesystem = os.statvfs(ROOT)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if free_bytes < ROOT_FREE_FLOOR:
        raise RuntimeError("stage root below immutable prewrite free-space floor")
    return {
        "hostname": socket.gethostname(), "uid": os.geteuid(), "gid": os.getegid(),
        "boot_id": boot_id, "mount_namespace": namespace, "mount": mount,
        "parent_device": parent.st_dev, "parent_inode": parent.st_ino,
        "parent_uid": parent.st_uid, "parent_gid": parent.st_gid,
        "parent_mode": oct(stat.S_IMODE(parent.st_mode)), "root_free_bytes": free_bytes,
    }


def _stable_identity(before, after):
    for key in ("hostname", "uid", "gid", "boot_id", "mount_namespace",
                "mount", "parent_device", "parent_inode", "parent_uid",
                "parent_gid", "parent_mode"):
        if before[key] != after[key]:
            raise RuntimeError("stage host/mount identity changed within session: " + key)


def _file_metadata(fd, parent_fd, name):
    opened = os.fstat(fd)
    linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise RuntimeError("stage target is not regular")
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        raise RuntimeError("stage target changed during verification")
    if opened.st_nlink != 1 or linked.st_nlink != 1:
        raise RuntimeError("stage target hard-link count is not one")
    if opened.st_uid != UID or linked.st_uid != UID:
        raise RuntimeError("stage target UID mismatch")
    if stat.S_IMODE(opened.st_mode) != 0o600 or stat.S_IMODE(linked.st_mode) != 0o600:
        raise RuntimeError("stage target must be private mode 0600")
    return {
        "device": opened.st_dev, "inode": opened.st_ino,
        "bytes": opened.st_size, "uid": opened.st_uid, "gid": opened.st_gid,
        "mode": oct(stat.S_IMODE(opened.st_mode)), "nlink": opened.st_nlink,
    }


def _digest_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, CHUNK)
        if not chunk:
            return size, digest.hexdigest()
        size += len(chunk)
        digest.update(chunk)


def _artifact(artifact):
    if artifact not in ARTIFACTS:
        raise ValueError("unknown fixed artifact")
    spec = ARTIFACTS[artifact]
    if "/" in spec["name"] or spec["name"] in ("", ".", ".."):
        raise RuntimeError("non-basename stage target")
    return spec


def _open_parent():
    return os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


def _receipt(operation, artifact, spec, before, after, metadata, observed_sha256,
             streamed_sha256=None):
    _stable_identity(before, after)
    if metadata["device"] != after["parent_device"]:
        raise RuntimeError("stage target escaped the ext4 parent device")
    if metadata["bytes"] != spec["bytes"] or observed_sha256 != spec["sha256"]:
        raise RuntimeError("stage target byte count or SHA-256 mismatch")
    with open("/proc/self/stat", encoding="ascii") as stream:
        process_stat = stream.read()
    process_start_ticks = int(process_stat.rsplit(")", 1)[1].split()[19])
    return {
        "schema": SCHEMA, "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY",
        "operation": operation, "artifact": artifact,
        "path": ROOT + "/" + spec["name"],
        "expected_bytes": spec["bytes"], "expected_sha256": spec["sha256"],
        "streamed_sha256": streamed_sha256, "readback_sha256": observed_sha256,
        "target_stat": metadata, "identity_before": before, "identity_after": after,
        "process_id": os.getpid(), "process_start_ticks": process_start_ticks,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def write_artifact(artifact, input_stream):
    spec = _artifact(artifact)
    parent_fd = _open_parent()
    try:
        before = _identity(parent_fd)
        try:
            os.stat(spec["name"], dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("stage target already exists; no overwrite or retry")
        fd = os.open(spec["name"], os.O_RDWR | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        try:
            streamed = hashlib.sha256()
            count = 0
            while True:
                chunk = input_stream.read(CHUNK)
                if not chunk:
                    break
                count += len(chunk)
                if count > spec["bytes"]:
                    raise RuntimeError("stream longer than fixed artifact; partial target retained")
                streamed.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise RuntimeError("zero-length stage write; partial target retained")
                    view = view[written:]
            if count != spec["bytes"] or streamed.hexdigest() != spec["sha256"]:
                raise RuntimeError("input bytes/hash mismatch; partial target retained")
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            os.fsync(parent_fd)
            metadata = _file_metadata(fd, parent_fd, spec["name"])
            read_bytes, read_sha256 = _digest_fd(fd)
            if read_bytes != count:
                raise RuntimeError("readback byte count mismatch; target retained")
            if _file_metadata(fd, parent_fd, spec["name"]) != metadata:
                raise RuntimeError("stage target changed after write readback; target retained")
            after = _identity(parent_fd)
            return _receipt("write", artifact, spec, before, after, metadata,
                            read_sha256, streamed.hexdigest())
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def verify_artifact(artifact):
    spec = _artifact(artifact)
    parent_fd = _open_parent()
    try:
        before = _identity(parent_fd)
        fd = os.open(spec["name"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=parent_fd)
        try:
            metadata = _file_metadata(fd, parent_fd, spec["name"])
            read_bytes, read_sha256 = _digest_fd(fd)
            if read_bytes != metadata["bytes"]:
                raise RuntimeError("readback byte count mismatch")
            if _file_metadata(fd, parent_fd, spec["name"]) != metadata:
                raise RuntimeError("stage target changed during read-only verification")
            after = _identity(parent_fd)
            return _receipt("verify", artifact, spec, before, after, metadata,
                            read_sha256)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "verify"):
        raise SystemExit("usage: python3 -c <reviewed-writer-source> write|verify capsule|preflight|contained")
    operation, artifact = sys.argv[1:]
    result = write_artifact(artifact, sys.stdin.buffer) if operation == "write" else verify_artifact(artifact)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
