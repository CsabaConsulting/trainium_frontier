"""Offline exact SSH command builder for trn21 source-stage r2; no dispatch CLI.

This file cannot transfer anything when executed. A separately authorized
Operator may import the reviewed functions, save the complete argv and source
hashes in an immutable request, then execute one SSH process per operation.
"""

import datetime
import hashlib
import json
import os
import shlex
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WRITER = HERE / "stage_r2_remote_writer.py"
WRITER_SHA256 = "a87fb4efe9b3e7b4c0b4f60855edb1ebf98f8a02cd66224cd36fb90d7d5b20eb"
SSH_IDENTITY = "/home/poojith-devan/Desktop/Trainium/trn2.pem"
ENDPOINT = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"
ARTIFACTS = {
    "capsule": {
        "source": Path("/dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz"),
        "target": "/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz",
        "bytes": 104005787,
        "sha256": "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264",
        "mode": 0o600,
    },
    "preflight": {
        "source": REPO / "workflow/tools/preflight_dryrun.py",
        "target": "/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py",
        "bytes": 23550,
        "sha256": "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8",
        "mode": None,
    },
    "contained": {
        "source": REPO / "workflow/tools/contained.py",
        "target": "/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py",
        "bytes": 9185,
        "sha256": "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d",
        "mode": None,
    },
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def reviewed_writer_source():
    source = WRITER.read_bytes()
    if _sha256_bytes(source) != WRITER_SHA256:
        raise RuntimeError("reviewed remote writer source drift")
    return source.decode("utf-8")


def checked_local_source(artifact):
    if artifact not in ARTIFACTS:
        raise ValueError("unknown fixed artifact")
    spec = ARTIFACTS[artifact]
    path = spec["source"]
    record = path.lstat()
    if not stat.S_ISREG(record.st_mode):
        raise RuntimeError("local source must be a regular non-symlink file")
    if spec["mode"] is not None and stat.S_IMODE(record.st_mode) != spec["mode"]:
        raise RuntimeError("local source mode drift")
    digest = hashlib.sha256()
    size = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        identity = ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")
        if not stat.S_ISREG(opened.st_mode) or any(
            getattr(opened, key) != getattr(record, key) for key in identity
        ):
            raise RuntimeError("local source changed between lstat and nofollow open")
        while True:
            chunk = os.read(fd, 4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        closed = os.fstat(fd)
        named = path.lstat()
        if any(
            getattr(snapshot, key) != getattr(opened, key)
            for snapshot in (closed, named) for key in identity
        ):
            raise RuntimeError("local source changed during hashing")
    finally:
        os.close(fd)
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise RuntimeError("local source byte count or SHA-256 drift")
    return {"source": str(path), "target": spec["target"], "bytes": size,
            "sha256": digest.hexdigest(), "device": record.st_dev, "inode": record.st_ino,
            "mode": oct(stat.S_IMODE(record.st_mode))}


def build_ssh_argv(operation, artifact):
    if operation not in ("write", "verify") or artifact not in ARTIFACTS:
        raise ValueError("unknown fixed operation/artifact")
    source = reviewed_writer_source()
    remote_words = ["exec", "python3", "-c", source, operation, artifact]
    remote_command = " ".join(shlex.quote(word) for word in remote_words)
    argv = ["ssh", "-T", "-i", SSH_IDENTITY, "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=20", ENDPOINT, remote_command]
    if shlex.split(remote_command) != remote_words:
        raise RuntimeError("remote command quoting is not reversible")
    return argv


def command_sha256(argv):
    return _sha256_bytes(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def compare_cross_session(write, verify):
    """Reject a missing/replaced target or changed boot/mount across sessions."""
    if write.get("schema") != "trn21-c5c-a0-stage-r2-transfer-v1" or verify.get("schema") != write["schema"]:
        raise ValueError("wrong transfer-receipt schema")
    if write.get("operation") != "write" or verify.get("operation") != "verify":
        raise ValueError("requires a write receipt and later read-only verify receipt")
    if write.get("status") != "HASH_VERIFIED_IN_THIS_SESSION_ONLY" or verify.get("status") != write["status"]:
        raise ValueError("missing completed in-session hash check")
    for key in ("artifact", "path", "expected_bytes", "expected_sha256", "target_stat"):
        if write.get(key) != verify.get(key):
            raise ValueError("target changed between sessions: " + key)
    for receipt in (write, verify):
        if receipt["readback_sha256"] != receipt["expected_sha256"]:
            raise ValueError("readback differs from fixed expected hash")
        if receipt["target_stat"]["bytes"] != receipt["expected_bytes"]:
            raise ValueError("byte count differs from fixed expected count")
    if write["streamed_sha256"] != write["expected_sha256"] or verify["streamed_sha256"] is not None:
        raise ValueError("wrong write stream/verify mode evidence")
    stable = ("hostname", "uid", "gid", "boot_id", "mount_namespace", "mount",
              "parent_device", "parent_inode", "parent_uid", "parent_gid", "parent_mode")
    for key in stable:
        value = write["identity_before"][key]
        if any(other[key] != value for other in
               (write["identity_after"], verify["identity_before"], verify["identity_after"])):
            raise ValueError("host/mount changed between sessions: " + key)
    if (write["process_id"], write["process_start_ticks"]) == (verify["process_id"], verify["process_start_ticks"]):
        raise ValueError("not proven to be two separate processes")
    first = datetime.datetime.fromisoformat(write["utc"])
    second = datetime.datetime.fromisoformat(verify["utc"])
    if first.tzinfo is None or second.tzinfo is None or second <= first:
        raise ValueError("cross-session check is not later than write")
    return {"status": "CROSS_SESSION_PERSISTENCE_HASH_PASS", "artifact": write["artifact"],
            "path": write["path"], "bytes": write["expected_bytes"],
            "sha256": write["expected_sha256"], "device": write["target_stat"]["device"],
            "inode": write["target_stat"]["inode"], "boot_id": write["identity_after"]["boot_id"],
            "mount_namespace": write["identity_after"]["mount_namespace"]}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this module builds/reviews exact commands but never opens SSH")
