"""Read-only, bounded trn22 r4 source-stage feasibility snapshot.

This script is streamed to ``python3 -`` over a pinned SSH connection.  It
does not create, alter, remove, or lock a host path, and prints only curated
non-credential facts.  Its output is planning evidence, never a preflight or
dispatch authorization.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import socket
import stat
import subprocess


LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r4"
ROOT = "/home/ubuntu"
ARCHIVE = ROOT + "/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz"
CHECKOUT = ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4"
MONITOR = ROOT + "/trainium_a0_disk_monitor_20260925_r4"
OBSERVER = ROOT + "/trainium_a0_boundary_observer_20260925_r1"
CHECKPOINT = (
    ROOT
    + "/trainium_freshstart_20260924_current_base_control_trn22_r1"
    + "/out/fs-20260924-current-base-control-trn22-r1-work/train/out/final.pt"
)
CACHE_ROOT = (
    ROOT
    + "/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1"
    + "/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train"
)
CACHE_INODES = {
    "neff": 2430420,
    "local": 2430421,
    "hlo": 2430422,
    "inductor": 2430423,
    "compiler": 2430424,
    "debug": 2430425,
}
CHECKPOINT_SHA256 = "1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b"
PRETRANSFER_MIN_BYTES = 13177973911
POSTSTAGE_MIN_BYTES = 10737418240
PRETRANSFER_MIN_INODES = 10000


def identity(path: str, *, hash_regular: bool = False) -> dict:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return {"exists": False}
    kind = (
        "directory"
        if stat.S_ISDIR(before.st_mode)
        else "regular"
        if stat.S_ISREG(before.st_mode)
        else "symlink"
        if stat.S_ISLNK(before.st_mode)
        else "other"
    )
    result = {
        "exists": True,
        "type": kind,
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "mode": format(stat.S_IMODE(before.st_mode), "04o"),
        "nlink": before.st_nlink,
        "bytes": before.st_size,
    }
    if hash_regular and kind == "regular":
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise RuntimeError("file identity changed on open")
            digest = hashlib.sha256()
            read_bytes = 0
            while True:
                chunk = os.read(fd, 4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                read_bytes += len(chunk)
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise RuntimeError("file identity changed during read")
            result["sha256"] = digest.hexdigest()
            result["read_bytes"] = read_bytes
        finally:
            os.close(fd)
    return result


def covering_mount(path: str) -> dict:
    candidates = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            left_fields = left.split()
            right_fields = right.split()
            if len(left_fields) < 5 or len(right_fields) < 2:
                continue
            mount_point = left_fields[4]
            if path == mount_point or path.startswith(mount_point.rstrip("/") + "/"):
                candidates.append(
                    (
                        len(mount_point),
                        {
                            "mount_id": left_fields[0],
                            "major_minor": left_fields[2],
                            "mount_point": mount_point,
                            "filesystem": right_fields[0],
                            "source": right_fields[1],
                        },
                    )
                )
    if not candidates:
        raise RuntimeError("no covering mount")
    return max(candidates, key=lambda value: value[0])[1]


def command(argv: list[str], *, timeout: int = 25) -> dict:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"rc": 124, "timeout": True}
    return {
        "rc": result.returncode,
        "stdout": result.stdout,
        "stderr_empty": not bool(result.stderr.strip()),
    }


def holder(path: str) -> dict:
    result = command(["fuser", path], timeout=20)
    output = result.pop("stdout", "")
    return {
        **result,
        "pids": sorted({int(x) for x in re.findall(r"\b\d+\b", output)}),
    }


def ancestor_pids() -> set[int]:
    result = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
                tail = stream.read().rsplit(")", 1)[1].split()
            pid = int(tail[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            break
    return result


def matched_processes() -> dict:
    needles = (
        "torchrun",
        "train.py",
        "prepare.py",
        "phase_timer",
        "reconcile",
        "run_experiment.py",
        "run_job.py",
        "fresh_start_guard.py",
        "fresh_start_launch.py",
        "fresh_start_monitor_interlock.py",
        "root_free_guard_monitor.py",
        "paired_root_free_guard_monitor.py",
        "a0_boundary_observer.py",
        "g2-probe",
        "g2_preflight",
        "contained.py",
        "neuronx-cc",
    )
    own = ancestor_pids()
    hits = []
    unreadable = 0
    for name in os.listdir("/proc"):
        if not name.isdecimal():
            continue
        pid = int(name)
        if pid in own:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                args = stream.read(65536).split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable += 1
            continue
        words = [arg.decode("utf-8", "replace") for arg in args if arg]
        categories = [needle for needle in needles if any(needle in word for word in words)]
        if categories:
            hits.append({"pid": pid, "categories": categories})
    return {"hits": hits, "unreadable_cmdline_count": unreadable}


def neuron_processes() -> dict:
    result = command(["neuron-ls", "-j", "-a"], timeout=30)
    output = result.pop("stdout", "")
    summary = {**result, "json_valid": False, "non_diagnostic_pids": []}
    if result["rc"] != 0:
        return summary
    try:
        devices = json.loads(output)
        if not isinstance(devices, list):
            return summary
        summary["json_valid"] = True
        for device in devices:
            for process in device.get("neuron_processes", []):
                if not str(process.get("command", "")).startswith("neuron-ls"):
                    summary["non_diagnostic_pids"].append(process.get("pid"))
    except (ValueError, TypeError, AttributeError):
        summary["json_valid"] = False
    return summary


def label_scan() -> dict:
    result = command(
        ["find", ROOT, "-xdev", "-name", "*" + LABEL + "*", "-print"],
        timeout=45,
    )
    paths = result.pop("stdout", "").splitlines()
    return {**result, "match_count": len(paths), "matches": paths[:20], "truncated": len(paths) > 20}


def main() -> None:
    root_fs = os.statvfs("/")
    ancestry = {path: identity(path) for path in ("/", "/home", ROOT)}
    for item in ancestry.values():
        item["safe_directory"] = (
            item.get("type") == "directory"
            and item.get("uid") in (0, 1000)
            and not bool(int(item.get("mode", "0000"), 8) & 0o022)
        )
    checkpoint = identity(CHECKPOINT, hash_regular=True)
    checkpoint["expected_match"] = (
        checkpoint.get("type") == "regular"
        and checkpoint.get("device") == 66305
        and checkpoint.get("inode") == 2417091
        and checkpoint.get("uid") == 1000
        and checkpoint.get("nlink") == 1
        and checkpoint.get("bytes") == 469831009
        and checkpoint.get("read_bytes") == 469831009
        and checkpoint.get("sha256") == CHECKPOINT_SHA256
    )
    cache = {}
    for name, expected_inode in CACHE_INODES.items():
        entry = identity(CACHE_ROOT + "/" + name)
        entry["expected_match"] = (
            entry.get("type") == "directory"
            and entry.get("device") == 66305
            and entry.get("inode") == expected_inode
            and entry.get("uid") == 1000
        )
        cache[name] = entry
    targets = [
        ARCHIVE,
        CHECKOUT,
        MONITOR,
        CHECKOUT + "/out/" + LABEL,
        CHECKOUT + "/research/run-logs/" + LABEL + ".log",
        CHECKOUT + "/research/run-logs/" + LABEL + ".steps.jsonl",
        CHECKOUT + "/research/run-logs/" + LABEL + ".process.json",
        CHECKOUT + "/research/run-logs/" + LABEL + ".postflight.json",
        OBSERVER + "/" + LABEL + ".observer.json",
        OBSERVER + "/" + LABEL + ".observer.stderr.log",
    ]
    stop_paths = [ROOT + "/research/agentic/STOP", ROOT + "/research/v2/STOP"]
    for checkout in (
        ROOT + "/trainium_freshstart_20260918",
        ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1",
        ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2",
        ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r3",
        CHECKOUT,
    ):
        stop_paths.extend((checkout + "/research/agentic/STOP", checkout + "/research/v2/STOP"))
    lock_path = ROOT + "/.cache/trainium_freshstart/device0.lock"
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot_id = stream.read().strip()
    output = {
        "schema": "trn22-r4-source-stage-readonly-feasibility-v1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "euid": os.geteuid(),
        "boot_id": boot_id,
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "root_mount": covering_mount("/"),
        "parent_mount": covering_mount(ROOT),
        "root_free_bytes": root_fs.f_bavail * root_fs.f_frsize,
        "root_free_inodes": root_fs.f_favail,
        "pretransfer_min_bytes": PRETRANSFER_MIN_BYTES,
        "poststage_min_bytes": POSTSTAGE_MIN_BYTES,
        "pretransfer_min_inodes": PRETRANSFER_MIN_INODES,
        "ancestry": ancestry,
        "targets": {path: identity(path) for path in targets},
        "stops": {path: identity(path) for path in stop_paths},
        "science_label_scan": label_scan(),
        "protected_checkpoint": checkpoint,
        "checkpoint_fuser": holder(CHECKPOINT),
        "cache_root": identity(CACHE_ROOT),
        "six_caches": cache,
        "host_lock": identity(lock_path),
        "host_lock_fuser": holder(lock_path),
        "neuron0_fuser": holder("/dev/neuron0"),
        "neuron_ls": neuron_processes(),
        "processes": matched_processes(),
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
