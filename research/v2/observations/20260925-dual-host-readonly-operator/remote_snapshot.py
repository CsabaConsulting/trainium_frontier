"""One bounded, read-only Trainium host status snapshot.

Stream this source to ``python3 -B - trn21|trn22`` over pinned SSH.  It only
reads procfs, mount/device metadata, fixed lock/STOP paths, and candidate
checkout identity/source hashes.  It never creates a path, takes a lock,
sends a signal, contacts AWS, or starts training or evaluation.
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
import sys


ROOT = "/home/ubuntu"
HOSTS = {
    "trn21": {
        "hostname": "ip-172-31-34-85",
        "e65_checkout": ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2",
        "r4_checkout": ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r4",
    },
    "trn22": {
        "hostname": "ip-172-31-32-253",
        "e65_checkout": ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2",
        "r4_checkout": ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4",
    },
}
LOCK = ROOT + "/.cache/trainium_freshstart/device0.lock"
PROCESS_MARKERS = (
    "torchrun", "train.py", "eval-public", "prepare.py", "phase_timer",
    "reconcile", "run_experiment.py", "run_job.py", "fresh_start_guard.py",
    "fresh_start_launch.py", "fresh_start_monitor_interlock.py",
    "root_free_guard_monitor.py", "paired_root_free_guard_monitor.py",
    "a0_boundary_observer.py", "g2-probe", "g2_preflight", "contained.py",
    "neuronx-cc",
)
SOURCE_FILES = ("train.py", "prepare.py", "research/fresh_start_launch.py")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def named(path: str) -> dict:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {"exists": False}
    except OSError as error:
        return {"exists": None, "error": type(error).__name__}
    kind = (
        "directory" if stat.S_ISDIR(info.st_mode) else
        "regular" if stat.S_ISREG(info.st_mode) else
        "symlink" if stat.S_ISLNK(info.st_mode) else
        "block" if stat.S_ISBLK(info.st_mode) else "other"
    )
    return {
        "exists": True, "type": kind, "device": info.st_dev,
        "inode": info.st_ino, "uid": info.st_uid, "gid": info.st_gid,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        "nlink": info.st_nlink, "bytes": info.st_size,
    }


def bounded_hash(path: str) -> dict:
    result = named(path)
    if result.get("type") != "regular":
        return result
    if result["bytes"] > 2_000_000:
        result["hash_error"] = "size_exceeds_2m_limit"
        return result
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            first = os.fstat(fd)
            if (first.st_dev, first.st_ino, first.st_size) != (
                result["device"], result["inode"], result["bytes"]
            ):
                raise RuntimeError("identity_changed_on_open")
            digest = hashlib.sha256()
            count = 0
            while count <= 2_000_000:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
            last = os.fstat(fd)
            if count != first.st_size or (
                last.st_dev, last.st_ino, last.st_size
            ) != (first.st_dev, first.st_ino, first.st_size):
                raise RuntimeError("identity_changed_during_read")
            result["sha256"] = digest.hexdigest()
            result["read_bytes"] = count
        finally:
            os.close(fd)
    except (OSError, RuntimeError) as error:
        result["hash_error"] = type(error).__name__ + ": " + str(error)
    return result


def run(argv: list[str], timeout: int = 20, env: dict | None = None) -> dict:
    try:
        done = subprocess.run(
            argv, capture_output=True, timeout=timeout, check=False, env=env
        )
    except subprocess.TimeoutExpired:
        return {"rc": None, "timeout": True, "stdout": b"", "stderr": b""}
    except OSError as error:
        return {
            "rc": None, "error": type(error).__name__,
            "stdout": b"", "stderr": b"",
        }
    return {"rc": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def fuser(path: str) -> dict:
    if not named(path).get("exists"):
        return {"path_exists": False, "rc": None, "pids": []}
    result = run(["/usr/bin/fuser", path], 20)
    return {
        "path_exists": True, "rc": result["rc"],
        "timeout": result.get("timeout", False),
        "pids": sorted({int(value) for value in re.findall(rb"\b\d+\b", result["stdout"])}),
        "stderr_bytes": len(result["stderr"]),
        "stderr_sha256": sha(result["stderr"]),
    }


def ancestor_pids() -> set[int]:
    values = set()
    current = os.getpid()
    while current > 1 and current not in values:
        values.add(current)
        try:
            with open(f"/proc/{current}/stat", encoding="ascii") as stream:
                tail = stream.read().rsplit(")", 1)[1].split()
            current = int(tail[1])
        except (OSError, ValueError, IndexError):
            break
    return values


def processes() -> dict:
    own = ancestor_pids()
    hits = []
    unreadable = 0
    for entry in os.listdir("/proc"):
        if not entry.isdecimal() or int(entry) in own:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as stream:
                words = stream.read(65536).split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable += 1
            continue
        matched = sorted({
            marker for marker in PROCESS_MARKERS
            if any(marker.encode("ascii") in word for word in words)
        })
        if matched:
            hits.append({"pid": int(entry), "markers": matched})
    return {"matched": sorted(hits, key=lambda item: item["pid"]),
            "unreadable_cmdlines": unreadable}


def neuron() -> dict:
    result = run(["neuron-ls", "-j", "-a"], 30)
    answer = {
        "rc": result["rc"], "timeout": result.get("timeout", False),
        "stderr_bytes": len(result["stderr"]),
        "stderr_sha256": sha(result["stderr"]), "json_valid": False,
        "device_count": None, "non_diagnostic_pids": [],
    }
    try:
        devices = json.loads(result["stdout"])
        if isinstance(devices, list):
            answer["json_valid"] = True
            answer["device_count"] = len(devices)
            for device in devices:
                for process in device.get("neuron_processes", []):
                    if not str(process.get("command", "")).startswith("neuron-ls"):
                        answer["non_diagnostic_pids"].append(process.get("pid"))
    except (TypeError, ValueError, AttributeError):
        pass
    return answer


def mount(path: str) -> dict:
    choices = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            fields, fs = left.split(), right.split()
            if len(fields) < 5 or len(fs) < 2:
                continue
            point = fields[4]
            if path == point or path.startswith(point.rstrip("/") + "/"):
                choices.append((len(point), {
                    "mount_id": fields[0], "major_minor": fields[2],
                    "point": point, "filesystem": fs[0], "source": fs[1],
                }))
    return max(choices, key=lambda item: item[0])[1] if choices else {"error": "no_covering_mount"}


def block_alias(path: str) -> dict:
    result = named(path)
    if not result.get("exists"):
        return result
    try:
        info = os.stat(path)
        result["resolved_type"] = "block" if stat.S_ISBLK(info.st_mode) else "other"
        result["resolved_major_minor"] = (
            f"{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}"
            if stat.S_ISBLK(info.st_mode) else None
        )
    except OSError as error:
        result["resolve_error"] = type(error).__name__
    return result


def git_head(path: str) -> dict:
    git_entry = named(path + "/.git")
    if not git_entry.get("exists"):
        return {"git_entry": git_entry, "commit": None}
    result = run(
        ["/usr/bin/git", "-c", "gc.auto=0", "-c", "core.fsmonitor=false",
         "-C", path, "rev-parse", "--verify", "HEAD^{commit}"],
        8,
        {"PATH": "/usr/bin:/bin", "GIT_OPTIONAL_LOCKS": "0",
         "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    output = result["stdout"].strip().decode("ascii", "replace")
    return {
        "git_entry": git_entry, "rc": result["rc"],
        "timeout": result.get("timeout", False),
        "commit": output if re.fullmatch(r"[0-9a-f]{40}", output) else None,
        "stderr_bytes": len(result["stderr"]),
        "stderr_sha256": sha(result["stderr"]),
    }


def git_clean(path: str) -> dict:
    result = run(
        ["/usr/bin/git", "-c", "gc.auto=0", "-c", "core.fsmonitor=false",
         "-C", path, "status", "--porcelain", "--untracked-files=all"],
        20,
        {"PATH": "/usr/bin:/bin", "GIT_OPTIONAL_LOCKS": "0",
         "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    return {
        "rc": result["rc"], "timeout": result.get("timeout", False),
        "clean": result["rc"] == 0 and not result["stdout"] and not result["stderr"],
        "stdout_bytes": len(result["stdout"]),
        "stdout_sha256": sha(result["stdout"]),
        "stderr_bytes": len(result["stderr"]),
        "stderr_sha256": sha(result["stderr"]),
    }


def checkout(path: str) -> dict:
    entry = named(path)
    answer = {"path": path, "entry": entry}
    if entry.get("type") != "directory":
        return answer
    answer["git"] = git_head(path)
    answer["source_hashes"] = {
        name: bounded_hash(path + "/" + name) for name in SOURCE_FILES
    }
    if path in (HOSTS["trn21"]["e65_checkout"], HOSTS["trn22"]["e65_checkout"]):
        answer["git_status"] = git_clean(path)
    return answer


def stop_and_checkout_inventory(host: str) -> tuple[dict, dict]:
    discovered = []
    stop_paths = {ROOT + "/research/agentic/STOP", ROOT + "/research/v2/STOP"}
    for entry in os.scandir(ROOT):
        name = entry.name
        if not name.startswith("trainium_freshstart_") or not entry.is_dir(follow_symlinks=False):
            continue
        stop_paths.add(entry.path + "/research/agentic/STOP")
        stop_paths.add(entry.path + "/research/v2/STOP")
        if ("_r9" in name or "446b" in name or "_r4" in name or
                (host == "trn22" and ("c5c" in name or "_a0_" in name))):
            discovered.append(entry.path)
    if len(discovered) > 80:
        discovered = sorted(discovered)[:80]
        truncated = True
    else:
        discovered.sort()
        truncated = False
    paths = {"r4_exact": HOSTS[host]["r4_checkout"]}
    paths["e65_exact"] = HOSTS[host]["e65_checkout"]
    for number, path in enumerate(discovered):
        if path not in paths.values():
            paths[f"discovered_{number:02d}"] = path
    checkouts = {name: checkout(path) for name, path in paths.items()}
    present_stops = {path: named(path) for path in sorted(stop_paths)
                     if named(path).get("exists") is not False}
    return (
        {"checked_count": len(stop_paths), "present": present_stops},
        {"discovered_names": discovered, "discovery_truncated": truncated,
         "entries": checkouts},
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in HOSTS:
        raise SystemExit("one known host id is required")
    host = sys.argv[1]
    started = utc()
    hostname = socket.gethostname()
    if hostname != HOSTS[host]["hostname"] or os.geteuid() != 1000:
        print(json.dumps({"schema": "dual-host-readonly-v1", "host": host,
                          "utc": started, "identity_error": "hostname_or_uid_mismatch"},
                         sort_keys=True))
        raise SystemExit(2)
    fs = os.statvfs("/")
    root_space_utc = utc()
    stops, checkouts = stop_and_checkout_inventory(host)
    result = {
        "schema": "dual-host-readonly-v1", "host": host,
        "started_utc": started, "ended_utc": utc(),
        "identity": {
            "hostname": hostname, "uid": os.geteuid(),
            "boot_id": open("/proc/sys/kernel/random/boot_id", encoding="ascii").read().strip(),
            "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "root_mount": mount("/"), "home_mount": mount(ROOT),
            "root_device": named("/"),
            "block_aliases": {name: block_alias(name) for name in
                              ("/dev/root", "/dev/nvme0n1p1")},
        },
        "root_space": {
            "observed_utc": root_space_utc,
            "free_bytes": fs.f_bavail * fs.f_frsize,
            "free_inodes": fs.f_favail,
            "total_bytes": fs.f_blocks * fs.f_frsize,
        },
        "activity": {
            "processes": processes(), "neuron": neuron(),
            "neuron0_fuser": fuser("/dev/neuron0"),
            "host_lock": named(LOCK), "host_lock_fuser": fuser(LOCK),
        },
        "stops": stops, "checkouts": checkouts,
    }
    result["ended_utc"] = utc()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
