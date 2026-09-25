"""Read-only timing model of the exact disabled trn22 r10 stage boundary.

No writer is imported or called, and no bundle/checkout is opened or created.
The would-open is a monotonic clock sample, not an O_EXCL syscall.  Even a
subsecond trial cannot prove actual open latency, rollback, or future safety.
"""

from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time


SCHEMA = "trn22-r10-stage-r1-timing-v1"
HOST = "ip-172-31-32-253"
UID = GID = 1000
DEV = 66305
ROOT = Path("/home/ubuntu")
SOURCE = ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
TARGET = ROOT / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1"
BUNDLE = ROOT / "trainium-c5c-e65-to-r10-20260925.trn22-stage-r1.bundle"
LABEL = "fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1"
BASE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
STAGE_R10_COMMIT = "020bfa7ebd048f947d5207cbedaf028a29a4b159"
STAGE_R10_REMOTE_SHA256 = "da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f"
STAGE_R10_OPERATOR_SHA256 = "7c04b0288de04a514b3da0a6a9c01f525551f2cfc92f69d4ba63f8941aa2d0b5"
STAGE_R10_HOLD_SHA256 = "625b4a53c0f67fba0a7b4ef5a6705f64f3935eb8f5841ef5e2212f6c8e820b6e"
TRAIN_SHA256 = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
SOURCE_INO = 2432738
SOURCE_MODE = 0o775
GIT_MARKER_INO = 2432740
GIT_MARKER_SHA256 = "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b"
LOCK_CACHE = ROOT / ".cache"
LOCK_DIR = LOCK_CACHE / "trainium_freshstart"
LOCK = LOCK_DIR / "device0.lock"
LOCK_INODES = (295736, 832588, 832589)
LOCK_MODES = (0o700, 0o775, 0o664)
STOP_PATHS = [ROOT / "research/agentic/STOP", ROOT / "research/v2/STOP",
              ROOT / "trainium_freshstart_20260918/research/agentic/STOP",
              ROOT / "trainium_freshstart_20260918/research/v2/STOP"]
for _name in ("r1", "r2", "r3", "r4", "r9_stage_r1", "r10_stage_r1"):
    _checkout = ROOT / ("trainium_freshstart_20260925_v2_c5c_a0_control_trn22_" + _name)
    STOP_PATHS += [_checkout / "research/agentic/STOP", _checkout / "research/v2/STOP"]
PROCESS_MARKERS = ("train.py", "torchrun", "eval-public", "prepare.py",
                   "run_experiment.py", "fresh_start_launch", "fresh_start_guard",
                   "root_free_guard_monitor", "paired_root_free_guard_monitor",
                   "run_job.py", "preflight_dryrun", "contained.py", "g2-probe",
                   "phase_timer", "reconcile", "neuronx-cc")
GIT_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "file",
           "GIT_OPTIONAL_LOCKS": "0"}
TRIALS = 20
MAX_AGE_NS = 1_000_000_000
MIN_FREE = 13_177_973_911
MIN_INODES = 10_000
MAX_OUTPUT_LINE = 8192


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _fail(message):
    raise RuntimeError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _emit(record):
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_OUTPUT_LINE:
        _fail("bounded JSONL record exceeded")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()  # Retain every finished trial on later timeout.


def _file_sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            _fail("unsafe source file")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(fd)
        named = os.lstat(path)
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) !=
                (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) or
                (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns) !=
                (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
            _fail("source file changed during hash")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _git(*words):
    result = subprocess.run(["/usr/bin/git", "-c", "core.fsmonitor=false",
                             "-c", "submodule.recurse=false", "-C", str(SOURCE), *words],
                            env=GIT_ENV, capture_output=True, timeout=8, check=False)
    if result.returncode != 0 or result.stderr:
        _fail("read-only source Git check failed")
    return result.stdout.strip().decode("ascii")


def _source_identity():
    source = os.lstat(SOURCE)
    marker = os.lstat(SOURCE / ".git")
    if (not stat.S_ISDIR(source.st_mode) or source.st_dev != DEV or
            source.st_ino != SOURCE_INO or source.st_uid != UID or source.st_gid != GID or
            stat.S_IMODE(source.st_mode) != SOURCE_MODE or
            not stat.S_ISREG(marker.st_mode) or marker.st_dev != DEV or
            marker.st_ino != GIT_MARKER_INO or marker.st_uid != UID or
            marker.st_nlink != 1 or marker.st_size != 148 or
            stat.S_IMODE(marker.st_mode) != 0o664 or
            _file_sha(SOURCE / ".git") != GIT_MARKER_SHA256):
        _fail("accepted linked source identity changed")
    if (_git("rev-parse", "HEAD") != BASE_COMMIT or
            _git("status", "--porcelain=v1", "--untracked-files=all",
                 "--ignore-submodules=all") or
            _file_sha(SOURCE / "train.py") != TRAIN_SHA256 or
            _file_sha(SOURCE / "prepare.py") != PREPARE_SHA256):
        _fail("accepted source commit, cleanliness or science files changed")
    return {"path": str(SOURCE), "commit": BASE_COMMIT,
            "inode": source.st_ino, "git_marker_inode": marker.st_ino,
            "train_sha256": TRAIN_SHA256, "prepare_sha256": PREPARE_SHA256}


@contextmanager
def _held_existing_lock():
    paths = (LOCK_CACHE, LOCK_DIR, LOCK)
    descriptors = []
    try:
        descriptors.append(os.open(paths[0], os.O_RDONLY | os.O_DIRECTORY |
                                   os.O_NOFOLLOW | os.O_CLOEXEC))
        descriptors.append(os.open("trainium_freshstart", os.O_RDONLY |
                                   os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                   dir_fd=descriptors[0]))
        descriptors.append(os.open("device0.lock", os.O_RDONLY | os.O_NONBLOCK |
                                   os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptors[1]))
        for path, fd, inode, mode, directory in zip(
                paths, descriptors, LOCK_INODES, LOCK_MODES, (True, True, False)):
            opened, named = os.fstat(fd), os.lstat(path)
            if ((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink) !=
                    (named.st_dev, named.st_ino, named.st_mode, named.st_nlink) or
                    opened.st_dev != DEV or opened.st_ino != inode or
                    (opened.st_uid, opened.st_gid) != (UID, GID) or
                    stat.S_IMODE(opened.st_mode) != mode or
                    (not stat.S_ISDIR(opened.st_mode) if directory else
                     not stat.S_ISREG(opened.st_mode))):
                _fail("cooperative read-only lock identity changed")
        if (os.fstat(descriptors[-1]).st_size != 0 or
                os.fstat(descriptors[-1]).st_nlink != 1):
            _fail("cooperative lock is not an empty regular file")
        fcntl.flock(descriptors[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield {"path": str(LOCK), "device": DEV, "inode": LOCK_INODES[-1]}
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _identity():
    if socket.gethostname() != HOST or os.geteuid() != UID:
        _fail("wrong trn22 hostname or UID")
    root = os.lstat(ROOT)
    if not stat.S_ISDIR(root.st_mode) or root.st_dev != DEV:
        _fail("wrong trn22 persistent root device")
    fs = os.statvfs("/")
    free, inodes = fs.f_bavail * fs.f_frsize, fs.f_favail
    if free < MIN_FREE or free < 6_979_321_856 or inodes < MIN_INODES:
        _fail("root byte/inode preflight floor failed")
    return {"hostname": HOST, "uid": UID, "root_device": DEV,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "root_free_bytes": free, "root_free_inodes": inodes,
            "source": _source_identity()}


def _stop_process():
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("host/source/target STOP is present")
    own = set()
    pid = os.getpid()
    while pid > 1 and pid not in own:
        own.add(pid)
        try:
            text = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
            pid = int(text.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            words = Path("/proc", entry.name, "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            _fail("process table cmdline unreadable")
        if any(marker.encode("ascii") in word for marker in PROCESS_MARKERS
               for word in words):
            _fail("train/eval/guard/process activity present")
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("STOP appeared during process scan")


def _activity():
    _stop_process()
    fuser = subprocess.run(["/usr/bin/fuser", "/dev/neuron0"],
                           capture_output=True, timeout=20, check=False)
    if fuser.returncode != 1 or fuser.stdout or fuser.stderr.strip():
        _fail("Neuron device holder or uncertain fuser")
    neuron = subprocess.run(["neuron-ls", "-j", "-a"],
                            env={"PATH": "/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin"},
                            capture_output=True, timeout=20, check=False)
    if neuron.returncode != 0 or neuron.stderr.strip():
        _fail("neuron-ls failed or emitted diagnostics")
    devices = json.loads(neuron.stdout)
    if not isinstance(devices, list):
        _fail("neuron-ls JSON is not a list")
    for device in devices:
        if not isinstance(device, dict):
            _fail("neuron-ls record malformed")
        for process in device.get("neuron_processes", []):
            if (not isinstance(process, dict) or
                    not str(process.get("command", "")).startswith("neuron-ls")):
                _fail("Neuron process active or malformed")
    _stop_process()


def _label_scan():
    result = subprocess.run(["/usr/bin/find", str(ROOT), "-xdev", "-name",
                             "*" + LABEL + "*", "-print"],
                            capture_output=True, timeout=45, check=False)
    if result.returncode != 0 or result.stdout or result.stderr.strip():
        _fail("stage label appeared or broad scan failed")


def _direct_stop_target():
    if any(os.path.lexists(path) for path in STOP_PATHS):
        _fail("STOP appeared during final broad scans")
    if os.path.lexists(BUNDLE) or os.path.lexists(TARGET):
        _fail("source-stage bundle or target appeared")


def _space_floor():
    """Same pre-write byte/inode floor as r10, without reserving any space."""
    fs = os.statvfs("/")
    free, inodes = fs.f_bavail * fs.f_frsize, fs.f_favail
    if free < MIN_FREE or free < 6_979_321_856 or inodes < MIN_INODES:
        _fail("root byte/inode pre-write floor failed")
    return {"root_free_bytes": free, "root_free_inodes": inodes}


def probe():
    with _held_existing_lock() as lock:
        return probe_without_nested_lock(lock)


def _measure_step(name, callback):
    started = time.monotonic_ns()
    observation = callback()
    ended = time.monotonic_ns()
    step = {"name": name, "started_ns": started, "ended_ns": ended,
            "duration_ns": ended - started}
    if name.startswith("space_"):
        step["observation"] = observation
    return step


def _ages(now, *, label_at, activity_at, space_at):
    ages = {"label_age_ns": now - label_at,
            "activity_age_ns": now - activity_at,
            "space_age_ns": now - space_at}
    if any(age < 0 for age in ages.values()):
        _fail("monotonic clock regressed")
    return ages


def _too_old(ages):
    return any(age > MAX_AGE_NS for age in ages.values())


def _trial(index):
    record = {"schema": SCHEMA, "kind": "trial", "index": index,
              "status": "ABORTED", "model": "CLOCK_ONLY_NO_OPEN",
              "steps": []}
    try:
        steps = record["steps"]
        steps.append(_measure_step("activity_pre_scan", _activity))
        steps.append(_measure_step("label_scan_1", _label_scan))
        steps.append(_measure_step("label_scan_2", _label_scan))
        label_end = steps[-1]["ended_ns"]
        scan_pair_duration = label_end - steps[1]["started_ns"]
        if scan_pair_duration < 0:
            _fail("monotonic clock regressed during double scan")
        record.update({"scan_pair_duration_ns": scan_pair_duration,
                       "max_age_ns": MAX_AGE_NS})
        if scan_pair_duration > MAX_AGE_NS:
            record.update({"status": "TIMING_MISS", "stopped_at": "double_scan"})
            return record
        steps.append(_measure_step("activity_post_scan", _activity))
        activity_end = steps[-1]["ended_ns"]
        steps.append(_measure_step("space_pre_open", _space_floor))
        space_end = steps[-1]["ended_ns"]
        steps.append(_measure_step("direct_pre_open", _direct_stop_target))
        pre_open_check = time.monotonic_ns()
        pre_ages = _ages(pre_open_check, label_at=label_end,
                         activity_at=activity_end, space_at=space_end)
        record.update({"pre_open_check_ns": pre_open_check,
                       "pre_open_ages_ns": pre_ages})
        if _too_old(pre_ages):
            record.update({"status": "TIMING_MISS", "stopped_at": "pre_open"})
            return record
        # Clock only.  No os.open, O_CREAT, O_EXCL, payload, or rollback.
        would_open = time.monotonic_ns()
        at_would_open = _ages(would_open, label_at=label_end,
                              activity_at=activity_end, space_at=space_end)
        record.update({"would_open_ns": would_open,
                       "would_open_ages_ns": at_would_open})
        if _too_old(at_would_open):
            record.update({"status": "TIMING_MISS", "stopped_at": "would_open"})
            return record
        steps.append(_measure_step("label_scan_3", _label_scan))
        post_label_end = steps[-1]["ended_ns"]
        steps.append(_measure_step("activity_post_would_open", _activity))
        post_activity_end = steps[-1]["ended_ns"]
        steps.append(_measure_step("space_post_would_open", _space_floor))
        post_space_end = steps[-1]["ended_ns"]
        # On an actual r10 write the provisional bundle exists here; this
        # diagnostic instead requires both remote paths to remain absent.
        steps.append(_measure_step("direct_post_would_open", _direct_stop_target))
        would_payload = time.monotonic_ns()
        post_ages = _ages(would_payload, label_at=post_label_end,
                           activity_at=post_activity_end, space_at=post_space_end)
        record.update({"would_payload_ns": would_payload,
                       "post_would_open_ages_ns": post_ages,
                       "status": "TIMING_MISS" if _too_old(post_ages) else
                                 "WITHIN_MODEL_NO_OPEN",
                       "stopped_at": "would_payload" if _too_old(post_ages) else "none"})
    except BaseException as error:
        record["error"] = {"type": type(error).__name__,
                           "message": str(error)[:256]}
    return record


def _compatible(prior, current):
    if not isinstance(prior, dict) or prior.get("schema") != SCHEMA + "-probe":
        _fail("wrong read-only preflight pin")
    stamp = datetime.datetime.fromisoformat(prior["utc"])
    age = (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds()
    if stamp.tzinfo is None or not -5 <= age <= 180:
        _fail("read-only preflight pin is stale")
    for key in ("lock", "label", "bundle", "target", "stop_paths_absent",
                "stage_r10"):
        if prior[key] != current[key]:
            _fail("before/timing read-only identity drift: " + key)
    for key in ("hostname", "uid", "root_device", "boot_id", "mount_namespace", "source"):
        if prior["identity"][key] != current["identity"][key]:
            _fail("before/timing host or source drift: " + key)
    if prior["process_identity"] == current["process_identity"]:
        _fail("before/timing probes did not use distinct processes")


def timing(prior):
    # Emit each complete trial before beginning another; stdout is JSONL, not one final JSON.
    with _held_existing_lock() as lock:
        current = probe_without_nested_lock(lock)
        _compatible(prior, current)
        _emit({"schema": SCHEMA, "kind": "start", "trial_count": TRIALS,
               "utc": _utc(), "preflight_sha256": _sha(json.dumps(
                   prior, sort_keys=True, separators=(",", ":")).encode()),
               "identity": current["identity"], "lock": current["lock"],
               "stage_r10": current["stage_r10"],
               "process_identity": current["process_identity"]})
        done = 0
        misses = 0
        for index in range(1, TRIALS + 1):
            trial = _trial(index)
            _emit(trial)
            done += 1
            misses += trial["status"] == "TIMING_MISS"
            if trial["status"] == "ABORTED":
                break
        status = ("COMPLETE_READ_ONLY" if done == TRIALS and
                  trial["status"] != "ABORTED" else "ABORTED_READ_ONLY")
        _emit({"schema": SCHEMA, "kind": "end", "trial_count": TRIALS,
               "completed_trials": done, "timing_misses": misses,
               "status": status,
               "utc": _utc()})
        return status


def probe_without_nested_lock(lock):
    """Timing owns the lock already; same read-only probe sequence."""
    before = _identity()
    _direct_stop_target()
    _activity()
    _label_scan()
    _activity()
    _direct_stop_target()
    after = _identity()
    if before["boot_id"] != after["boot_id"] or before["source"] != after["source"]:
        _fail("host/source identity drift during timing preflight")
    own_stat = Path("/proc/self/stat").read_text(encoding="ascii")
    process_identity = {"pid": os.getpid(),
                        "start_ticks": int(own_stat.rsplit(")", 1)[1].split()[19])}
    return {"schema": SCHEMA + "-probe", "utc": _utc(), "identity": after,
            "process_identity": process_identity, "lock": lock,
            "label": LABEL, "bundle": str(BUNDLE), "target": str(TARGET),
            "stage_r10": {"commit": STAGE_R10_COMMIT,
                         "remote_sha256": STAGE_R10_REMOTE_SHA256,
                         "operator_sha256": STAGE_R10_OPERATOR_SHA256,
                         "hold_sha256": STAGE_R10_HOLD_SHA256},
            "stop_paths_absent": True}
