"""Read-only timing of the frozen trn22 final-r3 label/post-label helpers.

The caller supplies a namespace compiled from the exact pinned final-r3 remote
source.  This module has no standalone host command, target open, or write path.
Each trial releases the existing nonblocking cooperative flock before the next.
"""

import datetime
import fcntl
import os
import time


SCHEMA = "trn22-fifo-final-r3-readonly-timing-v1"
PIN_SCHEMA = "trn22-fifo-final-r3-readonly-preflight-pin-v1"
TRIAL_COUNT = 20
SOURCE_COMMIT = "d571ea61b9fd2c3c7a21f254e044338ff873edba"
SOURCE_SHA256 = "fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a"
TIMING_ERRORS = {
    "final label expired before post-scan host safety",
    "post-label host safety expired before O_EXCL",
}
STABLE_KEYS = ("hostname", "uid", "boot_id", "mount_namespace", "mount",
               "block_aliases", "ancestry")


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _lock_stat(fd):
    info = os.fstat(fd)
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
            "gid": info.st_gid, "mode": info.st_mode, "links": info.st_nlink,
            "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns}


def _check_pin(remote, pin):
    expected = {"schema", "source_commit", "source_sha256", "preflight_utc",
                "preflight_sha256", "preflight_process_identity", "identity",
                "lock", "old_inventory", "targets", "protected"}
    if not isinstance(pin, dict) or set(pin) != expected or pin["schema"] != PIN_SCHEMA:
        raise RuntimeError("incomplete exact read-only preflight pin")
    if (pin["source_commit"] != SOURCE_COMMIT or
            pin["source_sha256"] != SOURCE_SHA256 or
            pin["identity"].get("hostname") != remote["HOST"] or
            pin["identity"].get("uid") != remote["UID"] or
            pin["lock"].get("path") != remote["HOST_LOCK"] or
            pin["targets"].get(remote["TARGET"], {}).get("exists") is not False):
        raise RuntimeError("wrong frozen source, host, lock or absent target pin")
    try:
        observed = datetime.datetime.fromisoformat(pin["preflight_utc"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("malformed preflight time") from error
    age = (datetime.datetime.now(datetime.timezone.utc) - observed).total_seconds()
    if observed.tzinfo is None or not -5 <= age <= 180:
        raise RuntimeError("read-only preflight stale before timing session")
    if pin["preflight_process_identity"] == remote["_process_identity"]():
        raise RuntimeError("timing did not enter a distinct SSH process")


def _check_snapshot(remote, pin):
    """Re-read the fixed preflight assets without opening the target."""
    parent_fd, ancestry = remote["_open_parent"]()
    try:
        identity = remote["_identity"](parent_fd, ancestry)
        if any(identity[key] != pin["identity"][key] for key in STABLE_KEYS):
            raise RuntimeError("host, boot, mount or ancestry drift during timing")
        if (identity["root_free_bytes"] < remote["PRE_FREE"] or
                identity["root_free_inodes"] < remote["MIN_INODES"]):
            raise RuntimeError("persistent root free-space or inode gate failed")
        lock = remote["_inspect_existing_lock"](parent_fd)
        if lock != pin["lock"]:
            raise RuntimeError("cooperative lock drift during timing")
        old = remote["_old_inventory"](parent_fd)
        if old != pin["old_inventory"]:
            raise RuntimeError("old source/cache/output inventory drift during timing")
        targets = {path: remote["_safe_path"](path) for path in pin["targets"]}
        if targets != pin["targets"] or targets[remote["TARGET"]]["exists"] is not False:
            raise RuntimeError("future target or archive path drift during timing")
        protected = remote["_protected"]()
        if protected != pin["protected"]:
            raise RuntimeError("protected checkpoint/cache drift during timing")
        return {"utc": _utc(), "identity": identity, "lock": lock,
                "old_inventory": old, "targets": targets, "protected": protected}
    finally:
        os.close(parent_fd)


def _timed_post(remote, parent_fd, lock_parts, lock_pin, final_label):
    """Call the exact source helper while recording its existing helper calls."""
    events = []
    originals = {name: remote[name] for name in
                 ("_fuser", "_neuron_ls", "_lock_identity",
                  "_require_no_stops_or_processes", "_processes",
                  "_require_no_host_stops")}

    def wrap(name, original):
        def observed(*args, **kwargs):
            event_name = name
            if name == "_fuser":
                event_name = ("checkpoint_fuser" if args[0] == remote["CHECKPOINT"] else
                              "neuron0_fuser" if args[0] == "/dev/neuron0" else
                              "unexpected_fuser")
            elif name == "_neuron_ls":
                event_name = "neuron_ls"
            elif name == "_lock_identity":
                event_name = "lock_acl"
            elif name == "_require_no_stops_or_processes":
                event_name = "stop_process"
            elif name == "_processes":
                event_name = "process_inventory"
            elif name == "_require_no_host_stops":
                event_name = "final_stops"
            event = {"name": event_name, "started_monotonic_ns": time.monotonic_ns()}
            events.append(event)
            try:
                result = original(*args, **kwargs)
                event["status"] = "OK"
                event["observation"] = result
                return result
            except BaseException as error:
                event["status"] = "ERROR"
                event["error"] = _error(error)
                raise
            finally:
                ended = time.monotonic_ns()
                event["completed_monotonic_ns"] = ended
                event["duration_ns"] = ended - event["started_monotonic_ns"]
        return observed

    for name, original in originals.items():
        remote[name] = wrap(name, original)
    started_ns = time.monotonic_ns()
    result = None
    error = None
    try:
        result = remote["_post_final_label_safety"](
            parent_fd, lock_parts, lock_pin, final_label)
    except BaseException as caught:
        error = caught
    finally:
        completed_ns = time.monotonic_ns()
        for name, original in originals.items():
            remote[name] = original
    stop_event = next((item for item in events if item["name"] == "stop_process"), None)
    process_event = next((item for item in events if item["name"] == "process_inventory"), None)
    return {"started_monotonic_ns": started_ns,
            "completed_monotonic_ns": completed_ns,
            "duration_ns": completed_ns - started_ns,
            "label_completion_to_post_end_ns": (
                completed_ns - final_label["completed_monotonic_ns"]),
            "initial_stop_sweep_ns": (
                process_event["started_monotonic_ns"] - stop_event["started_monotonic_ns"]
                if stop_event is not None and process_event is not None else None),
            "events": events, "safety": result,
            "error": _error(error) if error is not None else None}


def _trial(remote, pin, index):
    trial = {"index": index, "utc": _utc(), "status": "ABORTED",
             "prelabel": None, "final_label": None, "post_label": None}
    parent_fd = None
    lock_parts = None
    acquired = False
    try:
        parent_fd, ancestry = remote["_open_parent"]()
        identity = remote["_identity"](parent_fd, ancestry)
        if any(identity[key] != pin["identity"][key] for key in STABLE_KEYS):
            raise RuntimeError("host/boot/mount drift before timing trial")
        if remote["_safe_path"](remote["TARGET"])["exists"] is not False:
            raise RuntimeError("archive target appeared before timing trial")
        lock_parts, lock_pin = remote["_open_existing_lock"](parent_fd)
        if lock_pin != pin["lock"]:
            raise RuntimeError("cooperative lock drift before timing trial")
        trial["lock_stat_before"] = _lock_stat(lock_parts[2])
        fcntl.flock(lock_parts[2], fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        trial["lock_acquired_monotonic_ns"] = time.monotonic_ns()
        if remote["_lock_identity"](parent_fd, lock_parts) != pin["lock"]:
            raise RuntimeError("held cooperative lock identity drift")
        prelabel_started = time.monotonic_ns()
        trial["prelabel"] = remote["_require_no_stops_or_processes"]()
        trial["prelabel_duration_ns"] = time.monotonic_ns() - prelabel_started
        label_call_started = time.monotonic_ns()
        final_label = remote["_final_label_observation"]()
        label_call_completed = time.monotonic_ns()
        trial["final_label"] = final_label
        trial["final_label_call"] = {
            "started_monotonic_ns": label_call_started,
            "completed_monotonic_ns": label_call_completed,
            "duration_ns": label_call_completed - label_call_started,
            "source_scan_duration_ns": (final_label["completed_monotonic_ns"] -
                                        final_label["started_monotonic_ns"])}
        post = _timed_post(remote, parent_fd, lock_parts, pin["lock"], final_label)
        trial["post_label"] = post
        if post["error"] is not None:
            if post["error"]["type"] == "RuntimeError" and post["error"]["message"] in TIMING_ERRORS:
                trial["status"] = "TIMING_MISS"
                return trial
            raise RuntimeError("post-label safety failed: " + post["error"]["message"])
        would_open_ns = time.monotonic_ns()
        trial["would_open_attempt_monotonic_ns"] = would_open_ns
        elapsed = would_open_ns - final_label["completed_monotonic_ns"]
        trial["label_to_would_open_ns"] = elapsed
        trial["remaining_to_one_second_ns"] = remote["MAX_FINAL_SCAN_TO_OPEN_NS"] - elapsed
        if (would_open_ns < post["completed_monotonic_ns"] or
                elapsed > remote["MAX_FINAL_SCAN_TO_OPEN_NS"]):
            trial["status"] = "TIMING_MISS"
        else:
            trial["status"] = "WITHIN_SOURCE_WINDOW_NO_OPEN"
        return trial
    except BaseException as error:
        trial["error"] = _error(error)
        return trial
    finally:
        try:
            if lock_parts is not None:
                try:
                    if acquired:
                        trial["lock_release_started_monotonic_ns"] = time.monotonic_ns()
                        fcntl.flock(lock_parts[2], fcntl.LOCK_UN)
                        trial["lock_released_monotonic_ns"] = time.monotonic_ns()
                        trial["lock_held_ns"] = (trial["lock_released_monotonic_ns"] -
                                                 trial["lock_acquired_monotonic_ns"])
                    trial["lock_stat_after"] = _lock_stat(lock_parts[2])
                    if trial["lock_stat_after"] != trial["lock_stat_before"]:
                        raise RuntimeError("cooperative lock metadata changed during trial")
                except BaseException as error:
                    trial["status"] = "ABORTED"
                    trial["error"] = _error(error)
                finally:
                    remote["_close_lock"](lock_parts)
        finally:
            if parent_fd is not None:
                os.close(parent_fd)


def run(remote, pin):
    """One fixed 20-trial attempt; failures preserve partial timings."""
    record = {"schema": SCHEMA, "source_commit": SOURCE_COMMIT,
              "source_sha256": SOURCE_SHA256, "trial_count": TRIAL_COUNT,
              "process_identity": remote["_process_identity"](),
              "started_utc": _utc(), "status": "ABORTED_READONLY",
              "before": None, "initial_live_safety": None, "trials": [],
              "final_live_safety": None, "after": None, "errors": []}
    try:
        _check_pin(remote, pin)
        record["preflight_sha256"] = pin["preflight_sha256"]
        record["before"] = _check_snapshot(remote, pin)
        record["initial_live_safety"] = remote["_live_prewrite_safety"]()
        for index in range(1, TRIAL_COUNT + 1):
            trial = _trial(remote, pin, index)
            record["trials"].append(trial)
            if trial["status"] == "ABORTED":
                raise RuntimeError("timing trial %d failed closed: %s" %
                                   (index, trial.get("error", {}).get("message", "unknown")))
        record["final_live_safety"] = remote["_live_prewrite_safety"]()
    except BaseException as error:
        record["errors"].append(_error(error))
    finally:
        try:
            record["after"] = _check_snapshot(remote, pin)
        except BaseException as error:
            record["errors"].append(_error(error))
        record["completed_utc"] = _utc()
        if not record["errors"] and len(record["trials"]) == TRIAL_COUNT:
            misses = sum(item["status"] == "TIMING_MISS" for item in record["trials"])
            record["status"] = ("COMPLETED_20_WITH_TIMING_MISS" if misses else
                                "COMPLETED_20_WITHIN_SOURCE_WINDOW_NO_OPEN")
        record["timing_miss_count"] = sum(
            item["status"] == "TIMING_MISS" for item in record["trials"])
    return record
