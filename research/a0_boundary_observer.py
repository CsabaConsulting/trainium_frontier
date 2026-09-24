#!/usr/bin/env python3
"""Read-only arrival observer for one generic fresh-start A0 process.

Prints one terminal JSON object. It never launches, signals, or writes runtime state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time


def now():
    return time.monotonic_ns(), datetime.now(timezone.utc)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def read_file(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("not a regular single-link file")
        return b"".join(iter(lambda: os.read(fd, 1 << 20), b"")), (info.st_dev, info.st_ino)
    finally:
        os.close(fd)


def object_json(data, name):
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not an object")
    return value


def proc(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(fields) < 20:
        raise ValueError("short /proc stat")
    return {"state": fields[0], "session": int(fields[3]), "start_ticks": int(fields[19])}


def add(reasons, value):
    if value not in reasons:
        reasons.append(value)


def observe(a):
    start_mono, start_utc = now()
    deadline = start_mono + int(a.timeout_seconds * 1e9)
    bad, boundaries = [], []
    process_blob = process_inode = identity = launch_utc = None
    step_blob, step_inode, parsed_bytes, stream_bad = b"", None, 0, False
    child_terminal = post = None

    while post is None and time.monotonic_ns() < deadline:
        mono, utc = now()
        guard_missing = False
        if identity is None:
            try:
                blob, inode = read_file(a.process_json)
            except FileNotFoundError:
                blob = None
            except (OSError, ValueError) as exc:
                add(bad, f"process_file_invalid:{exc}")
                blob = None
            if blob is not None:
                try:
                    value = object_json(blob, "process")
                    if value.get("label") != a.label:
                        raise ValueError("label mismatch")
                    for key in ("guard_pid", "child_pid", "child_session"):
                        if type(value.get(key)) is not int or value[key] <= 0:
                            raise ValueError(f"invalid {key}")
                    command = value.get("command")
                    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
                        raise ValueError("invalid command")
                    launch_utc = timestamp(value.get("started_at"))
                    if launch_utc > utc or value["child_session"] != value["child_pid"]:
                        raise ValueError("invalid start/session identity")
                    guard, child = proc(value["guard_pid"]), proc(value["child_pid"])
                    if guard is None or child is None or child["state"] == "Z":
                        raise ValueError("live guard/child unavailable")
                    if child["session"] != value["child_session"]:
                        raise ValueError("live child session mismatch")
                except ValueError as exc:
                    add(bad, f"process_identity_invalid:{exc}")
                else:
                    process_blob, process_inode = blob, inode
                    identity = {
                        "label": a.label, "started_at": value["started_at"], "command": command,
                        "guard_pid": value["guard_pid"], "guard_start_ticks": guard["start_ticks"],
                        "child_pid": value["child_pid"], "child_session": value["child_session"],
                        "child_start_ticks": child["start_ticks"],
                        "process_json_sha256": hashlib.sha256(blob).hexdigest(),
                        "first_observed_monotonic_ns": mono, "first_observed_utc": utc.isoformat(),
                    }
        else:
            try:
                blob, inode = read_file(a.process_json)
                if blob != process_blob or inode != process_inode:
                    add(bad, "process_file_changed_after_identity")
            except (FileNotFoundError, OSError, ValueError):
                add(bad, "process_file_disappeared_or_became_unsafe")
            for role in ("guard", "child"):
                live = proc(identity[f"{role}_pid"])
                if live is not None and live["start_ticks"] != identity[f"{role}_start_ticks"]:
                    add(bad, f"{role}_pid_reused")
                if role == "guard" and live is None:
                    guard_missing = True
                if role == "child" and child_terminal is None and (live is None or live["state"] == "Z"):
                    child_terminal = {"first_observed_monotonic_ns": mono,
                                      "first_observed_utc": utc.isoformat()}

        try:
            blob, inode = read_file(a.boundary_file)
        except FileNotFoundError:
            blob = None
        except (OSError, ValueError) as exc:
            add(bad, f"boundary_file_invalid:{exc}")
            blob = None
        if blob is not None:
            if identity is None and blob:
                add(bad, "boundary_observed_before_process_identity")
            if step_inode is None:
                step_inode = inode
            elif inode != step_inode:
                add(bad, "boundary_file_identity_changed")
                stream_bad = True
            if not blob.startswith(step_blob):
                add(bad, "boundary_file_truncated_or_rewritten")
                stream_bad = True
            elif not stream_bad:
                unparsed = blob[parsed_bytes:]
                end = unparsed.rfind(b"\n") + 1
                lines = unparsed[:end].splitlines()
                if len(lines) > 1:
                    add(bad, "observation_gap_multiple_boundaries_in_one_poll")
                for raw in lines:
                    try:
                        text = raw.decode("ascii")
                        if not text.isdigit() or (len(text) > 1 and text[0] == "0"):
                            raise ValueError
                        value = int(text)
                    except (UnicodeDecodeError, ValueError):
                        add(bad, "boundary_line_not_canonical_integer")
                        stream_bad = True
                        break
                    if value != len(boundaries):
                        add(bad, "boundary_gap_or_duplicate")
                    boundaries.append({"ordinal": len(boundaries) + 1, "value": value,
                                       "first_observed_monotonic_ns": mono,
                                       "first_observed_utc": utc.isoformat()})
                parsed_bytes += end
            step_blob = blob

        try:
            blob, inode = read_file(a.postflight_json)
        except FileNotFoundError:
            blob = None
        except (OSError, ValueError) as exc:
            add(bad, f"postflight_file_invalid:{exc}")
            blob = None
        if blob is not None:
            try:
                value = object_json(blob, "postflight")
                completed = timestamp(value.get("completed_at"))
            except ValueError as exc:
                add(bad, f"postflight_invalid:{exc}")
            else:
                checks = {
                    "postflight_protocol_version_mismatch": value.get("protocol_version") == 2,
                    "postflight_label_or_stage_mismatch": value.get("label") == a.label and value.get("stage") == "postflight",
                    "postflight_command_not_successful": type(value.get("command_returncode")) is int and value["command_returncode"] == 0,
                    "postflight_guard_error": value.get("guard_error") is None,
                    "postflight_host_not_clean": value.get("host_clean") is True and value.get("failures") == [],
                    "postflight_cleanup_incomplete": isinstance(value.get("cleanup"), dict) and value["cleanup"].get("remaining") == [],
                }
                for reason, passed in checks.items():
                    if not passed:
                        add(bad, reason)
                if launch_utc is not None and completed < launch_utc:
                    add(bad, "postflight_completed_before_launch")
                post = {"sha256": hashlib.sha256(blob).hexdigest(), "device": inode[0],
                        "inode": inode[1], "completed_at": value.get("completed_at"),
                        "command_returncode": value.get("command_returncode"),
                        "host_clean": value.get("host_clean"),
                        "first_observed_monotonic_ns": mono, "first_observed_utc": utc.isoformat()}
        if guard_missing and post is None:
            add(bad, "guard_disappeared_before_postflight")
        if post is None:
            time.sleep(a.poll_seconds)

    timed_out = post is None
    if timed_out:
        add(bad, "observer_timeout_before_postflight")
    if identity is None:
        add(bad, "process_identity_never_observed")
    if child_terminal is None:
        add(bad, "child_terminal_state_not_observed")
    if len(boundaries) < 5 or [x["value"] for x in boundaries[:5]] != list(range(5)):
        add(bad, "first_five_boundaries_incomplete_or_nonsequential")
    if post is not None and step_blob and not step_blob.endswith(b"\n"):
        add(bad, "boundary_file_has_terminal_partial_line")

    fifth = boundaries[4] if len(boundaries) >= 5 else None
    timing = None
    if fifth is not None and identity is not None:
        timing = {
            "boundary5_value": fifth["value"],
            "launch_to_boundary5_seconds": (timestamp(fifth["first_observed_utc"]) - launch_utc).total_seconds(),
            "process_observed_to_boundary5_seconds": (fifth["first_observed_monotonic_ns"] - identity["first_observed_monotonic_ns"]) / 1e9,
            "boundary5_to_child_terminal_observed_seconds": None if child_terminal is None else (child_terminal["first_observed_monotonic_ns"] - fifth["first_observed_monotonic_ns"]) / 1e9,
            "boundary5_to_postflight_observed_seconds": None if post is None else (post["first_observed_monotonic_ns"] - fifth["first_observed_monotonic_ns"]) / 1e9,
        }
        if any(type(v) is float and (not math.isfinite(v) or v < 0) for v in timing.values()):
            add(bad, "timing_interval_invalid")

    status = "TIMEOUT_NON_SCORED" if timed_out else ("COMPLETE_NON_SCORED" if bad else "COMPLETE_SCORED_TIMING_PROOF")
    return {
        "schema": "a0-boundary-observer-v1", "label": a.label, "status": status,
        "scored_timing_proof": status == "COMPLETE_SCORED_TIMING_PROOF",
        "observer": {"started_monotonic_ns": start_mono, "started_utc": start_utc.isoformat(),
                     "poll_seconds": a.poll_seconds, "timeout_seconds": a.timeout_seconds},
        "paths": {"process_json": str(a.process_json), "boundary_file": str(a.boundary_file),
                  "postflight_json": str(a.postflight_json)},
        "process_identity": identity, "boundaries": boundaries, "timing": timing,
        "child_terminal": child_terminal, "postflight": post, "invalid_reasons": bad,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--process-json", required=True, type=Path)
    parser.add_argument("--boundary-file", required=True, type=Path)
    parser.add_argument("--postflight-json", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=4800.0)
    a = parser.parse_args()
    if not re.fullmatch(r"fs-[A-Za-z0-9_.-]+", a.label):
        parser.error("label must be a canonical fs-* request label")
    if a.poll_seconds <= 0 or a.timeout_seconds <= 0:
        parser.error("positive poll/timeout values are required")
    result = observe(a)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if result["scored_timing_proof"] else 2


if __name__ == "__main__":
    sys.exit(main())
