#!/usr/bin/env python3
"""Read-only arrival-time observer for one direct fresh-start A0 process.

The observer never launches, signals, or writes runtime state. It prints exactly one
terminal JSON object to stdout; the Operator may redirect that stdout outside the
science checkout for later archival.
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"not a regular single-link file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(fd)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proc_identity(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    tail = raw.rsplit(") ", 1)[1].split()
    if len(tail) < 20:
        raise ValueError(f"short /proc stat for pid {pid}")
    return {
        "pid": pid,
        "state": tail[0],
        "ppid": int(tail[1]),
        "process_group": int(tail[2]),
        "session": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def json_object(data: bytes, what: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {what} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{what} JSON is not an object")
    return value


def snapshot_time() -> tuple[int, int, str]:
    return time.monotonic_ns(), time.time_ns(), utc_now()


def add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def validate_process(value: dict, label: str, observed_wall_ns: int) -> tuple[dict, datetime]:
    if value.get("label") != label:
        raise ValueError("process label mismatch")
    for key in ("guard_pid", "child_pid", "child_session"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] <= 0:
            raise ValueError(f"invalid process {key}")
    if value["child_session"] != value["child_pid"]:
        raise ValueError("child session is not the child pid")
    command = value.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise ValueError("invalid process command")
    started = parse_utc(value.get("started_at"))
    observed = datetime.fromtimestamp(observed_wall_ns / 1e9, timezone.utc)
    if started > observed:
        raise ValueError("process started_at is later than first observation")
    guard, child = proc_identity(value["guard_pid"]), proc_identity(value["child_pid"])
    if guard is None or child is None or child["state"] == "Z":
        raise ValueError("guard or live child identity unavailable at first observation")
    if child["session"] != value["child_session"]:
        raise ValueError("live child session mismatch")
    return {
        "label": label,
        "guard_pid": value["guard_pid"],
        "guard_start_ticks": guard["start_ticks"],
        "child_pid": value["child_pid"],
        "child_session": value["child_session"],
        "child_start_ticks": child["start_ticks"],
        "command": command,
        "started_at": value["started_at"],
    }, started


def validate_postflight(value: dict, label: str) -> list[str]:
    reasons = []
    if value.get("protocol_version") != 2:
        reasons.append("postflight_protocol_version_mismatch")
    if value.get("label") != label or value.get("stage") != "postflight":
        reasons.append("postflight_label_or_stage_mismatch")
    try:
        parse_utc(value.get("completed_at"))
    except ValueError:
        reasons.append("postflight_completed_at_invalid")
    rc = value.get("command_returncode")
    if isinstance(rc, bool) or not isinstance(rc, int) or rc != 0:
        reasons.append("postflight_command_not_successful")
    if value.get("guard_error") is not None:
        reasons.append("postflight_guard_error")
    if value.get("host_clean") is not True or value.get("failures") != []:
        reasons.append("postflight_host_not_clean")
    cleanup = value.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("remaining") != []:
        reasons.append("postflight_cleanup_incomplete")
    return reasons


def run(args: argparse.Namespace) -> tuple[dict, int]:
    started_mono, started_wall, started_utc = snapshot_time()
    deadline = started_mono + int(args.timeout_seconds * 1e9)
    reasons: list[str] = []
    observations: list[dict] = []
    process_value = process_identity = process_bytes = process_stat = process_started = None
    boundary_bytes = b""
    boundary_parsed_bytes = 0
    boundary_stat = None
    boundary_stream_invalid = False
    child_terminal = None
    postflight_value = postflight_bytes = postflight_stat = postflight_observed = None

    while postflight_observed is None and time.monotonic_ns() < deadline:
        poll_mono, poll_wall, poll_utc = snapshot_time()
        guard_missing_this_poll = False

        if process_identity is None:
            try:
                current, info = read_regular(args.process_json)
            except FileNotFoundError:
                current = None
            except (OSError, ValueError) as exc:
                add_reason(reasons, f"process_file_invalid:{exc}")
                current = None
            if current is not None:
                try:
                    value = json_object(current, "process")
                    identity, launched = validate_process(value, args.label, poll_wall)
                except ValueError as exc:
                    add_reason(reasons, f"process_identity_invalid:{exc}")
                else:
                    process_value, process_identity, process_bytes = value, identity, current
                    process_stat, process_started = info, launched
                    process_identity.update({
                        "process_json_sha256": sha256(current),
                        "process_json_device": info.st_dev,
                        "process_json_inode": info.st_ino,
                        "first_observed_monotonic_ns": poll_mono,
                        "first_observed_utc": poll_utc,
                    })
        else:
            try:
                current, info = read_regular(args.process_json)
                if (current != process_bytes or info.st_dev != process_stat.st_dev
                        or info.st_ino != process_stat.st_ino):
                    add_reason(reasons, "process_file_changed_after_identity")
            except (FileNotFoundError, OSError, ValueError):
                add_reason(reasons, "process_file_disappeared_or_became_unsafe")

            for role, pid_key, ticks_key in (
                ("guard", "guard_pid", "guard_start_ticks"),
                ("child", "child_pid", "child_start_ticks"),
            ):
                live = proc_identity(process_identity[pid_key])
                if live is not None and live["start_ticks"] != process_identity[ticks_key]:
                    add_reason(reasons, f"{role}_pid_reused")
                if role == "guard" and live is None:
                    guard_missing_this_poll = True
                if role == "child" and child_terminal is None and (live is None or live["state"] == "Z"):
                    child_terminal = {
                        "first_observed_monotonic_ns": poll_mono,
                        "first_observed_utc": poll_utc,
                    }

        try:
            current, info = read_regular(args.boundary_file)
        except FileNotFoundError:
            current = None
        except (OSError, ValueError) as exc:
            add_reason(reasons, f"boundary_file_invalid:{exc}")
            current = None
        if current is not None:
            if process_identity is None and current:
                add_reason(reasons, "boundary_observed_before_process_identity")
            if boundary_stat is None:
                boundary_stat = info
            elif info.st_dev != boundary_stat.st_dev or info.st_ino != boundary_stat.st_ino:
                add_reason(reasons, "boundary_file_identity_changed")
                boundary_stream_invalid = True
            if not current.startswith(boundary_bytes):
                add_reason(reasons, "boundary_file_truncated_or_rewritten")
                boundary_stream_invalid = True
            elif not boundary_stream_invalid:
                unparsed = current[boundary_parsed_bytes:]
                complete_end = unparsed.rfind(b"\n") + 1
                complete = unparsed[:complete_end]
                new_lines = complete.splitlines()
                if len(new_lines) > 1:
                    add_reason(reasons, "observation_gap_multiple_boundaries_in_one_poll")
                for raw in new_lines:
                    try:
                        text = raw.decode("ascii")
                        if not text or not text.isdigit() or (len(text) > 1 and text[0] == "0"):
                            raise ValueError
                        value = int(text)
                    except (UnicodeDecodeError, ValueError):
                        add_reason(reasons, "boundary_line_not_canonical_integer")
                        boundary_stream_invalid = True
                        break
                    expected = len(observations)
                    if value != expected:
                        add_reason(reasons, "boundary_gap_or_duplicate")
                    observations.append({
                        "ordinal": len(observations) + 1,
                        "value": value,
                        "first_observed_monotonic_ns": poll_mono,
                        "first_observed_utc": poll_utc,
                    })
                boundary_parsed_bytes += complete_end
            boundary_bytes = current

        try:
            current, info = read_regular(args.postflight_json)
        except FileNotFoundError:
            current = None
        except (OSError, ValueError) as exc:
            add_reason(reasons, f"postflight_file_invalid:{exc}")
            current = None
        if current is not None:
            try:
                value = json_object(current, "postflight")
            except ValueError as exc:
                add_reason(reasons, f"postflight_invalid:{exc}")
            else:
                postflight_value, postflight_bytes, postflight_stat = value, current, info
                postflight_observed = {
                    "first_observed_monotonic_ns": poll_mono,
                    "first_observed_utc": poll_utc,
                }
                for reason in validate_postflight(value, args.label):
                    add_reason(reasons, reason)
        if guard_missing_this_poll and postflight_observed is None:
            add_reason(reasons, "guard_disappeared_before_postflight")
        if postflight_observed is None:
            time.sleep(args.poll_seconds)

    timed_out = postflight_observed is None
    if timed_out:
        add_reason(reasons, "observer_timeout_before_postflight")
    if process_identity is None:
        add_reason(reasons, "process_identity_never_observed")
    if child_terminal is None:
        add_reason(reasons, "child_terminal_state_not_observed")
    if len(observations) < 5 or [x["value"] for x in observations[:5]] != [0, 1, 2, 3, 4]:
        add_reason(reasons, "first_five_boundaries_incomplete_or_nonsequential")
    if postflight_observed is not None and boundary_bytes and not boundary_bytes.endswith(b"\n"):
        add_reason(reasons, "boundary_file_has_terminal_partial_line")

    fifth = observations[4] if len(observations) >= 5 else None
    timing = None
    if fifth is not None and process_identity is not None:
        fifth_wall = parse_utc(fifth["first_observed_utc"])
        timing = {
            "boundary5_value": fifth["value"],
            "launch_to_boundary5_seconds": (fifth_wall - process_started).total_seconds(),
            "process_observed_to_boundary5_seconds":
                (fifth["first_observed_monotonic_ns"]
                 - process_identity["first_observed_monotonic_ns"]) / 1e9,
            "boundary5_to_child_terminal_observed_seconds": None,
            "boundary5_to_postflight_observed_seconds": None,
        }
        if child_terminal is not None:
            timing["boundary5_to_child_terminal_observed_seconds"] = (
                child_terminal["first_observed_monotonic_ns"]
                - fifth["first_observed_monotonic_ns"]) / 1e9
        if postflight_observed is not None:
            timing["boundary5_to_postflight_observed_seconds"] = (
                postflight_observed["first_observed_monotonic_ns"]
                - fifth["first_observed_monotonic_ns"]) / 1e9
        if any(isinstance(x, float) and (not math.isfinite(x) or x < 0) for x in timing.values()):
            add_reason(reasons, "timing_interval_invalid")

    status = ("TIMEOUT_NON_SCORED" if timed_out else
              "COMPLETE_SCORED_TIMING_PROOF" if not reasons else "COMPLETE_NON_SCORED")
    result = {
        "schema": "a0-boundary-observer-v1",
        "label": args.label,
        "status": status,
        "scored_timing_proof": status == "COMPLETE_SCORED_TIMING_PROOF",
        "observer": {
            "started_monotonic_ns": started_mono,
            "started_wall_ns": started_wall,
            "started_utc": started_utc,
            "poll_seconds": args.poll_seconds,
            "timeout_seconds": args.timeout_seconds,
        },
        "paths": {
            "process_json": str(args.process_json),
            "boundary_file": str(args.boundary_file),
            "postflight_json": str(args.postflight_json),
        },
        "process_identity": process_identity,
        "boundaries": observations,
        "timing": timing,
        "child_terminal": child_terminal,
        "postflight": ({
            "sha256": sha256(postflight_bytes),
            "device": postflight_stat.st_dev,
            "inode": postflight_stat.st_ino,
            "first_observed_monotonic_ns": postflight_observed["first_observed_monotonic_ns"],
            "first_observed_utc": postflight_observed["first_observed_utc"],
            "completed_at": postflight_value.get("completed_at"),
            "command_returncode": postflight_value.get("command_returncode"),
            "host_clean": postflight_value.get("host_clean"),
        } if postflight_observed is not None else None),
        "invalid_reasons": reasons,
    }
    return result, 0 if result["scored_timing_proof"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--process-json", required=True, type=Path)
    parser.add_argument("--boundary-file", required=True, type=Path)
    parser.add_argument("--postflight-json", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=4800.0)
    args = parser.parse_args()
    if not re.fullmatch(r"fs-[A-Za-z0-9_.-]+", args.label):
        parser.error("label must be a canonical fs-* request label")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("positive poll/timeout values are required")
    result, code = run(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
