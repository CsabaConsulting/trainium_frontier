"""Read-only timing of the exact trn21 r10 stage r3 pre-open helper.

The collector supplies SHA-pinned, independently static-PASSed stage source
as a module namespace. This file calls only its existing read-only lock,
preflight and `_final_no_write_continuity` functions. It never calls the
writer, creates a target, or opens the bundle. A clock reading is merely the
earliest hypothetical O_EXCL point, not an actual-open or post-open proof.
"""

import datetime
import json
import os
import sys
import time


SCHEMA = "trn21-r10-stage-preopen-readonly-timing-v1"
TRIALS = 3
MAX_LINE = 8192
EXPECTED = {
    "HOST": "ip-172-31-34-85",
    "UID": 1000,
    "DEV": 66305,
    "SOURCE": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2",
    "TARGET": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10",
    "BUNDLE": "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle",
    "LABEL": "fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2",
    "LOCK": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
    "BASE_COMMIT": "e65a3097a5b00f6178188a935bb2acb9a5042c6c",
    "TARGET_COMMIT": "22bfbab5a77d6feb670d2d737504f8b70b08ddc1",
    "PRE_FREE": 13_177_973_911,
    "MIN_INODES": 10_000,
    "MAX_FINAL_CHECK_AGE_SECONDS": 1.0,
}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _emit(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_LINE:
        raise RuntimeError("bounded timing JSONL line exceeded")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def _source_identity(stage):
    for key, expected in EXPECTED.items():
        actual = stage[key]
        if key in ("SOURCE", "TARGET", "BUNDLE", "LOCK"):
            actual = str(actual)
        if actual != expected:
            raise RuntimeError("pinned stage identity changed: " + key)
    if sys.flags.optimize:
        raise RuntimeError("optimized Python is forbidden")
    return {"hostname": EXPECTED["HOST"], "uid": EXPECTED["UID"],
            "source": EXPECTED["SOURCE"], "target": EXPECTED["TARGET"],
            "bundle": EXPECTED["BUNDLE"], "label": EXPECTED["LABEL"],
            "lock": EXPECTED["LOCK"],
            "source_commit": EXPECTED["BASE_COMMIT"],
            "target_commit": EXPECTED["TARGET_COMMIT"]}


def run(stage):
    """Measure at most three original helper calls under its existing lock."""
    identity = _source_identity(stage)
    with stage["_host_lock"]() as lock:
        probe_started = time.monotonic()
        snapshot = stage["_probe"](lock, minimum=stage["PRE_FREE"])
        probe_seconds = time.monotonic() - probe_started
        if snapshot["bundle_exists"] or snapshot["target_exists"]:
            raise RuntimeError("bundle or target already exists; no timing")
        _emit({"schema": SCHEMA, "kind": "start", "utc": _utc(),
               "identity": identity, "boot_id": snapshot["boot_id"],
               "mount_namespace": snapshot["mount"]["namespace"],
               "root_free_bytes": snapshot["identity_after"]["root_free_bytes"],
               "root_free_inodes": snapshot["identity_after"]["root_free_inodes"],
               "full_readonly_preflight_seconds": probe_seconds,
               "max_trials": TRIALS, "actual_open": False})
        completed = 0
        for index in range(1, TRIALS + 1):
            started = time.monotonic()
            try:
                # This is the exact static-PASSed pre-open sequence: full
                # activity, two broad scans, post-scan activity, root space,
                # STOP/target absence and the source's one-second age gate.
                samples = stage["_final_no_write_continuity"]()
                would_open = time.monotonic()  # Clock only. No O_EXCL.
                ages = stage["_freshness_evidence"](samples, would_open)
                result = {"status": "WITHIN_WINDOW_NO_OPEN",
                          "label_scan_duration_seconds":
                              samples["label_scan_duration_seconds"],
                          "activity_age_seconds": ages["activity_age_seconds"],
                          "label_age_seconds": ages["label_age_seconds"],
                          "space_age_seconds": ages["space_age_seconds"],
                          "root_free_bytes":
                              samples["root_space_before_open"]["root_free_bytes"],
                          "root_free_inodes":
                              samples["root_space_before_open"]["root_free_inodes"]}
            except Exception as error:
                result = {"status": "HOLD_NO_OPEN", "error_type": type(error).__name__,
                          "error": str(error)[:256]}
            completed += 1
            _emit({"schema": SCHEMA, "kind": "trial", "index": index,
                   "utc": _utc(), "preopen_seconds": time.monotonic() - started,
                   **result})
            if result["status"] != "WITHIN_WINDOW_NO_OPEN":
                break
        terminal = ("READONLY_TIMING_COMPLETE_NO_OPEN" if completed == TRIALS and
                    result["status"] == "WITHIN_WINDOW_NO_OPEN" else
                    "READONLY_TIMING_HOLD_NO_OPEN")
        _emit({"schema": SCHEMA, "kind": "end", "utc": _utc(),
               "completed_trials": completed, "max_trials": TRIALS,
               "status": terminal, "actual_open": False})
        return terminal
