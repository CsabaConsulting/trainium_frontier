"""Read-only trn21 r10 r5 write-path diagnostic; not a staging engine.

``reference`` is the namespace made by executing the *hash-pinned* archived
``remote_incremental.py`` in an isolated remote Python process. Only its
preflight/postflight, probe, stable-binding and final *no-write* checks are
called. In particular this module never calls the archived writer,
reconstruction path, or an exclusive/open-for-write operation.
"""

import hashlib
import json
import math
import os
import stat
import subprocess
import time


SCHEMA = "trn21-r10-r5-write-hold-readonly-diagnostic-v1"
PHASES = frozenset(("preflight", "preopen", "postflight"))
MAX_TRACE_ITEMS = 12
TRACE_NAMES = ("_activity", "_label_scan", "_space", "_freshness_evidence")


def _elapsed(started):
    value = time.monotonic() - started
    if not math.isfinite(value) or value < 0:
        return None
    return round(value, 6)


def _code(error):
    """A finite, nonsecret category; never return exception text or stdout."""
    if isinstance(error, subprocess.TimeoutExpired):
        return "SUBPROCESS_TIMEOUT"
    if isinstance(error, OSError):
        if error.errno in (13, 1):
            return "OS_PERMISSION"
        if error.errno in (2, 20):
            return "OS_PATH_MISSING_OR_NOT_DIRECTORY"
        if error.errno in (17,):
            return "OS_TARGET_EXISTS"
        return "OS_OTHER"
    message = str(error)
    if message.startswith("final label scan too old"):
        return "FINAL_LABEL_SCAN_AGE"
    if message.startswith("final activity/label/space proof too old"):
        return "FINAL_FRESHNESS_AGE"
    if message.startswith("preflight-to-stage identity drift"):
        return "PREFLIGHT_IDENTITY_DRIFT"
    if message.startswith("fresh preflight has expired"):
        return "PREFLIGHT_EXPIRED"
    if message.startswith("preflight was not a distinct SSH process"):
        return "PREFLIGHT_NOT_DISTINCT"
    if message.startswith("missing exact preflight binding"):
        return "PREFLIGHT_BINDING_MISSING"
    if message.startswith("incoming exact bundle bytes/hash mismatch"):
        return "INPUT_BUNDLE_MISMATCH"
    if message.startswith("stage attempt label already appears"):
        return "LABEL_PRESENT_OR_SCAN_FAILED"
    if message.startswith("STOP") or " STOP " in message or message.endswith(" STOP is present"):
        return "STOP_PRESENT"
    if message.startswith("matched train/eval/guard/process activity"):
        return "PROCESS_ACTIVITY"
    if message.startswith("Neuron device holder") or message.startswith("neuron-ls failed"):
        return "NEURON_ACTIVITY_OR_PROBE_ERROR"
    if message.startswith("root byte/inode floor failed"):
        return "ROOT_SPACE_FLOOR"
    if message.startswith("one-shot stage target appeared") or message.startswith("one-shot bundle/target"):
        return "TARGET_OR_BUNDLE_PRESENT"
    if message.startswith("wrong trn21 host or UID"):
        return "HOST_OR_UID_MISMATCH"
    if message.startswith("absolute GNU timeout identity check failed"):
        return "REMOTE_TIMEOUT_IDENTITY"
    if message.startswith("Git command failed") or message.startswith("accepted e65 checkout"):
        return "SOURCE_GIT_DRIFT"
    if message.startswith("type/device/owner/inode/mode/link drift") or message.startswith("file bytes/hash changed"):
        return "SOURCE_OR_PROTECTED_IDENTITY_DRIFT"
    return "OTHER_CHECK_FAILED"


def _hold(phase, stage, error, started, trace):
    return {
        "schema": SCHEMA,
        "phase": phase,
        "status": "HOLD_READONLY",
        "stage": stage,
        "error_code": _code(error),
        "exception_class": type(error).__name__ if type(error).__name__ in
                           ("RuntimeError", "ValueError", "TypeError", "OSError",
                            "TimeoutExpired", "FileNotFoundError", "PermissionError")
                           else "OtherException",
        "elapsed_seconds": _elapsed(started),
        "trace": trace[:MAX_TRACE_ITEMS],
        "remote_write_attempted": False,
        "source_stage_authorized": False,
        "native_training_authorized": False,
    }


def _instrument(reference, trace, current):
    originals = {}
    for name in TRACE_NAMES:
        original = reference[name]
        originals[name] = original

        def wrapped(*args, _name=name, _original=original, **kwargs):
            current[0] = "continuity." + _name[1:]
            began = time.monotonic()
            try:
                return _original(*args, **kwargs)
            finally:
                if len(trace) < MAX_TRACE_ITEMS:
                    trace.append({"check": _name[1:],
                                  "elapsed_seconds": _elapsed(began)})

        reference[name] = wrapped
    return originals


def probe(reference, phase, prior=None, data=b""):
    """Observe exactly the r5 pre-open path up to, but not including, O_EXCL.

    The caller must use a new SSH process per phase. A PASS is diagnostic
    evidence only: it cannot predict a later actual exclusive-open result.
    """
    if phase not in PHASES or not isinstance(reference, dict):
        raise ValueError("unsupported read-only diagnostic phase")
    started = time.monotonic()
    stage, trace = "entry", []
    try:
        if phase in ("preflight", "postflight"):
            if prior is not None or data:
                raise ValueError("unexpected input for read-only inventory")
            stage = "archived_" + phase
            receipt = reference["perform"](phase, None, b"")
            if not isinstance(receipt, dict) or receipt.get("phase") != phase:
                raise ValueError("archived read-only receipt malformed")
            if (receipt.get("snapshot", {}).get("bundle_exists") is not False or
                    receipt.get("snapshot", {}).get("target_exists") is not False):
                raise RuntimeError("one-shot bundle/target already exists")
            return {"schema": SCHEMA, "phase": phase,
                    "status": "PASS_READONLY", "stage": "completed",
                    "snapshot": receipt["snapshot"],
                    "elapsed_seconds": _elapsed(started),
                    "remote_write_attempted": False,
                    "source_stage_authorized": False,
                    "native_training_authorized": False}

        if not isinstance(prior, dict) or not isinstance(data, bytes):
            raise ValueError("preopen needs exact prior snapshot and bytes")
        stage = "remote_gnu_timeout_identity"
        reference["_timeout_identity"]()
        stage = "host_lock"
        with reference["_host_lock"]() as lock:
            stage = "full_write_phase_probe"
            snapshot = reference["_probe"](lock, minimum=reference["PRE_FREE"])
            stage = "distinct_preflight_binding"
            reference["_stable"](prior, snapshot)
            stage = "target_absence"
            if snapshot["bundle_exists"] or snapshot["target_exists"]:
                raise RuntimeError("one-shot bundle/target already exists")
            stage = "exact_input_bundle"
            if (len(data) != reference["BUNDLE_BYTES"] or
                    hashlib.sha256(data).hexdigest() != reference["BUNDLE_SHA256"]):
                raise RuntimeError("incoming exact bundle bytes/hash mismatch")
            stage = "read_only_root_open"
            root = reference["ROOT"]
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                             os.O_CLOEXEC)
            try:
                opened, named = os.fstat(parent), os.lstat(root)
                if (not stat.S_ISDIR(opened.st_mode) or
                        (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino) or
                        (opened.st_uid, opened.st_gid) != (reference["UID"], reference["GID"])):
                    raise RuntimeError("read-only root identity changed")
                stage = "immediate_target_absence"
                if (os.path.lexists(reference["BUNDLE"]) or
                        os.path.lexists(reference["TARGET"])):
                    raise RuntimeError("one-shot stage target appeared during final checks")
                stage = "final_no_write_continuity"
                current = [stage]
                originals = _instrument(reference, trace, current)
                try:
                    samples = reference["_final_no_write_continuity"]()
                finally:
                    for name, original in originals.items():
                        reference[name] = original
                    stage = current[0]
                stage = "would_open_freshness"
                would_open = reference["_freshness_evidence"](samples,
                                                               time.monotonic())
                stage = "terminal_target_absence"
                if (os.path.lexists(reference["BUNDLE"]) or
                        os.path.lexists(reference["TARGET"]) or
                        any(os.path.lexists(path) for path in reference["STOP_PATHS"])):
                    raise RuntimeError("one-shot stage target appeared during final checks")
                stage = "completed"
                return {"schema": SCHEMA, "phase": phase,
                        "status": "PASS_READONLY", "stage": stage,
                        "snapshot_sha256": hashlib.sha256(json.dumps(
                            snapshot, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")).hexdigest(),
                        "would_open": would_open,
                        "elapsed_seconds": _elapsed(started),
                        "trace": trace[:MAX_TRACE_ITEMS],
                        "bundle_exists": False, "target_exists": False,
                        "remote_write_attempted": False,
                        "source_stage_authorized": False,
                        "native_training_authorized": False}
            finally:
                os.close(parent)
    except Exception as error:
        return _hold(phase, stage, error, started, trace)


if __name__ == "__main__":
    raise SystemExit("HOLD: this read-only diagnostic has no standalone remote entrypoint")
