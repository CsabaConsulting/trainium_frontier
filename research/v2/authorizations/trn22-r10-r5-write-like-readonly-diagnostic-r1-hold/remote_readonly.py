"""Read-only model of the r5 write phase up to, but never including, O_EXCL.

The caller supplies the exact, hash-checked r5 ``remote_incremental.py``
namespace. Its preflight and postflight run in separate SSH processes. This
module calls only its read-only gates and never calls ``_write_bundle`` or
``_reconstruct``. No host path is created, removed, renamed, or truncated.
"""

import hashlib
import os
import stat
import time


SCHEMA = "trn22-r10-r5-write-like-readonly-diagnostic-v1"


def _failure_class(message):
    """Return only a finite category; never include a raw host exception."""
    lower = message.lower()
    if "stop" in lower:
        return "STOP"
    if "fresh" in lower or "too old" in lower or "duration" in lower:
        return "FRESHNESS"
    if "label" in lower or "scan" in lower:
        return "LABEL_OR_SCAN"
    if "neuron" in lower or "process" in lower or "activity" in lower:
        return "ACTIVITY"
    if "byte/inode" in lower or "space" in lower or "free" in lower:
        return "SPACE"
    if "preflight" in lower or "identity drift" in lower:
        return "PREFLIGHT_STABILITY"
    if "bundle" in lower or "target" in lower or "checkout" in lower:
        return "TARGET_OR_BUNDLE"
    if "git" in lower or "source" in lower or "checkpoint" in lower:
        return "SOURCE_OR_PROTECTED"
    if "lock" in lower:
        return "LOCK"
    return "OTHER"


def perform_readonly(prior, data, stage):
    """Under the existing flock, sample the final *would-open* boundary.

    ``stage`` is the exact r5 remote source namespace. The returned snapshot
    is needed only in memory for cross-session comparison; the local Operator
    emits a sanitized summary, not this raw remote object.
    """
    started = time.monotonic()
    gate = "HOST_LOCK"
    timings = {}
    passed_gates = []
    try:
        tick = time.monotonic()
        with stage["_host_lock"]() as lock:
            timings["host_lock_seconds"] = time.monotonic() - tick
            passed_gates.append("HOST_LOCK")
            gate = "FULL_PROBE"
            tick = time.monotonic()
            snapshot = stage["_probe"](lock, minimum=stage["PRE_FREE"])
            timings["full_probe_seconds"] = time.monotonic() - tick
            passed_gates.append("FULL_PROBE")

            gate = "PREFLIGHT_STABLE"
            tick = time.monotonic()
            stage["_stable"](prior, snapshot)
            timings["preflight_stable_seconds"] = time.monotonic() - tick
            passed_gates.append("PREFLIGHT_STABLE")

            gate = "TARGET_BUNDLE_ABSENT"
            tick = time.monotonic()
            if (snapshot["bundle_exists"] or snapshot["target_exists"] or
                    os.path.lexists(stage["BUNDLE"]) or
                    os.path.lexists(stage["TARGET"])):
                raise RuntimeError("one-shot bundle/target already exists")
            timings["target_bundle_absent_seconds"] = time.monotonic() - tick
            passed_gates.append("TARGET_BUNDLE_ABSENT")

            gate = "BUNDLE_STDIN_IDENTITY"
            tick = time.monotonic()
            if (not isinstance(data, bytes) or
                    len(data) != stage["BUNDLE_BYTES"] or
                    hashlib.sha256(data).hexdigest() != stage["BUNDLE_SHA256"]):
                raise RuntimeError("incoming exact bundle bytes/hash mismatch")
            timings["bundle_stdin_identity_seconds"] = time.monotonic() - tick
            passed_gates.append("BUNDLE_STDIN_IDENTITY")

            gate = "ROOT_READONLY_FD"
            tick = time.monotonic()
            fd = os.open(stage["ROOT"], os.O_RDONLY | os.O_DIRECTORY |
                         os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                opened = os.fstat(fd)
                named = os.lstat(stage["ROOT"])
                if (not stat.S_ISDIR(opened.st_mode) or
                        (opened.st_dev, opened.st_ino, opened.st_mode) !=
                        (named.st_dev, named.st_ino, named.st_mode) or
                        (opened.st_dev, opened.st_uid, opened.st_gid) !=
                        (stage["DEV"], stage["UID"], stage["GID"])):
                    raise RuntimeError("read-only root parent identity changed")
                timings["root_readonly_fd_seconds"] = time.monotonic() - tick
                passed_gates.append("ROOT_READONLY_FD")

                gate = "FINAL_NO_WRITE_CONTINUITY"
                tick = time.monotonic()
                samples = stage["_final_no_write_continuity"]()
                timings["final_no_write_continuity_seconds"] = time.monotonic() - tick
                passed_gates.append("FINAL_NO_WRITE_CONTINUITY")

                # This is the first-instruction position corresponding to
                # the r5 O_EXCL boundary. It samples time only. No open of
                # BUNDLE and no payload/write operation follows it.
                gate = "WOULD_OPEN_FRESHNESS"
                tick = time.monotonic()
                would_open_at = time.monotonic()
                freshness = stage["_freshness_evidence"](samples, would_open_at)
                if (os.path.lexists(stage["BUNDLE"]) or
                        os.path.lexists(stage["TARGET"])):
                    raise RuntimeError("bundle/target appeared at would-open sample")
                timings["would_open_freshness_seconds"] = time.monotonic() - tick
                passed_gates.append("WOULD_OPEN_FRESHNESS")

                gate = "ROOT_FD_STABLE"
                after = os.fstat(fd)
                if (opened.st_dev, opened.st_ino, opened.st_mode) != (
                        after.st_dev, after.st_ino, after.st_mode):
                    raise RuntimeError("read-only root parent fd changed")
                passed_gates.append("ROOT_FD_STABLE")
            finally:
                os.close(fd)

        timings["total_seconds"] = time.monotonic() - started
        return {"schema": SCHEMA, "phase": "write_like", "status": "PASS_NO_WRITE",
                "gate": "ROOT_FD_STABLE", "passed_gates": passed_gates,
                "snapshot": snapshot, "freshness": freshness, "timings": timings,
                "source_stage_authorized": False,
                "native_training_authorized": False,
                "evaluation_authorized": False}
    except Exception as error:
        # The type, gate, timings and a hash suffice for diagnosis without
        # serializing an exception message, path, command or traceback.
        message = str(error).encode("utf-8", "backslashreplace")
        timings["total_seconds"] = time.monotonic() - started
        return {"schema": SCHEMA, "phase": "write_like", "status": "FAIL_NO_WRITE",
                "gate": gate, "reason_class": _failure_class(str(error)),
                "passed_gates": passed_gates,
                "error_type": type(error).__name__[:64],
                "error_sha256": hashlib.sha256(message).hexdigest(),
                "error_bytes": len(message), "timings": timings,
                "source_stage_authorized": False,
                "native_training_authorized": False,
                "evaluation_authorized": False}
