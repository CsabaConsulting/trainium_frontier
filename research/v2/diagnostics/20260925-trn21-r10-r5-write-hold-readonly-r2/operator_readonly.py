"""Disabled, no-write trn21 diagnostic wrapper for the consumed r5 stage.

This file has no live authority: its enabled request, independent review and
Supervisor assignment are deliberately absent. It reuses the hash-pinned r5
Operator's read-only STOP, SSH identity and *separate remote GNU-timeout probe*
for any future independently enabled diagnostic phase. It never calls the r5
staging entrypoint, creates a local marker, or writes on the host.
"""

import hashlib
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ARCHIVED = (HERE.parents[1] / "authorizations" /
            "trn21-c5c-a0-r10-incremental-stage-r2-hold")
R5_OPERATOR = ARCHIVED / "operator_once.py"
R5_REMOTE = ARCHIVED / "remote_incremental.py"
R5_BUNDLE = ARCHIVED / "e65-to-r10.bundle"
R5_JOURNAL = (HERE.parents[1] / "observations" /
              "20260925-trn21-r10-stage-r5-operator-result-hold-r1" /
              "RAW_ONE_ATTEMPT.jsonl")
R5_RESULT_REVIEW = (HERE.parents[1] / "verification" /
                    "20260925-trn21-r10-r5-one-shot-stage-result-independent-HOLD.md")
REMOTE_DIAGNOSTIC = HERE / "remote_readonly.py"
HOLD_REQUEST = HERE / "REQUEST.HOLD.json"
ENABLED_REQUEST = HERE / "REQUEST.ENABLED_PENDING_FINAL_REVIEW.json"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_READONLY_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_READONLY_ASSIGNMENT.json"

R5_OPERATOR_SHA256 = "f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b"
R5_REMOTE_SHA256 = "6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6"
R5_BUNDLE_SHA256 = "b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e"
R5_BUNDLE_BYTES = 109800
R5_JOURNAL_SHA256 = "54c65164271d895392c2e170b47dbdf07f62e0aab2cfa378e529afdbbeca3a4c"
R5_RESULT_REVIEW_SHA256 = "786f3f26179db7a31adb7b4440460e662427494f7cbfe192c76878803cea14bf"
HOST = "trn21"
HOSTNAME = "ip-172-31-34-85"
SOURCE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TARGET_COMMIT = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"
DIAGNOSTIC_LABEL = "fs-20260925-trn21-r10-r5-write-hold-readonly-d2"
SCOPE = "TRN21_R10_R5_CONSUMED_STAGE_READONLY_DIAGNOSIS_ONLY"
MAX_REMOTE_STDOUT = 128 * 1024
MAX_REMOTE_STDERR = 128 * 1024
MAX_REMOTE_SECONDS = 350.0
PRE_FREE = 13_177_973_911
POST_FREE = 10_737_418_240
MIN_INODES = 10_000
SNAPSHOT_IDENTITY_KEYS = ("hostname", "uid", "boot_id", "mount", "lock",
                          "source", "protected")
SNAPSHOT_KEYS = frozenset(("schema", "utc", "process_identity", "hostname",
                           "uid", "boot_id", "mount", "lock", "source",
                           "protected", "activity", "label", "identity_before",
                           "identity_after", "bundle_exists", "target_exists"))
TRACE_CHECKS = ("activity", "label_scan", "activity", "space",
                "freshness_evidence")
REMOTE_COMMON_KEYS = frozenset(("schema", "phase", "status", "stage",
                                "elapsed_seconds", "remote_write_attempted",
                                "source_stage_authorized",
                                "native_training_authorized"))
REMOTE_HOLD_KEYS = REMOTE_COMMON_KEYS | frozenset(("error_code",
                                                    "exception_class", "trace"))
REMOTE_INVENTORY_KEYS = REMOTE_COMMON_KEYS | frozenset(("snapshot",))
REMOTE_PREOPEN_KEYS = REMOTE_COMMON_KEYS | frozenset(("snapshot_sha256",
                                                    "would_open", "trace",
                                                    "bundle_exists", "target_exists"))
WOULD_OPEN_KEYS = frozenset(("activity_age_seconds", "label_age_seconds",
                             "space_age_seconds", "pre_open_broad_label_scan",
                             "post_label_activity_check",
                             "label_scan_duration_seconds",
                             "root_space_before_open"))
REMOTE_STAGES = frozenset(("entry", "archived_preflight", "archived_postflight",
                           "remote_gnu_timeout_identity", "host_lock",
                           "full_write_phase_probe", "distinct_preflight_binding",
                           "target_absence", "exact_input_bundle",
                           "read_only_root_open", "immediate_target_absence",
                           "final_no_write_continuity", "continuity.activity",
                           "continuity.label_scan", "continuity.space",
                           "continuity.freshness_evidence",
                           "would_open_freshness", "terminal_target_absence",
                           "completed"))
REMOTE_ERROR_CODES = frozenset(("SUBPROCESS_TIMEOUT", "OS_PERMISSION",
                                "OS_PATH_MISSING_OR_NOT_DIRECTORY",
                                "OS_TARGET_EXISTS", "OS_OTHER",
                                "FINAL_LABEL_SCAN_AGE", "FINAL_FRESHNESS_AGE",
                                "PREFLIGHT_IDENTITY_DRIFT", "PREFLIGHT_EXPIRED",
                                "PREFLIGHT_NOT_DISTINCT", "PREFLIGHT_BINDING_MISSING",
                                "INPUT_BUNDLE_MISMATCH", "LABEL_PRESENT_OR_SCAN_FAILED",
                                "STOP_PRESENT", "PROCESS_ACTIVITY",
                                "NEURON_ACTIVITY_OR_PROBE_ERROR", "ROOT_SPACE_FLOOR",
                                "TARGET_OR_BUNDLE_PRESENT", "HOST_OR_UID_MISMATCH",
                                "REMOTE_TIMEOUT_IDENTITY", "SOURCE_GIT_DRIFT",
                                "SOURCE_OR_PROTECTED_IDENTITY_DRIFT",
                                "OTHER_CHECK_FAILED"))
REMOTE_EXCEPTION_CLASSES = frozenset(("RuntimeError", "ValueError", "TypeError",
                                      "OSError", "TimeoutExpired",
                                      "FileNotFoundError", "PermissionError",
                                      "OtherException"))
TRANSPORT_ERROR_CODES = frozenset(("SSH_OUTPUT_TYPE_INVALID",
                                   "SSH_RETURN_CODE_INVALID", "SSH_NONZERO",
                                   "SSH_STDERR_NONEMPTY", "SSH_OUTPUT_OVERSIZE",
                                   "REMOTE_JSON_INVALID", "REMOTE_RECEIPT_INVALID",
                                   "LOCAL_SSH_TIMEOUT", "LOCAL_PREPHASE_FAILURE"))
LOCAL_PHASE_STAGES = frozenset(("local_stop_before_identity",
                                "pinned_ssh_identity",
                                "local_stop_before_remote_gnu_probe",
                                "separate_remote_gnu_timeout_probe",
                                "local_stop_after_remote_gnu_probe",
                                "diagnostic_ssh_phase"))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _regular(path, *, maximum=2 * 1024 * 1024):
    """Read only the named, owner-owned, nonlinked local reference file."""
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or named.st_size > maximum):
        raise RuntimeError("local diagnostic reference metadata changed")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("local diagnostic reference inode changed")
        out = bytearray()
        while True:
            piece = os.read(fd, min(1024 * 1024, maximum - len(out) + 1))
            if not piece:
                break
            out.extend(piece)
            if len(out) > maximum:
                raise RuntimeError("local diagnostic reference exceeds bound")
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise RuntimeError("local diagnostic reference changed while reading")
        return bytes(out)
    finally:
        os.close(fd)


def _exact(path, digest, *, maximum=2 * 1024 * 1024):
    raw = _regular(path, maximum=maximum)
    if _sha(raw) != digest:
        raise RuntimeError("pinned diagnostic reference SHA-256 changed")
    return raw


def _json(path):
    raw = _regular(path, maximum=16 * 1024)
    value = json.loads(raw)
    if not isinstance(value, dict) or raw.strip() != _canonical(value):
        raise RuntimeError("diagnostic authority JSON is not canonical")
    return value, _sha(raw)


def _archived_snapshot(journal_bytes):
    """Use the hash-pinned r5 preflight as the complete receipt shape."""
    matches = []
    for line in journal_bytes.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError("pinned r5 journal row malformed")
        if row.get("event") == "FRESH_PREFLIGHT":
            receipt = row.get("receipt")
            if not isinstance(receipt, dict) or receipt.get("phase") != "preflight":
                raise RuntimeError("pinned r5 preflight malformed")
            matches.append(receipt.get("snapshot"))
    if (len(matches) != 1 or not isinstance(matches[0], dict) or
            frozenset(matches[0]) != SNAPSHOT_KEYS or
            matches[0].get("schema") != "trn21-r10-incremental-full-readonly-v1"):
        raise RuntimeError("pinned r5 full snapshot unavailable")
    return matches[0]


def _load_reference():
    """Load exact r5 sources in memory without invoking the writer."""
    operator_source = _exact(R5_OPERATOR, R5_OPERATOR_SHA256)
    remote_source = _exact(R5_REMOTE, R5_REMOTE_SHA256)
    bundle = _exact(R5_BUNDLE, R5_BUNDLE_SHA256)
    if len(bundle) != R5_BUNDLE_BYTES:
        raise RuntimeError("pinned r5 bundle length changed")
    archived_snapshot = _archived_snapshot(_exact(R5_JOURNAL, R5_JOURNAL_SHA256))
    _exact(R5_RESULT_REVIEW, R5_RESULT_REVIEW_SHA256)
    diagnostic_source = _regular(REMOTE_DIAGNOSTIC)
    # The known r5 operator module has no import-time host contact. In an
    # isolated namespace, its __main__ block is not entered. Its safe SSH
    # identity/probe helpers are reused byte-for-byte rather than retyped.
    namespace = {"__name__": "trn21_r5_readonly_reference",
                 "__file__": str(R5_OPERATOR)}
    exec(compile(operator_source, str(R5_OPERATOR), "exec"), namespace)
    return namespace, remote_source, bundle, diagnostic_source, archived_snapshot


def _expected_request(remote_diagnostic_hash, local_operator_hash):
    return {
        "schema": "trn21-r10-r5-write-hold-readonly-request-v1",
        "dispatch_authorized": True,
        "read_only_diagnostic_authorized": True,
        "source_stage_authorized": False,
        "native_training_authorized": False,
        "evaluation_authorized": False,
        "aws_action_authorized": False,
        "host": HOST,
        "hostname": HOSTNAME,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "scope": SCOPE,
        "accepted_source_commit": SOURCE_COMMIT,
        "unstaged_target_commit": TARGET_COMMIT,
        "archived_r5_operator_sha256": R5_OPERATOR_SHA256,
        "archived_r5_remote_sha256": R5_REMOTE_SHA256,
        "archived_r5_bundle_sha256": R5_BUNDLE_SHA256,
        "consumed_r5_journal_sha256": R5_JOURNAL_SHA256,
        "independent_r5_result_review_sha256": R5_RESULT_REVIEW_SHA256,
        "remote_diagnostic_sha256": remote_diagnostic_hash,
        "local_diagnostic_operator_sha256": local_operator_hash,
        "phases": ["preflight", "preopen", "postflight"],
    }


def _authority(request, request_hash, review, review_hash, assignment,
               remote_diagnostic_hash, local_operator_hash):
    if request != _expected_request(remote_diagnostic_hash, local_operator_hash):
        raise RuntimeError("exact read-only diagnostic request absent or mismatched")
    if review != {
            "schema": "trn21-r10-r5-write-hold-readonly-final-review-v1",
            "verdict": "PASS_READONLY_DIAGNOSTIC_ONLY",
            "request_sha256": request_hash,
            "remote_diagnostic_sha256": remote_diagnostic_hash,
            "local_diagnostic_operator_sha256": local_operator_hash,
            "archived_r5_remote_sha256": R5_REMOTE_SHA256,
            "consumed_r5_journal_sha256": R5_JOURNAL_SHA256,
            "scope": SCOPE,
            "host": HOST,
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "source_stage_authorized": False,
            "native_training_authorized": False}:
        raise RuntimeError("independent exact read-only review absent or mismatched")
    if assignment != {
            "schema": "trn21-r10-r5-write-hold-readonly-assignment-v1",
            "request_sha256": request_hash,
            "review_sha256": review_hash,
            "scope": SCOPE,
            "host": HOST,
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "source_stage_authorized": False,
            "native_training_authorized": False}:
        raise RuntimeError("Supervisor exact read-only assignment absent or mismatched")


def _remote_command(phase, prior, remote_source, diagnostic_source):
    if phase not in ("preflight", "preopen", "postflight"):
        raise ValueError("unsupported diagnostic phase")
    program = (
        "import json,sys\n"
        "if sys.flags.optimize: raise SystemExit('optimized Python HOLD')\n"
        "reference={'__name__':'trn21_r5_archived_remote'}\n"
        "exec(" + repr(remote_source.decode("utf-8")) + ",reference)\n"
        "diagnostic={'__name__':'trn21_r5_readonly_diagnostic'}\n"
        "exec(" + repr(diagnostic_source.decode("utf-8")) + ",diagnostic)\n"
        "result=diagnostic['probe'](reference,sys.argv[1],json.loads(sys.argv[2]),"
        "sys.stdin.buffer.read())\n"
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))\n"
    )
    command = ["/usr/bin/timeout", "--foreground", "--signal=TERM",
               "--kill-after=15s", "320s", "/usr/bin/env", "-i",
               "PATH=/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
               "/usr/bin/python3", "-I", "-B", "-c", program, phase,
               json.dumps(prior, sort_keys=True, separators=(",", ":"))]
    return " ".join(shlex.quote(item) for item in command)


def _shape_matches(template, value):
    """The pinned r5 journal fixes every nested inventory key and JSON type."""
    if type(template) is not type(value):
        return False
    if isinstance(template, dict):
        return (template.keys() == value.keys() and
                all(_shape_matches(template[key], value[key]) for key in template))
    if isinstance(template, list):
        return (len(template) == len(value) and
                all(_shape_matches(left, right)
                    for left, right in zip(template, value)))
    return True


def _finite_between(value, ceiling):
    return (type(value) in (int, float) and math.isfinite(value) and
            0 <= value <= ceiling)


def _hex_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _utc_string(value):
    if type(value) is not str or len(value) > 48:
        return False
    try:
        stamp = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return stamp.utcoffset() == datetime.timedelta(0)


def _space_valid(space, minimum):
    return (type(space) is dict and
            space.keys() == {"root_free_bytes", "root_free_inodes"} and
            type(space["root_free_bytes"]) is int and
            type(space["root_free_inodes"]) is int and
            space["root_free_bytes"] >= minimum and
            space["root_free_inodes"] >= MIN_INODES)


def _snapshot_valid(snapshot, template, phase):
    minimum = PRE_FREE if phase == "preflight" else POST_FREE
    if (phase not in ("preflight", "postflight") or
            type(template) is not dict or type(snapshot) is not dict or
            not _shape_matches(template, snapshot) or
            frozenset(snapshot) != SNAPSHOT_KEYS or
            snapshot["schema"] != "trn21-r10-incremental-full-readonly-v1" or
            snapshot["hostname"] != HOSTNAME or
            type(snapshot["uid"]) is not int or snapshot["uid"] != 1000 or
            snapshot["bundle_exists"] is not False or
            snapshot["target_exists"] is not False or
            not _utc_string(snapshot["utc"])):
        return False
    process = snapshot["process_identity"]
    if (process.keys() != {"pid", "start_ticks"} or
            any(type(process[key]) is not int or process[key] <= 0
                for key in process)):
        return False
    if (type(snapshot["boot_id"]) is not str or
            re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                         snapshot["boot_id"]) is None):
        return False
    mount = snapshot["mount"]
    if (mount.keys() != {"root", "sha256", "namespace"} or
            mount["root"][1:] != template["mount"]["root"][1:] or
            re.fullmatch(r"[0-9]+", mount["root"][0]) is None or
            not _hex_sha(mount["sha256"]) or
            re.fullmatch(r"mnt:\[[0-9]+\]", mount["namespace"]) is None):
        return False
    for path, expected in template["lock"].items():
        observed = snapshot["lock"][path]
        if any(observed[key] != expected[key]
               for key in ("device", "inode", "mode")):
            return False
        if path.endswith("device0.lock") and observed["nlink"] != 1:
            return False
    source = snapshot["source"]
    if (source["commit"] != SOURCE_COMMIT or
            source["common_gitdir"] != template["source"]["common_gitdir"] or
            source["science_sha256"] != template["source"]["science_sha256"] or
            source["git_controls"] != template["source"]["git_controls"]):
        return False
    for name in ("source", "gitdir"):
        if any(source[name][key] != template["source"][name][key]
               for key in ("device", "inode", "uid", "gid", "mode", "nlink")):
            return False
    protected = snapshot["protected"]
    archived_protected = template["protected"]
    if (protected["checkpoints"] != archived_protected["checkpoints"] or
            protected["preserved_e65_stage"] !=
            archived_protected["preserved_e65_stage"]):
        return False
    for name in ("cache_root",):
        if any(protected[name][key] != archived_protected[name][key]
               for key in ("device", "inode", "uid", "gid")):
            return False
    for name, observed in protected["caches"].items():
        if any(observed[key] != archived_protected["caches"][name][key]
               for key in ("device", "inode", "uid", "gid")):
            return False
    if (snapshot["activity"] != {"stops_absent": True,
                                  "matched_processes": [],
                                  "neuron_holder_absent": True} or
            snapshot["label"]["label"] != template["label"]["label"] or
            snapshot["label"]["matches"] != [] or
            not _utc_string(snapshot["label"]["scan_utc"])):
        return False
    return (_space_valid(snapshot["identity_before"], minimum) and
            _space_valid(snapshot["identity_after"], minimum))


def _trace_valid(trace, *, complete):
    if type(trace) is not list or len(trace) > len(TRACE_CHECKS):
        return False
    if complete and len(trace) != len(TRACE_CHECKS):
        return False
    for index, item in enumerate(trace):
        if (type(item) is not dict or
                item.keys() != {"check", "elapsed_seconds"} or
                item["check"] != TRACE_CHECKS[index] or
                not _finite_between(item["elapsed_seconds"], MAX_REMOTE_SECONDS)):
            return False
    return True


def _would_open_valid(value):
    if (type(value) is not dict or frozenset(value) != WOULD_OPEN_KEYS or
            value["pre_open_broad_label_scan"] is not True or
            value["post_label_activity_check"] is not True or
            not _space_valid(value["root_space_before_open"], PRE_FREE)):
        return False
    for name in ("activity_age_seconds", "label_age_seconds",
                 "space_age_seconds", "label_scan_duration_seconds"):
        if not _finite_between(value[name], 1.0):
            return False
    return (value["label_age_seconds"] + 1e-6 >=
            value["activity_age_seconds"] and
            value["activity_age_seconds"] + 1e-6 >=
            value["space_age_seconds"])


def _inventory_projection(value, snapshot):
    identity = {key: snapshot[key] for key in SNAPSHOT_IDENTITY_KEYS}
    return {key: value[key] for key in REMOTE_COMMON_KEYS} | {
        "snapshot_sha256": _sha(_canonical(snapshot)),
        "identity_sha256": _sha(_canonical(identity)),
        "root_free_bytes_before": snapshot["identity_before"]["root_free_bytes"],
        "root_free_inodes_before": snapshot["identity_before"]["root_free_inodes"],
        "root_free_bytes_after": snapshot["identity_after"]["root_free_bytes"],
        "root_free_inodes_after": snapshot["identity_after"]["root_free_inodes"],
        "bundle_exists": False, "target_exists": False}


def _checked_remote(phase, value, template):
    """Return only bounded, typed evidence plus private full inventory."""
    if (type(value) is not dict or value.get("schema") !=
            "trn21-r10-r5-write-hold-readonly-diagnostic-v1" or
            value.get("phase") != phase or
            value.get("status") not in ("PASS_READONLY", "HOLD_READONLY") or
            value.get("remote_write_attempted") is not False or
            value.get("source_stage_authorized") is not False or
            value.get("native_training_authorized") is not False or
            type(value.get("stage")) is not str or
            value["stage"] not in REMOTE_STAGES):
        raise ValueError("invalid read-only phase receipt")
    elapsed = value.get("elapsed_seconds")
    if value["status"] == "HOLD_READONLY":
        if (frozenset(value) != REMOTE_HOLD_KEYS or
                value.get("error_code") not in REMOTE_ERROR_CODES or
                value.get("exception_class") not in REMOTE_EXCEPTION_CLASSES or
                (elapsed is not None and
                 not _finite_between(elapsed, MAX_REMOTE_SECONDS)) or
                not _trace_valid(value.get("trace"), complete=False)):
            raise ValueError("invalid read-only HOLD receipt")
        return dict(value), None
    if value["stage"] != "completed" or not _finite_between(elapsed, MAX_REMOTE_SECONDS):
        raise ValueError("incomplete read-only PASS receipt")
    if phase in ("preflight", "postflight"):
        if (frozenset(value) != REMOTE_INVENTORY_KEYS or
                not _snapshot_valid(value.get("snapshot"), template, phase)):
            raise ValueError("invalid full read-only inventory")
        snapshot = value["snapshot"]
        return _inventory_projection(value, snapshot), snapshot
    if (phase != "preopen" or frozenset(value) != REMOTE_PREOPEN_KEYS or
            not _hex_sha(value.get("snapshot_sha256")) or
            value.get("bundle_exists") is not False or
            value.get("target_exists") is not False or
            not _would_open_valid(value.get("would_open")) or
            not _trace_valid(value.get("trace"), complete=True)):
        raise ValueError("invalid pre-open read-only proof")
    return dict(value), None


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate remote JSON key")
        value[key] = item
    return value


def _reject_nonfinite_json(_value):
    raise ValueError("nonfinite remote JSON number")


def classify_transport(phase, returncode, stdout, stderr, elapsed_seconds,
                       snapshot_template):
    """Retain bounded cause *classes*, never raw stdout/stderr or exception text."""
    base = {"phase": phase,
            "returncode": returncode if type(returncode) is int else None,
            "elapsed_seconds": round(elapsed_seconds, 6) if
            type(elapsed_seconds) in (int, float) and
            math.isfinite(elapsed_seconds) and elapsed_seconds >= 0 else None,
            "stdout_bytes": len(stdout) if type(stdout) is bytes else None,
            "stderr_bytes": len(stderr) if type(stderr) is bytes else None}
    if type(stdout) is not bytes or type(stderr) is not bytes:
        return {**base, "transport": "HOLD", "error_code": "SSH_OUTPUT_TYPE_INVALID"}
    if type(returncode) is not int:
        return {**base, "transport": "HOLD", "error_code": "SSH_RETURN_CODE_INVALID"}
    if returncode != 0:
        return {**base, "transport": "HOLD", "error_code": "SSH_NONZERO"}
    if stderr:
        return {**base, "transport": "HOLD", "error_code": "SSH_STDERR_NONEMPTY"}
    if len(stdout) > MAX_REMOTE_STDOUT or len(stderr) > MAX_REMOTE_STDERR:
        return {**base, "transport": "HOLD", "error_code": "SSH_OUTPUT_OVERSIZE"}
    try:
        value = json.loads(stdout, object_pairs_hook=_unique_object,
                           parse_constant=_reject_nonfinite_json)
    except (ValueError, UnicodeDecodeError, TypeError, RecursionError):
        return {**base, "transport": "HOLD", "error_code": "REMOTE_JSON_INVALID"}
    try:
        projected, snapshot = _checked_remote(phase, value, snapshot_template)
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return {**base, "transport": "HOLD", "error_code": "REMOTE_RECEIPT_INVALID"}
    checked = {**base, "transport": "PASS", "remote": projected}
    if snapshot is not None:
        checked["_snapshot"] = snapshot  # Internal only; never returned by run_once.
    return checked


def _phase(reference, phase, prior, data, remote_source, diagnostic_source,
           snapshot_template):
    """Future read-only transport, gated through ``run_once`` authority.

    It probes remote /usr/bin/timeout in a *separate* SSH session before each
    phase. The original r5 pinned SSH identity and fresh local STOP/breaker
    checks surround that probe, so no phase can rely on an earlier check.
    """
    stage = "local_stop_before_identity"
    began = time.monotonic()
    try:
        reference["_local_stop_and_breakers"]()
        command = _remote_command(phase, prior, remote_source, diagnostic_source)
        stage = "pinned_ssh_identity"
        reference["_ssh_identity"]()
        stage = "local_stop_before_remote_gnu_probe"
        reference["_local_stop_and_breakers"]()
        stage = "separate_remote_gnu_timeout_probe"
        reference["_remote_timeout_identity"]()
        stage = "local_stop_after_remote_gnu_probe"
        reference["_local_stop_and_breakers"]()
        stage = "diagnostic_ssh_phase"
        try:
            result = subprocess.run(reference["_ssh_argv"](command),
                                    input=data, capture_output=True,
                                    timeout=350, check=False)
        except subprocess.TimeoutExpired:
            return {"phase": phase, "transport": "HOLD",
                    "error_code": "LOCAL_SSH_TIMEOUT", "stage": stage,
                    "elapsed_seconds": round(time.monotonic() - began, 6)}
        return classify_transport(phase, result.returncode, result.stdout,
                                  result.stderr, time.monotonic() - began,
                                  snapshot_template)
    except Exception as error:
        # Deliberately discard error text; it may contain a transport path.
        return {"phase": phase, "transport": "HOLD",
                "error_code": "LOCAL_PREPHASE_FAILURE",
                "stage": stage,
                "exception_class": type(error).__name__ if type(error).__name__ in
                                   ("RuntimeError", "ValueError", "OSError",
                                    "TimeoutExpired", "FileNotFoundError")
                                   else "OtherException",
                "elapsed_seconds": round(time.monotonic() - began, 6)}


def _public_phase(phase, candidate, template):
    """Recheck even a mocked/local phase before terminal PASS or serialization.

    ``_snapshot`` is private transport state used only to bind the next phase;
    full host paths and any unexpected remote fields cannot enter the result.
    """
    invalid = {"phase": phase, "transport": "HOLD",
               "error_code": "PHASE_RECEIPT_INVALID"}
    if type(candidate) is not dict or candidate.get("phase") != phase:
        return invalid, False, None
    if candidate.get("transport") == "HOLD":
        allowed = {"phase", "transport", "error_code", "stage",
                   "exception_class", "elapsed_seconds", "returncode",
                   "stdout_bytes", "stderr_bytes"}
        if (not candidate.keys() <= allowed or
                candidate.get("error_code") not in TRANSPORT_ERROR_CODES or
                ("stage" in candidate and
                 candidate["stage"] not in LOCAL_PHASE_STAGES) or
                ("exception_class" in candidate and
                 candidate["exception_class"] not in REMOTE_EXCEPTION_CLASSES) or
                ("returncode" in candidate and
                 candidate["returncode"] is not None and
                 type(candidate["returncode"]) is not int) or
                ("elapsed_seconds" in candidate and
                 candidate["elapsed_seconds"] is not None and
                 not _finite_between(candidate["elapsed_seconds"], 3600.0)) or
                any(type(candidate[key]) is not int or
                    not 0 <= candidate[key] <= 1_000_000_000
                    for key in ("stdout_bytes", "stderr_bytes")
                    if key in candidate and candidate[key] is not None)):
            return invalid, False, None
        # No raw remote record, stdout, stderr, path or exception message.
        return {key: candidate[key] for key in allowed if key in candidate}, False, None
    if candidate.get("transport") != "PASS":
        return invalid, False, None
    common = {"phase", "returncode", "elapsed_seconds", "stdout_bytes",
              "stderr_bytes", "transport", "remote"}
    has_inventory_snapshot = (phase in ("preflight", "postflight") and
                              type(candidate.get("remote")) is dict and
                              candidate["remote"].get("status") == "PASS_READONLY")
    expected_outer = common | ({"_snapshot"} if has_inventory_snapshot else set())
    if (candidate.keys() != expected_outer or
            type(candidate["returncode"]) is not int or
            candidate["returncode"] != 0 or
            not _finite_between(candidate["elapsed_seconds"], 500.0) or
            type(candidate["stdout_bytes"]) is not int or
            not 0 <= candidate["stdout_bytes"] <= MAX_REMOTE_STDOUT or
            type(candidate["stderr_bytes"]) is not int or
            candidate["stderr_bytes"] != 0 or
            type(candidate["remote"]) is not dict):
        return invalid, False, None
    remote = candidate["remote"]
    snapshot = candidate.get("_snapshot")
    try:
        if remote.get("status") == "HOLD_READONLY":
            if snapshot is not None:
                return invalid, False, None
            projected, _ = _checked_remote(phase, remote, template)
            if projected != remote:
                return invalid, False, None
            passed = False
        elif phase in ("preflight", "postflight"):
            if (not _snapshot_valid(snapshot, template, phase) or
                    remote.get("schema") !=
                    "trn21-r10-r5-write-hold-readonly-diagnostic-v1" or
                    remote.get("phase") != phase or
                    remote.get("status") != "PASS_READONLY" or
                    remote.get("stage") != "completed" or
                    remote.get("remote_write_attempted") is not False or
                    remote.get("source_stage_authorized") is not False or
                    remote.get("native_training_authorized") is not False or
                    not _finite_between(remote.get("elapsed_seconds"),
                                        MAX_REMOTE_SECONDS) or
                    remote != _inventory_projection(remote, snapshot)):
                return invalid, False, None
            passed = True
        else:
            if snapshot is not None:
                return invalid, False, None
            projected, _ = _checked_remote(phase, remote, template)
            if projected != remote or remote["status"] != "PASS_READONLY":
                return invalid, False, None
            passed = True
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return invalid, False, None
    public = {key: candidate[key] for key in common}
    return public, passed, snapshot


def _same_full_inventory_identity(preflight, postflight):
    if type(preflight) is not dict or type(postflight) is not dict:
        return False
    if preflight["process_identity"] == postflight["process_identity"]:
        return False
    return all(preflight[key] == postflight[key]
               for key in SNAPSHOT_IDENTITY_KEYS)


def run_once():
    """No dispatch in this package: three new authority files are absent."""
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python or inherited optimization is forbidden")
    # A checked-in HOLD request must never serve as implied permission.
    request, request_hash = _json(ENABLED_REQUEST)
    if request.get("dispatch_authorized") is not True:
        raise RuntimeError("read-only diagnostic is not enabled")
    reference, remote_source, bundle, diagnostic_source, template = _load_reference()
    remote_hash = _sha(diagnostic_source)
    local_hash = _sha(_regular(Path(__file__)))
    review, review_hash = _json(FINAL_REVIEW)
    assignment, _ = _json(ASSIGNMENT)
    _authority(request, request_hash, review, review_hash, assignment,
               remote_hash, local_hash)
    reference["_local_stop_and_breakers"]()
    raw_preflight = _phase(reference, "preflight", None, b"",
                           remote_source, diagnostic_source, template)
    preflight, preflight_pass, prior = _public_phase("preflight", raw_preflight,
                                                    template)
    phases = [preflight]
    preopen_pass = False
    if preflight_pass:
        raw_preopen = _phase(reference, "preopen", prior, bundle,
                             remote_source, diagnostic_source, template)
        preopen, preopen_pass, _ = _public_phase("preopen", raw_preopen,
                                               template)
        phases.append(preopen)
    # A separate full postflight is attempted even when preflight/preopen
    # HOLD; no write, cleanup, or retry follows from its result.
    raw_postflight = _phase(reference, "postflight", None, b"",
                            remote_source, diagnostic_source, template)
    postflight, postflight_pass, after = _public_phase("postflight",
                                                        raw_postflight, template)
    phases.append(postflight)
    identity = ("PASS" if preflight_pass and postflight_pass and
                _same_full_inventory_identity(prior, after) else "HOLD")
    distinct_preopen = (preflight_pass and preopen_pass and
                        preopen["remote"]["snapshot_sha256"] !=
                        preflight["remote"]["snapshot_sha256"])
    ok = (len(phases) == 3 and preflight_pass and preopen_pass and
          postflight_pass and identity == "PASS" and distinct_preopen)
    return {"schema": "trn21-r10-r5-write-hold-readonly-result-v1",
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "verdict": "READONLY_DIAGNOSTIC_PASS_NOT_STAGE_AUTHORITY" if ok else
                       "HOLD_READONLY_DIAGNOSTIC",
            "phases": phases, "full_snapshot_identity": identity,
            "distinct_preopen_snapshot": bool(distinct_preopen),
            "source_stage_authorized": False,
            "native_training_authorized": False,
            "original_consumed_marker_not_accessed": True}


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("HOLD: no ad-hoc diagnostic arguments")
    if not ENABLED_REQUEST.exists():
        raise SystemExit("HOLD: no enabled read-only request, independent review or assignment")
    try:
        print(json.dumps(run_once(), sort_keys=True, separators=(",", ":")))
    except Exception as error:
        raise SystemExit("HOLD: " + type(error).__name__ + ": " + str(error))
