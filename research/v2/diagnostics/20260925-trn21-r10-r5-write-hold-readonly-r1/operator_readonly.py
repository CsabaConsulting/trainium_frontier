"""Disabled, no-write trn21 diagnostic wrapper for the consumed r5 stage.

This file has no live authority: its enabled request, independent review and
Supervisor assignment are deliberately absent. It reuses the hash-pinned r5
Operator's read-only STOP, SSH identity and *separate remote GNU-timeout probe*
for any future independently enabled diagnostic phase. It never calls the r5
staging entrypoint, creates a local marker, or writes on the host.
"""

import hashlib
import json
import math
import os
from pathlib import Path
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
DIAGNOSTIC_LABEL = "fs-20260925-trn21-r10-r5-write-hold-readonly-d1"
SCOPE = "TRN21_R10_R5_CONSUMED_STAGE_READONLY_DIAGNOSIS_ONLY"
MAX_REMOTE_STDOUT = 128 * 1024
MAX_REMOTE_STDERR = 128 * 1024


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


def _load_reference():
    """Load exact r5 sources in memory without invoking the writer."""
    operator_source = _exact(R5_OPERATOR, R5_OPERATOR_SHA256)
    remote_source = _exact(R5_REMOTE, R5_REMOTE_SHA256)
    bundle = _exact(R5_BUNDLE, R5_BUNDLE_SHA256)
    if len(bundle) != R5_BUNDLE_BYTES:
        raise RuntimeError("pinned r5 bundle length changed")
    _exact(R5_JOURNAL, R5_JOURNAL_SHA256)
    _exact(R5_RESULT_REVIEW, R5_RESULT_REVIEW_SHA256)
    diagnostic_source = _regular(REMOTE_DIAGNOSTIC)
    # The known r5 operator module has no import-time host contact. In an
    # isolated namespace, its __main__ block is not entered. Its safe SSH
    # identity/probe helpers are reused byte-for-byte rather than retyped.
    namespace = {"__name__": "trn21_r5_readonly_reference",
                 "__file__": str(R5_OPERATOR)}
    exec(compile(operator_source, str(R5_OPERATOR), "exec"), namespace)
    return namespace, remote_source, bundle, diagnostic_source


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


def classify_transport(phase, returncode, stdout, stderr, elapsed_seconds):
    """Retain bounded cause *classes*, never raw stdout/stderr or exception text."""
    lengths = {"stdout_bytes": len(stdout), "stderr_bytes": len(stderr)}
    base = {"phase": phase, "returncode": returncode,
            "elapsed_seconds": round(elapsed_seconds, 6) if
            isinstance(elapsed_seconds, (int, float)) and
            math.isfinite(elapsed_seconds) and elapsed_seconds >= 0 else None,
            **lengths}
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        return {**base, "transport": "HOLD", "error_code": "SSH_RETURN_CODE_INVALID"}
    if returncode != 0:
        return {**base, "transport": "HOLD", "error_code": "SSH_NONZERO"}
    if stderr:
        return {**base, "transport": "HOLD", "error_code": "SSH_STDERR_NONEMPTY"}
    if len(stdout) > MAX_REMOTE_STDOUT or len(stderr) > MAX_REMOTE_STDERR:
        return {**base, "transport": "HOLD", "error_code": "SSH_OUTPUT_OVERSIZE"}
    try:
        value = json.loads(stdout)
    except (ValueError, UnicodeDecodeError):
        return {**base, "transport": "HOLD", "error_code": "REMOTE_JSON_INVALID"}
    if (not isinstance(value, dict) or value.get("schema") !=
            "trn21-r10-r5-write-hold-readonly-diagnostic-v1" or
            value.get("phase") != phase or
            value.get("status") not in ("PASS_READONLY", "HOLD_READONLY") or
            value.get("remote_write_attempted") is not False or
            value.get("source_stage_authorized") is not False or
            value.get("native_training_authorized") is not False):
        return {**base, "transport": "HOLD", "error_code": "REMOTE_RECEIPT_INVALID"}
    return {**base, "transport": "PASS", "remote": value}


def _phase(reference, phase, prior, data, remote_source, diagnostic_source):
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
                                  result.stderr, time.monotonic() - began)
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


def run_once():
    """No dispatch in this package: three new authority files are absent."""
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python or inherited optimization is forbidden")
    # A checked-in HOLD request must never serve as implied permission.
    request, request_hash = _json(ENABLED_REQUEST)
    if request.get("dispatch_authorized") is not True:
        raise RuntimeError("read-only diagnostic is not enabled")
    reference, remote_source, bundle, diagnostic_source = _load_reference()
    remote_hash = _sha(diagnostic_source)
    local_hash = _sha(_regular(Path(__file__)))
    review, review_hash = _json(FINAL_REVIEW)
    assignment, _ = _json(ASSIGNMENT)
    _authority(request, request_hash, review, review_hash, assignment,
               remote_hash, local_hash)
    reference["_local_stop_and_breakers"]()
    preflight = _phase(reference, "preflight", None, b"",
                       remote_source, diagnostic_source)
    phases = [preflight]
    if (preflight.get("transport") == "PASS" and
            preflight.get("remote", {}).get("status") == "PASS_READONLY"):
        prior = preflight["remote"].get("snapshot")
        if isinstance(prior, dict):
            phases.append(_phase(reference, "preopen", prior, bundle,
                                 remote_source, diagnostic_source))
    # A separate full postflight is attempted even when preflight/preopen
    # HOLD; no write, cleanup, or retry follows from its result.
    phases.append(_phase(reference, "postflight", None, b"",
                         remote_source, diagnostic_source))
    ok = (len(phases) == 3 and all(
        phase.get("transport") == "PASS" and
        phase.get("remote", {}).get("status") == "PASS_READONLY"
        for phase in phases))
    return {"schema": "trn21-r10-r5-write-hold-readonly-result-v1",
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "verdict": "READONLY_DIAGNOSTIC_PASS_NOT_STAGE_AUTHORITY" if ok else
                       "HOLD_READONLY_DIAGNOSTIC",
            "phases": phases, "source_stage_authorized": False,
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
