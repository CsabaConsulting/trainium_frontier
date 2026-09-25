"""Collect one read-only trn22 r4 feasibility snapshot.

This evidence helper invokes only existing read-only remote probe sources.
It never imports or calls a transfer wrapper's dispatch entry point, and it
never sends capsule bytes to the host. Its output is local planning evidence,
never a transfer preflight.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "trn22-c5c-a0-r4-transfer-only"
CANONICAL = Path("/home/poojith-devan/Desktop/Trainium")
ARTIFACTS = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts")
FINAL_R2_REQUEST = (
    CANONICAL
    / "research/v2/authorizations/trn22-c5c-a0-r4-transfer-only"
    / "TRANSFER_REQUEST.CONTINUITY_FINAL_R2.json"
)
FINAL_R2_REVIEW = FINAL_R2_REQUEST.with_name("INDEPENDENT_CONTINUITY_FINAL_R2_REVIEW.json")
FINAL_R2_ASSIGNMENT = FINAL_R2_REQUEST.with_name(
    "SUPERVISOR_CONTINUITY_TRANSFER_R2_ASSIGNMENT.json"
)
MARKER = ARTIFACTS / "trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker"
JOURNAL = ARTIFACTS / "trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl"
EXPECTED_TRANSPORT_SHA256 = "ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0"

sys.path.insert(0, str(PACKAGE))
transport = importlib.import_module("stage_r4_transport")
core = importlib.import_module("operator_transfer_once")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, check=True, timeout=10, text=True,
    )
    return result.stdout.strip()


def path_metadata(path: Path) -> dict:
    try:
        named = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    return {
        "exists": True,
        "type": (
            "regular" if stat.S_ISREG(named.st_mode) else
            "directory" if stat.S_ISDIR(named.st_mode) else
            "symlink" if stat.S_ISLNK(named.st_mode) else "other"
        ),
        "device": named.st_dev,
        "inode": named.st_ino,
        "mode": oct(stat.S_IMODE(named.st_mode)),
        "bytes": named.st_size,
    }


def local_state() -> dict:
    try:
        breaker = {"status": "PASS", **core._check_controller_breakers()}
    except Exception as error:
        breaker = {"status": "HOLD", "error": type(error).__name__ + ": " + str(error)}
    request = path_metadata(FINAL_R2_REQUEST)
    if request.get("type") == "regular":
        request["sha256"] = sha(FINAL_R2_REQUEST.read_bytes())
        request["dispatch_authorized_field"] = json.loads(
            FINAL_R2_REQUEST.read_text(encoding="utf-8")
        ).get("dispatch_authorized")
    return {
        "utc": utc(),
        "canonical_head": git_head(CANONICAL),
        "evidence_branch_head": git_head(HERE.parents[3]),
        "controller_breakers_and_stops": breaker,
        "final_r2_request": request,
        "final_r2_independent_review": path_metadata(FINAL_R2_REVIEW),
        "final_r2_supervisor_assignment": path_metadata(FINAL_R2_ASSIGNMENT),
        "one_shot_marker": path_metadata(MARKER),
        "transfer_journal": path_metadata(JOURNAL),
    }


def redact(raw: bytes) -> dict:
    text = raw.decode("utf-8", "replace")
    text = text.replace(transport.IDENTITY_FILE, "<private-key-path-redacted>")
    # The reviewed probe emits curated metadata, yet fail closed if unexpected
    # credential-looking text appears in transport diagnostics or path names.
    marker = re.search(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
        r"(?i:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|PRIVATE_KEY\s*[:=])",
        text,
    )
    return {
        "bytes": len(raw),
        "sha256": sha(raw),
        "redacted_text": "<suppressed: credential-like marker>" if marker else text,
        "suppressed_for_credential_marker": bool(marker),
    }


def readonly_argv(*, probe_file: Path | None) -> tuple[list[str], bytes | None, dict]:
    if sha((PACKAGE / "stage_r4_transport.py").read_bytes()) != EXPECTED_TRANSPORT_SHA256:
        raise RuntimeError("reviewed transport SHA-256 drift")
    argv = transport.build_ssh_argv("probe")
    command_words = shlex.split(argv[-1])
    if command_words != ["exec", "python3", "-c", transport.reviewed_remote_source(), "probe"]:
        raise RuntimeError("read-only probe command differs from reviewed transport")
    stdin_data = None
    if probe_file is not None:
        stdin_data = probe_file.read_bytes()
        command_words = ["exec", "python3", "-B", "-"]
    else:
        # Disable bytecode writes on the remote host while preserving the
        # exact reviewed source bytes and the probe-only operation.
        command_words.insert(2, "-B")
    argv[-1] = " ".join(shlex.quote(word) for word in command_words)
    if shlex.split(argv[-1]) != command_words:
        raise RuntimeError("read-only remote quoting drift")
    display = argv.copy()
    display[display.index("-i") + 1] = "<private-key-path-redacted>"
    executed_source_sha256 = (
        sha(stdin_data) if probe_file is not None else sha(command_words[4].encode("utf-8"))
    )
    display[-1] = (
        "exec python3 -B - <'read-only diagnostic source SHA256="
        + executed_source_sha256 + ">"
        if probe_file is not None else
        "exec python3 -B -c '<reviewed stage_r4_remote.py SHA256="
        + executed_source_sha256 + ">' probe"
    )
    return argv, stdin_data, {
        "argv_redacted": display,
        "argv_sha256": transport.argv_sha256(argv),
        "remote_command_sha256": sha(argv[-1].encode("utf-8")),
        "reviewed_remote_source_sha256": transport.REMOTE_SHA256,
        "executed_remote_source_sha256": executed_source_sha256,
        "executed_source_unmodified": True,
        "remote_operation": probe_file.name if probe_file is not None else "probe",
        "remote_python_no_bytecode": True,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main() -> None:
    if sys.argv[1:] not in ([], ["--legacy-probe"], ["--supplemental-probe"]):
        raise SystemExit("only unmodified read-only probe sources are accepted")
    mode = sys.argv[1:] or []
    probe_file = (
        HERE / "remote_readonly_probe.py" if mode == ["--legacy-probe"] else
        HERE / "supplemental_readonly_probe.py" if mode == ["--supplemental-probe"] else
        None
    )
    started = utc()
    source = transport.checked_local_source()
    ssh_identity = transport.checked_ssh_identity()
    before = local_state()
    argv, stdin_data, argv_record = readonly_argv(probe_file=probe_file)
    monotonic_start = time.monotonic()
    try:
        completed = subprocess.run(
            argv, input=stdin_data, capture_output=True, check=False, timeout=180
        )
        rc, stdout, stderr, timed_out = (
            completed.returncode, completed.stdout, completed.stderr, False
        )
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr, timed_out = (
            None, error.stdout or b"", error.stderr or b"", True
        )
    ended = utc()
    after = local_state()
    out, err = redact(stdout), redact(stderr)
    remote = None
    validation = {"status": "HOLD", "reason": "SSH failed, timed out, or stdout was redacted"}
    if rc == 0 and not timed_out and not out["suppressed_for_credential_marker"]:
        try:
            remote = json.loads(stdout)
            if mode == ["--legacy-probe"]:
                if remote.get("schema") != "trn22-r4-source-stage-readonly-feasibility-v1":
                    raise RuntimeError("wrong unmodified legacy probe schema")
                validation = {"status": "LIMITED_READ_ONLY_OBSERVATION_NOT_FULL_R4_PREFLIGHT"}
            elif mode == ["--supplemental-probe"]:
                if remote.get("schema") != "trn22-r4-readonly-alias-oldpath-diagnostic-v1":
                    raise RuntimeError("wrong supplemental probe schema")
                validation = {"status": "LIMITED_READ_ONLY_OBSERVATION_NOT_FULL_R4_PREFLIGHT"}
            else:
                transport.validate_probe(remote, before_write=True)
                validation = {"status": "READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME_ONLY"}
        except Exception as error:
            validation = {
                "status": "HOLD",
                "reason": type(error).__name__ + ": " + str(error),
            }
    record = {
        "schema": "trn22-r4-operator-readonly-refresh-v1",
        "scope": "planning_snapshot_only_no_transfer_authority",
        "started_utc": started,
        "ended_utc": ended,
        "duration_seconds": round(time.monotonic() - monotonic_start, 3),
        "local_before": before,
        "local_after": after,
        "local_source_capsule": source,
        "pinned_ssh_identity": ssh_identity,
        "transport": {
            **argv_record,
            "exit_code": rc,
            "timed_out": timed_out,
            "stdout": out,
            "stderr": err,
        },
        "remote_receipt": remote,
        "reviewed_probe_validation": validation,
    }
    timestamp = dt.datetime.fromisoformat(started).strftime("%Y%m%dT%H%M%SZ")
    kind = (
        "LEGACY_" if mode == ["--legacy-probe"] else
        "SUPPLEMENTAL_" if mode == ["--supplemental-probe"] else ""
    )
    destination = HERE / ("OPERATOR_RAW_READONLY_" + kind + timestamp + ".json")
    write_exclusive(destination, record)
    print(json.dumps({
        "evidence_path": str(destination),
        "status": validation["status"],
        "ssh_rc": rc,
        "ssh_stderr_bytes": len(stderr),
        "host_utc": remote.get("utc") if isinstance(remote, dict) else None,
        "root_free_bytes": (
            remote.get("identity_after", {}).get("root_free_bytes")
            if isinstance(remote, dict) else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
