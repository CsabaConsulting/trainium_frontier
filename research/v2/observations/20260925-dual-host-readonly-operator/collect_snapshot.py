"""Run one pinned SSH read-only snapshot and archive a redacted local receipt.

This program does not import a transfer operator or invoke a writer.  Its only
remote command is ``exec python3 -B - <host>`` with remote_snapshot.py on stdin.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SOURCE_REPO = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")
CANONICAL = Path("/home/poojith-devan/Desktop/Trainium")
TRANSPORT = {
    "trn21": (
        SOURCE_REPO / "research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/stage_r4_transport.py",
        "fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647",
        "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc",
    ),
    "trn22": (
        SOURCE_REPO / "research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/stage_r4_transport.py",
        "1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51",
        "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U",
    ),
}
SENSITIVE = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    rb"(?i:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|OPENAI_API_KEY|PRIVATE_KEY|"
    rb"BEARER_TOKEN|trn2\.pem|\.env(?:\b|/))"
)


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_state() -> dict:
    state_path = CANONICAL / "research/v2/state.json"
    raw = state_path.read_bytes()
    state = json.loads(raw)
    stop_paths = (
        CANONICAL / "research/agentic/STOP",
        CANONICAL / "research/v2/STOP",
        SOURCE_REPO / "research/agentic/STOP",
        SOURCE_REPO / "research/v2/STOP",
    )
    present_stops = []
    for path in stop_paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        present_stops.append(str(path))
    return {
        "utc": utc(), "controller_state_sha256": sha(raw),
        "launch_authorized": state.get("launch_authorized"),
        "host_launch_allowed": {
            host: state.get("hosts", {}).get(host, {}).get("launch_allowed")
            for host in TRANSPORT
        },
        "breaker_tripped": {
            name: state.get("breakers", {}).get(name, {}).get("tripped")
            for name in ("global_correctness", "trn21", "trn22")
        },
        "local_stops_present": present_stops,
    }


def load_transport(host: str):
    path, expected_sha, expected_fingerprint = TRANSPORT[host]
    actual_sha = sha(path.read_bytes())
    if actual_sha != expected_sha:
        raise RuntimeError("pinned transport source SHA-256 drift")
    spec = importlib.util.spec_from_file_location("readonly_pinned_transport_" + host, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("transport loader missing")
    spec.loader.exec_module(module)
    if module.HOST_FINGERPRINT != expected_fingerprint:
        raise RuntimeError("pinned host fingerprint drift")
    identity = module.checked_ssh_identity()
    if identity.get("host_fingerprint") != expected_fingerprint:
        raise RuntimeError("strict host-key identity check failed")
    # Probe argv is built by the already-pinned transport, then constrained
    # further to our own stdin-only read-only source and bytecode-disabled mode.
    argv = module.build_ssh_argv("probe")
    if argv[0] != "ssh" or argv[-2] != module.ENDPOINT:
        raise RuntimeError("unexpected SSH argv shape")
    argv[-1] = "exec python3 -B - " + host
    if not all(option in argv for option in (
        "BatchMode=yes", "IdentitiesOnly=yes", "StrictHostKeyChecking=yes",
        "HostKeyAlgorithms=ssh-ed25519", "UpdateHostKeys=no",
    )):
        raise RuntimeError("pinned SSH options incomplete")
    display = argv.copy()
    display[display.index("-i") + 1] = "<private-key-path-redacted>"
    return argv, display, identity, actual_sha


def archive_stream(raw: bytes) -> dict:
    blocked = bool(SENSITIVE.search(raw))
    return {
        "bytes": len(raw), "sha256": sha(raw),
        "base64": None if blocked else base64.b64encode(raw).decode("ascii"),
        "suppressed_for_sensitive_marker": blocked,
    }


def main() -> None:
    diagnostic = sys.argv[1:] == ["trn22", "--gitdir-acl"]
    if not diagnostic and (len(sys.argv) != 2 or sys.argv[1] not in TRANSPORT):
        raise SystemExit("one host id: trn21 or trn22; optional trn22 --gitdir-acl")
    host = sys.argv[1]
    source_name = "remote_gitdir_acl.py" if diagnostic else "remote_snapshot.py"
    expected_schema = (
        "trn22-e65-r2-gitdir-acl-readonly-v1" if diagnostic else "dual-host-readonly-v1"
    )
    source = (HERE / source_name).read_bytes()
    if SENSITIVE.search(source):
        raise RuntimeError("remote source unexpectedly contains a sensitive marker")
    before = local_state()
    if before["local_stops_present"]:
        raise RuntimeError("local STOP present; no SSH contact")
    argv, display, identity, transport_sha = load_transport(host)
    started = utc()
    try:
        done = subprocess.run(argv, input=source, capture_output=True,
                              timeout=120, check=False)
        rc, stdout, stderr, timed_out = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr, timed_out = (
            None, error.stdout or b"", error.stderr or b"", True
        )
    ended = utc()
    after = local_state()
    out, err = archive_stream(stdout), archive_stream(stderr)
    parsed = None
    parse_error = None
    if rc == 0 and not timed_out and not out["suppressed_for_sensitive_marker"]:
        try:
            parsed = json.loads(stdout)
            if parsed.get("schema") != expected_schema or (
                not diagnostic and parsed.get("host") != host
            ):
                raise ValueError("wrong read-only receipt schema or host")
        except (TypeError, ValueError) as error:
            parsed = None
            parse_error = type(error).__name__ + ": " + str(error)
    receipt = {
        "schema": "dual-host-operator-readonly-raw-v1",
        "scope": "point_in_time_snapshot_only_no_preflight_or_launch_authority",
        "host": host, "started_utc": started, "ended_utc": ended,
        "local_before": before, "local_after": after,
        "ssh": {
            "argv_redacted": display,
            "argv_sha256": sha(json.dumps(argv, separators=(",", ":")).encode()),
            "remote_command_sha256": sha(argv[-1].encode()),
            "transport_source_sha256": transport_sha,
            "remote_source_sha256": sha(source),
            "pinned_host_key": identity,
            "exit_code": rc, "timed_out": timed_out,
            "stdout": out, "stderr": err,
        },
        "remote": parsed, "parse_error": parse_error,
    }
    label = dt.datetime.fromisoformat(started).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_GITDIR_ACL" if diagnostic else ""
    destination = HERE / f"OPERATOR_RAW_{host.upper()}_READONLY{suffix}_{label}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary = {
        "host": host, "receipt": destination.name, "ssh_rc": rc,
        "stderr_bytes": len(stderr), "parsed": parsed is not None,
    }
    if parsed is not None and diagnostic:
        summary["observed_utc"] = parsed.get("ended_utc")
        summary["gitdir"] = parsed.get("pointer", {}).get("resolved_gitdir")
        summary["pointer_error"] = parsed.get("pointer", {}).get("error")
        summary["other_local_gid_1000_account"] = parsed.get(
            "gid_1000_local_facts", {}).get("other_local_account_shares_gid_1000")
    elif parsed is not None:
        summary["observed_utc"] = parsed.get("ended_utc")
        summary["root_free_bytes"] = parsed.get("root_space", {}).get("free_bytes")
        summary["root_free_inodes"] = parsed.get("root_space", {}).get("free_inodes")
        summary["checkouts"] = {
            name: {
                "exists": value.get("entry", {}).get("exists"),
                "commit": value.get("git", {}).get("commit"),
                "launcher_sha256": value.get("source_hashes", {}).get(
                    "research/fresh_start_launch.py", {}).get("sha256"),
            }
            for name, value in parsed.get("checkouts", {}).get("entries", {}).items()
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
