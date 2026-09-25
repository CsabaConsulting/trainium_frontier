"""Disabled-by-default, pinned trn22 A2 tool-path feasibility collector.

The current HOLD request cannot contact a host. A later independent review
and Supervisor read-only assignment must provide the fixed enablement files
before ``--capture`` can make three bounded, strictly host-key-checked,
read-only SSH observations. It never imports a transfer/CPU-run entry point.
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
REPO = HERE.parents[3]
ACTIVE_CONTROL = Path("/home/poojith-devan/Desktop/Trainium")
LEGACY_CONTROL = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")
REQUEST = HERE / "REQUEST.HOLD.json"
ENABLED = HERE / "READONLY_ENABLEMENT.FINAL.json"
REVIEW = HERE / "INDEPENDENT_READONLY_ENABLEMENT.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_READONLY_ASSIGNMENT.json"
TRANSPORT = REPO / "research/v2/authorizations/trn22-c5c-a0-r4-transfer-only/stage_r4_transport.py"
SNAPSHOT = REPO / "research/v2/observations/20260925-dual-host-readonly-operator/remote_snapshot.py"
GITDIR = REPO / "research/v2/observations/20260925-dual-host-readonly-operator/remote_gitdir_acl.py"
TOOLS = HERE / "remote_tool_paths.py"
PIN = {
    "request": (REQUEST, "65e63ffa09213ea5831f5a2797378bc67a7b93acd215d7a519a9c976ec446dc1"),
    "transport": (TRANSPORT, "1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51"),
    "snapshot": (SNAPSHOT, "4a8b8ed413240c229ce943dd9bd8d6781bbddf7c348aa76b5b496f5c5d14f70a"),
    "gitdir": (GITDIR, "676c1ed5c80b37a56df7f90a80150fe29ebc0d7c6e65813513af8b7f4f4b8499"),
    "tools": (TOOLS, "51d5f6be8b313a51efc56609dd391610ee1b6528e9f118f95176679484ef83b7"),
}
SENSITIVE = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    rb"(?i:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|OPENAI_API_KEY|PRIVATE_KEY|"
    rb"BEARER_TOKEN|trn2\.pem|\.env(?:\b|/))"
)
MAX_OUTPUT_BYTES = 2_000_000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def unique_json(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def pinned_sources() -> tuple[dict, dict[str, bytes]]:
    raw = {}
    for name, (path, expected) in PIN.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("pinned read-only input absent or symlinked: " + name)
        content = path.read_bytes()
        if sha(content) != expected:
            raise RuntimeError("pinned read-only input hash drift: " + name)
        raw[name] = content
    request = json.loads(raw["request"], object_pairs_hook=unique_json)
    if (request.get("schema") != "trn22-a2-toolpath-readonly-feasibility-v1"
            or request.get("host") != "trn22"
            or request.get("status") != "IMPLEMENTER_DISABLED_PENDING_INDEPENDENT_STATIC_REVIEW"
            or any(value is not False for value in request.get("authority", {}).values())
            or request.get("fresh_host_values") is not None):
        raise RuntimeError("disabled request authority or schema changed")
    return request, raw


def require_future_authority(raw: dict[str, bytes]) -> dict:
    """No enablement files exist in the Implementer package; fail before SSH."""
    for path in (ENABLED, REVIEW, ASSIGNMENT):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("read-only capture HOLD: independent enablement/assignment absent")
    enabled_raw, review_raw, assignment_raw = (
        path.read_bytes() for path in (ENABLED, REVIEW, ASSIGNMENT)
    )
    enabled, review, assignment = (
        json.loads(value, object_pairs_hook=unique_json)
        for value in (enabled_raw, review_raw, assignment_raw)
    )
    expected = {name: sha(value) for name, value in raw.items()}
    if (enabled.get("schema") != "trn22-a2-toolpath-readonly-enablement-v1"
            or enabled.get("host") != "trn22"
            or enabled.get("scope") != "read_only_tool_feasibility_only"
            or enabled.get("host_contact_authorized") is not True
            or enabled.get("no_marker_or_host_mutation") is not True
            or enabled.get("pinned_sha256") != expected
            or review.get("schema") != "trn22-a2-toolpath-independent-review-v1"
            or review.get("verdict") != "PASS_READONLY_CAPTURE_ONLY"
            or review.get("enabled_request_sha256") != sha(enabled_raw)
            or review.get("collector_sha256") != sha(Path(__file__).read_bytes())
            or assignment.get("schema") != "trn22-a2-toolpath-supervisor-assignment-v1"
            or assignment.get("scope") != "one_readonly_capture_only"
            or assignment.get("enabled_request_sha256") != sha(enabled_raw)
            or assignment.get("independent_review_sha256") != sha(review_raw)
            or assignment.get("host") != "trn22"):
        raise RuntimeError("read-only capture HOLD: enablement byte chain invalid")
    return {"enabled_sha256": sha(enabled_raw), "review_sha256": sha(review_raw),
            "assignment_sha256": sha(assignment_raw)}


def local_stops() -> list[str]:
    roots = (ACTIVE_CONTROL, LEGACY_CONTROL, REPO)
    paths = [root / name for root in roots
             for name in ("research/agentic/STOP", "research/v2/STOP")]
    return [str(path) for path in paths if os.path.lexists(path)]


def active_controller_gate() -> dict:
    """Desktop state alone governs read-only contact; legacy state never enables it."""
    if local_stops():
        raise RuntimeError("local STOP present before read-only SSH")
    path = ACTIVE_CONTROL / "research/v2/state.json"
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or named.st_size > 1024 * 1024):
        raise RuntimeError("active controller state metadata unsafe")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns):
            raise RuntimeError("active controller state changed during open")
        raw = os.read(fd, 1024 * 1024 + 1)
        after, named_after = os.fstat(fd), os.lstat(path)
        if (len(raw) != opened.st_size or
                any((item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) !=
                    (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                    for item in (after, named_after))):
            raise RuntimeError("active controller state changed during read")
    finally:
        os.close(fd)
    state = json.loads(raw, object_pairs_hook=unique_json)
    breakers = state.get("breakers") if isinstance(state, dict) else None
    host = state.get("hosts") if isinstance(state, dict) else None
    if (not isinstance(breakers, dict) or
            any(not isinstance(breakers.get(name), dict) or
                breakers[name].get("tripped") is not False
                for name in ("global_correctness", "trn22")) or
            not isinstance(host, dict) or
            not isinstance(host.get("trn22"), dict) or
            state.get("launch_authorized") is not False or
            host["trn22"].get("launch_allowed") is not False):
        raise RuntimeError("active Desktop controller breaker/launch gate is unsafe")
    if local_stops():
        raise RuntimeError("local STOP appeared during controller check")
    return {"active_state_sha256": sha(raw), "breakers_clear": True,
            "trn22_launch_allowed": False, "local_stops_absent": True}


def redacted_stream(raw: bytes) -> dict:
    suppressed = len(raw) > MAX_OUTPUT_BYTES or bool(SENSITIVE.search(raw))
    return {"bytes": len(raw), "sha256": sha(raw),
            "base64": None if suppressed else base64.b64encode(raw).decode("ascii"),
            "suppressed": suppressed}


def load_transport():
    spec = importlib.util.spec_from_file_location("a2_readonly_pinned_transport", TRANSPORT)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned transport cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
    if module.HOST_FINGERPRINT != expected:
        raise RuntimeError("pinned host fingerprint changed")
    identity = module.checked_ssh_identity()
    if identity.get("host_fingerprint") != expected:
        raise RuntimeError("strict trn22 SSH identity check failed")
    return module, identity


def one_probe(transport, name: str, source: bytes, schema: str) -> tuple[dict, dict | None]:
    argv = transport.build_ssh_argv("probe")
    if (argv[0] != "ssh" or argv[-2] != transport.ENDPOINT
            or "StrictHostKeyChecking=yes" not in argv
            or "BatchMode=yes" not in argv or "IdentitiesOnly=yes" not in argv):
        raise RuntimeError("pinned SSH argv drift")
    argv[-1] = "exec python3 -B - trn22"
    redacted_argv = argv.copy()
    redacted_argv[redacted_argv.index("-i") + 1] = "<private-key-path-redacted>"
    active_controller_gate()
    started = utc()
    try:
        done = subprocess.run(argv, input=source, capture_output=True, timeout=120, check=False)
        rc, stdout, stderr, timed_out = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
    output, errors = redacted_stream(stdout), redacted_stream(stderr)
    parsed = None
    parse_error = None
    if rc == 0 and not timed_out and not output["suppressed"]:
        try:
            parsed = json.loads(stdout, object_pairs_hook=unique_json)
            if parsed.get("schema") != schema:
                raise ValueError("wrong remote schema")
        except (TypeError, ValueError) as error:
            parsed = None
            parse_error = type(error).__name__ + ": " + str(error)
    record = {"name": name, "started_utc": started, "ended_utc": utc(),
              "ssh_argv_redacted": redacted_argv,
              "ssh_argv_sha256": sha(json.dumps(argv, separators=(",", ":")).encode()),
              "remote_source_sha256": sha(source), "remote_command_sha256": sha(argv[-1].encode()),
              "exit_code": rc, "timed_out": timed_out,
              "stdout": output, "stderr": errors, "parse_error": parse_error}
    return record, parsed


def planning_decision(request: dict, snapshot: dict | None,
                      gitdir: dict | None, tools: dict | None) -> dict:
    """Fail closed; even a PASS is only a dated planning observation."""
    reasons = []
    if any(item is None for item in (snapshot, gitdir, tools)):
        return {"status": "HOLD", "reasons": ["missing_or_redacted_remote_receipt"]}
    base = snapshot
    identity = base.get("identity", {})
    source = base.get("checkouts", {}).get("entries", {}).get("e65_exact", {})
    entry = source.get("entry", {})
    git = source.get("git", {})
    status = source.get("git_status", {})
    hashes = source.get("source_hashes", {})
    expected = request["source"]
    if (base.get("host") != "trn22" or tools.get("host") != "trn22"
            or identity.get("hostname") != "ip-172-31-32-253"
            or gitdir.get("hostname") != identity.get("hostname")
            or tools.get("hostname") != identity.get("hostname")
            or identity.get("uid") != 1000 or gitdir.get("uid") != 1000
            or tools.get("uid") != 1000 or not identity.get("boot_id")
            or identity.get("boot_id") != gitdir.get("boot_id")
            or identity.get("boot_id") != tools.get("boot_id")
            or identity.get("mount_namespace") != gitdir.get("mount_namespace")
            or identity.get("mount_namespace") != tools.get("mount_namespace")
            or identity.get("root_device", {}).get("device") != tools.get("root_device")):
        reasons.append("host_boot_mount_or_uid_mismatch")
    if (source.get("path") != expected["e65_checkout"]
            or entry.get("type") != "directory"
            or (entry.get("device"), entry.get("inode"), entry.get("uid")) != (
                expected["expected_checkout_device"], expected["expected_checkout_inode"],
                expected["expected_checkout_uid"])
            or git.get("commit") != expected["e65_commit"]
            or git.get("rc") != 0 or status.get("clean") is not True
            or hashes.get("train.py", {}).get("sha256") != expected["expected_train_sha256"]
            or hashes.get("prepare.py", {}).get("sha256") != expected["expected_prepare_sha256"]):
        reasons.append("e65_checkout_commit_cleanliness_or_source_hash_mismatch")
    pointer = gitdir.get("pointer", {})
    pointer_meta = pointer.get("metadata", {})
    linked = pointer.get("resolved_gitdir_metadata", {})
    if (pointer_meta.get("type") != "regular"
            or pointer_meta.get("device") != expected["expected_checkout_device"]
            or pointer_meta.get("inode") != expected["expected_git_pointer_inode"]
            or pointer_meta.get("uid") != 1000
            or pointer_meta.get("mode") != "0664"
            or pointer.get("sha256") != expected["expected_git_pointer_sha256"]
            or linked.get("type") != "directory"
            or linked.get("device") != expected["expected_checkout_device"]
            or linked.get("inode") != expected["expected_linked_gitdir_inode"]
            or linked.get("uid") != 1000
            or linked.get("mode") != "0775"
            or pointer.get("resolved_gitdir") is None):
        reasons.append("linked_git_pointer_or_gitdir_mismatch")
    activity = base.get("activity", {})
    if (base.get("stops", {}).get("present") != {}
            or base.get("checkouts", {}).get("discovery_truncated") is not False
            or activity.get("processes", {}).get("matched") != []
            or activity.get("processes", {}).get("unreadable_cmdlines") != 0
            or activity.get("neuron", {}).get("rc") != 0
            or activity.get("neuron", {}).get("json_valid") is not True
            or activity.get("neuron", {}).get("non_diagnostic_pids") != []
            or activity.get("host_lock_fuser", {}).get("rc") != 1
            or activity.get("host_lock_fuser", {}).get("pids") != []
            or activity.get("neuron0_fuser", {}).get("rc") != 1
            or activity.get("neuron0_fuser", {}).get("pids") != []):
        reasons.append("stop_process_neuron_or_lock_occupancy")
    floors = request["planning_floor"]
    space = base.get("root_space", {})
    if (not isinstance(space.get("free_bytes"), int)
            or space["free_bytes"] < floors["root_free_bytes"]
            or not isinstance(space.get("free_inodes"), int)
            or space["free_inodes"] < floors["root_free_inodes"]
            or not isinstance(tools.get("memory_available_bytes"), int)
            or tools["memory_available_bytes"] < floors["memory_available_bytes"]):
        reasons.append("root_disk_inode_or_memory_below_planning_floor")
    exact = request["exact_tools"]
    found = tools.get("tools", {})
    for name in ("g2", "contained"):
        target = found.get(name, {})
        if (target.get("path") != exact[name]["path"]
                or target.get("exists") is not True
                or target.get("type") != "regular"
                or target.get("uid") != 1000
                or target.get("device") != expected["expected_checkout_device"]
                or target.get("nlink") != 1
                or target.get("sha256") != exact[name]["expected_sha256"]
                or target.get("read_bytes") != target.get("bytes")
                or target.get("hash_error") is not None):
            reasons.append(name + "_exact_nofollow_tool_path_absent_or_mismatched")
    tool_checkout = tools.get("checkout", {})
    if (tools.get("checkout_path") != expected["e65_checkout"]
            or tool_checkout.get("type") != "directory"
            or tool_checkout.get("device") != expected["expected_checkout_device"]
            or tool_checkout.get("inode") != expected["expected_checkout_inode"]
            or tool_checkout.get("opened_same_inode") is not True):
        reasons.append("tool_probe_checkout_identity_mismatch")
    return {"status": "READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME_ONLY" if not reasons else "HOLD",
            "reasons": reasons}


def write_receipt(record: dict, started: str) -> Path:
    filename = "OPERATOR_RAW_TRN22_A2_TOOLPATH_READONLY_" + (
        dt.datetime.fromisoformat(started).strftime("%Y%m%dT%H%M%SZ")) + ".json"
    destination = HERE / filename
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination


def capture() -> Path:
    request, raw = pinned_sources()
    authority = require_future_authority(raw)
    controller_before = active_controller_gate()
    transport, identity = load_transport()
    started = utc()
    probes = []
    parsed = {}
    for name, schema in (("snapshot", "dual-host-readonly-v1"),
                         ("gitdir", "trn22-e65-r2-gitdir-acl-readonly-v1"),
                         ("tools", "trn22-a2-toolpath-readonly-v1")):
        try:
            entry, value = one_probe(transport, name, raw[name], schema)
        except (OSError, RuntimeError, ValueError) as error:
            probes.append({"name": name, "collection_error": type(error).__name__ +
                           ": " + str(error)[:200], "remote_source_sha256": sha(raw[name])})
            parsed[name] = None
            break
        probes.append(entry)
        parsed[name] = value
        if value is None:
            break
    decision = planning_decision(request, parsed.get("snapshot"),
                                 parsed.get("gitdir"), parsed.get("tools"))
    stop_after = local_stops()
    if stop_after:
        decision = {"status": "HOLD", "reasons": decision["reasons"] +
                    ["local_STOP_present_after_readonly_capture"]}
    try:
        controller_after = active_controller_gate()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        controller_after = {"status": "HOLD", "error": type(error).__name__}
        decision = {"status": "HOLD", "reasons": decision["reasons"] +
                    ["active_controller_gate_failed_after_readonly_capture"]}
    else:
        if controller_after["active_state_sha256"] != controller_before["active_state_sha256"]:
            decision = {"status": "HOLD", "reasons": decision["reasons"] +
                        ["active_controller_state_changed_after_readonly_capture"]}
    record = {
        "schema": "trn22-a2-toolpath-operator-readonly-raw-v1",
        "scope": "point_in_time_planning_only_no_preflight_or_cpu_or_native_authority",
        "started_utc": started, "ended_utc": utc(), "host": "trn22",
        "authority_byte_chain": authority,
        "pinned_source_sha256": {name: sha(value) for name, value in raw.items()},
        "host_key": identity, "probes": probes, "remote": parsed,
        "planning_decision": decision,
        "active_controller_before": controller_before,
        "active_controller_after": controller_after,
        "local_stops_after": stop_after,
    }
    return write_receipt(record, started)


def main() -> int:
    if sys.argv[1:] == ["--check"]:
        _, raw = pinned_sources()
        print(json.dumps({"status": "DISABLED_NO_HOST_CONTACT",
                          "pinned_sha256": {name: sha(value) for name, value in raw.items()}},
                         sort_keys=True))
        return 0
    if sys.argv[1:] == ["--capture"]:
        try:
            path = capture()
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"status": "HOLD_NO_SUCCESS_CLAIM",
                              "error": type(error).__name__ + ": " + str(error)[:200]},
                             sort_keys=True))
            return 3
        print(json.dumps({"status": "RECEIPT_CREATED_PLANNING_ONLY", "path": str(path)},
                         sort_keys=True))
        return 0
    raise SystemExit("--check (offline) or --capture (requires future reviewed enablement)")


if __name__ == "__main__":
    raise SystemExit(main())
