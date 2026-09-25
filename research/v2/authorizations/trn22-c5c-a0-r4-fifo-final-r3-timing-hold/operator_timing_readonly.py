"""Disabled one-attempt collector for the exact trn22 final-r3 timing study.

Only a separate enabled request, independent review, and Supervisor assignment
can reach SSH.  The remote command has probe/timing operations only, uses
python -B, and passes /dev/null as stdin.  Its local exclusive output is the
raw evidence artifact, never a host marker or journal.
"""

import base64
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FROZEN = HERE.parent / "trn22-c5c-a0-r4-transfer-lock-final-r3"
SOURCE = FROZEN / "stage_r4_remote_lock_final_r3.py"
TRANSPORT = FROZEN / "stage_r4_transport_lock_final_r3.py"
DIAGNOSTIC = HERE / "remote_timing_readonly.py"
HOLD = HERE / "GATE_CONTRACT.HOLD.json"
ENABLED = HERE / "TIMING_REQUEST.ENABLED.json"
REVIEW = HERE / "INDEPENDENT_TIMING_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_TIMING_ASSIGNMENT.json"
OUTPUT = HERE / "OPERATOR_RAW_TIMING_RESULT.json"
HOLD_RECEIPT = REPO / "research/v2/verification/20260925-trn22-fifo-final-r3-independent-HOLD.md"
CONTROLLER_STATE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/state.json")

SOURCE_COMMIT = "d571ea61b9fd2c3c7a21f254e044338ff873edba"
SOURCE_SHA256 = "fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a"
TRANSPORT_SHA256 = "a12b8b202e18a8e809355c1fe0499ac7b2b68a94c5766ddb383f331f07b0c6c1"
DIAGNOSTIC_SHA256 = "452113ab31188bf84211f0d59c9ae78b8c27f96b8e080a673b349051a79750b9"
HOLD_RECEIPT_SHA256 = "34377be826b5cad75759082f94dcb7bd1539f5f93c388b4a22cedac529d87fdd"
HOST_FINGERPRINT = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
RESULT_SCHEMA = "trn22-fifo-final-r3-operator-raw-timing-v1"
PIN_SCHEMA = "trn22-fifo-final-r3-readonly-preflight-pin-v1"
TRIALS = 20
STABLE_KEYS = ("hostname", "uid", "boot_id", "mount_namespace", "mount",
               "block_aliases", "ancestry")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_exact(path):
    named = path.lstat()
    if (not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
            named.st_nlink != 1):
        raise RuntimeError("local evidence is not an owner-owned regular file: " + path.name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(named, key) != getattr(opened, key) for key in keys):
            raise RuntimeError("local evidence changed during open: " + path.name)
        data = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data.extend(block)
        if any(getattr(opened, key) != getattr(snapshot, key)
               for snapshot in (os.fstat(fd), path.lstat()) for key in keys):
            raise RuntimeError("local evidence changed during read: " + path.name)
        return bytes(data)
    finally:
        os.close(fd)


def _pinned(path, digest):
    data = _read_exact(path)
    if _sha(data) != digest:
        raise RuntimeError("exact file hash drift: " + path.name)
    return data


def _json_exact(path):
    data = _read_exact(path)
    value = json.loads(data)
    if not isinstance(value, dict) or data != _canonical(value) + b"\n":
        raise RuntimeError("noncanonical fixed authority file: " + path.name)
    return data, value


def _local_stops_and_breakers():
    paths = (CONTROLLER_STATE.parent / "STOP",
             CONTROLLER_STATE.parent.parent / "agentic/STOP",
             REPO / "research/agentic/STOP", REPO / "research/v2/STOP")
    for path in paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP present: " + str(path))
    state_bytes = _read_exact(CONTROLLER_STATE)
    state = json.loads(state_bytes)
    if any(state["breakers"][name].get("tripped") is not False
           for name in ("global_correctness", "trn22")):
        raise RuntimeError("local global/trn22 breaker is tripped or unknown")
    if _read_exact(CONTROLLER_STATE) != state_bytes:
        raise RuntimeError("local controller state changed during check")
    for path in paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP appeared: " + str(path))
    return {"state_sha256": _sha(state_bytes), "stop_paths_absent": True,
            "global_tripped": False, "trn22_tripped": False}


def _transport():
    _pinned(TRANSPORT, TRANSPORT_SHA256)
    spec = importlib.util.spec_from_file_location("trn22_frozen_timing_transport", TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.REMOTE_SHA256 != SOURCE_SHA256 or
            module.KNOWN_HOSTS_SHA256 != KNOWN_HOSTS_SHA256 or
            module.HOST_FINGERPRINT != HOST_FINGERPRINT or
            module.ENDPOINT != ENDPOINT):
        raise RuntimeError("frozen transport host/source binding drift")
    return module


def _authorize():
    """All fixed authority files must exist and match before host contact."""
    _pinned(SOURCE, SOURCE_SHA256)
    _pinned(DIAGNOSTIC, DIAGNOSTIC_SHA256)
    _pinned(HOLD_RECEIPT, HOLD_RECEIPT_SHA256)
    hold_bytes, hold = _json_exact(HOLD)
    if (hold.get("dispatch_authorized") is not False or
            hold.get("trial_count") != TRIALS or
            hold.get("source_commit") != SOURCE_COMMIT or
            hold.get("source_sha256") != SOURCE_SHA256 or
            hold.get("diagnostic_sha256") != DIAGNOSTIC_SHA256 or
            hold.get("operator_sha256") != _sha(_read_exact(Path(__file__))) or
            hold.get("transport_sha256") != TRANSPORT_SHA256 or
            hold.get("hold_receipt_sha256") != HOLD_RECEIPT_SHA256):
        raise RuntimeError("disabled timing contract drift")
    request_bytes, request = _json_exact(ENABLED)
    own_sha = _sha(_read_exact(Path(__file__)))
    expected = {"schema": "trn22-fifo-final-r3-readonly-timing-enabled-v1",
                "dispatch_authorized": True, "scope": "TRN22_READONLY_FINAL_R3_20_TRIAL_TIMING",
                "host": "trn22", "endpoint": ENDPOINT,
                "host_fingerprint": HOST_FINGERPRINT,
                "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
                "transport_sha256": TRANSPORT_SHA256,
                "diagnostic_sha256": DIAGNOSTIC_SHA256,
                "operator_sha256": own_sha, "hold_contract_sha256": _sha(hold_bytes),
                "hold_receipt_sha256": HOLD_RECEIPT_SHA256,
                "trial_count": TRIALS, "one_attempt": True,
                "stdin": "/dev/null", "output_basename": OUTPUT.name,
                "host_operations": ["probe", "timing", "probe"]}
    if request != expected:
        raise RuntimeError("enabled timing request does not match exact read-only package")
    review_bytes, review = _json_exact(REVIEW)
    if review != {"schema": "trn22-fifo-final-r3-readonly-timing-review-v1",
                  "verdict": "PASS_READ_ONLY_TIMING_ONLY",
                  "reviewer_role": "verifier", "request_sha256": _sha(request_bytes),
                  "operator_sha256": own_sha,
                  "diagnostic_sha256": DIAGNOSTIC_SHA256}:
        raise RuntimeError("independent timing review missing or mismatched")
    _, assignment = _json_exact(ASSIGNMENT)
    if assignment != {"schema": "trn22-fifo-final-r3-readonly-timing-assignment-v1",
                      "status": "ASSIGNED_ONE_ATTEMPT_READ_ONLY_TIMING",
                      "assigned_by_role": "supervisor",
                      "scope": expected["scope"],
                      "request_sha256": _sha(request_bytes),
                      "review_sha256": _sha(review_bytes),
                      "output_basename": OUTPUT.name}:
        raise RuntimeError("Supervisor one-attempt timing assignment missing or mismatched")
    local_gates = _local_stops_and_breakers()
    transport = _transport()
    host_key = transport.checked_ssh_identity()
    if host_key != {"known_hosts_sha256": KNOWN_HOSTS_SHA256,
                    "host_fingerprint": HOST_FINGERPRINT,
                    "host_key_algorithm": "ssh-ed25519"}:
        raise RuntimeError("trn22 SSH host-key identity mismatch")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError("fixed local result already exists; no retry")
    return {"request_sha256": _sha(request_bytes), "review_sha256": _sha(review_bytes),
            "assignment_sha256": _sha(_read_exact(ASSIGNMENT)),
            "hold_contract_sha256": _sha(hold_bytes),
            "host_key": host_key, "local_gates": local_gates,
            "transport": transport}


_BOOTSTRAP = (
    "import base64,hashlib,json,sys\n"
    "if len(sys.argv) not in (3,5): raise SystemExit('wrong read-only operation shape')\n"
    "source=base64.b64decode(sys.argv[1],validate=True)\n"
    "if hashlib.sha256(source).hexdigest()!=%r: raise SystemExit('frozen source SHA drift')\n"
    "ns={'__name__':'trn22_frozen_remote_readonly','__file__':'<frozen-final-r3>'}\n"
    "exec(compile(source,ns['__file__'],'exec'),ns)\n"
    "operation=sys.argv[2]\n"
    "if operation=='probe' and len(sys.argv)==3:\n"
    " result=ns['probe']()\n"
    "elif operation=='timing' and len(sys.argv)==5:\n"
    " diagnostic=base64.b64decode(sys.argv[3],validate=True)\n"
    " if hashlib.sha256(diagnostic).hexdigest()!=%r: raise SystemExit('diagnostic SHA drift')\n"
    " pin_raw=base64.b64decode(sys.argv[4],validate=True)\n"
    " pin=json.loads(pin_raw)\n"
    " if json.dumps(pin,sort_keys=True,separators=(',',':')).encode()!=pin_raw: raise SystemExit('noncanonical preflight pin')\n"
    " diag={'__name__':'trn22_readonly_timing','__file__':'<pinned-readonly-timing>'}\n"
    " exec(compile(diagnostic,diag['__file__'],'exec'),diag)\n"
    " result=diag['run'](ns,pin)\n"
    "else: raise SystemExit('only probe/timing/probe are permitted')\n"
    "print(json.dumps(result,sort_keys=True,separators=(',',':')))\n"
    "if operation=='timing' and result['status']=='ABORTED_READONLY': raise SystemExit(3)\n"
) % (SOURCE_SHA256, DIAGNOSTIC_SHA256)


def _ssh_argv(operation, *, pin=None, transport):
    if operation not in ("probe", "timing") or (operation == "probe") != (pin is None):
        raise RuntimeError("wrong exact read-only SSH operation shape")
    transport.checked_ssh_identity()
    source = _pinned(SOURCE, SOURCE_SHA256)
    words = ["exec", "python3", "-B", "-c", _BOOTSTRAP,
             base64.b64encode(source).decode("ascii"), operation]
    if operation == "timing":
        words.extend((base64.b64encode(_pinned(DIAGNOSTIC, DIAGNOSTIC_SHA256)).decode("ascii"),
                      base64.b64encode(_canonical(pin)).decode("ascii")))
    command = " ".join(shlex.quote(word) for word in words)
    if shlex.split(command) != words or len(command.encode("utf-8")) > 120000:
        raise RuntimeError("read-only SSH command quoting/length drift")
    return ["ssh", "-F", "/dev/null", "-T", "-i", transport.IDENTITY_FILE,
            "-o", "UserKnownHostsFile=" + str(transport.KNOWN_HOSTS),
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UpdateHostKeys=no", "-o", "VerifyHostKeyDNS=no",
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
            ENDPOINT, command]


def _run_ssh(operation, *, pin=None, transport):
    argv = _ssh_argv(operation, pin=pin, transport=transport)
    started = time.monotonic_ns()
    timed_out = False
    try:
        process = subprocess.run(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=600, check=False)
        stdout, stderr, rc = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, rc = error.stdout or b"", error.stderr or b"", None
        timed_out = True
    ended = time.monotonic_ns()
    redacted = list(argv)
    redacted[5] = "<identity-file-path-redacted>"
    redacted[-1] = "<remote-command-sha256:" + _sha(argv[-1].encode()) + ">"
    raw = {"operation": operation, "argv_redacted": redacted,
           "argv_sha256": _sha(_canonical(argv)),
           "remote_command_sha256": _sha(argv[-1].encode()),
           "stdin": "/dev/null", "returncode": rc, "timed_out": timed_out,
           "started_monotonic_ns": started, "completed_monotonic_ns": ended,
           "duration_ns": ended - started,
           "stdout_bytes": len(stdout), "stdout_sha256": _sha(stdout),
           "stdout_base64": base64.b64encode(stdout).decode("ascii"),
           "stderr_bytes": len(stderr), "stderr_sha256": _sha(stderr),
           "stderr_base64": base64.b64encode(stderr).decode("ascii")}
    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            raise ValueError("remote receipt is not an object")
    except (UnicodeError, ValueError) as error:
        raw["parse_error"] = str(error)
        parsed = None
    raw["parsed"] = parsed
    return raw


def _require_probe(raw, transport):
    if (raw["returncode"] != 0 or raw["timed_out"] or raw["stderr_bytes"] != 0 or
            raw["parsed"] is None):
        raise RuntimeError("read-only full host probe failed")
    return transport.validate_probe(raw["parsed"], before_write=True)


def _pin(pre):
    return {"schema": PIN_SCHEMA, "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "preflight_utc": pre["utc"],
            "preflight_sha256": _sha(_canonical(pre)),
            "preflight_process_identity": pre["process_identity"],
            "identity": {key: pre["identity_after"][key] for key in STABLE_KEYS},
            "lock": pre["host_lock_identity"],
            "old_inventory": pre["old_inventory"],
            "targets": pre["targets"], "protected": pre["protected"]}


def _compare_clean(pre, timing, post):
    pre_utc = datetime.datetime.fromisoformat(pre["utc"])
    timing_start = datetime.datetime.fromisoformat(timing["started_utc"])
    timing_end = datetime.datetime.fromisoformat(timing["completed_utc"])
    post_utc = datetime.datetime.fromisoformat(post["utc"])
    if (any(value.tzinfo is None for value in
            (pre_utc, timing_start, timing_end, post_utc)) or
            not pre_utc < timing_start <= timing_end < post_utc):
        raise RuntimeError("before/timing/after UTC order failed")
    for key in STABLE_KEYS:
        if pre["identity_after"][key] != post["identity_before"][key]:
            raise RuntimeError("before/after host identity drift: " + key)
    for key in ("old_inventory", "targets", "protected", "host_lock_identity", "stops"):
        if pre[key] != post[key]:
            raise RuntimeError("before/after read-only snapshot drift: " + key)
    if (pre["process_identity"] == timing["process_identity"] or
            post["process_identity"] == timing["process_identity"] or
            pre["process_identity"] == post["process_identity"]):
        raise RuntimeError("before/timing/after did not use distinct SSH processes")
    if (timing["schema"] != "trn22-fifo-final-r3-readonly-timing-v1" or
            timing["source_commit"] != SOURCE_COMMIT or
            timing["source_sha256"] != SOURCE_SHA256 or
            timing.get("preflight_sha256") != _sha(_canonical(pre)) or
            timing["trial_count"] != TRIALS or len(timing["trials"]) != TRIALS or
            timing["before"] is None or timing["after"] is None or
            timing["initial_live_safety"] is None or
            timing["final_live_safety"] is None or timing["errors"]):
        raise RuntimeError("read-only 20-trial receipt incomplete or aborted")
    for snapshot in (timing["before"], timing["after"]):
        for key in STABLE_KEYS:
            if snapshot["identity"][key] != pre["identity_after"][key]:
                raise RuntimeError("timing-session host identity drift: " + key)
        for key, pin_key in (("lock", "host_lock_identity"),
                             ("old_inventory", "old_inventory"),
                             ("targets", "targets"), ("protected", "protected")):
            if snapshot[key] != pre[pin_key]:
                raise RuntimeError("timing-session protected path drift: " + key)
    expected_statuses = {"TIMING_MISS", "WITHIN_SOURCE_WINDOW_NO_OPEN"}
    for index, trial in enumerate(timing["trials"], 1):
        if trial["index"] != index or trial["status"] not in expected_statuses:
            raise RuntimeError("timing trial absent or unsafe")
        label = trial["final_label"]
        if (label["scan"]["rc"] != 0 or label["scan"]["match_count"] != 0 or
                label["scan"]["matches"] != [] or
                label["scan"]["stderr_empty"] is not True):
            raise RuntimeError("final label scan is not empty")
        if (trial["lock_stat_before"] != trial["lock_stat_after"] or
                trial["lock_held_ns"] < 0 or
                trial["prelabel_duration_ns"] < 0 or
                trial["final_label_call"]["duration_ns"] < 0 or
                trial["final_label_call"]["source_scan_duration_ns"] < 0 or
                trial["post_label"]["label_completion_to_post_end_ns"] < 0 or
                trial["post_label"]["initial_stop_sweep_ns"] is None or
                trial["post_label"]["initial_stop_sweep_ns"] < 0 or
                trial["post_label"]["started_monotonic_ns"] <
                label["completed_monotonic_ns"]):
            raise RuntimeError("trial lock or monotonic timing drift")
        names = [event["name"] for event in trial["post_label"]["events"]]
        if names != ["checkpoint_fuser", "neuron0_fuser", "neuron_ls", "lock_acl",
                     "stop_process", "process_inventory", "final_stops"]:
            raise RuntimeError("exact post-label substage sequence incomplete")
        if any(event["status"] != "OK" or event["duration_ns"] < 0
               for event in trial["post_label"]["events"]):
            raise RuntimeError("post-label substage failure or clock reversal")
        if trial["status"] == "WITHIN_SOURCE_WINDOW_NO_OPEN" and (
                trial["post_label"]["error"] is not None or
                trial["post_label"]["safety"] is None or
                trial["label_to_would_open_ns"] > 1_000_000_000):
            raise RuntimeError("claimed within-window trial is malformed")
        if trial["status"] == "TIMING_MISS" and (
                trial["post_label"]["error"] is None and
                trial.get("label_to_would_open_ns", 0) <= 1_000_000_000):
            raise RuntimeError("claimed timing miss lacks a timing cause")
    if timing["status"] not in ("COMPLETED_20_WITH_TIMING_MISS",
                                "COMPLETED_20_WITHIN_SOURCE_WINDOW_NO_OPEN"):
        raise RuntimeError("timing receipt has no complete result")
    if timing["timing_miss_count"] != sum(
            trial["status"] == "TIMING_MISS" for trial in timing["trials"]):
        raise RuntimeError("timing-miss count does not match raw trials")
    return {"status": "RAW_20_TRIAL_TIMING_RETRIEVED_PENDING_INDEPENDENT_REVIEW",
            "timing_miss_count": timing["timing_miss_count"],
            "max_label_to_would_open_ns": max(
                (trial.get("label_to_would_open_ns", 0) for trial in timing["trials"]),
                default=0)}


def collect_once(authority):
    """Consume one fixed local raw receipt path; never retry host operations."""
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    record = {"schema": RESULT_SCHEMA, "status": "FAILED_CLOSED",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "authority": {key: value for key, value in authority.items()
                            if key != "transport"},
              "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
              "diagnostic_sha256": DIAGNOSTIC_SHA256, "phases": [],
              "comparison": None, "errors": []}
    pre = timing = post = None
    timing_attempted = False
    try:
        if _local_stops_and_breakers() != authority["local_gates"]:
            raise RuntimeError("local STOP/breaker state drift before read-only preflight")
        pre_raw = _run_ssh("probe", transport=authority["transport"])
        record["phases"].append(pre_raw)
        pre = _require_probe(pre_raw, authority["transport"])
        if _local_stops_and_breakers() != authority["local_gates"]:
            raise RuntimeError("local STOP/breaker state drift before timing")
        timing_attempted = True
        timing_raw = _run_ssh("timing", pin=_pin(pre), transport=authority["transport"])
        record["phases"].append(timing_raw)
        timing = timing_raw["parsed"]
        if (timing_raw["returncode"] != 0 or timing_raw["timed_out"] or
                timing_raw["stderr_bytes"] != 0):
            raise RuntimeError("read-only timing SSH failed or emitted stderr")
    except BaseException as error:
        record["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if timing_attempted:
            try:
                post_raw = _run_ssh("probe", transport=authority["transport"])
                record["phases"].append(post_raw)
                post = _require_probe(post_raw, authority["transport"])
            except BaseException as error:
                record["errors"].append({"type": type(error).__name__, "message": str(error)})
        if pre is not None and timing is not None and post is not None:
            try:
                record["comparison"] = _compare_clean(pre, timing, post)
                if not record["errors"]:
                    record["status"] = "RAW_TIMING_RETRIEVED_PENDING_INDEPENDENT_REVIEW"
            except BaseException as error:
                record["errors"].append({"type": type(error).__name__, "message": str(error)})
        record["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = _canonical(record) + b"\n"
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise RuntimeError("local raw timing receipt write failed")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    return record["status"]


def main():
    authority = _authorize()  # Absent enabled request/review/assignment: no SSH or local receipt.
    status = collect_once(authority)
    print(status + ": " + str(OUTPUT))
    if status != "RAW_TIMING_RETRIEVED_PENDING_INDEPENDENT_REVIEW":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
