"""Gated trn21 r10 stage pre-open timing collector; no host contact by default.

The exact enabled request cannot dispatch without a distinct independent final
review and Supervisor assignment. Future capture is read-only: it sends pinned
code on SSH stdin, calls no stage writer, and creates no host file or marker.
"""

import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE = HERE.parent / "trn21-c5c-a0-r10-incremental-stage-r2-hold/remote_incremental.py"
STAGE_OPERATOR = HERE.parent / "trn21-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py"
RECEIPT = REPO / "research/v2/verification/20260925-trn21-r10-incremental-stage-r3-active-root-independent-static-PASS.md"
TIMING_REVIEW = REPO / "research/v2/verification/20260925-trn21-r10-stage-readonly-timing-r3-independent-static-PASS.md"
TIMING = HERE / "remote_timing.py"
HOLD = HERE / "REQUEST.HOLD.json"
ENABLED = HERE / "READONLY_ENABLEMENT.FINAL.json"
REVIEW = HERE / "INDEPENDENT_READONLY_ENABLEMENT.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_READONLY_ASSIGNMENT.json"
INVOKER = HERE / "invoke_timing_readonly.py"
OUTPUT = HERE / "OPERATOR_RAW_TRN21_R10_TIMING_RESULT.json"
EXECUTION_ROOT = Path("/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root")
ACTIVE_CONTROL = Path("/home/poojith-devan/Desktop/Trainium")
LEGACY_CONTROL = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
IDENTITY_FILE = Path("/home/poojith-devan/Desktop/Trainium/trn2.pem")
ENDPOINT = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"
FINGERPRINT = "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
STAGE_COMMIT = "7205abf5469edc616c4e4b58caee27bc1957bbd4"
REVIEW_COMMIT = "79bac32c0ce8dbd6e343e27fccf9def272234deb"
STAGE_SHA256 = "47e2c28bd6fd735904398a48d6c91ef95b93454cef9558b121c26b0ee5b66a0b"
STAGE_OPERATOR_SHA256 = "6ecad35400199b7c1cba41c78d5d4f14363d37180c40e8aab5e9b485846db74f"
RECEIPT_SHA256 = "ab0a63e47ece6d9055167b62679b40bee949b1d8e1e948195967960a77bf859e"
TIMING_SOURCE_COMMIT = "3aa258d31d62b5f59f2dfcf34149b63f8eabd453"
TIMING_REVIEW_COMMIT = "adee3b47cbeb8d82b5b716ce6f5aafcd80a4aee3"
TIMING_REVIEW_SHA256 = "78aab14b15c88431974322ad0bfe6e5ee96f1873dfb53266290c19db71f8de2c"
TIMING_SHA256 = "5a1298febfb67f6a61eb1f6fa02f974c7cbe15c6eb0bc221338b85d892be0db7"
HOLD_SHA256 = "9f2ed81e1c04decc16e0747d6e71f52037da759241ea1c8774dbbf790604d897"
TRAIN_SHA256 = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
SCHEMA = "trn21-r10-stage-preopen-readonly-timing-v1"
SCOPE = "TRN21_R10_STAGE_R3_READONLY_PREOPEN_TIMING_ONLY"
TRIALS = 3
MAX_REMOTE_STDOUT = 64 * 1024
MAX_REMOTE_LINE = 8192
PRE_FREE = 13_177_973_911
MIN_INODES = 10_000
MAX_AGE_SECONDS = 1.0
MAX_REMOTE_SECONDS = 330.0
# The remote fields subtract floating-point monotonic readings. One microsecond
# covers subtraction roundoff even at long uptimes and is negligible beside
# the one-second final freshness gate.
CHRONOLOGY_TOLERANCE_SECONDS = 1e-6
SENSITIVE = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
                       rb"(?i:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|OPENAI_API_KEY|PRIVATE_KEY|BEARER_TOKEN|trn2\.pem|\.env(?:\b|/))")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _regular(path, *, private=False, maximum=2 * 1024 * 1024):
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or named.st_size > maximum or
            (private and stat.S_IMODE(named.st_mode) & 0o077)):
        raise RuntimeError("unsafe local evidence metadata: " + path.name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(opened, key) != getattr(named, key) for key in fields):
            raise RuntimeError("local evidence changed during open: " + path.name)
        data = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data.extend(block)
            if len(data) > maximum:
                raise RuntimeError("local evidence exceeded byte limit: " + path.name)
        after, named_after = os.fstat(fd), os.lstat(path)
        if any(getattr(other, key) != getattr(opened, key)
               for other in (after, named_after) for key in fields):
            raise RuntimeError("local evidence changed during read: " + path.name)
        return bytes(data)
    finally:
        os.close(fd)


def _exact_json(path):
    raw = _regular(path)
    value = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(value, dict) or raw != _canonical(value) + b"\n":
        raise RuntimeError("authority JSON is not exact canonical bytes: " + path.name)
    return raw, value


def _pins():
    pinned = {"stage": (STAGE, STAGE_SHA256),
              "stage_operator": (STAGE_OPERATOR, STAGE_OPERATOR_SHA256),
              "r3_review": (RECEIPT, RECEIPT_SHA256),
              "timing_r3_review": (TIMING_REVIEW, TIMING_REVIEW_SHA256),
              "timing": (TIMING, TIMING_SHA256),
              "hold": (HOLD, HOLD_SHA256)}
    raw = {}
    for name, (path, expected) in pinned.items():
        raw[name] = _regular(path)
        if _sha(raw[name]) != expected:
            raise RuntimeError("pinned timing source changed: " + name)
    hold_raw, hold = _exact_json(HOLD)
    if hold_raw != raw["hold"] or hold != {
            "schema": SCHEMA + "-hold", "status": "DISABLED_PENDING_INDEPENDENT_REVIEW",
            "scope": SCOPE, "host": "trn21", "hostname": "ip-172-31-34-85",
            "stage_r3_commit": STAGE_COMMIT, "r3_review_commit": REVIEW_COMMIT,
            "stage_remote_sha256": STAGE_SHA256,
            "stage_operator_sha256": STAGE_OPERATOR_SHA256,
            "r3_review_sha256": RECEIPT_SHA256,
            "timing_source_sha256": TIMING_SHA256,
            "source": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2",
            "target": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10",
            "bundle": "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle",
            "label": "fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2",
            "lock": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
            "max_trials": TRIALS, "host_contact_authorized": False,
            "host_write_authorized": False, "one_shot_marker_authorized": False,
            "training_authorized": False, "evaluation_authorized": False,
            "aws_action_authorized": False, "fresh_host_values": None}:
        raise RuntimeError("disabled timing request changed")
    return raw


def _authority(raw):
    # The final review and Supervisor assignment are intentionally absent.
    enabled_raw, enabled = _exact_json(ENABLED)
    own_hash = _sha(_regular(Path(__file__)))
    invoker_hash = _sha(_regular(INVOKER))
    expected = {
        "schema": SCHEMA + "-enabled", "scope": SCOPE,
        "status": "ENABLED_REQUEST_PENDING_FINAL_REVIEW_AND_ASSIGNMENT",
        "host": "trn21", "hostname": "ip-172-31-34-85",
        "endpoint": ENDPOINT, "fingerprint": FINGERPRINT,
        "known_hosts_sha256": KNOWN_HOSTS_SHA256,
        "host_contact_authorized": True, "host_write_authorized": False,
        "host_process_signals_authorized": False,
        "archive_target_open_authorized": False,
        "one_shot_marker_authorized": False, "training_authorized": False,
        "evaluation_authorized": False, "aws_action_authorized": False,
        "hold_sha256": _sha(raw["hold"]),
        "stage_r3_commit": STAGE_COMMIT,
        "stage_r3_review_commit": REVIEW_COMMIT,
        "stage_remote_sha256": STAGE_SHA256,
        "stage_operator_sha256": STAGE_OPERATOR_SHA256,
        "r3_review_sha256": RECEIPT_SHA256,
        "timing_source_commit": TIMING_SOURCE_COMMIT,
        "timing_r3_review_commit": TIMING_REVIEW_COMMIT,
        "timing_r3_review_sha256": TIMING_REVIEW_SHA256,
        "timing_source_sha256": TIMING_SHA256,
        "collector_sha256": own_hash, "invoker_sha256": invoker_hash,
        "execution_root": str(EXECUTION_ROOT), "execution_head": STAGE_COMMIT,
        "science_target_commit": "22bfbab5a77d6feb670d2d737504f8b70b08ddc1",
        "train_sha256": TRAIN_SHA256, "prepare_sha256": PREPARE_SHA256,
        "source": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2",
        "target": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10",
        "bundle": "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle",
        "label": "fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2",
        "lock": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
        "output_basename": OUTPUT.name, "one_attempt": True,
        "max_trials": TRIALS}
    if enabled != expected:
        raise RuntimeError("exact read-only enablement missing or mismatched")
    review_raw, review = _exact_json(REVIEW)
    if review != {"schema": SCHEMA + "-review", "role": "independent_verifier",
                  "verdict": "PASS_READONLY_TIMING_ONLY", "host": "trn21",
                  "scope": SCOPE, "enabled_sha256": _sha(enabled_raw),
                  "collector_sha256": own_hash, "invoker_sha256": invoker_hash,
                  "timing_r3_review_sha256": TIMING_REVIEW_SHA256,
                  "timing_source_sha256": TIMING_SHA256,
                  "stage_remote_sha256": STAGE_SHA256,
                  "execution_root": str(EXECUTION_ROOT),
                  "execution_head": STAGE_COMMIT,
                  "output_basename": OUTPUT.name}:
        raise RuntimeError("exact independent timing review missing or mismatched")
    assignment_raw, assignment = _exact_json(ASSIGNMENT)
    if assignment != {"schema": SCHEMA + "-assignment", "role": "supervisor",
                      "status": "ASSIGNED_ONE_READONLY_CAPTURE", "host": "trn21",
                      "scope": SCOPE, "enabled_sha256": _sha(enabled_raw),
                      "review_sha256": _sha(review_raw), "max_trials": TRIALS,
                      "output_basename": OUTPUT.name, "one_attempt": True}:
        raise RuntimeError("exact Supervisor timing assignment missing or mismatched")
    return {"enabled_sha256": _sha(enabled_raw), "review_sha256": _sha(review_raw),
            "assignment_sha256": _sha(assignment_raw)}


def _local_stops():
    return [str(root / suffix)
            for root in (ACTIVE_CONTROL, LEGACY_CONTROL, REPO)
            for suffix in ("research/agentic/STOP", "research/v2/STOP")
            if os.path.lexists(root / suffix)]


def _controller_gate():
    if _local_stops():
        raise RuntimeError("local STOP present before SSH")
    raw = _regular(ACTIVE_CONTROL / "research/v2/state.json", maximum=1024 * 1024)
    state = json.loads(raw, object_pairs_hook=_unique)
    breakers = state.get("breakers") if isinstance(state, dict) else None
    hosts = state.get("hosts") if isinstance(state, dict) else None
    if (not isinstance(breakers, dict) or
            any(not isinstance(breakers.get(name), dict) or
                breakers[name].get("tripped") is not False
                for name in ("global_correctness", "trn21")) or
            not isinstance(hosts, dict) or not isinstance(hosts.get("trn21"), dict) or
            state.get("launch_authorized") is not False or
            hosts["trn21"].get("launch_allowed") is not False):
        raise RuntimeError("active Desktop controller breaker/launch gate unsafe")
    if _local_stops():
        raise RuntimeError("local STOP appeared during controller gate")
    return {"active_state_sha256": _sha(raw), "breakers_clear": True,
            "trn21_launch_allowed": False, "stops_absent": True}


def _ssh_identity():
    parent, key = os.lstat(KNOWN_HOSTS.parent), os.lstat(IDENTITY_FILE)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077 or
            not stat.S_ISREG(key.st_mode) or key.st_uid != os.getuid() or
            key.st_nlink != 1 or stat.S_IMODE(key.st_mode) != 0o600):
        raise RuntimeError("SSH identity metadata unsafe")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("pinned known-hosts bytes changed")
    hostname = ENDPOINT.split("@", 1)[1]
    found = subprocess.run(["/usr/bin/ssh-keygen", "-F", hostname, "-f",
                            str(KNOWN_HOSTS)], capture_output=True,
                           timeout=10, check=False)
    if found.returncode != 0:
        raise RuntimeError("pinned SSH host entry missing")
    entries = [line for line in found.stdout.splitlines()
               if line and not line.startswith(b"#")]
    ed = [line for line in entries if len(line.split()) >= 3 and
          line.split()[1] == b"ssh-ed25519"]
    if len(ed) != 1:
        raise RuntimeError("pinned ED25519 entry not unique")
    fp = subprocess.run(["/usr/bin/ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
                        input=ed[0] + b"\n", capture_output=True,
                        timeout=10, check=False)
    if fp.returncode != 0 or FINGERPRINT.encode("ascii") not in fp.stdout.split():
        raise RuntimeError("trn21 host fingerprint changed")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts changed during identity check")
    return {"algorithm": "ssh-ed25519", "fingerprint": FINGERPRINT,
            "known_hosts_sha256": KNOWN_HOSTS_SHA256}


BOOTSTRAP = (
    "import base64,hashlib,json,sys\n"
    "if sys.flags.optimize: raise SystemExit('optimized Python HOLD')\n"
    "raw=sys.stdin.buffer.read(131073)\n"
    "if len(raw)>131072: raise SystemExit('oversized timing payload')\n"
    "p=json.loads(raw)\n"
    "if set(p)!={'stage','timing'}: raise SystemExit('wrong payload shape')\n"
    "s=base64.b64decode(p['stage'],validate=True)\n"
    "t=base64.b64decode(p['timing'],validate=True)\n"
    "if hashlib.sha256(s).hexdigest()!=%r or hashlib.sha256(t).hexdigest()!=%r: raise SystemExit('pinned source drift')\n"
    "stage={'__name__':'trn21_r10_stage_r3_readonly_source','__file__':'<pinned-stage-source>'}\n"
    "exec(compile(s,stage['__file__'],'exec'),stage)\n"
    "timing={'__name__':'trn21_r10_readonly_timing','__file__':'<pinned-timing>'}\n"
    "exec(compile(t,timing['__file__'],'exec'),timing)\n"
    "timing['run'](stage)\n"
) % (STAGE_SHA256, TIMING_SHA256)


def _ssh_argv():
    words = ["/usr/bin/timeout", "--foreground", "--signal=TERM",
             "--kill-after=10s", "330s", "/usr/bin/env", "-i",
             "PATH=/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
             "HOME=/nonexistent", "/usr/bin/python3", "-I", "-B", "-c", BOOTSTRAP]
    command = " ".join(shlex.quote(word) for word in words)
    if shlex.split(command) != words or len(command.encode()) > 12000:
        raise RuntimeError("timing command quoting or size changed")
    return ["/usr/bin/ssh", "-T", "-F", "/dev/null", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o",
            "UserKnownHostsFile=" + str(KNOWN_HOSTS), "-o",
            "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UpdateHostKeys=no", "-o", "VerifyHostKeyDNS=no",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ForwardAgent=no", "-o", "ConnectTimeout=10",
            "-o", "IdentitiesOnly=yes", "-i", str(IDENTITY_FILE), ENDPOINT, command]


def _parse_jsonl(raw):
    if not raw or len(raw) > MAX_REMOTE_STDOUT or not raw.endswith(b"\n"):
        raise RuntimeError("timing stdout absent, oversized or truncated")
    records = []
    for line in raw.splitlines():
        if not line or len(line) > MAX_REMOTE_LINE:
            raise RuntimeError("empty or oversized timing line")
        value = json.loads(line, object_pairs_hook=_unique)
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA or
                _canonical(value) != line):
            raise RuntimeError("malformed timing JSONL record")
        records.append(value)
    if (len(records) < 3 or len(records) > TRIALS + 2 or
            records[0].get("kind") != "start" or
            records[-1].get("kind") != "end" or
            any(value.get("kind") != "trial" or value.get("index") != index
                for index, value in enumerate(records[1:-1], 1)) or
            records[-1].get("completed_trials") != len(records) - 2):
        raise RuntimeError("timing record order or trial count malformed")
    start, end = records[0], records[-1]
    expected_identity = {
        "hostname": "ip-172-31-34-85", "uid": 1000,
        "source": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2",
        "target": "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10",
        "bundle": "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle",
        "label": "fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2",
        "lock": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
        "source_commit": "e65a3097a5b00f6178188a935bb2acb9a5042c6c",
        "target_commit": "22bfbab5a77d6feb670d2d737504f8b70b08ddc1",
    }
    if (set(start) != {"schema", "kind", "utc", "identity", "boot_id",
                       "mount_namespace", "root_free_bytes", "root_free_inodes",
                       "full_readonly_preflight_seconds", "max_trials", "actual_open"} or
            start["identity"] != expected_identity or
            type(start["boot_id"]) is not str or
            re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                         start["boot_id"]) is None or
            type(start["mount_namespace"]) is not str or
            re.fullmatch(r"mnt:\[[0-9]+\]", start["mount_namespace"]) is None or
            start["max_trials"] != TRIALS or type(start["max_trials"]) is not int or
            start["actual_open"] is not False or
            not _space_ok(start) or
            not _finite_range(start["full_readonly_preflight_seconds"], MAX_REMOTE_SECONDS) or
            not _utc_ok(start["utc"])):
        raise RuntimeError("timing start/preflight identity or resources malformed")
    if (set(end) != {"schema", "kind", "utc", "completed_trials", "max_trials",
                     "status", "actual_open"} or
            type(end["completed_trials"]) is not int or
            type(end["max_trials"]) is not int or end["max_trials"] != TRIALS or
            end["actual_open"] is not False or not _utc_ok(end["utc"])):
        raise RuntimeError("timing terminal fields malformed")
    trials = records[1:-1]
    terminal = end["status"]
    for trial in trials:
        if (not _finite_range(trial.get("preopen_seconds"), MAX_REMOTE_SECONDS) or
                not _utc_ok(trial.get("utc"))):
            raise RuntimeError("timing trial elapsed/UTC malformed")
        if trial.get("status") == "WITHIN_WINDOW_NO_OPEN":
            if (set(trial) != {"schema", "kind", "index", "utc", "preopen_seconds",
                              "status", "label_scan_duration_seconds",
                              "activity_age_seconds", "label_age_seconds",
                              "space_age_seconds", "root_free_bytes", "root_free_inodes"} or
                    not _space_ok(trial) or
                    any(not _finite_range(trial[name], MAX_AGE_SECONDS) for name in
                        ("label_scan_duration_seconds", "activity_age_seconds",
                         "label_age_seconds", "space_age_seconds"))):
                raise RuntimeError("timing trial freshness or space malformed")
            if (trial["preopen_seconds"] + CHRONOLOGY_TOLERANCE_SECONDS <
                    trial["label_scan_duration_seconds"] + trial["label_age_seconds"] or
                    trial["label_age_seconds"] + CHRONOLOGY_TOLERANCE_SECONDS <
                    trial["activity_age_seconds"] or
                    trial["activity_age_seconds"] + CHRONOLOGY_TOLERANCE_SECONDS <
                    trial["space_age_seconds"]):
                raise RuntimeError("timing trial remote monotonic chronology contradictory")
        elif trial.get("status") == "HOLD_NO_OPEN":
            if (set(trial) != {"schema", "kind", "index", "utc", "preopen_seconds",
                              "status", "error_type", "error"} or
                    type(trial["error_type"]) is not str or not trial["error_type"] or
                    type(trial["error"]) is not str or not trial["error"] or
                    len(trial["error"]) > 256):
                raise RuntimeError("timing HOLD trial malformed")
        else:
            raise RuntimeError("timing trial status malformed")
    if (
            (terminal == "READONLY_TIMING_COMPLETE_NO_OPEN" and
             (len(trials) != TRIALS or
              any(item["status"] != "WITHIN_WINDOW_NO_OPEN" for item in trials))) or
            (terminal == "READONLY_TIMING_HOLD_NO_OPEN" and
             (trials[-1]["status"] != "HOLD_NO_OPEN" or
              any(item["status"] != "WITHIN_WINDOW_NO_OPEN" for item in trials[:-1]))) or
            terminal not in ("READONLY_TIMING_COMPLETE_NO_OPEN", "READONLY_TIMING_HOLD_NO_OPEN")):
        raise RuntimeError("timing trial/terminal status mismatch")
    return records


def _finite_range(value, maximum):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= maximum
    except (OverflowError, ValueError, TypeError):
        return False


def _space_ok(value):
    return (type(value.get("root_free_bytes")) is int and
            type(value.get("root_free_inodes")) is int and
            value["root_free_bytes"] >= PRE_FREE and
            value["root_free_inodes"] >= MIN_INODES)


def _utc_ok(value):
    if type(value) is not str:
        return False
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == datetime.timedelta(0)


def _raw_summary(raw):
    suppressed = len(raw) > MAX_REMOTE_STDOUT or bool(SENSITIVE.search(raw))
    return {"bytes": len(raw), "sha256": _sha(raw), "suppressed": suppressed,
            "base64": None if suppressed else base64.b64encode(raw).decode("ascii")}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _reject_optimized():
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python is forbidden")


def _execution_root_gate():
    from invoke_timing_readonly import check_execution_root
    return check_execution_root()


def _write_receipt(fd, record):
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise RuntimeError("local timing receipt write failed")
        view = view[written:]
    os.fsync(fd)


def _recorded_status(destination):
    record = json.loads(_regular(destination, private=True), object_pairs_hook=_unique)
    return record.get("status") if isinstance(record, dict) else None


def capture():
    _reject_optimized()
    inputs = _pins()
    authority = _authority(inputs)  # Final review/assignment absent by design.
    root_before = _execution_root_gate()
    before = _controller_gate()
    identity = _ssh_identity()
    if os.path.lexists(OUTPUT) or any(HERE.glob("OPERATOR_RAW_TRN21_R10_TIMING_*.json")):
        raise RuntimeError("prior local timing receipt exists; no duplicate capture")
    payload = _canonical({"stage": base64.b64encode(inputs["stage"]).decode("ascii"),
                          "timing": base64.b64encode(inputs["timing"]).decode("ascii")})
    if len(payload) > 131072:
        raise RuntimeError("timing payload exceeded bound")
    argv = _ssh_argv()
    if _execution_root_gate() != root_before:
        raise RuntimeError("local execution root changed before SSH")
    if _controller_gate() != before:
        raise RuntimeError("active controller state changed before SSH")
    started = _utc()
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        _write_receipt(fd, {"schema": SCHEMA + "-operator-raw", "scope": SCOPE,
                            "status": "HOLD_IN_PROGRESS", "started_utc": started,
                            "authority": authority, "host_key": identity,
                            "execution_root_before": root_before,
                            "active_controller_before": before,
                            "no_host_write_or_marker": True})
        if _execution_root_gate() != root_before or _controller_gate() != before:
            raise RuntimeError("local execution/controller gate changed before SSH")
        try:
            done = subprocess.run(argv, input=payload, capture_output=True,
                                  timeout=350, check=False)
            rc, stdout, stderr, timed_out = done.returncode, done.stdout, done.stderr, False
        except subprocess.TimeoutExpired as error:
            rc, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
        try:
            after = _controller_gate()
            controller_stable = after == before
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            after = {"status": "HOLD", "error_type": type(error).__name__}
            controller_stable = False
        try:
            root_after = _execution_root_gate()
            root_stable = root_after == root_before
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            root_after = {"status": "HOLD", "error_type": type(error).__name__}
            root_stable = False
        records, parse_error = None, None
        if rc == 0 and not timed_out and not stderr and not SENSITIVE.search(stdout):
            try:
                records = _parse_jsonl(stdout)
            except (RuntimeError, ValueError, TypeError, KeyError) as error:
                parse_error = type(error).__name__ + ": " + str(error)[:200]
        elif stderr:
            parse_error = "nonempty_remote_stderr"
        status = ("READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY"
                  if root_stable and controller_stable and rc == 0 and not timed_out and
                  not stderr and records and
                  records[-1]["status"] == "READONLY_TIMING_COMPLETE_NO_OPEN" else "HOLD")
        record = {"schema": SCHEMA + "-operator-raw", "scope": SCOPE,
                  "started_utc": started, "ended_utc": _utc(),
                  "authority": authority, "host_key": identity,
                  "execution_root_before": root_before,
                  "execution_root_after": root_after,
                  "execution_root_stable": root_stable,
                  "active_controller_before": before, "active_controller_after": after,
                  "active_controller_stable": controller_stable,
                  "source_sha256": {name: _sha(value) for name, value in inputs.items()},
                  "remote_command_sha256": _sha(argv[-1].encode()),
                  "remote_payload_sha256": _sha(payload), "exit_code": rc,
                  "timed_out": timed_out, "stdout": _raw_summary(stdout),
                  "stderr": _raw_summary(stderr), "parsed": records,
                  "parse_error": parse_error, "status": status,
                  "no_host_write_or_marker": True}
        _write_receipt(fd, record)
        return OUTPUT
    finally:
        os.close(fd)


def main():
    if sys.argv[1:] == ["--check"]:
        inputs = _pins()
        print(json.dumps({"status": "DISABLED_NO_HOST_CONTACT",
                          "pinned_sha256": {name: _sha(raw)
                                            for name, raw in inputs.items()}},
                         sort_keys=True))
        return 0
    if sys.argv[1:] == ["--capture"]:
        try:
            destination = capture()
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            print("HOLD: " + type(error).__name__ + ": " + str(error)[:200],
                  file=sys.stderr)
            return 2
        print(str(destination))
        return (0 if _recorded_status(destination) ==
                "READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY" else 3)
    print("HOLD: use --check; capture needs final review and assignment",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
