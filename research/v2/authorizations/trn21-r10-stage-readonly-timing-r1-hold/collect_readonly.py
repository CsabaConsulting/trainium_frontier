"""Disabled trn21 r10 stage pre-open timing collector; no host contact by default.

The exact enabled request, independent review and Supervisor assignment do
not exist in this package. Future capture is read-only: it sends pinned code
on SSH stdin, calls no stage writer, and creates no host file or marker.
"""

import base64
import datetime
import hashlib
import json
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
TIMING = HERE / "remote_timing.py"
HOLD = HERE / "REQUEST.HOLD.json"
ENABLED = HERE / "READONLY_ENABLEMENT.FINAL.json"
REVIEW = HERE / "INDEPENDENT_READONLY_ENABLEMENT.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_READONLY_ASSIGNMENT.json"
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
TIMING_SHA256 = "5a1298febfb67f6a61eb1f6fa02f974c7cbe15c6eb0bc221338b85d892be0db7"
HOLD_SHA256 = "9f2ed81e1c04decc16e0747d6e71f52037da759241ea1c8774dbbf790604d897"
SCHEMA = "trn21-r10-stage-preopen-readonly-timing-v1"
SCOPE = "TRN21_R10_STAGE_R3_READONLY_PREOPEN_TIMING_ONLY"
TRIALS = 3
MAX_REMOTE_STDOUT = 64 * 1024
MAX_REMOTE_LINE = 8192
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
    # These three files are intentionally absent. Missing one fails before SSH.
    enabled_raw, enabled = _exact_json(ENABLED)
    expected = {
        "schema": SCHEMA + "-enabled", "scope": SCOPE, "host": "trn21",
        "host_contact_authorized": True, "host_write_authorized": False,
        "one_shot_marker_authorized": False, "training_authorized": False,
        "evaluation_authorized": False, "aws_action_authorized": False,
        "hold_sha256": _sha(raw["hold"]),
        "stage_remote_sha256": STAGE_SHA256,
        "stage_operator_sha256": STAGE_OPERATOR_SHA256,
        "r3_review_sha256": RECEIPT_SHA256,
        "timing_source_sha256": TIMING_SHA256,
        "collector_sha256": _sha(_regular(Path(__file__))),
        "max_trials": TRIALS}
    if enabled != expected:
        raise RuntimeError("exact read-only enablement missing or mismatched")
    review_raw, review = _exact_json(REVIEW)
    if review != {"schema": SCHEMA + "-review", "role": "independent_verifier",
                  "verdict": "PASS_READONLY_TIMING_ONLY", "host": "trn21",
                  "scope": SCOPE, "enabled_sha256": _sha(enabled_raw),
                  "collector_sha256": expected["collector_sha256"],
                  "timing_source_sha256": TIMING_SHA256,
                  "stage_remote_sha256": STAGE_SHA256}:
        raise RuntimeError("exact independent timing review missing or mismatched")
    assignment_raw, assignment = _exact_json(ASSIGNMENT)
    if assignment != {"schema": SCHEMA + "-assignment", "role": "supervisor",
                      "status": "ASSIGNED_ONE_READONLY_CAPTURE", "host": "trn21",
                      "scope": SCOPE, "enabled_sha256": _sha(enabled_raw),
                      "review_sha256": _sha(review_raw), "max_trials": TRIALS}:
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
            records[0].get("actual_open") is not False or
            records[-1].get("kind") != "end" or
            any(value.get("kind") != "trial" or value.get("index") != index
                for index, value in enumerate(records[1:-1], 1)) or
            records[-1].get("completed_trials") != len(records) - 2 or
            records[-1].get("actual_open") is not False):
        raise RuntimeError("timing record order or trial count malformed")
    trials = records[1:-1]
    terminal = records[-1].get("status")
    if (any(item.get("status") not in ("WITHIN_WINDOW_NO_OPEN", "HOLD_NO_OPEN")
            for item in trials) or
            (terminal == "READONLY_TIMING_COMPLETE_NO_OPEN" and
             (len(trials) != TRIALS or
              any(item["status"] != "WITHIN_WINDOW_NO_OPEN" for item in trials))) or
            (terminal == "READONLY_TIMING_HOLD_NO_OPEN" and
             (trials[-1]["status"] != "HOLD_NO_OPEN" or
              any(item["status"] != "WITHIN_WINDOW_NO_OPEN" for item in trials[:-1]))) or
            terminal not in ("READONLY_TIMING_COMPLETE_NO_OPEN", "READONLY_TIMING_HOLD_NO_OPEN")):
        raise RuntimeError("timing trial/terminal status mismatch")
    return records


def _raw_summary(raw):
    suppressed = len(raw) > MAX_REMOTE_STDOUT or bool(SENSITIVE.search(raw))
    return {"bytes": len(raw), "sha256": _sha(raw), "suppressed": suppressed,
            "base64": None if suppressed else base64.b64encode(raw).decode("ascii")}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture():
    inputs = _pins()
    authority = _authority(inputs)  # Missing by design: no host contact.
    before = _controller_gate()
    identity = _ssh_identity()
    if any(HERE.glob("OPERATOR_RAW_TRN21_R10_TIMING_*.json")):
        raise RuntimeError("prior local timing receipt exists; no duplicate capture")
    payload = _canonical({"stage": base64.b64encode(inputs["stage"]).decode("ascii"),
                          "timing": base64.b64encode(inputs["timing"]).decode("ascii")})
    if len(payload) > 131072:
        raise RuntimeError("timing payload exceeded bound")
    argv = _ssh_argv()
    if _controller_gate() != before:
        raise RuntimeError("active controller state changed before SSH")
    started = _utc()
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
    records, parse_error = None, None
    if rc == 0 and not timed_out and not SENSITIVE.search(stdout):
        try:
            records = _parse_jsonl(stdout)
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            parse_error = type(error).__name__ + ": " + str(error)[:200]
    status = ("READONLY_TIMING_OBSERVED_NOT_STAGE_AUTHORITY"
              if controller_stable and rc == 0 and not timed_out and records and
              records[-1]["status"] == "READONLY_TIMING_COMPLETE_NO_OPEN" else "HOLD")
    record = {"schema": SCHEMA + "-operator-raw", "scope": SCOPE,
              "started_utc": started, "ended_utc": _utc(),
              "authority": authority, "host_key": identity,
              "active_controller_before": before, "active_controller_after": after,
              "active_controller_stable": controller_stable,
              "source_sha256": {name: _sha(value) for name, value in inputs.items()},
              "remote_command_sha256": _sha(argv[-1].encode()),
              "remote_payload_sha256": _sha(payload), "exit_code": rc,
              "timed_out": timed_out, "stdout": _raw_summary(stdout),
              "stderr": _raw_summary(stderr), "parsed": records,
              "parse_error": parse_error, "status": status,
              "no_host_write_or_marker": True}
    filename = "OPERATOR_RAW_TRN21_R10_TIMING_" + datetime.datetime.fromisoformat(
        started).strftime("%Y%m%dT%H%M%SZ") + ".json"
    destination = HERE / filename
    encoded = json.dumps(record, sort_keys=True, indent=2).encode() + b"\n"
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        while encoded:
            encoded = encoded[os.write(fd, encoded):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return destination


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
        return 0
    print("HOLD: use --check; capture needs a separate reviewed enablement",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
