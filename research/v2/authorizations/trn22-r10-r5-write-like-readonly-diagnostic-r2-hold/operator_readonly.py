"""Disabled, read-only trn22 r10 write-like diagnostic; no stage authority.

This is a proposed Operator entrypoint, not permission to contact a host.
It requires a *separate*, currently absent enabled request, independent
final review and Supervisor assignment before its first SSH. It never
creates a local marker or a remote file. Its output contains only bounded
typed summaries and hashes/sizes of SSH streams, never raw host output.
"""

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
STAGE = HERE.parent / "trn22-c5c-a0-r10-incremental-stage-r1-hold"
STAGE_REMOTE = STAGE / "remote_incremental.py"
STAGE_BUNDLE = STAGE / "e65-to-r10.bundle"
DIAG_REMOTE = HERE / "remote_readonly.py"
ENABLED_REQUEST = HERE / "REQUEST.ENABLED_AFTER_REVIEW.json"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_READONLY_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_READONLY_ASSIGNMENT.json"
CONTROL_ROOT = Path("/home/poojith-devan/Desktop/Trainium")
FORMER_CONTROL_ROOT = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
IDENTITY_FILE = CONTROL_ROOT / "trn2.pem"
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
HOST_FINGERPRINT = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
STAGE_REMOTE_SHA256 = "da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f"
BUNDLE_SHA256 = "b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e"
BUNDLE_BYTES = 109800
PRE_FREE = 13_177_973_911
POST_FREE = 10_737_418_240
MIN_INODES = 10_000
TRAIN_SHA256 = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
R5_RESULT_COMMIT = "b1387247faf1451dbfd04aadf72a821658b92063"
R5_REVIEWED_SOURCE_COMMIT = "ed2ca7fc43d29f6909c001ee97e2875ef8ae4ee7"
R5_RESULT_RECEIPT_SHA256 = "58461a36b0f6c0a9901be40139cf3da75d65f5c08727f55f16e2fd1ad20d9dc6"
BASE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TARGET_COMMIT = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"
R5_JOURNAL_SHA256 = "229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8"
CONSUMED_MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
                       "trn22-c5c-a0-r10-incremental-stage-r1.one-attempt.jsonl")
R5_RESULT_RECEIPT = REPO / "research/v2/verification/20260925-trn22-r10-r5-one-shot-stage-result-independent-HOLD.md"
R5_ARCHIVED_JOURNAL = REPO / "research/v2/observations/20260925-trn22-r10-stage-r5-operator-result-hold-r1/RAW_ONE_ATTEMPT.jsonl"
LABEL = "fs-20260925-trn22-c5c-a0-r10-readonly-write-like-diagnostic-r2"
SCOPE = "TRN22_R10_R5_READONLY_WRITE_LIKE_DIAGNOSTIC_ONLY"
MAX_STREAM = 2 * 1024 * 1024
REMOTE_TIMEOUT_SECONDS = 320
LOCAL_TIMEOUT_SECONDS = 350
PHASES = ("preflight", "write_like", "postflight")
GATES = ("HOST_LOCK", "FULL_PROBE", "PREFLIGHT_STABLE",
         "TARGET_BUNDLE_ABSENT", "BUNDLE_STDIN_IDENTITY", "ROOT_READONLY_FD",
         "FINAL_NO_WRITE_CONTINUITY", "WOULD_OPEN_FRESHNESS", "ROOT_FD_STABLE")
TIMING_FIELDS = ("host_lock_seconds", "full_probe_seconds",
                 "preflight_stable_seconds", "target_bundle_absent_seconds",
                 "bundle_stdin_identity_seconds", "root_readonly_fd_seconds",
                 "final_no_write_continuity_seconds",
                 "would_open_freshness_seconds", "total_seconds")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _regular(path, *, private=False, max_bytes=MAX_STREAM):
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or
            (private and stat.S_IMODE(named.st_mode) & 0o077) or
            named.st_size > max_bytes):
        raise RuntimeError("unsafe local evidence metadata")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("local evidence inode drift")
        data = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data.extend(block)
            if len(data) > max_bytes:
                raise RuntimeError("local evidence unexpectedly large")
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise RuntimeError("local evidence changed during read")
        return bytes(data)
    finally:
        os.close(fd)


def _json(path):
    raw = _regular(path)
    value = json.loads(raw)
    if not isinstance(value, dict) or raw.strip() != _canonical(value):
        raise RuntimeError("authority file is not canonical JSON")
    return value, _sha(raw)


def _package_bytes():
    stage_remote = _regular(STAGE_REMOTE)
    bundle = _regular(STAGE_BUNDLE)
    diag_remote = _regular(DIAG_REMOTE)
    if (_sha(stage_remote) != STAGE_REMOTE_SHA256 or
            len(bundle) != BUNDLE_BYTES or _sha(bundle) != BUNDLE_SHA256):
        raise RuntimeError("pinned stage source or bundle changed")
    if (_sha(_regular(REPO / "train.py")) != TRAIN_SHA256 or
            _sha(_regular(REPO / "prepare.py")) != PREPARE_SHA256):
        raise RuntimeError("local science bytes changed")
    if (_sha(_regular(R5_RESULT_RECEIPT)) != R5_RESULT_RECEIPT_SHA256 or
            _sha(_regular(R5_ARCHIVED_JOURNAL)) != R5_JOURNAL_SHA256):
        raise RuntimeError("r5 independent result evidence changed")
    clean = subprocess.run(
        ["/usr/bin/git", "-C", str(REPO), "status", "--porcelain=v1",
         "--untracked-files=all", "--ignore-submodules=all"],
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
             "GIT_OPTIONAL_LOCKS": "0"},
        capture_output=True, timeout=20, check=False)
    if clean.returncode != 0 or clean.stdout or clean.stderr:
        raise RuntimeError("diagnostic source checkout is not clean")
    return stage_remote, bundle, diag_remote


def _authority(diag_remote_hash, operator_hash):
    request, request_hash = _json(ENABLED_REQUEST)
    expected = {
        "schema": "trn22-r10-r5-readonly-write-like-diagnostic-r2-enabled-v1",
        "dispatch_authorized": True,
        "scope": SCOPE,
        "host": "trn22",
        "hostname": "ip-172-31-32-253",
        "diagnostic_label": LABEL,
        "source_stage_authorized": False,
        "native_training_authorized": False,
        "evaluation_authorized": False,
        "aws_action_authorized": False,
        "cleanup_authorized": False,
        "marker_reset_authorized": False,
        "r5_result_commit": R5_RESULT_COMMIT,
        "r5_reviewed_source_commit": R5_REVIEWED_SOURCE_COMMIT,
        "remote_base_commit": BASE_COMMIT,
        "intended_target_commit": TARGET_COMMIT,
        "r5_journal_sha256": R5_JOURNAL_SHA256,
        "r5_result_receipt_sha256": R5_RESULT_RECEIPT_SHA256,
        "stage_remote_sha256": STAGE_REMOTE_SHA256,
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_bytes": BUNDLE_BYTES,
        "diagnostic_remote_sha256": diag_remote_hash,
        "operator_source_sha256": operator_hash,
        "preflight_phase": "exact-r5-perform-preflight",
        "second_phase": "r5-readonly-would-open-under-flock",
        "postflight_phase": "exact-r5-perform-postflight",
        "max_final_check_age_seconds": 1.0,
    }
    if request != expected:
        raise RuntimeError("no exact enabled read-only diagnostic request")
    review, review_hash = _json(FINAL_REVIEW)
    if review != {
            "schema": "trn22-r10-r5-readonly-write-like-diagnostic-r2-independent-final-v1",
            "verdict": "PASS_READONLY_DIAGNOSTIC_ONLY",
            "request_sha256": request_hash,
            "operator_source_sha256": operator_hash,
            "diagnostic_remote_sha256": diag_remote_hash,
            "stage_remote_sha256": STAGE_REMOTE_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "scope": SCOPE,
            "diagnostic_label": LABEL,
            "source_stage_authorized": False}:
        raise RuntimeError("independent diagnostic review absent or mismatched")
    assignment, _ = _json(ASSIGNMENT)
    if assignment != {
            "schema": "trn22-r10-r5-readonly-write-like-diagnostic-r2-supervisor-v1",
            "assignment_id": "trn22-r10-r5-readonly-write-like-diagnostic-r2",
            "request_sha256": request_hash,
            "review_sha256": review_hash,
            "scope": SCOPE,
            "host": "trn22",
            "diagnostic_label": LABEL,
            "source_stage_authorized": False,
            "native_training_authorized": False}:
        raise RuntimeError("Supervisor read-only assignment absent or mismatched")
    return request_hash, review_hash


def _local_interlock():
    for root in (CONTROL_ROOT, FORMER_CONTROL_ROOT, REPO):
        for suffix in ("research/agentic/STOP", "research/v2/STOP"):
            if os.path.lexists(root / suffix):
                raise RuntimeError("local STOP present")
    state = json.loads(_regular(CONTROL_ROOT / "research/v2/state.json"))
    breakers = state.get("breakers") if isinstance(state, dict) else None
    if (not isinstance(breakers, dict) or
            any(breakers.get(name, {}).get("tripped") is not False
                for name in ("global_correctness", "trn22")) or
            state.get("launch_authorized") is not False or
            state.get("hosts", {}).get("trn22", {}).get("launch_allowed") is not False):
        raise RuntimeError("controller STOP/breaker/launch state is not safe")
    if _sha(_regular(CONSUMED_MARKER, private=True, max_bytes=64 * 1024)) != R5_JOURNAL_SHA256:
        raise RuntimeError("consumed r5 marker audit bytes changed")


def _ssh_identity():
    parent = os.lstat(KNOWN_HOSTS.parent)
    key = os.lstat(IDENTITY_FILE)  # Never read or print the PEM bytes.
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077 or
            not stat.S_ISREG(key.st_mode) or key.st_uid != os.getuid() or
            key.st_nlink != 1 or stat.S_IMODE(key.st_mode) != 0o600):
        raise RuntimeError("SSH identity metadata unsafe")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts bytes changed")
    hostname = ENDPOINT.split("@", 1)[1]
    found = subprocess.run(["/usr/bin/ssh-keygen", "-F", hostname, "-f",
                            str(KNOWN_HOSTS)], capture_output=True,
                           timeout=20, check=False)
    if found.returncode != 0:
        raise RuntimeError("pinned SSH host entry not found")
    entries = [line for line in found.stdout.splitlines()
               if line and not line.startswith(b"#")]
    ed = [line for line in entries if len(line.split()) >= 3 and
          line.split()[1] == b"ssh-ed25519"]
    if len(ed) != 1:
        raise RuntimeError("pinned ED25519 host entry is not unique")
    fp = subprocess.run(["/usr/bin/ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
                        input=ed[0] + b"\n", capture_output=True, timeout=20,
                        check=False)
    if fp.returncode != 0 or HOST_FINGERPRINT.encode("ascii") not in fp.stdout.split():
        raise RuntimeError("trn22 host fingerprint changed")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts changed during identity check")


def _pinned_ssh_argv(remote_command):
    return ["/usr/bin/ssh", "-T", "-F", "/dev/null", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o",
            "UserKnownHostsFile=" + str(KNOWN_HOSTS), "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=10", "-i", str(IDENTITY_FILE), ENDPOINT,
            remote_command]


def _remote_timeout_identity():
    """Authenticate the *remote* deadline binary before each timed phase.

    This is a distinct, pinned, read-only SSH process; a check inside the
    timed Python child cannot authenticate the binary that started it.
    """
    command = "/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version"
    try:
        version = subprocess.run(_pinned_ssh_argv(command), input=b"",
                                 capture_output=True, timeout=25, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("remote absolute timeout identity probe timed out") from error
    first_line = version.stdout.split(b"\n", 1)[0]
    if (version.returncode != 0 or version.stderr or len(version.stdout) > 4096 or
            re.fullmatch(rb"timeout \(GNU coreutils\) [0-9]+(?:\.[0-9]+){1,3}",
                         first_line) is None):
        raise RuntimeError("remote absolute GNU timeout identity check failed")


def _remote_program(stage_source, diag_source):
    # r5's exact perform('preflight') and perform('postflight') are called.
    # The second branch executes only remote_readonly.perform_readonly.
    return ("import hashlib,json,re,subprocess,sys\n"
            "if sys.flags.optimize: raise SystemExit('optimized Python HOLD')\n"
            "stage={'__name__':'trn22_incremental_remote'}\n"
            "exec(" + repr(stage_source.decode("utf-8")) + ",stage)\n"
            "phase=sys.argv[1]\n"
            "gate='REMOTE_TIMEOUT_IDENTITY'\n"
            "try:\n"
            " version=subprocess.run(['/usr/bin/timeout','--version'],"
            "capture_output=True,timeout=10,check=False)\n"
            " first=version.stdout.split(b'\\n',1)[0]\n"
            " if (version.returncode!=0 or version.stderr or len(version.stdout)>4096"
            " or re.fullmatch(rb'timeout \\(GNU coreutils\\) [0-9]+(?:\\.[0-9]+){1,3}',"
            "first) is None):\n"
            "  raise RuntimeError('remote absolute GNU timeout identity failed')\n"
            " if phase=='write_like':\n"
            "  gate='READONLY_WRAPPER'\n"
            "  diag={'__name__':'trn22_readonly_diagnostic_remote'}\n"
            "  exec(" + repr(diag_source.decode("utf-8")) + ",diag)\n"
            "  result=diag['perform_readonly'](json.loads(sys.argv[2]),"
            "sys.stdin.buffer.read(" + str(BUNDLE_BYTES + 1) + "),stage)\n"
            " else:\n"
            "  gate='EXACT_R5_PERFORM'\n"
            "  result=stage['perform'](phase)\n"
            "  result={'schema':'trn22-r10-r5-readonly-phase-v1','phase':phase,"
            "'status':'PASS_NO_WRITE','result':result}\n"
            "except Exception as error:\n"
            " message=str(error).encode('utf-8','backslashreplace')\n"
            " failure={'phase':phase,'status':'FAIL_NO_WRITE','gate':gate,"
            "'error_type':type(error).__name__[:64],"
            "'error_sha256':hashlib.sha256(message).hexdigest(),"
            "'error_bytes':len(message)}\n"
            " if phase=='write_like':\n"
            "  failure.update({'schema':'trn22-r10-r5-write-like-readonly-diagnostic-v1',"
            "'passed_gates':[],'timings':{},'reason_class':'OTHER',"
            "'source_stage_authorized':False,'native_training_authorized':False,"
            "'evaluation_authorized':False})\n"
            " else:\n"
            "  failure['schema']='trn22-r10-r5-readonly-phase-v1'\n"
            " result=failure\n"
            "print(json.dumps(result,sort_keys=True,separators=(',',':'),"
            "allow_nan=False))\n")


def _stream_summary(result):
    return {"returncode": result.returncode,
            "stdout_bytes": len(result.stdout),
            "stdout_sha256": _sha(result.stdout),
            "stderr_bytes": len(result.stderr),
            "stderr_sha256": _sha(result.stderr)}


def _safe_float_fields(value, names, *, ceiling):
    if not isinstance(value, dict):
        raise ValueError("remote timing object malformed")
    out = {}
    for name in names:
        if name in value:
            number = value[name]
            if (type(number) not in (int, float) or not math.isfinite(number) or
                    not 0 <= number <= ceiling):
                raise ValueError("remote timing value malformed")
            out[name] = number
    return out


def _safe_error_fields(value, *, with_reason):
    name = value.get("error_type")
    digest = value.get("error_sha256")
    count = value.get("error_bytes")
    if (type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", name) is None or
            type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or
            type(count) is not int or not 0 <= count <= 1_000_000):
        raise ValueError("remote failure fields malformed")
    out = {"error_type": name, "error_sha256": digest, "error_bytes": count}
    if with_reason:
        reason = value.get("reason_class")
        if reason not in ("STOP", "FRESHNESS", "LABEL_OR_SCAN", "ACTIVITY",
                          "SPACE", "PREFLIGHT_STABILITY", "TARGET_OR_BUNDLE",
                          "SOURCE_OR_PROTECTED", "LOCK", "OTHER"):
            raise ValueError("remote reason class malformed")
        out["reason_class"] = reason
    return out


def _snapshot_valid(snapshot, *, minimum):
    if (not isinstance(snapshot, dict) or
            snapshot.get("schema") != "trn22-r10-incremental-full-readonly-v1" or
            snapshot.get("hostname") != "ip-172-31-32-253" or
            snapshot.get("uid") != 1000 or
            not isinstance(snapshot.get("boot_id"), str) or
            not isinstance(snapshot.get("mount"), dict) or
            not isinstance(snapshot.get("lock"), dict) or
            not isinstance(snapshot.get("protected"), dict) or
            not isinstance(snapshot.get("process_identity"), dict)):
        return False
    source = snapshot.get("source")
    space = snapshot.get("identity_after")
    activity = snapshot.get("activity")
    label = snapshot.get("label")
    if (not isinstance(source, dict) or source.get("commit") != BASE_COMMIT or
            source.get("science_sha256") != {"train.py": TRAIN_SHA256,
                                              "prepare.py": PREPARE_SHA256} or
            not isinstance(space, dict) or
            type(space.get("root_free_bytes")) is not int or
            space["root_free_bytes"] < minimum or
            type(space.get("root_free_inodes")) is not int or
            space["root_free_inodes"] < MIN_INODES or
            not isinstance(activity, dict) or
            activity.get("neuron_holder_absent") is not True or
            activity.get("stops_absent") is not True or
            not isinstance(label, dict) or label.get("matches") != []):
        return False
    return True


def _ssh_phase(phase, prior, data, stage_source, diag_source):
    if phase not in PHASES:
        raise RuntimeError("unknown diagnostic phase")
    _local_interlock()
    _ssh_identity()
    _local_interlock()
    _remote_timeout_identity()
    _local_interlock()  # STOP or breaker after the separate SSH probe.
    program = _remote_program(stage_source, diag_source)
    command = ["/usr/bin/timeout", "--foreground", "--signal=TERM",
               "--kill-after=15s", str(REMOTE_TIMEOUT_SECONDS) + "s",
               "/usr/bin/env", "-i",
               "PATH=/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
               "/usr/bin/python3", "-I", "-B", "-c", program, phase,
               json.dumps(prior, sort_keys=True, separators=(",", ":"))]
    argv = _pinned_ssh_argv(" ".join(shlex.quote(item) for item in command))
    _local_interlock()  # Late STOP/breaker before this phase's SSH.
    try:
        result = subprocess.run(argv, input=data, capture_output=True,
                                timeout=LOCAL_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        partial = type("Partial", (), {
            "returncode": None,
            "stdout": error.stdout or b"",
            "stderr": error.stderr or b"",
        })()
        return {"phase": phase, "status": "SSH_TIMEOUT",
                "transport": _stream_summary(partial)}, None
    summary = _stream_summary(result)
    if result.returncode != 0 or result.stderr or len(result.stdout) > MAX_STREAM:
        causes = []
        if result.returncode != 0:
            causes.append("NONZERO_RC")
        if result.stderr:
            causes.append("NONEMPTY_STDERR")
        if len(result.stdout) > MAX_STREAM:
            causes.append("STDOUT_OVERSIZE")
        return {"phase": phase, "status": "SSH_OR_STREAM_FAIL",
                "transport": summary, "failure_modes": causes}, None
    try:
        value = json.loads(result.stdout)
        if (not isinstance(value, dict) or
                result.stdout.strip() != _canonical(value) or
                value.get("phase") != phase):
            raise ValueError("noncanonical or wrong phase receipt")
    except (ValueError, TypeError):
        return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                "transport": summary}, None
    if phase == "write_like":
        if value.get("schema") != "trn22-r10-r5-write-like-readonly-diagnostic-v1":
            return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                    "transport": summary}, None
        status = value.get("status")
        if status not in ("PASS_NO_WRITE", "FAIL_NO_WRITE"):
            return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                    "transport": summary}, None
        try:
            timings = _safe_float_fields(
                value.get("timings"), TIMING_FIELDS,
                ceiling=REMOTE_TIMEOUT_SECONDS)
            gate = value.get("gate")
            passed = value.get("passed_gates")
            special = status == "FAIL_NO_WRITE" and gate in (
                "REMOTE_TIMEOUT_IDENTITY", "READONLY_WRAPPER")
            if (type(passed) is not list or
                    (special and passed != []) or
                    (not special and (gate not in GATES or
                     passed != list(GATES[:len(passed)]) or len(passed) > len(GATES)))):
                raise ValueError("remote gate malformed")
            if status == "PASS_NO_WRITE" and (
                    gate != GATES[-1] or passed != list(GATES) or
                    len(timings) != len(TIMING_FIELDS)):
                raise ValueError("remote pass proof incomplete")
        except ValueError:
            return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                    "transport": summary}, None
        detail = {"phase": phase, "status": status, "transport": summary,
                  "gate": gate, "passed_gates": passed, "timings": timings}
        if status == "PASS_NO_WRITE":
            try:
                fresh = value.get("freshness")
                ages = _safe_float_fields(
                    fresh,
                    ("activity_age_seconds", "label_age_seconds",
                     "space_age_seconds", "label_scan_duration_seconds"),
                    ceiling=1.0)
                if (len(ages) != 4 or fresh.get("double_label_scan") is not True or
                        fresh.get("post_label_activity_check") is not True):
                    raise ValueError("freshness proof malformed")
                detail["freshness"] = {**ages, "double_label_scan": True,
                                       "post_label_activity_check": True}
                if not _snapshot_valid(value.get("snapshot"), minimum=PRE_FREE):
                    raise ValueError("remote snapshot malformed")
            except ValueError:
                return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                        "transport": summary}, None
        else:
            try:
                detail.update(_safe_error_fields(value, with_reason=True))
            except ValueError:
                return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                        "transport": summary}, None
    else:
        if (value.get("schema") != "trn22-r10-r5-readonly-phase-v1" or
                value.get("status") not in ("PASS_NO_WRITE", "FAIL_NO_WRITE")):
            return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                    "transport": summary}, None
        detail = {"phase": phase, "status": value["status"],
                  "transport": summary}
        if detail["status"] == "FAIL_NO_WRITE":
            try:
                if value.get("gate") not in ("EXACT_R5_PERFORM", "REMOTE_TIMEOUT_IDENTITY"):
                    raise ValueError("remote pre/post failure gate malformed")
                detail["gate"] = value["gate"]
                detail.update(_safe_error_fields(value, with_reason=False))
            except ValueError:
                return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                        "transport": summary}, None
        else:
            payload = value.get("result")
            if (not isinstance(payload, dict) or payload.get("phase") != phase or
                    not isinstance(payload.get("snapshot"), dict)):
                return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                        "transport": summary}, None
            snap = payload["snapshot"]
            if not _snapshot_valid(
                    snap, minimum=PRE_FREE if phase == "preflight" else POST_FREE):
                return {"phase": phase, "status": "MALFORMED_REMOTE_RECEIPT",
                        "transport": summary}, None
            source = snap["source"]
            space = snap["identity_after"]
            detail["snapshot_sha256"] = _sha(_canonical(snap))
            detail["source_commit"] = source.get("commit")
            detail["bundle_exists"] = snap.get("bundle_exists")
            detail["target_exists"] = snap.get("target_exists")
            detail["root_space"] = {
                "root_free_bytes": space["root_free_bytes"],
                "root_free_inodes": space["root_free_inodes"]}
    return detail, value


def _finiteness(value):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0)


def _write_like_receipt_valid(value, preflight):
    if (not isinstance(value, dict) or value.get("status") != "PASS_NO_WRITE" or
            value.get("source_stage_authorized") is not False or
            value.get("native_training_authorized") is not False or
            value.get("evaluation_authorized") is not False):
        return False
    snap = value.get("snapshot", {})
    prior = preflight.get("result", {}).get("snapshot", {})
    if (not _snapshot_valid(snap, minimum=PRE_FREE) or
            not _snapshot_valid(prior, minimum=PRE_FREE)):
        return False
    if snap.get("process_identity") == prior.get("process_identity"):
        return False
    for key in ("hostname", "uid", "boot_id", "mount", "lock", "source", "protected"):
        if snap.get(key) != prior.get(key):
            return False
    fresh = value.get("freshness", {})
    if not isinstance(fresh, dict):
        return False
    for key in ("activity_age_seconds", "label_age_seconds", "space_age_seconds",
                "label_scan_duration_seconds"):
        if not _finiteness(fresh.get(key)) or fresh[key] > 1.0:
            return False
    if (fresh.get("double_label_scan") is not True or
            fresh.get("post_label_activity_check") is not True):
        return False
    return (snap.get("bundle_exists") is False and
            snap.get("target_exists") is False)


def run_diagnostic(remote_call=_ssh_phase):
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python forbidden")
    # The enabled request and its review/assignment are currently absent.
    # This refusal precedes any SSH, marker, or remote process.
    stage_source, bundle, diag_source = _package_bytes()
    _authority(_sha(diag_source), _sha(_regular(Path(__file__))))
    _local_interlock()
    _ssh_identity()
    _local_interlock()

    phases = []
    preflight = diagnostic = postflight = None
    try:
        first, preflight = remote_call("preflight", None, b"", stage_source, diag_source)
        phases.append(first)
        if first["status"] == "PASS_NO_WRITE":
            prior = preflight["result"]["snapshot"]
            if prior.get("bundle_exists") or prior.get("target_exists"):
                raise RuntimeError("preflight found stage target/bundle")
            second, diagnostic = remote_call("write_like", prior, bundle,
                                             stage_source, diag_source)
            phases.append(second)
            if second["status"] == "PASS_NO_WRITE" and not _write_like_receipt_valid(
                    diagnostic, preflight):
                phases[-1] = {**second, "status": "INVALID_CROSS_SESSION_PROOF"}
    except Exception as error:
        message = str(error).encode("utf-8", "backslashreplace")
        phases.append({"phase": "local", "status": "LOCAL_HOLD",
                       "error_type": type(error).__name__[:64],
                       "error_sha256": _sha(message), "error_bytes": len(message)})
    finally:
        # A separate read-only postflight is attempted after any dispatched
        # diagnostic failure. STOP/breaker/identity checks still apply.
        try:
            last, postflight = remote_call("postflight", None, b"",
                                           stage_source, diag_source)
            phases.append(last)
        except Exception as error:
            message = str(error).encode("utf-8", "backslashreplace")
            phases.append({"phase": "postflight", "status": "LOCAL_HOLD",
                           "error_type": type(error).__name__[:64],
                           "error_sha256": _sha(message),
                           "error_bytes": len(message)})
    okay = (len(phases) == 3 and
            [item.get("phase") for item in phases] == list(PHASES) and
            all(item.get("status") == "PASS_NO_WRITE" for item in phases))
    if okay:
        first_snap = preflight["result"]["snapshot"]
        second_snap = diagnostic["snapshot"]
        last_snap = postflight["result"]["snapshot"]
        if (first_snap.get("process_identity") == last_snap.get("process_identity") or
                second_snap.get("process_identity") == last_snap.get("process_identity") or
                any(first_snap.get(key) != last_snap.get(key) for key in
                    ("hostname", "uid", "boot_id", "mount", "lock", "source", "protected")) or
                last_snap.get("bundle_exists") is not False or
                last_snap.get("target_exists") is not False):
            okay = False
    return {"schema": "trn22-r10-r5-readonly-write-like-result-v1",
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "READONLY_DIAGNOSTIC_COMPLETE" if okay else "HOLD",
            "diagnostic_label": LABEL, "scope": SCOPE,
            "r5_result_commit": R5_RESULT_COMMIT,
            "stage_remote_sha256": STAGE_REMOTE_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "phases": phases, "source_stage_authorized": False,
            "native_training_authorized": False,
            "evaluation_authorized": False,
            "cleanup_authorized": False}


def main():
    if len(sys.argv) != 1:
        raise SystemExit("HOLD: no ad-hoc arguments")
    try:
        receipt = run_diagnostic()
    except Exception as error:
        # No raw exception text, host output, PEM contents, or environment.
        receipt = {"schema": "trn22-r10-r5-readonly-write-like-result-v1",
                   "status": "HOLD_PRE_DISPATCH", "diagnostic_label": LABEL,
                   "error_type": type(error).__name__[:64],
                   "source_stage_authorized": False}
    print(_canonical(receipt).decode("utf-8"))
    return 0 if receipt["status"] == "READONLY_DIAGNOSTIC_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
